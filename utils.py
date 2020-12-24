from __future__ import print_function
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np
from deeplabv3p import Deeplabv3
import os
import multiprocessing
workers = multiprocessing.cpu_count()//2
import tensorflow as tf

size_def = (576,576)

if tf.__version__[0] == "2":
    _IS_TF_2 = True
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
    from tensorflow.keras.layers import *
    from subpixel import *
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.client import device_lib
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import to_categorical
else:
    _IS_TF_2 = False
    import keras
    import keras.backend as K
    from keras.utils.data_utils import Sequence
    from keras.optimizers import Adam, SGD, RMSprop
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
    from keras.layers import *
    from subpixel import *
    from keras.models import Model, Sequential
    from keras.callbacks import TensorBoard
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.client import device_lib
    from keras.regularizers import l2
    from keras.utils import to_categorical
    
from collections import Counter

from sklearn.utils import class_weight
import cv2
import glob
import random
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import itertools

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)
    return cm

# Fully connected CRF post processing function
def do_crf(im, mask, zero_unsure=True):
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=.7, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5) # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    for u in unique_map: # get original labels back
        np.putmask(MAP, MAP == u, colors[u])
    return MAP
    # MAP = do_crf(frame, labels.astype('int32'), zero_unsure=False)
    
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    

def get_VOC2012_classes():
    PASCAL_VOC_classes = {
        0: 'background', 
        1: 'airplane',
        2: 'bicycle',
        3: 'bird', 
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted_plant',
        17: 'sheep',
        18: 'sofa',
        19 : 'train',
        20 : 'tv',
        21 : 'void'
    }
    return PASCAL_VOC_classes

def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(tf.cast(y_true[:,:,0], tf.int32), nb_classes+1)[:,:,:-1]
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.flatten(y_true), tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    value = K.sum(tf.cast(legal_labels & K.equal(y_true, K.argmax(y_pred, axis=-1)), tf.float32))
    divisor = K.sum(tf.cast(legal_labels, tf.float32))
    return value / divisor
def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)

        tf.cast(y_true[:, :, 0], tf.int32)

        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        if _IS_TF_2:
            iou.append(K.mean(ious[legal_batches]))
        else:
            iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou) if _IS_TF_2 else ~tf.debugging.is_nan(iou)
    iou = iou[legal_labels] if _IS_TF_2 else tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

        
class SegModel:
    epochs = 20
    batch_size = 16
    def __init__(self, dataset='VOCdevkit/VOC2012', image_size=size_def):
        self.sz = image_size
        self.mainpath = dataset
        self.crop = False
            
    
    def create_seg_model(self, net, n=4, backbone='mobilenetv2', load_weights=False, multi_gpu=False):
        
        '''
        Net is:
        1. original deeplab v3+
        2. original deeplab v3+ and subpixel upsampling layer
        '''
        
        model = Deeplabv3(weights=None, input_tensor=None, infer=False,
                          input_shape=self.sz + (3,), classes=4,
                          backbone=backbone, OS=16, alpha=1)
        
        base_model = Model(model.input, model.layers[-5].output)
        self.net = net
        self.modelpath = 'weights/{}_{}.h5'.format(backbone, net)
        if backbone=='xception':
            scale = 4
        else:
            scale = 8
        if net == 'original':
            x = Conv2D(n, (1, 1), padding='same', name='conv_upsample')(base_model.output)
            x = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x,size=(self.sz[0],self.sz[1])))(x)
            x = Reshape((self.sz[0]*self.sz[1], -1)) (x)
            x = Activation('softmax', name = 'pred_mask')(x)
            model = Model(base_model.input, x, name='deeplabv3p')
        elif net == 'subpixel':
            x = Subpixel(n, 1, scale, padding='same')(base_model.output)
            x = Reshape((self.sz[0]*self.sz[1], -1)) (x)
            x = Activation('softmax', name = 'pred_mask')(x)
            model = Model(base_model.input, x, name='deeplabv3p_subpixel')
        # Do ICNR
        for layer in model.layers:
            if type(layer) == Subpixel:
                c, b = layer.get_weights()
                w = icnr_weights(scale=scale, shape=c.shape)
                layer.set_weights([w, b])
                
        if load_weights:
            model.load_weights('weights/{}_{}.h5'.format(backbone, net))

        if multi_gpu:
            from keras.utils import multi_gpu_model
            model = multi_gpu_model(model, gpus = len(get_available_gpus()))
            
        self.model = model
        return model

    def create_generators(self, crop_shape=False, mode='train', do_ahisteq=True, n_classes=4, horizontal_flip=True,
                          vertical_flip=False, blur=False, with_bg=True, brightness=0.1, rotation=5.0, 
                          zoom=0.1, validation_split=.2, seed=7):
                
        generator = SegmentationGenerator(folder = self.mainpath, mode = mode, n_classes = n_classes, do_ahisteq = do_ahisteq,
                                       batch_size=self.batch_size, resize_shape=self.sz[::-1], crop_shape=crop_shape, 
                                       horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, blur = blur,
                                       brightness=brightness, rotation=rotation, zoom=zoom,
                                       validation_split = validation_split, seed = seed)
                
        return generator

    def load_weights(self, model):
        model.load_weights(self.modelpath)
        
    def train_generator(self, model, train_generator, valid_generator, callbacks, mp = True):
        steps = len(train_generator)
        h = model.fit_generator(train_generator,
                                steps_per_epoch=steps, 
                                epochs = self.epochs, verbose=1, 
                                callbacks = callbacks, 
                                validation_data=valid_generator, 
                                validation_steps=len(valid_generator), 
                                max_queue_size=10, 
                                workers=workers, use_multiprocessing=mp)
        return h
    
    def train(self, model, X, y, val_data, tf_board = False, plot_train_process = True):
        h = model.fit(X, y, validation_data = val_data, verbose=1, 
                      batch_size = self.batch_size, epochs = self.epochs, 
                      callbacks = self.build_callbacks(tf_board = tf_board, plot_process = plot_train_process))
        return h
    
    @classmethod
    def set_num_epochs(cls, new_epochs):
        cls.epochs = new_epochs
    @classmethod
    def set_batch_size(cls, new_batch_size):
        cls.batch_size = new_batch_size

from PIL import Image
import numpy as np
def get_shape_img(path_img: str):
    return np.asarray(Image.open(path_img)).shape
    
class SegmentationGenerator(Sequence):
    path_dataset = '/content/drive/MyDrive/dataset/576x576/custom_576x576_75_15_10_transf-morf_voc_cor'
    def __init__(
            self, folder='path_dataset', mode='train', n_classes=4, batch_size=1, resize_shape=None,
            validation_split = .1, seed = 7, crop_shape=size_def, horizontal_flip=True, blur = 0,
            vertical_flip=0, brightness=0.1, rotation=5.0, zoom=0.1, do_ahisteq = True, img_ext='jpeg',
            mask_ext='png'
    ):
        
        self.blur = blur
        self.histeq = do_ahisteq

        ids = []
        if (mode == 'train'):
            with open(os.path.join(self.path_dataset, 'ImageSets', 'train.txt'), 'r') as file_train:
                ids = [l.strip() for l in file_train.readlines()]
        elif (mode.lower() == 'val' or mode.lower() == 'validation'):
            with open(os.path.join(self.path_dataset, 'ImageSets', 'val.txt'), 'r') as file_train:
                ids = [l.strip() for l in file_train.readlines()]
        self.image_path_list = sorted([
            os.path.join(self.path_dataset, 'JPEGImages', f'{id}.{img_ext}') for id in ids
        ])
        self.label_path_list = sorted([
            os.path.join(self.path_dataset, 'SegmentationClass', f'{id}.{mask_ext}') for id in ids
        ])
        np.random.seed(seed)
        
        n_images_to_select = round(len(self.image_path_list) * validation_split)
        x = np.random.permutation(len(self.image_path_list))[:n_images_to_select]
        if mode == 'train':
            x = np.setxor1d(x, np.arange(len(self.image_path_list)))
            
        self.image_path_list = [self.image_path_list[j] for j in x]
        self.label_path_list = [self.label_path_list[j] for j in x]
        
        if mode.lower() == 'test' or mode.lower() == 'validation' or mode.lower() == 'val':
            ids_test_images = []
            with open(os.path.join(self.path_dataset, 'ImageSets', 'val.txt'), 'r') as file_train:
                ids_test_images = [l.strip() for l in file_train.readlines()]
            self.image_path_list = sorted([
                os.path.join(self.path_dataset, 'JPEGImages', f'{id}.{img_ext}')
                for id in ids_test_images
            ])[:100]

        print('LEN IMGs', len(self.image_path_list), self.image_path_list[0], get_shape_img(self.image_path_list[0]))
        print('LEN MASKs', len(self.label_path_list), self.label_path_list[0], get_shape_img(self.label_path_list[0]))

        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom
        # Preallocate memory
        if self.crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.SW = np.zeros((batch_size, crop_shape[1]*crop_shape[0]), dtype='float32')
            self.Y = np.zeros((batch_size, crop_shape[1]*crop_shape[0], 1), dtype='float32')
            self.F = np.zeros((batch_size, crop_shape[1]*crop_shape[0], 1), dtype='float32')
            self.F_SW = np.zeros((batch_size, crop_shape[1]*crop_shape[0]), dtype='float32')
        elif self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.SW = np.zeros((batch_size, resize_shape[1]*resize_shape[0]), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='float32')
            self.F = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='float32')
            self.F_SW = np.zeros((batch_size, resize_shape[1]*resize_shape[0]), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):
        
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            labels = np.unique(label)
            label = label.astype('int32')
            for j in np.setxor1d(np.unique(label), labels):
                label[label==j] = self.n_classes
            
            y = label.flatten()
            y[y>(self.n_classes-1)]=self.n_classes
                            
            self.Y[n]  = np.expand_dims(y, -1)
            self.F[n]  = (self.Y[n]!=0).astype('float32') # get all pixels that aren't background
            valid_pixels = self.F[n][self.Y[n]!=self.n_classes] # get all pixels (bg and foregroud) that aren't void
            u_classes = np.unique(valid_pixels)
            class_weights = class_weight.compute_class_weight('balanced', u_classes, valid_pixels)
            class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
            if len(class_weights)==1: # no bg\no fg
                if 1 in u_classes:
                    class_weights[0] = 0.
                else:
                    class_weights[1] = 0.
            elif not len(class_weights):
                class_weights[0] = 0.
                class_weights[1] = 0.
        
            sw_valid = np.ones(y.shape)
            np.putmask(sw_valid, self.Y[n]==0, class_weights[0]) # background weights
            np.putmask(sw_valid, self.F[n], class_weights[1]) # foreground wegihts 
            np.putmask(sw_valid, self.Y[n]==self.n_classes, 0)
            self.F_SW[n] = sw_valid
            self.X[n] = image    
        
            # Create adaptive pixels weights
            filt_y = y[y!=self.n_classes]
            u_classes = np.unique(filt_y)
            if len(u_classes):
                class_weights = class_weight.compute_class_weight('balanced', u_classes, filt_y)
                class_weights = {class_id : w for class_id, w in zip(u_classes, class_weights)}
            class_weights[self.n_classes] = 0.
            for yy in u_classes:
                np.putmask(self.SW[n], y==yy, class_weights[yy])
                
            np.putmask(self.SW[n], y==self.n_classes, 0)

        sample_dict = {'pred_mask' : self.SW}
        return self.X, self.Y, sample_dict
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)
                
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)
        return image, label
       
