import os
import cv2
import h5py
import keras
import logging
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import SGD, RMSprop
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.utils.conv_utils import convert_kernel
from keras.layers.local import LocallyConnected2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout

from data import Database
from loss import custom_loss_2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloNet(object):
    def __init__(self, weights, mode='detection'):
        keras.backend.set_image_dim_ordering('th')
        logger.info("Yolo model initialization...")
        self.order = 'th'
        self.model = Sequential()

        # Mode select
        if mode == 'detection':
            self.init_model_detection()
        elif mode == 'classifier':
            self.init_model_classifier()
        elif mode == 'darknet19':
            self.init_model_darknet19()
        else:
            raise ValueError("Not implemented mode!")

        # Load ImageNet Labels
        dir_path = os.path.dirname(os.path.realpath(__file__))
        im_net_labels = os.path.join(dir_path, "..", "data", "imagenet.labels.list")
        with open(im_net_labels, 'r') as imagenet_labels:
            self.inet_lab = imagenet_labels.readlines()

        # Load ImageNet Names
        im_net_names = os.path.join(dir_path, "..", "data", "imagenet.shortnames.list")
        with open(im_net_names, 'r') as imagenet_names:
            self.inet_nm = imagenet_names.readlines()

        # Load YOLO weights
        logger.info("Loading weights for Convo features...")
        self.load_weights(weights, len(self.model.layers))
        # self.load_weights(weights, 44)

    def init_model_detection(self, v1_version=True):
        channel_axis = 1 if self.order == "th" else -1
        self.model.add(Conv2D(filters=64, kernel_size=(7, 7), input_shape=(3, 448, 448), border_mode='same',
                              strides=2, use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))

        # Detection model
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version, strides=2))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=v1_version))
        if not v1_version:
            self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))

        # Last layers
        if v1_version:
            self.model.add(Flatten())
            self.model.add(Dense(4096))
            self.model.add(LeakyReLU(alpha=0.1))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1470, activation='linear'))
        else:
            # Locally connected
            self.model.add(LocallyConnected2D(filters=256, kernel_size=(3, 3), border_mode='valid', use_bias=True))
            self.model.add(LeakyReLU(alpha=0.1))
            self.model.add(Dropout(0.5))
            self.model.add(Flatten())
            self.model.add(Dense(1715))
            self.model.add(LeakyReLU(alpha=0.1))

    def init_model_classifier(self):
        channel_axis = 1 if self.order == "th" else -1
        self.model.add(Conv2D(filters=64, kernel_size=(7, 7), input_shape=(3, 224, 224), border_mode='same',
                              strides=2, use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))

        # Classifier
        # self.model.add(Conv2D(filters=1000, kernel_size=(3, 3), border_mode='same'))
        # self.model.add(LeakyReLU(alpha=0.1))
        # self.model.add(GlobalAveragePooling2D())
        # self.model.add(Activation('softmax'))

    def init_model_darknet19(self):
        channel_axis = 1 if self.order == "th" else -1
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(3, 448, 448), border_mode='same',
                              strides=1, use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=64, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', use_bias=False))
        self.model.add(BatchNormalization(axis=channel_axis))
        self.model.add(LeakyReLU(alpha=0.1))

        # Classifier
        self.model.add(Conv2D(filters=1000, kernel_size=(1, 1), border_mode='same', activation='linear'))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Activation('softmax'))

    def load_weights(self, yolo_weight_file, weight_num, number_of_layers_to_load=10000):
        # Read header
        weights_file = open(yolo_weight_file, 'rb')
        weights_header = np.ndarray(shape=(4,), dtype='int32', buffer=weights_file.read(16))
        logger.info("Weights header: {}".format(weights_header))

        # Read weights
        layer_index = 0
        prev_layer = self.model.layers[0].batch_input_shape[1]
        while layer_index < weight_num and layer_index < number_of_layers_to_load:
            layer = self.model.layers[layer_index]
            # Loading weights for convolution layer
            if isinstance(layer, Conv2D):
                # Check Batch Normalisation
                shape = [(layer.kernel_size[0], layer.kernel_size[0], prev_layer, layer.filters)]
                batch_normalize = isinstance(self.model.layers[layer_index + 1], BatchNormalization)
                w_shape = shape[0]

                # Read Bias weights
                conv_bias = np.ndarray(
                    shape=(w_shape[-1],),
                    dtype='float32',
                    buffer=weights_file.read(w_shape[-1] * 4)
                )

                if batch_normalize:
                    # Read batch normalisation weights
                    bn_weights = np.ndarray(
                        shape=(3, w_shape[-1]),
                        dtype='float32',
                        buffer=weights_file.read(w_shape[-1] * 12))
                    bn_weight_list = [
                        bn_weights[0],
                        conv_bias,
                        bn_weights[1],
                        bn_weights[2]
                    ]

                # Read convolution weights
                conv_weights = np.ndarray(
                    shape=(w_shape[-1], w_shape[2], w_shape[0], w_shape[1]),
                    # shape=(w_shape[-1], w_shape[2], w_shape[1], w_shape[0]), #theano
                    dtype='float32',
                    buffer=weights_file.read(np.product(w_shape) * 4))
                conv_weights = np.transpose(conv_weights, (3, 2, 1, 0))
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]
                if batch_normalize:
                    self.model.layers[layer_index + 1].set_weights(bn_weight_list)
                layer.set_weights(conv_weights)
                prev_layer = layer.filters
            # Loading weights for dense layer
            if isinstance(layer, Dense):
                # Read Dense Bias
                dense_bias = np.ndarray(
                    shape=(layer.output_shape[1],),
                    dtype='float32',
                    buffer=weights_file.read(layer.output_shape[1] * 4)
                )

                # Read convolution weights
                dense_weights = [layer.input_shape[1], layer.output_shape[1]]
                dense_weights = np.ndarray(
                    shape=(dense_weights[0], dense_weights[1]),
                    dtype='float32',
                    buffer=weights_file.read(np.product(dense_weights) * 4))
                layer.set_weights([dense_weights, dense_bias])
            layer_index += 1
        remaining_weights = len(weights_file.read()) / 4
        logger.info("Remaining weights {}".format(remaining_weights))
        weights_file.close()

        if K.backend() == 'theano':
            for layer in self.model.layers:
                if layer.__class__.__name__ in ['Conv2D']:
                    original_w = K.get_value(layer.kernel)
                    converted_w = convert_kernel(original_w)
                    K.set_value(layer.kernel, converted_w)

    def model_info(self):
        # plot_model(self.model, to_file='model.png', show_shapes=True)
        self.model.summary()

    def classify(self, image):
        img_resized = cv2.resize(image, (448, 448))
        image_data = np.array(img_resized, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)
        out = self.model.predict(image_data)
        return np.argsort(out[0])[::-1][:5], out

    def show_results(self, image):
        best_obj, score = self.classify(image)
        for obj_idx in best_obj:
            print("{}| {}: {}".format(self.inet_lab[obj_idx],
                                      self.inet_nm[obj_idx],
                                      score[0][obj_idx]))

    def predict(self, image):
        img_resized = cv2.resize(image, (448, 448))
        image_data = np.array(img_resized, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)
        out = self.model.predict(image_data)
        boxes_list = self.yolo2boxes(out)
        self.plot_boxes(boxes_list, image)

    def plot_boxes(self, boxes, image):
        clsss_dict = {
            0: "aeroplane",
            1: "bicycle",
            2: "bird",
            3: "boat"
        }
        # Iterate over boxes
        for box in boxes:
            h, w, _ = image.shape
            left = int((box['x'] - box['width'] / 2.) * w)
            right = int((box['x'] + box['width'] / 2.) * w)
            top = int((box['y'] - box['width'] / 2.) * h)
            bot = int((box['y'] + box['width'] / 2.) * h)
            # print(clsss_dict[box['class']])
            cv2.rectangle(image, (left, top), (right, bot), (255, 0, 0), 2)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def yolo2boxes(net_out, threshold=0.3, sqrt=1.8, classes=20, boxes=2, cells=7):
        # Define parameters
        boxes_final = []
        all_cells = cells * cells
        prob_size = all_cells * classes
        conf_size = all_cells * boxes

        # Reconstruct output
        probs = net_out[0, 0:prob_size]
        confs = net_out[0, prob_size:(prob_size + conf_size)]
        cords = net_out[0, (prob_size + conf_size):]
        probs = probs.reshape([all_cells, classes])
        confs = confs.reshape([all_cells, boxes])
        cords = cords.reshape([all_cells, boxes, 4])

        # Create dictionaries
        for grid in range(all_cells):
            for b in range(boxes):
                bx = {}
                bx['grid'] = grid
                bx['box_num'] = b
                bx['confidence'] = confs[grid, b]
                bx['x'] = (cords[grid, b, 0] + grid % cells) / cells
                bx['y'] = (cords[grid, b, 1] + grid // cells) / cells
                bx['width'] = cords[grid, b, 2] * sqrt
                bx['height'] = cords[grid, b, 3] * sqrt
                p = probs[grid, :] * bx['confidence']

                for class_num in range(classes):
                    if p[class_num] >= threshold:
                        bx['probability'] = p[class_num]
                        bx['class'] = class_num
                        boxes_final.append(bx)
        return boxes_final

    def learn(self, batch_size):
        # TODO this loss function works only on theano!
        self.model.compile(loss=custom_loss_2, optimizer=SGD(0.0001), metrics=['mae', 'mse'])
        # self.model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.0001), metrics=['accuracy'])
        # freeze first classification layers
        db = Database(name='voc2012')
        samples_per_epoch = db.set_size / batch_size
        for x, y in db.sb_batch_iter(boxes=2, batch_size=1, type='train'):
            img = np.transpose(x[0], (1, 2, 0))
            img = (img * 255).astype(np.uint8).copy()
            boxes_list = self.yolo2boxes(y, sqrt=1)
            self.plot_boxes(boxes_list, img)
            print()
        # self.model.fit_generator(db.sb_batch_iter(boxes=2, batch_size=batch_size, type='train'),
        #                          validation_data=db.sb_batch_iter(boxes=2, batch_size=batch_size, type='val'),
        #                          validation_steps=5 * batch_size,
        #                          samples_per_epoch=samples_per_epoch,
        #                          nb_epoch=500)
        # # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("../data/model_file.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # self.model.save_weights("../data/model_weights.h5")
        # logger.info("Saved model to disk")


if __name__ == '__main__':
    yn = YoloNet(mode='detection', weights='../data/yolo-full.weights')
    # img = cv2.imread('../data/bird.jpg', 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # yn.model_info()
    # print(yn.predict(img))
    yn.learn(batch_size=8)
