import cv2
import glob
import keras
import logging
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D

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
        with open('../data/imagenet.labels.list', 'r') as imagenet_labels:
            self.inet_lab = imagenet_labels.readlines()
        # Load ImageNet Names
        with open('../data/imagenet.shortnames.list', 'r') as imagenet_names:
            self.inet_nm = imagenet_names.readlines()

        # Load YOLO weights
        logger.info("Loading weights for Convo features...")
        self.load_weights(weights, len(self.model.layers))

    def init_model_detection(self):
        self.model.add(Conv2D(filters=64, kernel_size=(7, 7), input_shape=(3, 448, 448), border_mode='same', strides=2))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=128, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='valid'))

        self.model.add(Conv2D(filters=256, kernel_size=(1,1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))

        # end of pretrained model
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=(3, 3), border_mode='same', strides=2))
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

    def load_weights(self, yolo_weight_file, weight_num):
        # Read header
        weights_file = open(yolo_weight_file, 'rb')
        weights_header = np.ndarray(shape=(4, ), dtype='int32', buffer=weights_file.read(16))
        logger.info("Weights header: {}".format(weights_header))

        # Read weights
        layer_index = 0
        while layer_index < weight_num:
            layer = self.model.layers[layer_index]
            if isinstance(layer, Conv2D):
                # Check Batch Normalisation
                shape = [w.shape for w in layer.get_weights()]
                batch_normalize = isinstance(self.model.layers[layer_index + 1], BatchNormalization)
                if not batch_normalize:
                    print()
                w_shape = shape[0]

                # Read Bias weights
                conv_bias = np.ndarray(
                    shape=(w_shape[-1], ),
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
                    dtype='float32',
                    buffer=weights_file.read(np.product(w_shape) * 4))
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [
                    conv_weights, conv_bias
                ]
                if batch_normalize:
                    self.model.layers[layer_index + 1].set_weights(bn_weight_list)
                layer.set_weights(conv_weights)
            layer_index += 1
        remaining_weights = len(weights_file.read()) / 4
        logger.info("Remaining weights {}".format(remaining_weights))
        weights_file.close()

    def model_info(self):
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

if __name__ == '__main__':
    yn = YoloNet(mode='darknet19', weights='../data/darknet19_448.weights')
    img = cv2.imread('../data/dog.jpg', 1)
    yn.model_info()
    yn.show_results(img)
