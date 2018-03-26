from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Activation, Convolution2D, MaxPooling2D, Lambda, Input
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras import backend as K
import tensorflow as tf
import os

class CODEC:
    def __init__(self, img_size, num_channels, compress_mode=1, clip_value=0.5, resize=None):

        self.compress_mode = compress_mode
        working_img_size = img_size

        encoder_model = Sequential()
        # resize input to a size easy for down-sampled
        if resize:
            encoder_model.add( Lambda(lambda image: tf.image.resize_images(image, (resize, resize)), 
            input_shape=(img_size, img_size, num_channels)))
        else:
            encoder_model.add(Convolution2D( 16, 3, strides=1,padding='same', input_shape=(img_size, img_size, num_channels)))
        
        BatchNormalization(axis=3)
        encoder_model.add(Activation("tanh"))
        encoder_model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        working_img_size //= 2

        if compress_mode >=2:
            encoder_model.add(Convolution2D( 16, 3, strides=1,padding='same'))
            BatchNormalization(axis=3)
            encoder_model.add(Activation("tanh"))
            encoder_model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
            working_img_size //= 2

        if compress_mode >=3:
            encoder_model.add(Convolution2D( 16, 3, strides=1,padding='same'))
            BatchNormalization(axis=3)
            encoder_model.add(Activation("tanh"))
            encoder_model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
            working_img_size //= 2

        encoder_model.add(Convolution2D(num_channels, 3, strides=1, padding='same'))
        BatchNormalization(axis=3)
        decoder_model = Sequential()
        decoder_model.add(encoder_model)

        if compress_mode >=3:
            working_img_size *= 2
            decoder_model.add(Convolution2D(16, 3, strides=1, padding='same'))
            BatchNormalization(axis=3)
            decoder_model.add(Activation("tanh"))
            decoder_model.add(Lambda(lambda image: tf.image.resize_images(image, (working_img_size, working_img_size))))

        if compress_mode >=2:
            working_img_size *= 2
            decoder_model.add(Convolution2D(16, 3, strides=1, padding='same'))
            BatchNormalization(axis=3)
            decoder_model.add(Activation("tanh"))
            decoder_model.add(Lambda(lambda image: tf.image.resize_images(image, (working_img_size, working_img_size))))


        working_img_size *= 2
        decoder_model.add(Convolution2D(16, 3, strides=1, padding='same'))
        BatchNormalization(axis=3)
        decoder_model.add(Activation("tanh"))
        decoder_model.add(Lambda(lambda image: tf.image.resize_images(image, (img_size, img_size))))


        decoder_model.add(Convolution2D(num_channels, 3, strides=1, padding='same'))
        decoder_model.add(Lambda(lambda image: K.clip(image, -clip_value, clip_value) ))


        print('Encoder model:')
        encoder_model.summary()

        print('Decoder model:')
        decoder_model.summary()

        self.encoder = encoder_model
        self.decoder = decoder_model


    def load_codec(self, weights_prefix):
        encoder_weight_filename = weights_prefix + "_encoder.h5"
        decoder_weight_filename = weights_prefix + "_decoder.h5"

        if not os.path.isfile(encoder_weight_filename):
            raise Exception("The file for encoder weights does not exist:{}".format(encoder_weight_filename))
        self.encoder.load_weights(encoder_weight_filename)

        if not os.path.isfile(decoder_weight_filename):
            raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
        self.decoder.load_weights(decoder_weight_filename)

        print("Encoder summaries")
        self.encoder.summary()

        _, encode_H, encode_W, numChannels = self.encoder.output_shape
        config = self.decoder.get_config()
        config2 = config[1::]
        config2[0]['config']['batch_input_shape'] = (None, encode_H, encode_W, numChannels)
        decoder_temp = Sequential.from_config(config2, custom_objects={"tf": tf})

        # set weights
        cnt = -1
        for l in self.decoder.layers:
            cnt += 1
            if cnt == 0:
                continue
            weights = l.get_weights()
            decoder_temp.layers[cnt - 1].set_weights(weights)

        self.decoder = decoder_temp
        print("Decoder summaries")
        self.decoder.summary()
