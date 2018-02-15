from keras.models import Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, Lambda, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf
import os

class CODEC:
    def __init__(self, img_size, num_channels, compress_mode=1, clip_value=0.5):

        self.compress_mode = compress_mode

        encoder_model = Sequential()
        encoder_model.add(Convolution2D( 16, 3, strides=1,padding='same', input_shape=(img_size, img_size, num_channels)))
        encoder_model.add(Activation("relu"))
        encoder_model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        encoder_model.add(Convolution2D(num_channels, 3, strides=1, padding='same'))

        decoder_model = Sequential()
        decoder_model.add(encoder_model)
        decoder_model.add(Convolution2D(16, 3, strides=1, padding='same'))
        decoder_model.add(Lambda(lambda image: tf.image.resize_images(image, (img_size, img_size))))
        decoder_model.add(Convolution2D(16, 3, strides=1, padding='same'))
        decoder_model.add(Activation("relu"))
        decoder_model.add(Convolution2D(num_channels, 3, strides=1, padding='same'))
        decoder_model.add(Lambda(lambda image: tf.clip_by_value(image, -clip_value, clip_value)))
        print('Encoder model:')
        encoder_model.summary()

        print('Decoder model:')
        decoder_model.summary()

        self.encoder = encoder_model
        self.decoder = decoder_model


    def load_codec(self, weights_prefix):
        encoder_weight_filename = weights_prefix + "encoder.h5"
        decoder_weight_filename = weights_prefix + "decoder.h5"

        if not os.path.isfile(encoder_weight_filename):
            raise Exception("The file for encoder weights does not exist:{}".format(encoder_weight_filename))
        self.encoder_model.load_weights(encoder_weight_filename)

        if not os.path.isfile(decoder_weight_filename):
            raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
        decoder_temp.load_weights(decoder_weight_filename)

        print("Encoder summaries")
        self.encoder.summary()

        _, encode_H, encode_W, numChannels = encoder.output_shape
        config = decoder_temp.get_config()
        config2 = config[1::]
        config2[0]['config']['batch_input_shape'] = (None, encode_H, encode_W, numChannels)
        self.decoder = Sequential.from_config(config2, custom_objects={"tf": tf})

        # set weights
        cnt = -1
        for l in decoder_temp.layers:
            cnt += 1
            if cnt == 0:
                continue
            weights = l.get_weights()
            decoder.layers[cnt - 1].set_weights(weights)

        print("Decoder summaries")
        self.decoder.summary()
