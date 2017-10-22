import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, UpSampling2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
from keras.optimizers import SGD
from setup_mnist import MNIST
from setup_cifar import CIFAR
filename = 'CS_model_gpu/MNIST_1414_b2'

#load data
data = MNIST()
def train(data, saveModelName=None, ):
x_train = data.train_data.reshape(-1, 1, 28, 28)
y_train = x_train
#reshape to vector

x_test = data.test_data.reshape(-1, 1, 28, 28)
y_test = x_test

# build a neural network from the 1st layer to the last layer
encode_model = Sequential()

# Conv layer 1 output shape (16, 28, 28)
encode_model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
encode_model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (16, 14, 14)
encode_model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))


# Conv layer 2 output shape (1, 14, 14)

encode_model.add(Convolution2D(1, 5, strides=1, padding='same', data_format='channels_first'))
encode_model.add(Activation('relu'))


# end of encoding


# start of decoding
decode_model = Sequential()

decode_model.add(encode_model)
# Conv layer 3 output shape (16, 14, 14)
decode_model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
decode_model.add(Activation('relu'))

# Upsampling layer 3 output shape (16, 28, 28)
decode_model.add(UpSampling2D((2, 2), data_format='channels_first'))

# Conv layer 4 output shape (16, 28, 28)
decode_model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
decode_model.add(Activation('relu'))


# Conv layer 6 output shape (1, 28, 28)
decode_model.add(Convolution2D(1, 5, strides=1, padding='same', data_format='channels_first'))


decode_model.summary()

if os.path.exists(filename):
    decode_model.load_weights(filename)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
decode_model.compile(loss='mse', optimizer=sgd)


checkpointer = ModelCheckpoint(filepath='CS_model/ckpt.hdf5', verbose=1, save_best_only=True)

decode_model.fit(x_train, y_train,
          batch_size=500,
          validation_data=(x_test, y_test),
          epochs=1000,
          shuffle=True,
          callbacks = [checkpointer])
decode_model.save(filename)

