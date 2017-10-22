import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, UpSampling2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
from keras.optimizers import SGD
from setup_mnist import MNIST
from setup_cifar import CIFAR

def train(data, compressMode=1, batch_size=1000, epochs=1000, saveModelFileName=None, ckptFileName=None):

    """Train autoencoder

    Keyword arguments:
    modeFileName -- if specified, pre-trained model would be loaded if exist, model would be saved after training
    compressMode -- currently accept two modes. When 1, the compression ratio is 1/4,
                                                When 2, the compression ratio is 1/16

    """

    # data.train_data may have different definition of the shape, need to check
    # for MNIST is fine so far
    trainNum, imgH, imgW, _ = data.train_data.shape


    x_train = data.train_data.reshape(-1, 1, imgH, imgW)
    y_train = x_train

    x_test = data.test_data.reshape(-1, 1, imgH, imgW)
    y_test = x_test

    # build a neural network from the 1st layer to the last layer
    encode_model = Sequential()

    # Conv layer output shape (16, imgH, imgW)
    encode_model.add(Convolution2D(
        batch_input_shape=(None, 1, imgH, imgW),
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_first',
    ))
    encode_model.add(Activation('relu'))

    # Pooling layer (max pooling) output shape (16, imgH/2, imgW/2)
    encode_model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_first',
    ))

    # Conv layer output shape (1, imgH/2, imgW/2)
    encode_model.add(Convolution2D(1, 5, strides=1, padding='same', data_format='channels_first'))
    encode_model.add(Activation('relu'))

    if compressMode == 2:
        # Pooling layer (max pooling) output shape (16, imgH/4, imgW/4)
        encode_model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
            data_format='channels_first',
        ))

        # Conv layer 3 output shape (1, imgH/4, imgW/4)
        encode_model.add(Convolution2D(1, 5, strides=1, padding='same', data_format='channels_first'))
        encode_model.add(Activation('relu'))


    # end of encoding


    # start of decoding
    decode_model = Sequential()

    decode_model.add(encode_model)

    if compressMode == 2:
        # Conv layer output shape (16, imgH/4, imgW/4)
        decode_model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
        decode_model.add(Activation('relu'))

        # Upsampling layer  output shape (16, imgH/2, imgW/2)
        decode_model.add(UpSampling2D((2, 2), data_format='channels_first'))

    # Conv layer output shape (16, imgH/2, imgW/2)
    decode_model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
    decode_model.add(Activation('relu'))

    # Upsampling layer output shape (16, imgH, imgW)
    decode_model.add(UpSampling2D((2, 2), data_format='channels_first'))

    # Conv layer output shape (16, imgH, imgW)
    decode_model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
    decode_model.add(Activation('relu'))


    # Conv layer output shape (1, imgH, v)
    decode_model.add(Convolution2D(1, 5, strides=1, padding='same', data_format='channels_first'))


    decode_model.summary()

    if os.path.exists(saveModelFileName):
        decode_model.load_weights(saveModelFileName)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    decode_model.compile(loss='mse', optimizer=sgd)

    if ckptFileName is None:
        ckptFileName = 'ckpt'
    checkpointer = ModelCheckpoint(filepath=ckptFileName, verbose=1, save_best_only=True)

    decode_model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              shuffle=True,
              callbacks = [checkpointer])
    decode_model.save(saveModelFileName)

    print("Checkpoint is saved to %s\n".format(ckptFileName))
    print("Model is saved to %s\n".format(saveModelFileName))


def main(args):
    # load data
    print('Loading model', args['dataset'])
    if args['dataset'] == "mnist":
        data = MNIST()
    elif args['dataset'] == "cifar10":
        data = CIFAR()
    print('Done...')

    print('Start training autoencoder')
    train(data, compressMode=args['compress_mode'], batch_size=args['batch_size'], epochs=args['epochs'],
          saveModelFileName=args['save_model'], ckptFileName=args['save_ckpts'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10"], default="mnist", help="the dataset to train")
    parser.add_argument("--save_model", default="model/trainedModel", help="path to save trained model")
    parser.add_argument("--save_ckpts", default="model/ckpt", help="path to save checkpoint file")
    parser.add_argument("--compress_mode", choices=[1, 2], default=1, help="the compress mode, 1:25% 2:6.25%")
    parser.add_argument("--batch_size", default=1000, type=int, help="the batch size when training autoencoder")
    parser.add_argument("--epochs", default=1000, type=int, help="the number of training epochs")
    parser.add_argument("--seed", type=int, default=9487)

    args = vars(parser.parse_args())

    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    print(args)
    main(args)