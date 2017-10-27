import numpy as np
import random
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, UpSampling2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
from keras.optimizers import SGD
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_inception import ImageNet


def train(data, compressMode=1, batch_size=1000, epochs=1000, saveFilePrefix=None, use_tanh=True):

    """Train autoencoder

    Keyword arguments:
    modeFileName -- if specified, pre-trained model would be loaded if exist, model would be saved after training
    compressMode -- currently accept two modes. When 1, the compression ratio is 1/4,
                                                When 2, the compression ratio is 1/16

    """

    # data.train_data may have different definition of the shape, need to check
    # for MNIST and CIFAR

    trainNum, imgH, imgW, numChannels = data.train_data.shape

    x_train = data.train_data
    y_train = x_train

    x_test = data.test_data
    y_test = x_test

    print("Shape of training data:{}".format(x_train.shape))
    print("Shape of testing data:{}".format(x_test.shape))


    # build a neural network from the 1st layer to the last layer
    encoder_model = Sequential()

    # Conv layer output shape (imgH, imgW, 16)
    encoder_model.add(Convolution2D(
        batch_input_shape=(None, imgH, imgW, numChannels),
        filters=16,
        kernel_size=3,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_last',
    ))

    if use_tanh:
        encoder_model.add(Activation('tanh'))
    else:
        encoder_model.add(Activation('relu'))

    # Conv layer output shape (imgH, imgW, 16)
    encoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    if use_tanh:
        encoder_model.add(Activation('tanh'))
    else:
        encoder_model.add(Activation('relu'))

    # Pooling layer (max pooling) output shape (imgH/2, imgW/2, 16)
    encoder_model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_last',
    ))

    # Conv layer output shape (imgH/2, imgW/2, numChannels)
    encoder_model.add(Convolution2D(numChannels, 3, strides=1, padding='same', data_format='channels_last'))
    if use_tanh:
        encoder_model.add(Activation('tanh'))
    else:
        encoder_model.add(Activation('relu'))

    if compressMode == 2:
        # Pooling layer (max pooling) output shape (imgH/4, imgW/4, numChannels)
        encoder_model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
            data_format='channels_last',
        ))

        # Conv layer 3 output shape (imgH/4, imgW/4, numChannels)
        encoder_model.add(Convolution2D(numChannels, 3, strides=1, padding='same', data_format='channels_last'))
        if use_tanh:
            encoder_model.add(Activation('tanh'))
        else:
            encoder_model.add(Activation('relu'))


    # end of encoding


    # start of decoding
    decoder_model = Sequential()

    decoder_model.add(encoder_model)

    if compressMode == 2:
        # Conv layer output shape (imgH/4, imgW/4, 16)
        decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
        if use_tanh:
            decoder_model.add(Activation('tanh'))
        else:
            decoder_model.add(Activation('relu'))

        # Upsampling layer  output shape (imgH/2, imgW/2, 16)
        decoder_model.add(UpSampling2D((2, 2), data_format='channels_last'))

    # Conv layer output shape (imgH/2, imgW/2, 16)
    decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    if use_tanh:
        decoder_model.add(Activation('tanh'))
    else:
        decoder_model.add(Activation('relu'))

    # Upsampling layer output shape (imgH, imgW, 16)
    decoder_model.add(UpSampling2D((2, 2), data_format='channels_last'))

    # Conv layer output shape (imgH, imgW, 16)
    decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    if use_tanh:
        decoder_model.add(Activation('tanh'))
    else:
        decoder_model.add(Activation('relu'))


    # Conv layer output shape (1, imgH, v)
    decoder_model.add(Convolution2D(numChannels, 3, strides=1, padding='same', data_format='channels_last'))
    decoder_model.add(Activation('sigmoid'))

    # print model information
    print('Encoder model:')
    encoder_model.summary()

    print('Decoder model:')
    decoder_model.summary()

    ckptFileName = saveFilePrefix + "ckpt"

    encoder_model_filename = saveFilePrefix + "encoder.json"
    decoder_model_filename = saveFilePrefix + "decoder.json"
    encoder_weight_filename = saveFilePrefix + "encoder.h5"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"
    if os.path.exists(decoder_weight_filename):
        print("Load the pre-trained model.")
        decoder_model.load_weights(decoder_weight_filename)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    decoder_model.compile(loss='mse', optimizer=sgd)

    checkpointer = ModelCheckpoint(filepath=ckptFileName, verbose=1, save_best_only=True)

    decoder_model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              shuffle=True,
              callbacks = [checkpointer])

    print("Checkpoint is saved to {}\n".format(ckptFileName))


    
    model_json = encoder_model.to_json()
    with open(encoder_model_filename, "w") as json_file:
        json_file.write(model_json)
    print("Encoder specification is saved to {}".format(encoder_model_filename))

    encoder_model.save_weights(encoder_weight_filename)
    print("Encoder weight is saved to {}\n".format(encoder_weight_filename))

    model_json = decoder_model.to_json()
    with open(decoder_model_filename, "w") as json_file:
        json_file.write(model_json)
    print("Decoder specification is saved to {}".format(decoder_model_filename))

    decoder_model.save_weights(decoder_weight_filename)
    print("Decoder weight is saved to {}\n".format(decoder_weight_filename))


def main(args):
    # load data
    print("Loading data", args["dataset"])
    if args["dataset"] == "mnist":
        data = MNIST()
    elif args["dataset"] == "cifar10":
        data = CIFAR()
    elif args["dataset"] == "imagenet":
        data = ImageNet(datasetSize=args["imagenet_data_size"], testRatio=0.1)
    print("Done...")

    print("Start training autoencoder")
    train(data, compressMode=args["compress_mode"], batch_size=args["batch_size"], epochs=args["epochs"],
          saveFilePrefix=args["save_prefix"], use_tanh=args["use_tanh"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet"], default="mnist", help="the dataset to train")
    parser.add_argument("--save_prefix", default="codec", help="prefix of file name to save trained model/weights under model folder")
    parser.add_argument("--compress_mode", type=int, choices=[1, 2], default=1, help="the compress mode, 1:25% 2:6.25%")
    parser.add_argument("--batch_size", default=1000, type=int, help="the batch size when training autoencoder")
    parser.add_argument("--epochs", default=1000, type=int, help="the number of training epochs")
    parser.add_argument("--seed", type=int, default=9487)
    parser.add_argument("--imagenet_data_size", type=int,  default=10000, help="the size of imagenet loaded for training, Max 50,000")
    parser.add_argument("--use_tanh", action='store_true', help = "use tanh as activation function")
    args = vars(parser.parse_args())
    if not os.path.isdir("model"):
        print("Folder for saving models does not exist. The folder is created.")
        os.makedirs("model")

    args["save_prefix"] = "model/" + args["save_prefix"] + "_" + str(args["compress_mode"]) + "_"
    # setup random seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    print(args)



    main(args)