import numpy as np
import random
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, UpSampling2D, MaxPooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.backend import tf as ktf
import os
from keras.optimizers import SGD
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_inception import ImageNet
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def train(data, compressMode=1, batch_size=1000, epochs=1000, saveFilePrefix=None, use_tanh=True, train_imagenet=False, imagenet_path=None):

    """Train autoencoder

    Keyword arguments:
    modeFileName -- if specified, pre-trained model would be loaded if exist, model would be saved after training
    compressMode -- currently accept two modes. When 1, the compression ratio is 1/4,
                                                When 2, the compression ratio is 1/16

    """

    # data.train_data may have different definition of the shape, need to check
    # for MNIST and CIFAR

    

    if not train_imagenet:
        trainNum, imgH, imgW, numChannels = data.train_data.shape
        x_train = data.train_data
        y_train = x_train

        x_test = data.test_data
        y_test = x_test

        print("Shape of training data:{}".format(x_train.shape))
        print("Shape of testing data:{}".format(x_test.shape))
    else:
        train_datagen = data.train_datagen
        test_datagen = data.test_datagen
        imgH = 299
        imgW = 299
        numChannels = 3


    # build a neural network from the 1st layer to the last layer
    encoder_model = Sequential()

    if train_imagenet:
        # need to define function and import tf to avoid loading problem
        # see https://github.com/fchollet/keras/issues/5298
        #def resize_input(image):
        #    import tensorflow as tf
        #    output = tf.image.resize_images(image, (256, 256))
        #    return output

        #encoder_model.add( Lambda(resize_input, input_shape=(imgH, imgW, numChannels)))
        encoder_model.add( Lambda(lambda image: tf.image.resize_images(image, (256, 256)), 
            input_shape=(imgH, imgW, numChannels)))
        encoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    else:
        # Conv layer output shape (imgH, imgW, 16)
        encoder_model.add(Convolution2D(
            batch_input_shape=(None, imgH, imgW, numChannels),
            filters=16,
            kernel_size=3,
            strides=1,
            padding='same',     # Padding method
            data_format='channels_last',
        ))
    BatchNormalization(axis=3)
    if use_tanh:
        encoder_model.add(Activation('tanh'))
    else:
        encoder_model.add(Activation('relu'))

    # Conv layer output shape (imgH, imgW, 16)
    encoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    BatchNormalization(axis=3)
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
    BatchNormalization(axis=3)
    #if use_tanh:
    #    encoder_model.add(Activation('tanh'))
    #else:
    #    encoder_model.add(Activation('relu'))

    if compressMode == 2:
        if train_imagenet:
            encoder_model.add(MaxPooling2D(
                pool_size=4,
                strides=4,
                padding='same',  # Padding method
                data_format='channels_last',
                ))
        else:
            # Pooling layer (max pooling) output shape (imgH/4, imgW/4, numChannels)
            encoder_model.add(MaxPooling2D(
                pool_size=2,
                strides=2,
                padding='same',  # Padding method
                data_format='channels_last',
            ))

        # Conv layer 3 output shape (imgH/4, imgW/4, numChannels)
        encoder_model.add(Convolution2D(numChannels, 3, strides=1, padding='same', data_format='channels_last'))
        BatchNormalization(axis=3)
        #if use_tanh:
        #    encoder_model.add(Activation('tanh'))
        #else:
        #    encoder_model.add(Activation('relu'))


    # end of encoding


    # start of decoding
    decoder_model = Sequential()

    decoder_model.add(encoder_model)

    if compressMode == 2:
        # Conv layer output shape (imgH/4, imgW/4, 16)
        decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
        BatchNormalization(axis=3)
        if use_tanh:
            decoder_model.add(Activation('tanh'))
        else:
            decoder_model.add(Activation('relu'))

        if train_imagenet:
            decoder_model.add(UpSampling2D((4, 4), data_format='channels_last'))
        else:
            # Upsampling layer  output shape (imgH/2, imgW/2, 16)
            decoder_model.add(UpSampling2D((2, 2), data_format='channels_last'))

    # Conv layer output shape (imgH/2, imgW/2, 16)
    decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    BatchNormalization(axis=3)
    if use_tanh:
        decoder_model.add(Activation('tanh'))
    else:
        decoder_model.add(Activation('relu'))

    # Upsampling layer output shape (imgH, imgW, 16)
    decoder_model.add(UpSampling2D((2, 2), data_format='channels_last'))

    # Conv layer output shape (imgH, imgW, 16)
    decoder_model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    BatchNormalization(axis=3)
    if use_tanh:
        decoder_model.add(Activation('tanh'))
    else:
        decoder_model.add(Activation('relu'))


    # Conv layer output shape (1, imgH, imgW)

    if train_imagenet:
        #def resize_output(image):
        #    import tensorflow as tf
        #    output = tf.image.resize_images(image, (imgH, imgW))
        #    return output
        #decoder_model.add( Lambda(resize_output))

        # the original code would resize the image back to (imgH, imgW)
        # however, keras seems to have some issue with deserialization
        # currently resize to fixed size 299x299 (assuming imagenet is used)
        #decoder_model.add( Lambda(lambda image: tf.image.resize_images(image, (imgH, imgW))))
        decoder_model.add( Lambda(lambda image: tf.image.resize_images(image, (299, 299))))

    decoder_model.add(Convolution2D(numChannels, 3, strides=1, padding='same', data_format='channels_last'))
    
    #BatchNormalization(axis=3) 
    # the final activation layer would cause problem
    #if use_tanh:
    #    decoder_model.add(Activation('tanh'))
    #else:
    #    decoder_model.add(Activation('relu'))

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


    if not train_imagenet:
        print("Train without data generater")
        decoder_model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=epochs,
                  shuffle=True,
                  callbacks = [checkpointer])
    else:
        print("Train imagenet with data generater")
        

        decoder_model.fit_generator(
                    train_datagen,
                    steps_per_epoch= 50,
                    epochs=epochs,
                    validation_data=test_datagen,
                    validation_steps=100,
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

# a small class for imagenet image image generator
class imagenet_imageGen:
    def __init__(self, train_datagen, test_datagen):
        self.train_datagen = train_datagen
        self.test_datagen = test_datagen


def main(args):
    # load data
    print("Loading data", args["dataset"])
    if args["dataset"] == "mnist":
        data = MNIST()
    elif args["dataset"] == "cifar10":
        data = CIFAR()
    elif args["dataset"] == "imagenet":
        # the following code is the old function for loading imagenet data
        # data = ImageNet(datasetSize=args["imagenet_data_size"], testRatio=0.1)
        
        # new version uses ImageDataGenerator provided by Keras
        train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
                                "../imagenetdata/train_dir",
                                target_size=(299, 299),  
                                batch_size=args["batch_size"],
                                class_mode="input")  

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                                "../imagenetdata/test_dir",
                                target_size=(299, 299),
                                batch_size=args["batch_size"],
                                class_mode="input")
        data = imagenet_imageGen(train_generator, validation_generator)
        print(data.train_datagen)
    print("Done...")

    print("Start training autoencoder")
    train(data, compressMode=args["compress_mode"], batch_size=args["batch_size"], 
            epochs=args["epochs"], saveFilePrefix=args["save_prefix"], 
            use_tanh=args["use_tanh"], train_imagenet=args["train_imagenet"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet"], default="mnist", help="the dataset to train")
    parser.add_argument("--save_prefix", default="codec", help="prefix of file name to save trained model/weights under model folder")
    parser.add_argument("--compress_mode", type=int, choices=[1, 2], default=1, help="the compress mode, 1:25% 2:6.25%")
    parser.add_argument("--batch_size", default=1000, type=int, help="the batch size when training autoencoder")
    parser.add_argument("--epochs", default=1000, type=int, help="the number of training epochs")
    parser.add_argument("--seed", type=int, default=9487)
    parser.add_argument("--train_imagenet", action='store_true', help = "the encoder for imagenet would be different")
    parser.add_argument("--imagenet_data_size", type=int,  default=10000, help="the size of imagenet loaded for training, Max 50,000")
    parser.add_argument("--use_tanh", action='store_true', help = "use tanh as activation function")
    parser.add_argument("--imagenet_path", help="the path to imagenet images")
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