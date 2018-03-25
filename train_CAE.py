import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Activation, Convolution2D, MaxPooling2D, Lambda, Input
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras import backend as K
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_inception import ImageNet, ImageNetDataGen
# from setup_facial import FACIAL
from setup_codec import CODEC
import tensorflow as tf
import os

def train_autoencoder(data, codec, batch_size=1000, epochs=1000, saveFilePrefix=None, use_tanh=True, train_imagenet=False):

    """Train autoencoder

    Keyword arguments:
    modeFileName -- if specified, pre-trained model would be loaded if exist, model would be saved after training
    compressMode -- currently accept two modes. When 1, the compression ratio is 1/4,
                                                When 2, the compression ratio is 1/16

    """



    

    ckptFileName = saveFilePrefix + "ckpt"

    encoder_model_filename = saveFilePrefix + "encoder.json"
    decoder_model_filename = saveFilePrefix + "decoder.json"
    encoder_weight_filename = saveFilePrefix + "encoder.h5"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    

    if os.path.exists(decoder_weight_filename):
        print("Load the pre-trained model.")
        codec.decoder.load_weights(decoder_weight_filename)
    elif os.path.exists(ckptFileName):
        print("Load the previous checkpoint")
        codec.decoder.load_weights(ckptFileName)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    codec.decoder.compile(loss='mse', optimizer=sgd)

    checkpointer = ModelCheckpoint(filepath=ckptFileName, verbose=1, save_best_only=True)

    if train_imagenet:
        # train using Keras imageGenerator
        print("Training using imageGenerator:")
        codec.decoder.fit_generator(
                    data.train_generator_flow,
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data.validation_generator_flow,
                    validation_steps=100,
                    callbacks = [checkpointer])
    else: 
        # in-memory training
        x_train = data.validation_data
        y_train = x_train
        x_test = data.test_data
        y_test = x_test
        print("In-memory training:")
        print("Shape of training data:{}".format(x_train.shape))
        print("Shape of testing data:{}".format(x_test.shape))

        
        codec.decoder.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              shuffle=True,
              callbacks = [checkpointer])

    print("Checkpoint is saved to {}\n".format(ckptFileName))


    
    model_json = codec.encoder.to_json()
    with open(encoder_model_filename, "w") as json_file:
        json_file.write(model_json)
    print("Encoder specification is saved to {}".format(encoder_model_filename))

    codec.encoder.save_weights(encoder_weight_filename)
    print("Encoder weight is saved to {}\n".format(encoder_weight_filename))

    model_json = codec.decoder.to_json()
    with open(decoder_model_filename, "w") as json_file:
        json_file.write(model_json)

    print("Decoder specification is saved to {}".format(decoder_model_filename))

    codec.decoder.save_weights(decoder_weight_filename)
    print("Decoder weight is saved to {}\n".format(decoder_weight_filename))


def main(args):
    # load data
    print("Loading data", args["dataset"])
    if args["dataset"] == "mnist":
        data = MNIST()
        if args["train_data_source"]:
            print("Using data from {}".format(args["train_data_source"]))
            img = np.load("{}_data.npy".format(args["train_data_source"]))
            labels = np.load("{}_data.npy".format(args["train_data_source"]))
            data.validation_data = img
            data.validation_labels = labels


    elif args["dataset"] == "cifar10":
        data = CIFAR()
    elif args["dataset"] == "fe":
        data = FACIAL()
    elif args["dataset"] == "imagenet":
        # use ImageDataGenerate provided by Keras
        data = ImageNetDataGen(args["imagenet_train_dir"], args["imagenet_validation_dir"])

    print("Done...")

    if args["dataset"] == "imagenet":
        data_shape = (None, 299, 299, 3)
        resize = 256
    else:
        data_shape = data.train_data.shape
        resize = None

    print("Start training autoencoder")
    codec = CODEC(img_size=data_shape[1], num_channels=data_shape[3], 
                compress_mode=args["compress_mode"], resize=resize, clip_value=args["clip_value"])
    train_autoencoder(data, codec,  batch_size=args["batch_size"], 
            epochs=args["epochs"], saveFilePrefix=args["save_prefix"], train_imagenet= (args["dataset"] == "imagenet") )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "fe", "imagenet"], default="mnist", help="the dataset to train")
    parser.add_argument("--save_prefix", default="codec", help="prefix of file name to save trained model/weights under model folder")
    parser.add_argument("--compress_mode", type=int, choices=[1, 2, 3], default=1, help="the compress mode, 1:1/4 2:1/16, 3:1/64")
    parser.add_argument("--batch_size", default=1000, type=int, help="the batch size when training autoencoder")
    parser.add_argument("--epochs", default=1000, type=int, help="the number of training epochs")
    parser.add_argument("--seed", type=int, default=9487)
    parser.add_argument("--imagenet_train_dir", default=None, help="The path to training data for imagenet")
    parser.add_argument("--imagenet_validation_dir", default=None, help="The path to validation data for imagenet")
    parser.add_argument("--clip_value", default=1, type=float, help="The clipping value for the output of the decoder")
    parser.add_argument("--train_data_source", help="the training data other than the default dataset, MNIST only")
    # parser.add_argument("--train_imagenet", action='store_true', help = "the encoder for imagenet would be different")
    # parser.add_argument("--imagenet_data_size", type=int,  default=10000, help="the size of imagenet loaded for training, Max 50,000")
    # parser.add_argument("--use_tanh", action='store_true', help = "use tanh as activation function")
    # parser.add_argument("--imagenet_path", help="the path to imagenet images")
    # parser.add_argument("--train_on_test", action="store_true", help="use only testing data to train the autoencoder")
    # parser.add_argument("--train_on_test_ratio", type=float, default=0.99, help="the ratio of testing data to train the autoencoder; only used when train_on_test is set")
    # parser.add_argument("--augment_data", action="store_true", help="apply image augmentation on mnist dataset")
    
    args = vars(parser.parse_args())
    if not os.path.isdir("codec"):
        print("Folder for saving models does not exist. The folder is created.")
        os.makedirs("codec")
    args["save_prefix"] = "codec/" + args["save_prefix"] + "_" + str(args["compress_mode"]) + "_"

    if args["dataset"] == "imagenet":
        if args["imagenet_train_dir"] is None or args["imagenet_validation_dir"] is None:
            raise Exception("Using dataset imagenet but the training/validation dir is not set")


    # setup random seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    print(args)



    main(args)
