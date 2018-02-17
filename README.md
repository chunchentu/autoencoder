# autoencoder
Train the Convolutional AutoEncoder (CAE).

# Required packages
``` bash
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

Note:

Check if the keras is using tensorflow backend, if not, modify ~/.keras/keras.json

# mnist
To train the autoencoder on mnist dataset:

```bash
python3 train_CAE.py --dataset mnist --compress_mode 1 --save_prefix mnist --batch_size 1000 --epochs 1000 --clip_value 0.5
```

Several arguments:

- dataset: You can choose between **mnist**, **cifar10**, ***fe* and ***imagenet*

- compress_mode: Either 1, 2 or 3 if set to 1, data is compressed to 25%; if set to 2, data is set to 6.25% and if set to 3, data is set to 1.5%. Please make sure the original image size is large enough. 

- save_prefix: The prefix of file name to save the model. The autoencoder would be saved under "codec" folder. Checkpoint file would be saved under "codec" folder with the *.ckpt* extension.

- clip_value: Clip the output so to make sure output values lie in [-*clip_value*, *clip_value*].

# CIFAR10
To train the autoencoder on cifar10 dataset:

```bash
python3 train_CAE.py --dataset cifar10 --compress_mode 2 --save_prefix cifar10 --batch_size 1000 --epochs 1000
```

Arguments are the same as mnist

# Facial expression

The data can be downloaded from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

An example to train the autoencoder:
```bash
python3 train_CAE.py --dataset fe --compress_mode 2 --save_prefix fe --batch_size 1000 --epochs 1000
```



# ImageNet

## Download the ImageNet dataset

To prepare the ImageNet dataset, download the file from the following link

[ImageNet test data](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)

untar and put the imgs folder in ../imagenetdata. This path can be changed in setup_inception.py.

## Split the dataset into testing and training

```bash
python3 separate_imagenet.py --img_directory "../imagenetdata/imgs"
```

This would randomly separate the data in `--img_directory` into training and testing dataset. The default ratio of the testing samples is set to 0.1. This ratio can be changed by providing addtional `--test_ratio` argument. An optional random seed argument `--seed` is also available for reproducibility.

Two text files `train_file_list.txt` and `test_file_list.txt` would be generated listing all the files in the training and testing dataset.

When executing this python script, you would be asked if you wish the script to move the data. If yes, your files would be automatically moved to the corresponding sub-directories. **This step is currently designed for imagenet dataset and Keras ImageGenerator**. See [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information about the batch training in Keras.


## Batch training the autoencoder on ImgetNet 
To train the autoencoder using the following commands:

```bash
python3 train_CAE.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 100 --epochs 100 --imagenet_train_dir ../imagenetdata/train_dir --imagenet_validation_dir ../imagenetdata/test_dir
```
This would load and traingin data in `../imagenetdata/train_dir` and `../imagenetdata/test_dir` for training and validation.



# Update history

- Update 02/17/2018: Change code structure.

- Update 12/16/2017: Add options for building autoencoder using other data source

- Update 12/15/2017: Add augmentation for mnist and cifar10.

- Update 12/14/2017: Add the feature to build autoencoder on the testing dataset

- Update 10/30/2017: Batching training on imagenet using Keras ImageDataGenerator is now available.

- Update 10/28/2017: Add additional resize layer for training imagenet. Solve some issues with the saving/loading Lambda layers in Keras. 

- Update 10/27/2017: Add arguments for selecting `tanh` activation function (default `relu`)

- Update 10/24/2017: Add imagenet support. Change setup_inception to accept differnt number of training samples. Resize the image to 300x300 for down sampling

- Update 10/23/2017: Change the order of dimensions( previously channel first, changed to channel last). Fix reshaping bugs for multi-channel data. Change the function arguments






# TODO:

