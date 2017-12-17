# autoencoder

# Required packages
``` bash
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

Note:

Check if the keras is using tensorflow backend, if not, modify ~/.keras/keras.json

# mnist
To train the autoencoder on mnist dataset:

```bash
python3 train_autoencoder.py --dataset mnist --compress_mode 2 --save_prefix mnist --batch_size 1000 --epochs 1000
```

Several arguments:

- dataset: You can choose between **mnist** and **cifar10**

- compress_mode: Either 1 or 2, if set to 1, data is compressed to 25%; if set to 2, data is set to 6.25%

- save_model: The file name to save the model. The autoencoder would be saved under "model" folder

- save_ckpt: The file name to save the checkpoint. The checkpoint file would be saved under "model" folder

# CIFAR10
To train the autoencoder on cifar10 dataset:

```bash
python3 train_autoencoder.py --dataset cifar10 --compress_mode 2 --save_prefix cifar10 --batch_size 1000 --epochs 1000
```

Arguments are the same as mnist


# ImageNet

## Download the ImageNet dataset

To prepare the ImageNet dataset, download the file from the following link

[ImageNet test data](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)

tar and put the imgs folder in ../imagenetdata. This path can be changed in setup_inception.py.

## Split the dataset into testing and training

```bash
python3 separate_imagenet.py --img_directory "../imagenetdata/imgs"
```

This would randomly separate the data in `--img_directory` into training and testing dataset. The default ratio of the testing samples is set to 0.1. This ratio can be changed by providing addtional `--test_ratio` argument. An optional random seed argument `--seed` is also available for reproducibility.

Two text files `train_file_list.txt` and `test_file_list.txt` would be generated listing all the files in the training and testing dataset.

When executing this python script, you would be asked if you wish the script to move the data. If yes, your files would be automatically moved to the corresponding sub-directories. **This step is currently designed for imagenet dataset and Keras ImageGenerator**. See [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information about the batch training in Keras.


## Batch training the autoencoder on ImgetNet 
To train the autoencoder using the following commands:

```bash
python3 train_autoencoder.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 100 --epochs 100 --use_tanh --train_imagenet
```
This would load and traingin data in `../imagenetdata/train_dir` and `../imagenetdata/test_dir` for training.

# Building autoencoder on the testing dataset

Under some situation (e.g. blackbox attack), it is assumed users have no access to the training data. This can be done by setting `--train_on_test` and specify the ratio of splitting the testing data through `--train_on_test_ratio` (default 0.99)

For mnist

```bash
python3 train_autoencoder.py --dataset mnist --compress_mode 1 --save_prefix test_mnist --batch_size 5000 --epochs 10000 --train_on_test
```

For cifar10

```bash
python3 train_autoencoder.py --dataset cifar10 --compress_mode 1 --save_prefix test_cifar10 --batch_size 5000 --epochs 10000 --train_on_test
```

The size of the test dataset could be insufficient to train the model. We can do the image augmentation by turning on the option `--augment_data`.

For mnist
```bash
python3 train_autoencoder.py --dataset mnist --compress_mode 1 --save_prefix aug_mnist --batch_size 1000 --epochs 1000 --train_on_test --augment_data
```

For cifar10
```bash
python3 train_autoencoder.py --dataset cifar10 --compress_mode 1 --save_prefix aug_cifar10 --batch_size 1000 --epochs 1000 --train_on_test --augment_data
```

# Building autoencoder using other data source

The option `--use_other_data_name` allows users to build autoencoder using other data from current folder. Currently, this option only supports mnist dataset. Data should be store in `.npy` format (saved by `numpy` package) and with postfix `_data.npy`. For example, to use the data named `mnist8m_data.npy` you can

```bash
python3 train_autoencoder.py --dataset mnist --compress_mode 1 --save_prefix 8m_mnist --batch_size 5000 --epochs 5000 --use_other_data_name mnist8m
```


# Update history

- Update 12/16/2017: Add options for building autoencoder using other data source

- Update 12/15/2017: Add augmentation for mnist and cifar10.

- Update 12/14/2017: Add the feature to build autoencoder on the testing dataset

- Update 10/30/2017: Batching training on imagenet using Keras ImageDataGenerator is now available.

- Update 10/28/2017: Add additional resize layer for training imagenet. Solve some issues with the saving/loading Lambda layers in Keras. 

- Update 10/27/2017: Add arguments for selecting `tanh` activation function (default `relu`)

- Update 10/24/2017: Add imagenet support. Change setup_inception to accept differnt number of training samples. Resize the image to 300x300 for down sampling

- Update 10/23/2017: Change the order of dimensions( previously channel first, changed to channel last). Fix reshaping bugs for multi-channel data. Change the function arguments






# TODO:

