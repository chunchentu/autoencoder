# autoencoder

# Required packages
``` bash
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

Note:

Check if the keras is using tensorflow backend, if not, modify ~/.keras/keras.json


# CIFAR10
To train the autoencoder on cifar10 dataset:

```bash
python3 train_autoencoder.py --dataset cifar10 --compress_mode 2 --save_prefix cifar10 --batch_size 1000 --epochs 1000
```

Several arguments:

- dataset: You can choose between **mnist** and **cifar10**

- compress_mode: Either 1 or 2, if set to 1, data is compressed to 25%; if set to 2, data is set to 6.25%

- save_model: The file name to save the model. The autoencoder would be saved under "model" folder

- save_ckpt: The file name to save the checkpoint. The checkpoint file would be saved under "model" folder


# ImageNet

## Download the ImageNet dataset

To prepare the ImageNet dataset, download the file from the following link

[ImageNet test data](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)

tar and put the imgs folder in ../imagenetdata. This path can be changed in setup_inception.py.

## Split the dataset into testing and training

```bash
python3 separate_imagenet.py --img_directory "../imagenetdata/imgs"
```

This would randomly separate the data in `--mg_directory` into training and testing dataset. The default ratio of the testing samples is set to 0.1. This ratio can be changed by providing addtional `--test_ratio` argument. An optional random seed argument `--seed` is also available for reproducibility.

Two text files `train_file_list.txt` and `test_file_list.txt` would be generated listing all the files in the training and testing dataset.

When executing this python script, you would be asked if you wish the script to move the data. If yes, your files would be automatically moved to the corresponding sub-directories. **This step is currently designed for imagenet dataset and Keras ImageGenerator**. See [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information about the batch training in Keras.


## Batch training the autoencoder on ImgetNet 
To train the autoencoder using the following commands:

```bash
python3 train_autoencoder.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 100 --epochs 100 --use_tanh --train_imagenet
```
This would load and traingin data in `../imagenetdata/train_dir` and `../imagenetdata/test_dir` for training.

# Update history

- Update 10/30/2017: Batching training on imagenet using Keras ImageDataGenerator is now available.

- Update 10/28/2017: Add additional resize layer for training imagenet. Solve some issues with the saving/loading Lambda layers in Keras. 

- Update 10/27/2017: Add arguments for selecting `tanh` activation function (default `relu`)

- Update 10/24/2017: Add imagenet support. Change setup_inception to accept differnt number of training samples. Resize the image to 300x300 for down sampling

- Update 10/23/2017: Change the order of dimensions( previously channel first, changed to channel last). Fix reshaping bugs for multi-channel data. Change the function arguments






# TODO:


<!--
image generator
```
python3 train_autoencoder.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 100 --epochs 100 --imagenet_data_size 2500 --use_tanh --train_imagenet --imagenet_path ../imagenetdata/ | tee logfile/imagenet_32
```



## keras image generator
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


python3 train_autoencoder.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 100 --epochs 100 --use_tanh --train_imagenet | tee logfile/imagenet_32

-->