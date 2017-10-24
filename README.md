# autoencoder

# required packages
``` bash
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

Note:

Check if the keras is using tensorflow backend, if not, modify ~/.keras.


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

## Using inception model

``` bash
python3 setup_inception.py
```

To prepare the ImageNet dataset, download the file from the following link

[ImageNet test data](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)

tar and put the imgs folder in ../imagenetdata. This path can be changed in setup_inception.py.

To train the autoencoder 

```bash
python3 train_autoencoder.py --dataset imagenet --compress_mode 2 --save_prefix imagenet --batch_size 1000 --epochs 500 --imagenet_data_size 30
```
specify the number of imagenet figures to load through **imagenet_data_size**

# Update history

- Update 10/23/2017: Change the order of dimensions( previously channel first, changed to channel last). Fix reshaping bugs for multi-channel data. Change the function arguments

- Update 10/24/2017: Add imagenet support. Change setup_inception to accept differnt number of training samples. Resize the image to 300x300 for down sampling


# TODO: