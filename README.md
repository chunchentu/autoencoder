# autoencoder

# required packages
``` bash
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

# Notes

Check if the keras is using tensorflow backend, if not, modify ~/.keras.
To train the autoencoder:

```bash
python3 train_autoencoder.py --dataset cifar10 --compress_mode 2 --save_prefix  cifar10 --batch_size 1000 --epochs 1000
```

Several arguments:

- dataset: You can choose between **mnist** and **cifar10**

- compress_mode: Either 1 or 2, if set to 1, data is compressed to 25%; if set to 2, data is set to 6.25%

- save_model: The file name to save the model. The autoencoder would be saved under "model" folder

- save_ckpt: The file name to save the checkpoint. The checkpoint file would be saved under "model" folder

# TODO

- need to add the inception dataset
- need to revise the codec loading



