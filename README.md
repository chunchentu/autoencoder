# autoencoder

To train the autoencoder:

```bash
python3 train_autoencoder.py --dataset mnist --compress_mode 1 --save_model model/mnistModel --save_ckpt model/mnistCkpt --batch_size 1000 --epochs 1000 
```

Several arguments:

- dataset: You can choose between mnist and cifar10

- compress_mode: Either 1 or 2, if set to 1, data is compressed to 25%; if set to 2, data is set to 6.25%

- save_model: The file name to save the model. The autoencoder would be saved under "model" folder

- save_ckpt: The file name to save the checkpoint. The checkpoint file would be saved under "model" folder