# STLB-GN: Spatio-Temporal Dual Graph Network with Learnable Bases

This is a PyTorch implementation of STLB-GN.
![STLB-GN](https://github.com/linskyy/STLB-GN/blob/main/model_architecture.png)
## Folders

* configs: Configurations for training and model specifications tailored to individual datasets.

* lib: This folder encompasses custom-built modules essential for our project, including data loading, preprocessing, normalization, and evaluation metrics.

* model: The code and trainer that brings our model to life.

* pre-trained:  Parameters of models already trained beforehand, you can use them by configuring the `--mode` option to test in the `run.py` script.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
python run.py --datasets {DATASET_NAME} --mode {MODE_NAME}
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`

such as `python run.py --datasets PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the `pre-trained` folder.



