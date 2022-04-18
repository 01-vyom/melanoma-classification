## Experiments on the BYOL self-supervision using ResNet-101 and Efficientnet-B5

## SOFTWARE (python packages are detailed separately in `requirements.txt`)

- Python 3.9.5

- CUDA Version 11.1

- cuddn 8.0.5

- nvidia Driver Version: 418.116.00

## Dataset Setup

The dataset setup is same as the [CNN-Ha experiment](https://github.com/01-vyom/melanoma-classification/blob/main/src/CNN-Ha/README.md).

## Model Illustration

![](https://github.com/01-vyom/melanoma-classification/blob/main/src/BYOL/BYOL_Experiment.png)

More details can be found here: 

https://medium.com/the-dl/easy-self-supervised-learning-with-byol-53b8ad8185d

https://arxiv.org/pdf/2006.07733.pdf

## Training

### ResNet-101

To train the BYOL with Resnet101 model, run the following script:

```
python BYOL_SIIM-ISIC_RESNET101.py
```

To train the BYOL with Resnet101 model on the hipergator, run the following script:

```
sbatch resnet.sh
```

### EfficientNet-b5

To train the BYOL with Efficientnetb5 model, run the following script:

```
python BYOL_SIIM-ISIC_Efficientnetb5.py
```

To train the BYOL with Efficientnetb5 model on the hipergator, run the following script:

```
sbatch efficientnet.sh
```

One can look at the [README](https://github.com/01-vyom/melanoma-classification/blob/main/results/README_BYOL.md) file for more experiment information.

## Evaluation

Evaluation for both the models over their validation set can be done by running the following command:

```
python metric_calculation_BYOL.py
```

Note: 

- For both evaluation, and training the variable `data_dir` can be changed if needed to point to a directory containing the data and the variable `savepath` can be changed to point to a directory where the models will be saved.

## Acknowledgement

This implementation is based on [1](https://colab.research.google.com/drive/15HVEcDh-LRUn-kjxCCFWlEy1UFIiGUX3?usp=sharing#scrollTo=YoU-3B4FmtrZ) as described in the [Medium article](https://medium.com/the-dl/easy-self-supervised-learning-with-byol-53b8ad8185d).