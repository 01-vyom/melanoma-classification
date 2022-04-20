# Image Based Melanoma Classification using Deep Learning

### [[Paper]](https://github.com/01-vyom/melanoma-classification/blob/main/Image_Based_Melanoma_Detection.pdf) | [[Slides]](https://docs.google.com/presentation/d/1z-i3WToPgWQUGqSf5X4jM8-88gFHtJcnodxfjnoYBlM/edit?usp=sharing)

[Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Sanjana Rao](https://www.linkedin.com/in/sanjanaraogp/)<sup>1</sup>

<sup>1</sup>[Computer & Information Science & Engineering, University of Florida, Gainesville, Florida, USA](https://www.cise.ufl.edu/)

We present an ensemble of image-only convolutions neural network (CNN) models with different backbones and input sizes along with a self-supervised model to classify skin lesions. We have devised the first ensemble based on the winning solution to Kaggle's SIIM-ISIC Melanoma Classification challenge (We will be referring to this as the Ha-CNN model going further) and Bootstrap your own latent (BYOL) model which is based on self-supervised learning. The models have experimented with the SIIM-ISIC Melanoma dataset (2018-2020). Using specificity and sensitivity as the performance metrics, nine top-performing models were selected out of the eighteen models proposed in the Ha-CNN paper. We experimented BYOL model with two different backbones - ResNet and EfficientNet. The Ha-CNN model achieves a specificity and sensitivity of 94.3% and 92.1%  with a negative predictive value of 99.2%. As with the BYOL model, our results show an increase of 1.00% for the ResNet-101 model supervision (94.73% and 93.40%) and an increase of 
1.00% for the Efficient-B5 model (97.24% and 96.34%) with and without BYOL-self-supervision.

Two deep learning techniques are used to perform the melanoma classification. Each of them is described separately in the `./src/` folder. The CNN-Ensemble source code along with the repository setup is described in the [./src/CNN-Ha/](https://github.com/01-vyom/melanoma-classification/tree/main/src/CNN-Ha) folder and the BYOL-CNN is described in the [.src/BYOL/](https://github.com/01-vyom/melanoma-classification/tree/main/src/BYOL) Furthermore, the models results are saved in the `./results/` folder with different technique names.

## Results

The following table shows the comparison of different techniques on the task of Melanoma Classification:

| Technique name            | Sensitivity (%) | Specificity (MCRMSE) | G-means | NPV    | AUC  |
| ------------------------- | --------------- | -------------------- | ------- | ------ | ---- |
| Dermatologists (Haenssle) | 88.9 ± 9.6      | 75.7 ± 11.7          | 82.0    | -      | 82.0 |
| EGIR                      | 88              | 100                  | 93.8    | > 99.0 | 95.5 |
| CNN-Thissen               | 78              | 80                   | 78.9    | -      | -    |
| CNN-Haenssle              | -               | 82.5 (@88.9 sen.)    | 85.6    | -      | 95.3 |
| CNN-Ha (paper)            | -               | -                    | -       | -      | 94.9 |
| CNN-Ha (experiments)*     | 92.10           | 94.74                | 93.823  | 99.3   | 98.2 |
| BYOL-ResNet-101*          | 42.81           | 97.94                | 64.75   | 94.73  | -    |
| ResNet-101*               | 27.05           | 98.44                | 51.6    | 93.40  | -    |
| BYOL-EfficientNet-B5*     | 59.16           | 98.44                | 76.3    | 96.24  | -    |
| EfficientNet-B5*          | 60.35           | 98.40                | 77.1    | 96.34  | -    |


*The deep learning techniques show the results where we selected the threshold by maximizing the G-Means value.
More details on the experiments and their results can be found in the results folder for the experiments [CNN-Ensemble](https://github.com/01-vyom/melanoma-classification/blob/main/results/README_CNN-Ha.md) and the [BYOL-CNN](https://github.com/01-vyom/melanoma-classification/blob/main/results/README_BYOL.md) experiment. 