# Image Based Melanoma Classification using Deep Learning

### [[Paper]]() | [[Slides]]()

[Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Sanjana Rao](https://www.linkedin.com/in/sanjanaraogp/)<sup>1</sup>

<sup>1</sup>[Computer & Information Science & Engineering, University of Florida, Gainesville, Florida, USA](https://www.cise.ufl.edu/)

Two deep learning techniques are used to perform the melanoma classification. Each of them is described separately in the `./src/` folder. Furthermore, the models results are saved in the `./results/` folder with different technique names.

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
| BYOL-ResNet-101           | 42.81           | 97.94                | 64.75   | 94.73  | -    |
| ResNet-101                | 27.05           | 98.44                | 51.6    | 93.40  | -    |
| BYOL-EfficientNet-B5      | 59.16           | 98.44                | 76.3    | 96.24  | -    |
| EfficientNet-B5           | 60.35           | 98.40                | 77.1    | 96.34  | -    |


*The deep learning techniques show the results where we selected the threshold by maximizing the G-Means value. 