# Experiment on BYOL

This experiment was conducted to compare the effect of BYOL on melanoma classification for imaged of size 512X512. Furthermore, two CNN backbone architecture are compared namely ResNet-101, and EfficientNetB5. The BYOL experiment has the following results:

## ResNet-101

### BYOL_RESNET101: 

- G-means Maximized: AUC_Score: 90.73 Best Threshold=0.007711, G-Mean=82.210, NPV=97.760 Sensitivity: 79.58 Specificity: 84.83 Accuracy: 84.37

- Normal: Sensitivity: 42.81467 Specificity: 97.94974 Accuracy: 93.15104 NPV: 94.7277 GMeans 64.75867

### RESNET101:

- G-means Maximized: AUC_Score: 87.82 Best Threshold=0.094646, G-Mean=79.310, NPV=97.990 Sensitivity: 83.85 Specificity: 74.92 Accuracy: 75.7

- Normal: Sensitivity: 27.05649 Specificity: 98.44104 Accuracy: 92.22807 NPV: 93.40206 GMeans 51.60881

## EfficientNet-B5

### BYOL_EfficientnetB5: 

- G-means Maximized: AUC_Score: 93.16 Best Threshold=0.000490, G-Mean=85.740, NPV=98.500 Sensitivity: 86.42 Specificity: 84.96 Accuracy: 85.09

- Normal: Sensitivity: 60.75322 Specificity: 98.50718 Accuracy: 95.22125 NPV: 96.34079 GMeans 77.36038

### EfficientnetB5:

- G-means Maximized: AUC_Score: 93.36 Best Threshold=0.004039, G-Mean=86.440, NPV=98.230 Sensitivity: 82.95 Specificity: 89.98 Accuracy: 89.36

- Normal: Sensitivity: 59.66303 Specificity: 98.32766 Accuracy: 94.96248 NPV: 96.23636 GMeans 76.59325
