# Ensemble Results

- 7500 Samples equally from each of the 5 folds for each of the models were used to find the performance of the models. Also, we used the models which do not use meta-data.

- The 9 models which were ensembled were:

  - 9c_b5ns_1.5e_640_ext_15ep
  - 9c_b4ns_768_768_ext_15ep
  - 9c_nest101_2e_640_ext_15ep
  - 9c_b4ns_2e_896_ext_15ep
  - 9c_b6ns_640_ext_15ep
  - 9c_b7ns_1e_640_ext_15ep
  - 9c_b4ns_768_640_ext_15ep
  - 4c_b5ns_1.5e_640_ext_15ep
  - 9c_se_x101_640_ext_15ep

- The results are as follows:
  - AUC_Score: 98.3 Best Threshold=0.117822, G-Mean=93.820, NPV=99.300 Sensitivity: 92.74 Specificity: 94.74 Accuracy: 94.57.


## Below are the top ten gmeans and their corresponding threshold values for each model

## Model (1): 9c_meta_b3_768_512_ext_18ep

Best Threshold=0.059741, G-Mean=0.919

Best Threshold=0.060298, G-Mean=0.918

Best Threshold=0.077242, G-Mean=0.918

Best Threshold=0.059745, G-Mean=0.918

Best Threshold=0.067590, G-Mean=0.918

Best Threshold=0.081131, G-Mean=0.918

Best Threshold=0.067590, G-Mean=0.918

Best Threshold=0.067270, G-Mean=0.918

Best Threshold=0.083309, G-Mean=0.918

Best Threshold=0.077256, G-Mean=0.918

## Model (2):9c_b4ns_2e_896_ext_15ep

Best Threshold=0.072974, G-Mean=0.912

Best Threshold=0.073489, G-Mean=0.912

Best Threshold=0.073489, G-Mean=0.911

Best Threshold=0.046697, G-Mean=0.911

Best Threshold=0.060024, G-Mean=0.911

Best Threshold=0.074086, G-Mean=0.911

Best Threshold=0.046462, G-Mean=0.911

Best Threshold=0.047740, G-Mean=0.911

Best Threshold=0.077524, G-Mean=0.911

Best Threshold=0.074267, G-Mean=0.911

## Model (3):9c_b4ns_448_ext_15ep-newfold

Best Threshold=0.076472, G-Mean=0.928

Best Threshold=0.067855, G-Mean=0.928

Best Threshold=0.079087, G-Mean=0.927

Best Threshold=0.082552, G-Mean=0.927

Best Threshold=0.079197, G-Mean=0.927

Best Threshold=0.077019, G-Mean=0.927

Best Threshold=0.076472, G-Mean=0.927

Best Threshold=0.083782, G-Mean=0.927

Best Threshold=0.082955, G-Mean=0.927

Best Threshold=0.083782, G-Mean=0.926

## Model (4): 9c_b4ns_768_640_ext_15ep

Best Threshold=0.059184, G-Mean=0.932

Best Threshold=0.063318, G-Mean=0.932

Best Threshold=0.059184, G-Mean=0.932

Best Threshold=0.043624, G-Mean=0.932

Best Threshold=0.064031, G-Mean=0.931

Best Threshold=0.056257, G-Mean=0.931

Best Threshold=0.063736, G-Mean=0.931

Best Threshold=0.064031, G-Mean=0.931

Best Threshold=0.048044, G-Mean=0.931

Best Threshold=0.064031, G-Mean=0.931

## Model (5):9c_b4ns_768_768_ext_15ep

Best Threshold=0.053348, G-Mean=0.926

Best Threshold=0.055974, G-Mean=0.926

Best Threshold=0.057804, G-Mean=0.925

Best Threshold=0.067336, G-Mean=0.925

Best Threshold=0.062206, G-Mean=0.925

Best Threshold=0.095509, G-Mean=0.925

Best Threshold=0.107586, G-Mean=0.925

Best Threshold=0.058240, G-Mean=0.925

Best Threshold=0.062203, G-Mean=0.925

Best Threshold=0.071290, G-Mean=0.925

## Model (6):9c_meta_b4ns_640_ext_15ep

Best Threshold=0.171233, G-Mean=0.921

Best Threshold=0.049031, G-Mean=0.920

Best Threshold=0.082297, G-Mean=0.920

Best Threshold=0.091697, G-Mean=0.920

Best Threshold=0.150899, G-Mean=0.920

Best Threshold=0.050874, G-Mean=0.920

Best Threshold=0.096617, G-Mean=0.920

Best Threshold=0.133185, G-Mean=0.920

Best Threshold=0.091360, G-Mean=0.920

Best Threshold=0.061267, G-Mean=0.920

## Model (7):4c_b5ns_1.5e_640_ext_15ep

Best Threshold=0.053973, G-Mean=0.926

Best Threshold=0.055661, G-Mean=0.925

Best Threshold=0.043739, G-Mean=0.925

Best Threshold=0.055661, G-Mean=0.925

Best Threshold=0.056944, G-Mean=0.925

Best Threshold=0.056944, G-Mean=0.925

Best Threshold=0.054282, G-Mean=0.925

Best Threshold=0.053973, G-Mean=0.924

Best Threshold=0.057982, G-Mean=0.924

Best Threshold=0.054282, G-Mean=0.924

## Model (8):9c_b5ns_1.5e_640_ext_15ep

Best Threshold=0.071496, G-Mean=0.927

Best Threshold=0.065612, G-Mean=0.927

Best Threshold=0.073150, G-Mean=0.927

Best Threshold=0.065612, G-Mean=0.927

Best Threshold=0.065439, G-Mean=0.926

Best Threshold=0.075024, G-Mean=0.926

Best Threshold=0.073318, G-Mean=0.926

Best Threshold=0.057145, G-Mean=0.926

Best Threshold=0.073318, G-Mean=0.926

Best Threshold=0.075024, G-Mean=0.926

## Model (9):9c_b5ns_448_ext_15ep-newfold

Best Threshold=0.054568, G-Mean=0.932

Best Threshold=0.056335, G-Mean=0.931

Best Threshold=0.057778, G-Mean=0.931

Best Threshold=0.056620, G-Mean=0.931

Best Threshold=0.063060, G-Mean=0.931

Best Threshold=0.058198, G-Mean=0.931

Best Threshold=0.059083, G-Mean=0.930

Best Threshold=0.059083, G-Mean=0.930

Best Threshold=0.063953, G-Mean=0.930

Best Threshold=0.063953, G-Mean=0.930

## Model (10):9c_meta128_32_b5ns_384_ext_15ep

Best Threshold=0.050452, G-Mean=0.924

Best Threshold=0.051685, G-Mean=0.923

Best Threshold=0.050452, G-Mean=0.923

Best Threshold=0.051685, G-Mean=0.923

Best Threshold=0.055848, G-Mean=0.923

Best Threshold=0.041333, G-Mean=0.923

Best Threshold=0.052343, G-Mean=0.923

Best Threshold=0.052343, G-Mean=0.923

Best Threshold=0.051890, G-Mean=0.922

Best Threshold=0.051685, G-Mean=0.922

## Model (11):9c_b6ns_448_ext_15ep-newfold

Best Threshold=0.031112, G-Mean=0.930

Best Threshold=0.033520, G-Mean=0.929

Best Threshold=0.031812, G-Mean=0.929

Best Threshold=0.031812, G-Mean=0.929

Best Threshold=0.033775, G-Mean=0.929

Best Threshold=0.033775, G-Mean=0.928

Best Threshold=0.033775, G-Mean=0.928

Best Threshold=0.033991, G-Mean=0.928

Best Threshold=0.033991, G-Mean=0.928

Best Threshold=0.034971, G-Mean=0.928

## Model (12):9c_b6ns_576_ext_15ep_oldfold

Best Threshold=0.056124, G-Mean=0.914

Best Threshold=0.059967, G-Mean=0.913

Best Threshold=0.059967, G-Mean=0.913

Best Threshold=0.043333, G-Mean=0.913

Best Threshold=0.045785, G-Mean=0.913

Best Threshold=0.025692, G-Mean=0.913

Best Threshold=0.043333, G-Mean=0.913

Best Threshold=0.043234, G-Mean=0.913

Best Threshold=0.062425, G-Mean=0.912

Best Threshold=0.028874, G-Mean=0.912

## Model (13):9c_b6ns_640_ext_15ep

Best Threshold=0.040705, G-Mean=0.918

Best Threshold=0.053154, G-Mean=0.918

Best Threshold=0.041159, G-Mean=0.918

Best Threshold=0.040732, G-Mean=0.918

Best Threshold=0.054983, G-Mean=0.918

Best Threshold=0.042183, G-Mean=0.917

Best Threshold=0.042823, G-Mean=0.917

Best Threshold=0.056108, G-Mean=0.917

Best Threshold=0.055140, G-Mean=0.917

Best Threshold=0.042196, G-Mean=0.917

## Model (14):9c_b7ns_1e_576_ext_15ep_oldfold

Best Threshold=0.052887, G-Mean=0.914

Best Threshold=0.053059, G-Mean=0.914

Best Threshold=0.053059, G-Mean=0.913

Best Threshold=0.053427, G-Mean=0.912

Best Threshold=0.053427, G-Mean=0.912

Best Threshold=0.054217, G-Mean=0.912

Best Threshold=0.032904, G-Mean=0.912

Best Threshold=0.057036, G-Mean=0.911

Best Threshold=0.033750, G-Mean=0.911

Best Threshold=0.034153, G-Mean=0.911

## Model (15):9c_b7ns_1e_640_ext_15ep

Best Threshold=0.071994, G-Mean=0.925

Best Threshold=0.073875, G-Mean=0.925

Best Threshold=0.074969, G-Mean=0.925

Best Threshold=0.074220, G-Mean=0.924

Best Threshold=0.074969, G-Mean=0.924

Best Threshold=0.085966, G-Mean=0.924

Best Threshold=0.077603, G-Mean=0.924

Best Threshold=0.077603, G-Mean=0.924

Best Threshold=0.088904, G-Mean=0.924

Best Threshold=0.082232, G-Mean=0.923

## Model (16):9c_meta_1.5e-5_b7ns_384_ext_15ep

Best Threshold=0.034351, G-Mean=0.929

Best Threshold=0.038046, G-Mean=0.929

Best Threshold=0.038547, G-Mean=0.928

Best Threshold=0.038547, G-Mean=0.928

Best Threshold=0.049465, G-Mean=0.928

Best Threshold=0.044621, G-Mean=0.928

Best Threshold=0.038589, G-Mean=0.928

Best Threshold=0.040455, G-Mean=0.928

Best Threshold=0.040455, G-Mean=0.928

Best Threshold=0.039347, G-Mean=0.927

## Model (17):9c_nest101_2e_640_ext_15ep

Best Threshold=0.080347, G-Mean=0.917

Best Threshold=0.070683, G-Mean=0.917

Best Threshold=0.073539, G-Mean=0.917

Best Threshold=0.112194, G-Mean=0.916

Best Threshold=0.081641, G-Mean=0.916

Best Threshold=0.081641, G-Mean=0.916

Best Threshold=0.081201, G-Mean=0.916

Best Threshold=0.060356, G-Mean=0.916

Best Threshold=0.062705, G-Mean=0.915

Best Threshold=0.113671, G-Mean=0.915

## Model (18):9c_se_x101_640_ext_15ep

Best Threshold=0.093386, G-Mean=0.918

Best Threshold=0.080389, G-Mean=0.917

Best Threshold=0.129510, G-Mean=0.917

Best Threshold=0.083395, G-Mean=0.917

Best Threshold=0.123908, G-Mean=0.917

Best Threshold=0.130222, G-Mean=0.916

Best Threshold=0.093386, G-Mean=0.916

Best Threshold=0.094846, G-Mean=0.916

Best Threshold=0.098983, G-Mean=0.916

Best Threshold=0.099914, G-Mean=0.916

