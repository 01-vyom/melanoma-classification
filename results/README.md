# Ensemble Results

- 7500 Samples equally from each of the 5 folds for each of the models were used to find the performance of the models.

- The 13 models which were ensembled were:

  - 9c_b5ns_1.5e_640_ext_15ep
  - 9c_b4ns_768_768_ext_15ep
  - 9c_nest101_2e_640_ext_15ep
  - 9c_meta_b4ns_640_ext_15ep
  - 9c_b4ns_2e_896_ext_15ep
  - 9c_b6ns_640_ext_15ep
  - 9c_b7ns_1e_640_ext_15ep
  - 9c_meta_1.5e-5_b7ns_384_ext_15ep
  - 9c_b4ns_768_640_ext_15ep
  - 4c_b5ns_1.5e_640_ext_15ep
  - 9c_se_x101_640_ext_15ep
  - 9c_meta_b3_768_512_ext_18ep
  - 9c_meta128_32_b5ns_384_ext_15ep

- The results are as follows:
  - AUC_Score: 98.42, Best Threshold=0.119311, G-Mean=93.88, Sensitivity: 99.33, and Specificity: 61.2.
