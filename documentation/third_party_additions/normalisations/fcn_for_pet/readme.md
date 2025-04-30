# Segmentation of Prostate Tumour Volumes from PET Images is a Different Ball Game

## Introduction

This file contains the steps to use the FCN algorithm to retrieve the clipping-normalisation values. There are two types of clipping values:
1. SUV value based - this is better when you know that the labelling was done with some kind of thresholding. I would not recommend this, as this static value. Also, this is the old version of my work, yuck!!! What was I thinking!!!
2. SUV percent based - this is better since the threshold value is unique for each image based on the maximum SUV value of each image.

## Steps

### Download and convert the data
1. Convert the dataset to nnU-Net format: for our example I will use dataset ID: `500`.
2. In the raw dataset folder, modify the `dataset.json` file in the following manner. An example for dataset with suffixes for CT: 0000 and PET 0001:
```
"channel_names": {'0': 'CT', '1': 'suv_clip'} # This is version 1 - not recommended

OR

"channel_names": {'0': 'CT', '1':'suv_percent_clip'} # This is version 2 - recommended
```
3. Rather than executing `plan_and_preprocess`, we will split it in the following manner:
   - `nnUNetv2_extract_fingerprint -d 500` extracts the dataset fingerprint 
   - `nnUNetv2_plan_experiment -d 500` does the planning for the plain U-Net (you can use any architecture you like, but we need the plans file stored in the preprocessed folder).
4. Execute the [run_fcn.py](run_fcn.py) file. An example for dataset with suffixes for CT: 0000 and PET 0001: ```python_fcn.py --in_folder nnUNet_raw/Dataset500 --result_folder jsonFileDataset500 --pet_file_suffix 0001 --intra```
5. Based on the result on the fcn algorithm, you can choose the mean value(s) for different metric(s) you like. If you choose multiple metrics, average them.
6. Edit the plans file stored in the preprocessed folder for Dataset500. In the `intensityproperties` dict, add the following key and values:
     ```
     If using 'suv_clip', "minSUV"=0 and "maxSUV" = threshold-based average (single metric or average across multiple metrics)
     If using 'suv_percent_clip', "percentSUV" = percent-based average (single metric or average across multiple metrics)
     ```
7. Then run the preprocessing step `nnUNetv2_preprocess -d 500` 
8. Continue normally with training, cross-validation and inference.