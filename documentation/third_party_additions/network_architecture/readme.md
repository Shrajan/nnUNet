# Additional Architectures

```
Authored by Shrajan
```

## Introduction

You are here because you want to try different architectures with the nnUNet framework. The following U-Net variants have been implemented:

### Attention U-Net
Based on the paper [Attention U-Net](https://arxiv.org/pdf/1804.03999), and the codes are modified using the [repository](https://github.com/ozan-oktay/Attention-Gated-Networks). 

### IB U-Net
Based on the paper [IB U-Net](https://arxiv.org/abs/2210.15949), and the codes are modified using the [repository](https://github.com/Shrajan/IB_U_Nets). 

### DWS U-Net
All the normal convolutions are replaced with depthwise-separable convolutions in the U-Net model. There are two versions:
1. First convolution is a normal convolution (patch-based), remaining are depthwise-separable convolutions (first patch - FpDWS).
2. All are are depthwise-separable convolutions (DWS).

## Prerequisites
nnUNet v2 differs from v1 by implementing the U-Net architectures in a separate repository called [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures), which is downloaded as part of the nnUNet package installation. So, any new architecture that one needs to use, they would need to create their model here. As such, I have implemented `Attention U-Net`, `IB U-Net`, and `DWS U-Net` in my [forked repository](https://github.com/Shrajan/dynamic-network-architectures) of the `dynamic-network-architectures`.

To make this work, you need to download my forked version of `nnUNet v2` as well as `dynamic-network-architectures`. The instructions are given [here](https://github.com/Shrajan/nnUNet/blob/master/documentation/installation_instructions.md). Please note the steps slightly differ from the original version.

## Usage
The original authors of nnUNet v2 have provided us with a detailed explanation of ways to [extend nnU-Net](https://github.com/Shrajan/nnUNet/blob/master/documentation/extending_nnunet.md), including using additional architecture presets, such as [residual encoder UNet](https://github.com/Shrajan/nnUNet/blob/master/documentation/resenc_presets.md). It is good to peruse them to get a better understanding, but it is not mandatory.

If you have already trained and tested the default U-Net architecture on your desired dataset, then you would have already executed the following commands. 
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID UNET_CONFIGURATION FOLD [additional options, see -h]
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION 
```
1. However, it is worth noting that, `nnUNetv2_plan_and_preprocess` uses the default `ExperimentPlanner`. This command creates a plans file called `nnUNetPlans` in the preprocessed dataset folder. By default `nnUNetv2_train` and `nnUNetv2_predict` make use of `nnUNetPlans`, unless you modify using the command line input.
2. Additionally, when you run `nnUNetv2_plan_and_preprocess`, in the background, these are used individually: `nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment` and `nnUNetv2_preprocess` (in that order).
3. So, to use extra U-Net variants, we need to use architecture-specific ***planners*** during preprocessing, and their corresponding ***plans*** file when training and testing.
4. The various architecture-specific ***planners*** can be found in `nnunetv2/experiment_planning/experiment_planners/`. For example, the planner for `Attention U-Net` is called as ***nnUNetPlannerAttUNet***, and the plans file are saved as ***nnUNetPlansAttUNet***.

Here, there are two possible courses of action during preprocessing, based on your previous activities:
1. Dataset has already been preprocessed (for 3d_fullres and/or 2d configurations).
```
nnUNetv2_plan_experiment -d DATASET_ID -pl nnUNetPlannerAttUNet
```
2. Dataset has not been preprocessed.
```
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerAttUNet
```

From now on, you can proceed with the new plans file.
```
nnUNetv2_train DATASET_ID UNET_CONFIGURATION FOLD -p nnUNetPlansAttUNet
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -p nnUNetPlansAttUNet
```

