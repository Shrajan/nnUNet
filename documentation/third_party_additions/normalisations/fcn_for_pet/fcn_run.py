import os, argparse, json, copy
import numpy as np
from medpy.metric.binary import dc, hd95
from amos_challenge_metrics import compute_surface_distances, compute_surface_dice_at_tolerance, compute_robust_hausdorff
from batchgenerators.utilities.file_and_folder_operations import save_json
import SimpleITK as sitk 
from math import nan
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple

def get_file_paths(in_folder: Path, pet_file_suffix: str)->Dict:
    
    # Read all the label file names.
    labelFiles = os.listdir(os.path.join(in_folder, "labelsTr"))
    labelFiles.sort()
    
    # Saves the full file paths of the required pet and label volumes.
    filePaths = {}
    
    # Get the full paths of all the required files: PET and label.
    for labelFile in labelFiles:
        patient_name = labelFile[:-7] # Remove the suffix ".nii.gz"
        labelFilePath = os.path.join(in_folder, "labelsTr", labelFile) # Full label file path
        petFile = patient_name + f"_{pet_file_suffix}.nii.gz" # Get pet file name
        petFilePath = os.path.join(in_folder, "imagesTr", petFile) # Full pet file path
    
        # Append to the dictionary that stores all the information.
        filePaths[patient_name] = {"petFilePath": petFilePath, 
                                   "labelFilePath": labelFilePath}

    print(f"Found {len(filePaths)} files in the source folder.")
    return filePaths
    
def fcn(filePaths: Dict, jsonFilePath: Path, intra: bool):
    
    intra_label = intra
    threshold_percent_vals = list(np.arange(0,100,1))

    allHighestDices = []
    allHighestDSCs = []
    allLowestHD95s = []
    
    allHighestDicesThresholds = []
    allHighestDSCsThresholds = []
    allLowestHD95sThresholds = []
    
    allHighestDicesPercents = []
    allHighestDSCsPercents = []
    allLowestHD95sPercents = []
        
    for patient_name in filePaths.keys():
        
        petFilePath = filePaths[patient_name]["petFilePath"]
        labelFilePath = filePaths[patient_name]["labelFilePath"]                          
        
        labelImage = sitk.ReadImage(labelFilePath)
        label_array = sitk.GetArrayFromImage(labelImage)
        
        if intra_label:
            labelLabels = np.unique(label_array)
            if len(labelLabels) < 2:
                print(f"Skipping {patient_name} as it does not contain any annotations and intra is True. label labels: {labelLabels}.")
                continue
        
        # Make sure the labels have values only 0 and 1.
        label_array = np.where(label_array != 0, 1, label_array)

        petImage = sitk.ReadImage(petFilePath)
        pet_array = sitk.GetArrayFromImage(petImage)
 
        suv_max = np.max(pet_array)
        highest_suv_inside_label = np.max(np.where(label_array==1, pet_array, 0))
        
        if intra_label:
            # Pet values outside label can be any value above SUX max. 
            # This is to distinguish regions outside the label during threshold.
            # Don't set Pet values outside label to 0 if '<=' is used in the comparison line. For example like this: prediction_array = np.where(pet_array<=threshold_val, 1, 0).
            # Set Pet values outside label to 0 if '>=' is used in the comparison line. For example like this: prediction_array = np.where(pet_array>=threshold_val, 1, 0).
            pet_array = np.where(label_array==1, pet_array, suv_max+1) 

        highest_dice = 0.0
        highest_dsc = 0.0
        lowest_hd95 = float('inf')
        
        highest_dice_threshold = 0.0
        highest_dsc_threshold = 0.0
        lowest_hd95_threshold = 0.0
        
        highest_dice_percent = 0.0
        highest_dsc_percent = 0.0
        lowest_hd95_percent = 0.0
        
        for each_threshold_percent in threshold_percent_vals:
            
            threshold_val = (each_threshold_percent/100) * suv_max
            prediction_array = np.where(pet_array<=threshold_val, 1, 0)

            prediction_array = prediction_array.astype(np.uint8)
            label_array = label_array.astype(np.uint8)
            
            prediction_array_sum = prediction_array.sum()
            label_array_sum = label_array.sum()
            
            # Checking if both arrays are empty.
            if prediction_array_sum + label_array_sum == 0:
                imgDSC = 1.0
                imgDC = 1.0
                imgHD95 = 0.0
            else:
                # Calculate the DSC for each patient.
                imgDSC = dc(reference=label_array, result=prediction_array)

                # Calculate the dice score.
                spacing_mm = labelImage.GetSpacing()
                surface_distances = compute_surface_distances(mask_gt=label_array, mask_pred=prediction_array, spacing_mm=spacing_mm)
                imgDC = compute_surface_dice_at_tolerance(surface_distances, max(spacing_mm))

                # Calculate hd-95.
                # If either of the array is empty, then Hausdorff is not possible.
                if prediction_array_sum == 0 or label_array_sum == 0:
                    imgHD95 = float('inf')
                else:
                    imgHD95 = compute_robust_hausdorff(surface_distances, 95)
                
                
            if imgDC > highest_dice:
                highest_dice = copy.copy(float(imgDC))
                highest_dice_threshold = copy.copy(float(threshold_val))
                highest_dice_percent = copy.copy(float(each_threshold_percent))

            if imgDSC > highest_dsc:
                highest_dsc = copy.copy(float(imgDSC))
                highest_dsc_threshold = copy.copy(float(threshold_val))
                highest_dsc_percent = copy.copy(float(each_threshold_percent))
                
            if imgHD95 < lowest_hd95:
                lowest_hd95 = copy.copy(float(imgHD95))
                lowest_hd95_threshold = copy.copy(float(threshold_val))
                lowest_hd95_percent = copy.copy(float(each_threshold_percent))
                
            print("P_ID:", patient_name,",",
                  "Percent:",round(each_threshold_percent,3),",",
                  "Max SUV",round(suv_max,3),",",
                  "Threshold @ Percent:",round(threshold_val,3),",",
                  "DSC:", round(imgDSC,3),
                  "DC:", round(imgDC,3),
                  "HD95:", round(imgHD95,3))
            
            # DSC and Dice greater than 1.0 is not possible, and HD95 lower than 0.0 is not possible.
            # Once reached, further percents don't matter.
            if highest_dice == 1.0 and highest_dsc== 1.0 and lowest_hd95 == 0.0:
                break
	        
        allHighestDices.append(highest_dice)
        allHighestDSCs.append(highest_dsc) 
        allLowestHD95s.append(lowest_hd95) 
           
        allHighestDicesThresholds.append(highest_dice_threshold)
        allHighestDSCsThresholds.append(highest_dsc_threshold)
        allLowestHD95sThresholds.append(lowest_hd95_threshold)
        
        allHighestDicesPercents.append(highest_dice_percent)
        allHighestDSCsPercents.append(highest_dsc_percent)  
        allLowestHD95sPercents.append(lowest_hd95_percent) 
    
    allInformation = {}
    allInformation["allHighestDices"] = allHighestDices
    allInformation["allHighestDSCs"] = allHighestDSCs
    allInformation["allLowestHD95s"] = allLowestHD95s
    allInformation["allHighestDicesThresholds"] = allHighestDicesThresholds
    allInformation["allHighestDSCsThresholds"] = allHighestDSCsThresholds
    allInformation["allLowestHD95sThresholds"] = allLowestHD95sThresholds
    allInformation["allHighestDicesPercents"] = allHighestDicesPercents
    allInformation["allHighestDSCsPercents"] = allHighestDSCsPercents
    allInformation["allLowestHD95sPercents"] = allLowestHD95sPercents
        
    save_json(allInformation, jsonFilePath)   
    
def aggregate_fcn(jsonFilePath: Path):
    fileInfo = open(jsonFilePath)
    allInformation = json.load(fileInfo)
    allHighestDices = allInformation["allHighestDices"]
    allHighestDSCs = allInformation["allHighestDSCs"]
    allLowestHD95s = allInformation["allLowestHD95s"]
    allHighestDicesThresholds = allInformation["allHighestDicesThresholds"]
    allHighestDSCsThresholds = allInformation["allHighestDSCsThresholds"]
    allLowestHD95sThresholds = allInformation["allLowestHD95sThresholds"]
    allHighestDicesPercents = allInformation["allHighestDicesPercents"]
    allHighestDSCsPercents = allInformation["allHighestDSCsPercents"]
    allLowestHD95sPercents = allInformation["allLowestHD95sPercents"]
    
    print(f"Number of files: {len(allHighestDices)}, {len(allHighestDicesThresholds)}, {len(allHighestDSCs)}, {len(allHighestDSCsThresholds)}")
     
    print("---------------Dice---------------")
    print(f"Metrics - median: {np.median(np.array(allHighestDices))}, mean: {np.mean(np.array(allHighestDices))}, mode: {stats.mode(np.array(allHighestDices))} ")
    print(f"Thresholds - median: {np.median(np.array(allHighestDicesThresholds))}, mean: {np.mean(np.array(allHighestDicesThresholds))}, mode: {stats.mode(np.array(allHighestDicesThresholds))} ") 
    print(f"Percents - median: {np.median(np.array(allHighestDicesPercents))}, mean: {np.mean(np.array(allHighestDicesPercents))}, mode: {stats.mode(np.array(allHighestDicesPercents))} ") 
    
    print("---------------DSC---------------")
    print(f"Metrics - median: {np.median(np.array(allHighestDSCs))}, mean: {np.mean(np.array(allHighestDSCs))}, mode: {stats.mode(np.array(allHighestDSCs))} ")
    print(f"Thresholds - median: {np.median(np.array(allHighestDSCsThresholds))}, mean: {np.mean(np.array(allHighestDSCsThresholds))}, mode: {stats.mode(np.array(allHighestDSCsThresholds))} ")
    print(f"Percents - median: {np.median(np.array(allHighestDSCsPercents))}, mean: {np.mean(np.array(allHighestDSCsPercents))}, mode: {stats.mode(np.array(allHighestDSCsPercents))} ")
    
    print("---------------HD95---------------")
    print(f"Metrics - median: {np.median(np.array(allLowestHD95s))}, mean: {np.mean(np.array(allLowestHD95s))}, mode: {stats.mode(np.array(allLowestHD95s))} ")
    print(f"Thresholds - median: {np.median(np.array(allLowestHD95sThresholds))}, mean: {np.mean(np.array(allLowestHD95sThresholds))}, mode: {stats.mode(np.array(allLowestHD95sThresholds))} ")
    print(f"Percents - median: {np.median(np.array(allLowestHD95sPercents))}, mean: {np.mean(np.array(allLowestHD95sPercents))}, mode: {stats.mode(np.array(allLowestHD95sPercents))} ")


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="gtv", required=True, help="Folder that contains the dataset in the nnUNet format.")
    parser.add_argument("--result_folder", type=str, default="petInfo", required=False, help="Folder to save the information in JSON format.")
    parser.add_argument("--pet_file_suffix", type=str, default="0000", required=False, 
                        help="""nnUNet format requires image files (CT, MR, PET, etc.) as 'filePrefix_ID_fileSuffix.nii.gz and label files as 'filePrefix_ID.nii.gz'.
                                We need the suffix assigned to the PET modality. Default: 0000""")
    parser.add_argument("--intra", action='store_true', required=False, help='If set, only intra-label SUV from the PET image will be considered. Default: False')
    parser.add_argument("--only_aggregate", action='store_true', required=False, help='If set, the saved json file will be read and aggregates will be printed. Default: False')
    opt = parser.parse_args()
    
    os.makedirs(opt.result_folder, exist_ok=True)
    jsonFilePath =  os.path.join(opt.result_folder, 
                                   f"summary_intra{str(opt.intra)}.json")

    print("All information will be stored in ", jsonFilePath)

    if opt.only_aggregate is False:
        print("Running FCN")
        filePaths = get_file_paths(in_folder=opt.in_folder, 
                                   pet_file_suffix=opt.pet_file_suffix)
        fcn(filePaths=filePaths, jsonFilePath=jsonFilePath, intra=opt.intra)
    
    print("Aggregating FCN results")
    aggregate_fcn(jsonFilePath=jsonFilePath)
