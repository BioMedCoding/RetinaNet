import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from tqdm import tqdm
import os
import threading
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations import Compose
from filelock import SoftFileLock
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from multiprocessing import Pool
from pathlib import Path

# =====================================================================================================
# CODE INFO
'''
This code is used to evaluate the inference results performed by a network according to the choosen metrics. 
Given the intended operation of the various scripts, in this case, the evaluation is performed using all the images present in the various 
subfolders relative to the reference folder. Multiprocessing is used to speed up the analysis in the case of multiple subfolders, 
which correspond to multiple trials or experiments. At the end of the analysis, the values are then written to a file with a specific structure, 
created to facilitate subsequent analysis via Excel or the specific script "Results_analysis", which allows automatic analysis 
of all results related to the various trials to create a ranking that orders the various trials/experiments based on the quality of the results.

Uncomment the section at line 274 to evaluate a single folder. If nothing is changed, multiprocess analysis is the standard
Uncomment also 96 and 171 if single folder analysis is selected
'''
# =====================================================================================================

# working path definition
working_folder = os.path.dirname(os.path.abspath(__file__))

# ground truth definition, the images inside this folder will be used as reference to calculate the metrics
gt_dir = working_folder+"/Dataset/RETINA/Ground_truth"

# ==============================================================
# USEFUL PARAMETERS DEFINITION

# This is the folder in which there are multiple trials to evaluate using multiprocess
cartella_maschere = working_folder + '/Output_per_tabelle'    

images_have_postprocess = True

# ==============================================================

# Function to compute the IoU metric
def IOU(outputs, labels):
    
    # Convert PyTorch tensors to NumPy if necessary
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Check if the output and labels are of the same size
    if outputs.shape != labels.shape:
        raise ValueError(f"Expected output size ({outputs.shape}) to be same as target size ({labels.shape})")

    # Convert labels to boolean array
    labels = labels.astype(bool)
    # Threshold outputs to obtain binary values
    outputs = outputs > 0.5
    # Compute intersection and union
    intersection = np.logical_and(labels, outputs)
    union = np.logical_or(labels, outputs)
    # Compute IOU
    somma_unione = np.sum(union)
    if somma_unione >0:                             # This is done to avoid NaN in case of empty masks
        iou = np.sum(intersection) / somma_unione
        #if iou==0:           
        #    iou=0.001        
    else:
        iou=1
    return iou

# Function to compute the Dice metric
def dice(outputs, labels):
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                    # This is done to avoid NaN in case of empty masks
        dice_score = 2.0 * intersection / (union)    
    else:
        dice_score=1.0
        
    return dice_score

# Function called to avaluate both metrics, considering also 80th and 20th percentile
def evaluate_metrics(pred_dir, gt_dir):
    #pred_dir = pred_dir + '/noPost'
    pred_filenames = os.listdir(pred_dir)
    iou_scores = []
    dice_scores = []
    
    # Transformation of images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        lambda x: torch.round(x)
    ])
    
    # Calculation of IoU and Dice scores
    for filename in tqdm(pred_filenames):
        pred_path = os.path.join(pred_dir, filename)
        
        base_filename, file_extension = os.path.splitext(filename)
        
        # This section can be used in case of a system with different images format
        '''
        if file_extension.lower() == '.jpg':  # Check if file extension is .jpg
            gt_filename = base_filename + '.png'  # Substitute with .tif
        else:
        '''
        gt_filename = filename  
        gt_path = os.path.join(gt_dir, gt_filename)
        
        print(gt_path)
        pred_mask = transform(Image.open(pred_path))
        gt_mask = transform(Image.open(gt_path))

        iou_score = IOU(pred_mask.squeeze(), gt_mask.squeeze())
        dice_score = dice(pred_mask.squeeze(), gt_mask.squeeze())
        
    
        if not np.isnan(iou_score):
            iou_scores.append(iou_score)
            dice_scores.append(dice_score)
            
        
    # Calculation of overall mean values
    mean_iou_total = np.mean(iou_scores)
    mean_dice_total = np.mean(dice_scores)

    # Calculation of standard deviation for mean_iou and mean_dice
    std_iou_total = np.std(iou_scores)
    std_dice_total = np.std(dice_scores)

    # Calculation of means within percentiles
    low_iou_threshold = np.percentile(iou_scores, 20)
    high_iou_threshold = np.percentile(iou_scores, 80)
    low_dice_threshold = np.percentile(dice_scores, 20)
    high_dice_threshold = np.percentile(dice_scores, 80)

    mean_iou_low = np.mean([score for score in iou_scores if score <= low_iou_threshold])
    mean_iou_high = np.mean([score for score in iou_scores if score >= high_iou_threshold])
    mean_dice_low = np.mean([score for score in dice_scores if score <= low_dice_threshold])
    mean_dice_high = np.mean([score for score in dice_scores if score >= high_dice_threshold])

    metrics = {
        'mean_iou_total': mean_iou_total,
        'mean_dice_total': mean_dice_total,
        'std_iou_total': std_iou_total,  
        'std_dice_total': std_dice_total,  
        'mean_iou_low': mean_iou_low,
        'mean_iou_high': mean_iou_high,
        'mean_dice_low': mean_dice_low,
        'mean_dice_high': mean_dice_high,
    }
    return metrics

# Function to compute the J2J (Junction to Junction) and J2E (Junction to End) errors
def evaluate_errors(pred_dir, gt_dir):
    #pred_dir = pred_dir + '/noPost' 
    results_pred = {}
    results_gt = {}
    
    for dataset, results in [(pred_dir, results_pred), (gt_dir, results_gt)]:
        for file in (os.listdir(dataset)):
            if file.endswith('.png'):
                file_path = os.path.join(dataset, file)
                
                # Reading and skeletonizing the image
                try:
                    img = Image.open(file_path).convert('L')
                    img_array = np.array(img)
                    skeleton = skeletonize(img_array)
                    if np.max(skeleton) == 0:
                        print(f"No structure found in {file_path}, skipping...")
                        continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
                
                # Analyzing the skeleton
                try:
                    skeleton_data = Skeleton(skeleton)
                    skel_summary = summarize(skeleton_data)
                    N_J2E = len(skel_summary[skel_summary['branch-type'] == 1])
                    N_J2J = len(skel_summary[skel_summary['branch-type'] == 2])
                    results[file] = {'N_J2E': N_J2E, 'N_J2J': N_J2J}
                except ValueError as e:
                    print(f"Error in skeleton analysis: {e}")
                    continue

    common_files = set(results_pred.keys()).intersection(set(results_gt.keys()))
    
    diffs_N_J2E = []
    diffs_N_J2J = []
    
    for file in common_files:
        original = results_gt[file]
        predicted = results_pred[file]
        diff_N_J2E = abs((original['N_J2E'] - predicted['N_J2E']) / original['N_J2E'])
        diff_N_J2J = abs((original['N_J2J'] - predicted['N_J2J']) / original['N_J2J'])
        diffs_N_J2E.append(diff_N_J2E)
        diffs_N_J2J.append(diff_N_J2J)
    
    mean_diff_N_J2E = np.mean(diffs_N_J2E)
    std_diff_N_J2E = np.std(diffs_N_J2E)
    mean_diff_N_J2J = np.mean(diffs_N_J2J)
    std_diff_N_J2J = np.std(diffs_N_J2J)
    
    percentile_20_N_J2E = np.percentile(diffs_N_J2E, 20)
    percentile_80_N_J2E = np.percentile(diffs_N_J2E, 80)
    percentile_20_N_J2J = np.percentile(diffs_N_J2J, 20)
    percentile_80_N_J2J = np.percentile(diffs_N_J2J, 80)
    
    mean_diff_low_N_J2E = np.mean([diff for diff in diffs_N_J2E if diff <= percentile_20_N_J2E])
    mean_diff_high_N_J2E = np.mean([diff for diff in diffs_N_J2E if diff >= percentile_80_N_J2E])
    mean_diff_low_N_J2J = np.mean([diff for diff in diffs_N_J2J if diff <= percentile_20_N_J2J])
    mean_diff_high_N_J2J = np.mean([diff for diff in diffs_N_J2J if diff >= percentile_80_N_J2J])
    
    errors = {
        'mean_diff_N_J2E': mean_diff_N_J2E,
        'std_diff_N_J2E': std_diff_N_J2E,
        'mean_diff_N_J2J': mean_diff_N_J2J,
        'std_diff_N_J2J': std_diff_N_J2J,
        'mean_diff_low_N_J2E': mean_diff_low_N_J2E,
        'mean_diff_high_N_J2E': mean_diff_high_N_J2E,
        'mean_diff_low_N_J2J': mean_diff_low_N_J2J,
        'mean_diff_high_N_J2J': mean_diff_high_N_J2J
    }
    
    return errors

# Function to write results in a dedied file, with a file locking mechanism to avoid problem in case of multiprocess
def write_results_to_file(pred_dir, results, file_path):
    lock_path = file_path + ".lock"
    lock = SoftFileLock(lock_path)

    with lock:
        try:
            with open(file_path, 'a') as file:
                file.write(f"{os.path.basename(pred_dir)}: \n")  # Write pred_dir as first line
                for key, value in results.items():
                    file.write(f"{key}: {value}\n")
                file.write('\n\n\n')  # Add a blank paragraph after each line
            print(f"Results successfully written to {file_path}")
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

# Function to process every subfolder (including results evaluation and writing)
def process_folder(pred_dir, gt_dir):
    path_temp = Path(pred_dir)
    general_folder = path_temp.parent
    file_path = os.path.join(general_folder, 'results.txt')
    evaluation_metrics = evaluate_metrics(pred_dir, gt_dir)
    print(evaluation_metrics)
    evaluation_results = evaluate_errors(pred_dir, gt_dir)
    combined_results = {**evaluation_metrics, **evaluation_results}
    write_results_to_file(pred_dir, combined_results, file_path)
    
# Function to process the directory
def process_directory(pred_dir, gt_dir):
    if os.path.isdir(pred_dir):
        process_folder(pred_dir, gt_dir)
    else:
        print(f"The path {pred_dir} is not a directory.")


# Uncomment this section to process a single folder
'''
pred_dir = working_folder + '/Output_metriche_val/us1zgf45_AuZf5'
process_directory(pred_dir,gt_dir)
'''
# Ottieni la lista delle sottocartelle in cartella_maschere
# Get a list of all subfolders in the cartelle_maschere folder (the folder intended to be analysed)
subfolders = [os.path.join(cartella_maschere, o) for o in os.listdir(cartella_maschere) 
              if os.path.isdir(os.path.join(cartella_maschere, o))]

all_params = list(zip([d for d in subfolders], [gt_dir] * len(subfolders)))
num_workers = 3   # Number of processes to be used in parallel

# Process the combinations in parallel using Pool
with Pool(processes=num_workers) as pool:
    try:
        pool.starmap(process_directory, all_params)
    except Exception as e:
        print(f"Error in multiprocessing block: {e}")