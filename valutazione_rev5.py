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
#INFO CODICE
'''
Questo codice è il codice utilizzato per valutare i risultati dell'inference eseguita da una rete secondo le metriche utili a questa challenge (inclusa anche l'IoU).
Vista il funzionamento previsto dei vari script, in questo caso la valutazione viene eseguita usando tutte le immagini presenti nelle varie sottocartelle relative alla cartella data come riferimento,
utilizzando il multiprocess per velocizzare l'analisi in caso di più sottocartelle, relative quindi a più trial o esperimenti. Al termine dell'analisi i valori vengono poi scritti
all'interno di un file con una precisa struttura, creata per poter agevolare la successiva analisi tramite Excel oppure dall'apposito script "Analisi_risultati_rev5", che permette
di analizzare automaticamente tutti i risultati relativi ai vari trial per creare una classifica che ordini i vari trial/esperimenti in base alla qualità dei risultati
'''
# =====================================================================================================

# Definizione cartella di lavoro
working_folder = os.path.dirname(os.path.abspath(__file__))
#gt_dir = working_folder+"/Dataset/test/Ground_truth"
gt_dir = working_folder+"/Dataset/RETINA/Ground_truth"

# ==============================================================
# DEFINIZIONE PARAMETRI UTILI

#cartella_maschere = working_folder + '/Output_final_code/'
#cartella_maschere = working_folder + '/Output_test/'
cartella_maschere = working_folder + '/Output_per_tabelle'    # Questa sarà la cartella dove andare ad applicare il multiprocess

# N.B. Righe 93 e 156 vanno commentate quando si esegue l'analisi di immagini con postprocess (debito tecnico)

# Alla riga 268 si trova invece il codice necessario per analizzare una cartella specifica
# ==============================================================

# Funzione per il calcolo della metrica IoU
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
    if somma_unione >0:                                 #Modifica alla funzione effettuata per evitare NaN in caso di assenza di maschere
        iou = np.sum(intersection) / somma_unione
        #if iou==0:           
        #    iou=0.001        
    else:
        iou=1
    return iou

# Funzione per il calcolo del Dice
def dice(outputs, labels):
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                 #Modifica alla funzione effettuata per evitare NaN in caso di assenza di maschere
        dice_score = 2.0 * intersection / (union)    
    else:
        dice_score=1.0
        
    return dice_score

# Funzione per il calcolo delle metriche IoU e Dice, compreso il valore medio sopra 80 percentile e sotto 20 percentile
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
        
        #if file_extension.lower() == '.jpg':  # Controlla se l'estensione è .jpg indipendentemente dal caso
        #    gt_filename = base_filename + '.png'  # Sostituisci con .tif
        #else:
        gt_filename = filename  # Se l'estensione non è .jpg, lascia il nome del file inalterato, fallo rientrare se attivi costrutti sopra
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

# Funzione per il calcolo degli errori J2J e J2E, compreso nei valori percentili come indicato in precedenza
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


# Funzione per scrivere i risultati in un file con file locking (meccanismo che garantisce la scrittura senza problemi sullo stesso file da parte dei vari processi)
def write_results_to_file(pred_dir, results, file_path):
    lock_path = file_path + ".lock"
    lock = SoftFileLock(lock_path)

    with lock:
        try:
            with open(file_path, 'a') as file:
                file.write(f"{os.path.basename(pred_dir)}: \n")  # Scrive pred_dir come prima riga
                for key, value in results.items():
                    file.write(f"{key}: {value}\n")
                file.write('\n\n\n')  # Aggiunge una riga vuota dopo ogni coppia chiave-valore
            print(f"Results successfully written to {file_path}")
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

# Funzione per processare ogni sottocartella (inclusa la valutazione e la scrittura dei risultati)
def process_folder(pred_dir, gt_dir):
    path_temp = Path(pred_dir)
    general_folder = path_temp.parent
    file_path = os.path.join(general_folder, 'results.txt')
    evaluation_metrics = evaluate_metrics(pred_dir, gt_dir)
    print(evaluation_metrics)
    evaluation_results = evaluate_errors(pred_dir, gt_dir)
    combined_results = {**evaluation_metrics, **evaluation_results}
    write_results_to_file(pred_dir, combined_results, file_path)
    
# Funzione richiamata per il process della directory
def process_directory(pred_dir, gt_dir):
    if os.path.isdir(pred_dir):
        process_folder(pred_dir, gt_dir)
    else:
        print(f"The path {pred_dir} is not a directory.")


# Parte per esecuzione singola della funzione per process
'''
pred_dir = working_folder + '/Output_metriche_val/us1zgf45_AuZf5'
process_directory(pred_dir,gt_dir)
'''
# Ottieni la lista delle sottocartelle in cartella_maschere
subfolders = [os.path.join(cartella_maschere, o) for o in os.listdir(cartella_maschere) 
              if os.path.isdir(os.path.join(cartella_maschere, o))]

all_params = list(zip([d for d in subfolders], [gt_dir] * len(subfolders)))
num_workers = 3   # Numero di processi da avviare in parallelo

# Process the combinations in parallel using Pool
with Pool(processes=num_workers) as pool:
    try:
        pool.starmap(process_directory, all_params)
    except Exception as e:
        print(f"Error in multiprocessing block: {e}")