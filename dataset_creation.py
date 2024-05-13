import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from skimage import io,color, img_as_ubyte, exposure, filters, util, img_as_float, morphology, measure, img_as_float
from scipy.ndimage import gaussian_filter
from skimage import io, morphology, measure, img_as_ubyte

# =====================================================================================================
# CODE INFO
'''
This code aims to split the original dataset into training, validation, and test sets. The division is done randomly using a specific skLearn function. 
After dividing the images based on their names, the test set images are simply copied into the appropriate folder without any preprocessing, 
so they can be used to test the complete pipeline of the final code. Subsequently, the creation of the training and validation sets is addressed; 
in this case, the defined preprocessing is applied, and the images are divided into 256x256 pixel blocks. To fully study the effects of 
the various preprocessing stages, the images are saved not only at the end of the complete preprocessing but also after the various intermediate steps.
This will allow for individually testing the effects of each step to fully verify their effectiveness.
'''
# =====================================================================================================



# =====================================================================================================
# FUNCTION DEFINITION
# =====================================================================================================

def sigmoid(x):
    """
Function for calculating a simple sigmoid function, called by other functions later.

    Parameters:
        - x: parameter to pass to the sigmoid function
 
    Returns:
        - parameter processed through the sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def adaptive_gamma_lightening(img, block_size, gamma_min, gamma_max):
    """
Function for localized adaptive calculation of the gamma function

    Parameters:

        - img: image to process
        - block_size: size of the blocks into which the image is divided for local gamma calculation
        - gamma_min: minimum value that the calculated gamma can take
        - gamma_max: maximum value that the calculated gamma can take
    Returns:

        - corrected_img: image with modified gamma values ranging between 0 and 1
    """
      
    h, w = img.shape[:2]
    corrected_img = np.zeros_like(img, dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]
            avg_brightness = np.mean(block)

            # Mappare avg_brightness a un intervallo adatto per la sigmoide
            mapped_brightness = (avg_brightness - 0.5) * 12  

            # Applicare la sigmoide
            sigmoid_value = sigmoid(mapped_brightness)

            # Scala e trasforma l'output della sigmoide per rientrare nel range [gamma_min, gamma_max]
            gamma = gamma_min + (gamma_max - gamma_min) * sigmoid_value

            corrected_block = np.power(block, gamma)
            corrected_img[y:y+block_size, x:x+block_size] = corrected_block

    return np.clip(corrected_img, 0, 1)

def divide_into_blocks(img, block_size=(256,256)):
    """
    Function to divide the image into sub-images

        Parameters:
        - img: image to process
        - block_size: size of the blocks into which the image is divided for local gamma calculation

    Returns:
        - blocks: list containing the sub-images into which the original image has been divided
    """
    blocks = []
    for i in range(0, img.shape[0], block_size[0]):
        for j in range(0, img.shape[1], block_size[1]):
            block = img[i:i + block_size[0], j:j + block_size[1]]
            if block.shape[0] != block_size[0] or block.shape[1] != block_size[1]:
                block = np.pad(block, ((0, block_size[0] - block.shape[0]), (0, block_size[1] - block.shape[1])), 'constant')
            blocks.append(block)
    return blocks

def save_image(image, image_path, results_folder):
    """
    Function to save the image, either in complete form or by dividing it into blocks and saving them in their respective folders

    Parameters:
        - image: image to save
        - image_path: full path of the image, useful for isolating the image name
        - results_folder: path where the image will be saved

    Returns:
        - None (only performs saving, does not return anything)
    """
    
    image = np.clip(image,0,1)
    SaveImage = img_as_ubyte(image)
    
    filename_image = os.path.basename(image_path)
    
    complete_filename_image = results_folder+"/Complete_Original/"+filename_image
    if not os.path.exists(results_folder+'/Complete_Original'):
        os.makedirs(results_folder+'/Complete_Original')
    io.imsave(complete_filename_image, SaveImage)
    
    blocks_preprocessed = divide_into_blocks(SaveImage, block_size=(256,256))
    
    if not os.path.exists(results_folder+'/Blocks_Original'):
            os.makedirs(results_folder+'/Blocks_Original')
    for i, block in enumerate(blocks_preprocessed):
        block_filename = results_folder+"/Blocks_Original/"+filename_image+"_"+str(i)+".png"
        io.imsave(block_filename, block)
    return None

def preprocess(image_path, mask_path, results_folder):
    """
    Function to perform image preprocessing

    Parameters:
        - image_path: full path of the image
        - mask_path: full path of the mask related to the image
        - results_folder: path where the image will be saved

    Returns:
        - None (only performs saving, does not return anything)
    """
    
    image = io.imread(image_path)
    # Eye masks
    dim=image.shape
    gray_image = color.rgb2gray(image)
    gray_image = img_as_float(gray_image)
    gray_image=gaussian_filter(gray_image,sigma=5,mode='constant', cval=0.0,truncate=2.0)
    gray_image = exposure.equalize_adapthist(gray_image,clip_limit=0.019)
    threshold = filters.threshold_minimum(gray_image)
    bin_img = gray_image>threshold
    closed_image = morphology.remove_small_holes(bin_img,area_threshold=(dim[0]/8)**2*3)
    label_image = measure.label(closed_image)
    regions = measure.regionprops(label_image)

    if regions:
        largest_region = max(regions, key=lambda r: r.area)  
        eye_mask = label_image == largest_region.label
    else:
        eye_mask = closed_image

    image_green = image[:, :, 1]*eye_mask  # Estrai solo il canale verde
    image_green = img_as_float(image_green)
    
    save_image(image_green,image_path,results_folder=results_folder+'/v1')

    image_green = min_max_normalization(image_green)

    # Apply gaussian filter
    ris_min = min(image_green.shape)
    # Calculate the result of ris_min / 100
    value = ris_min / 100

    # Convert result to the nearest integer
    nearest_int = int(value)

    # If nearest_int is odd and not greater than result, return it
    if nearest_int % 2 != 0 and nearest_int <= value:
        dimension = nearest_int
    else:
        # If nearest_int is even or greater than result, return the previous odd number
        dimension = (nearest_int - 1 if nearest_int % 2 == 0 else nearest_int-2)

    #sigma = 0.15*dimension
    sigma=2.5
    truncate = ((dimension-1)/2-0.5)/sigma

    image_green_gaussian = filters.gaussian(image_green, sigma=sigma, truncate=truncate)
    
    save_image(image_green_gaussian,image_path,results_folder=results_folder+'/v2')

    # Apply CLAHE equalisation
    image_green_clahe = exposure.equalize_adapthist(image_green_gaussian, clip_limit=0.01)
    save_image(image_green_clahe,image_path,results_folder=results_folder+'/v3')
    
    # Apply gamma correction
    image_gamma = (adaptive_gamma_lightening(image_green_clahe, block_size=5, gamma_min=0.7, gamma_max=1))*eye_mask
    save_image(image_gamma,image_path,results_folder=results_folder+'/v4')
    # Apply Black Top-Hat Transform
    radius=int(max(image_gamma.shape)*0.0045)
    radius=np.clip(radius,8,15)
    transformed_img = exposure.rescale_intensity(morphology.black_tophat(image_gamma, footprint=morphology.disk(radius)))
    image_pre = (image_gamma-transformed_img)
    save_image(image_pre,image_path,results_folder=results_folder+'/final')
    
    # Elaborazione delle maschere, serve solo a salvare le maschere divise
    mask = io.imread(mask_path)
    filename_image = os.path.basename(mask_path)
    blocks_preprocessed = divide_into_blocks(mask, block_size=(256,256))
    if not os.path.exists(results_folder+'/Blocks_Mask'):
            os.makedirs(results_folder+'/Blocks_Mask')
    for i, block in enumerate(blocks_preprocessed):
        block_filename = results_folder+"/Blocks_Mask/"+filename_image+"_"+str(i)+".png"
        io.imsave(block_filename, block)

# Definizione funzioni utilizzate in seguito
def min_max_normalization(matrix):   #Funzione di normalizzazione usata in seguito
    """
    Normalize a matrix using Min-Max normalization.

    Parameters:
        - matrix: ndarray, the matrix to normalize

    Returns:
        - normalized_matrix: ndarray, the normalized matrix
    """
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    matrix_normalizzata = (matrix - min_value) / (max_value - min_value)

    return matrix_normalizzata


def get_file_paths(directory):
    '''
    Function to generate a list of full file paths contained in the directory passed as a parameter

    Parameters:
        - directory: path of the directory to scan

    Returns:
        - file_paths: list of full file paths contained in the directory
    '''
    return [os.path.join(directory, file) for file in os.listdir(directory)]

# =====================================================================================================
# PATHS DEFINITION
# =====================================================================================================
working_folder = os.path.dirname(os.path.abspath(__file__))
original_dataset = working_folder+'/Dataset/RETINA/'
original_images_dir = original_dataset+'Original'
original_masks_dir = original_dataset+'Ground_truth'

print('Percorso original_images: '+original_images_dir)
print('Percorso original_masks: '+original_masks_dir)

# Get the file paths
original_images = get_file_paths(original_images_dir)
original_masks = get_file_paths(original_masks_dir)



# =====================================================================================================
# DATASET DIVISION AND PREPARATION
# =====================================================================================================
train_image_paths, temp_image_paths, train_mask_paths, temp_mask_paths = train_test_split(original_images, original_masks, test_size=0.3, random_state=2023)
val_image_paths, test_image_paths, val_mask_paths, test_mask_paths = train_test_split(temp_image_paths, temp_mask_paths, test_size=0.5, random_state=2023)

train_images_names = [os.path.basename(path) for path in train_image_paths]
validation_images_names = [os.path.basename(path) for path in val_image_paths]           
test_images_names = [os.path.basename(path) for path in test_image_paths]



# =====================================================================================================
# TEST SET
# =====================================================================================================
test_folder = working_folder+'/Dataset'
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Cycle to save the images
files_copied = 0  
for file_name in tqdm(test_images_names, desc="Immagini test set"):
    source_file = os.path.join(original_images_dir, file_name)
    images_folder = test_folder+'/test/Original'
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    destination_file = os.path.join(images_folder, file_name)
    
    # Check if the file exists in the source folder before copying it
    if os.path.exists(source_file):
        try:
            shutil.copy(source_file, destination_file)
            files_copied += 1
        except Exception as e:
            print(f"Errore durante la copia di {file_name}: {e}")
    else:
        print(f"Il file {file_name} non esiste in {original_images}")
print(f"Totale file copiati nella cartella test/Original: {files_copied}")

# Cycle to save masks
files_copied = 0  
for file_name in tqdm(test_images_names, desc="Maschere test set"): # The names must be common between masks and images, so a single variable is used
    source_file = os.path.join(original_masks_dir, file_name)
    mask_folder = test_folder+'/test/Ground_truth'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    destination_file = os.path.join(mask_folder, file_name)
    
    # Check if the file exists in the source folder before copying it
    if os.path.exists(source_file):
        try:
            shutil.copy(source_file, destination_file)
            files_copied += 1
        except Exception as e:
            print(f"Errore durante la copia di {file_name}: {e}")
    else:
        print(f"Il file {file_name} non esiste in {original_images}")
print(f"Totale file copiati nella cartella test/Ground_truth: {files_copied}")
print('Terminata sezione per salvataggio test set')




# =====================================================================================================
# TRAINING SET
# =====================================================================================================
print('Avvio preparazione training set')
for file_name in tqdm(train_images_names, desc="Elaborazione training set"):
        image_file = os.path.join(original_images_dir, file_name)
        mask_file = os.path.join(original_masks_dir, file_name)
        results_folder = test_folder+'/train'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        try:
            preprocess(image_file,mask_file,results_folder)
        except Exception as e:
            print(f"Errore durante il preprocess dell'immagine {source_file}: {e}")
            
            
# =====================================================================================================
# VALIDATION SET
# =====================================================================================================
print('Avvio preparazione validation set')
for file_name in tqdm(validation_images_names, desc="Elaborazione validation set"):
        image_file = os.path.join(original_images_dir, file_name)
        mask_file = os.path.join(original_masks_dir, file_name)
        results_folder = test_folder+'/val'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        try:
            preprocess(image_file,mask_file,results_folder)
        except Exception as e:
            print(f"Errore durante il preprocess dell'immagine {source_file}: {e}")