import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io,color, img_as_ubyte, exposure, filters, util, img_as_float, morphology, measure, restoration, img_as_float
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import torch 
import torch.nn as nn
from torchvision import transforms
from skimage import io, morphology, measure, draw, img_as_ubyte
from scipy.spatial import KDTree
import skimage.draw
import numpy as np
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.filters import threshold_otsu
import torch.nn.functional as F
from multiprocessing import Pool
from itertools import product

# =====================================================================================================
#INFO CODICE
'''
Questo codice è il codice utilizzato per testare il sistema sul test set e trovare i parametri migliori per il post-process
Prende quindi le immagini presente nella cartella, applica il preprocess, divide le immagini in blocchi da 256x256,
esegue l'inference tramite la rete, ricompone un'unica immagine e a questo punto \, usando il multiprocess, 
testa tutte le possibili combinazioni dei parametri di post-process, salvando ogni volta i risultati. 
Questi vengono poi analizzati dagli appositi script per individuare in maniera semi-automatica i parametri più
adatti al post-process e da inserire quindi nel codice finale
Trattandosi di un codice usato solo per ottenere i valori di postprocess ottimizzati e non essendo pertanto
destinato ad essere eseguito in seguito, potrebbe contenere porzioni commentate relative a scelte progettuali 
esplorate e poi abbandonate
'''
# =====================================================================================================

# Per avviare il codice:
# 1- Cambia experiment e trial id, selezionando quelli di interesse
# 3- Se usi architettura non presente nel codice, aggiungila

# NB LA FUNZIONE DI INFERENCE ADESSO NON ESEGUE SIGMOIDE SU LOGITS DI OUTPUT PERCHÉ SI STA USANDO RESTNET CHE FA GIÀ SIGMOIDE, MA PER USARE RESNET BISOGNA CAMBIARLA SCOMMENTANDO QUELLO CHE ADESSO È COMMENTATO!!!!

# =====================================================================================================
# Definizione search_space dei parametri del post-process
# =====================================================================================================

# Definizione search space per i parametri del postprocess
v_bin_options = [0.4, 0.5, 0.6]
v_bin = 0.5
distance_threshold_options = [35, 40, 45, 50, 55, 60]
thickness_options = [4,5,6,7]
block_size = (256,256) # Definizione dei blocchi in cui dividere
rete = "test"
versione = "1"
isOtsu = False
v_otsu_options = [2]

# Definizione parametri per valutare esperimento 
experiment_id = '8rkvczpq'      # Esperimento finale: 8rkvczpq

# Definizione percorsi di riferimento
working_folder = os.path.dirname(os.path.abspath(__file__))
dataset_folder = working_folder+'/Dataset/test/'
experiment_path =   working_folder + '/modelli_allenati/'+experiment_id

# Per testare un singolo trial
trial_id = 'VXh13'             

single_trial = True  # Booleano che se vero testa solo la rete indicata tramite trial_id, altrimenti testa tutte le reti disponibili all'interno dell'experiment_path; creato per agevolare il testing di esperimenti con numerosi trial

# Definizione nome trial, nel caso di esecuzione su singolo trial
nome_trial = experiment_id+'_'+trial_id
model_path = experiment_path+'/'+trial_id
print(nome_trial) # Stampa nome per conferma e controllo

def apply_postprocess(params):
    image_path, working_folder, versione, isOtsu, v_otsu, v_bin, distance_threshold, thickness, rete, final_mask, block_size = params
    try:
        final_mask_uint8 = postprocess(final_mask, isOtsu, v_otsu, v_bin, distance_threshold, thickness, rete, block_size)
        if isOtsu:
            method = '_otsu'+str(v_otsu)
        else:
            method = '_sigmoid'+str(v_bin)
        results_folder = working_folder+'/cestino/Output_'+rete+'_v'+versione+'_d'+str(distance_threshold)+method+'_t'+str(thickness)    
        if not os.path.exists(results_folder):
                os.makedirs(results_folder)
        filename_image = os.path.basename(image_path)
        complete_filename_image = results_folder+'/'+filename_image
        io.imsave(complete_filename_image, final_mask_uint8)
        return None
    except Exception as e:
        print(f"An error occurred with parameters ({isOtsu}, {v_otsu}, {v_bin}, {distance_threshold}, {thickness}, {rete}): {e}")
        return None
        
# Definizione funzioni utilizzate in seguito
def min_max_normalization(matrix):   #Funzione di normalizzazione usata in seguito
    """
    Normalizza una matrice utilizzando la normalizzazione Min-Max.
    
    Parametri:
    - matrix: ndarray, la matrice da normalizzare

    Restituisce:
    - matrix_normalizzata: ndarray, la matrice normalizzata
    """
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    matrix_normalizzata = (matrix - min_value) / (max_value - min_value)

    return matrix_normalizzata
            
def draw_thick_line(pt1, pt2, thickness):
    """
    Traccia una linea di spessore scelto tra i due punti indicati
    
    Parametri:
    - pt1, pt2: punti da colleggare
    - thickness: spessore della linea da tracciare 

    Restituisce:
    - rr_all, cc_all: coordinate modificate
    """
    rr, cc = skimage.draw.line(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
    coords = [(rr, cc)]

    for i in range(1, thickness):
        # Calcolare le coordinate per linee aggiuntive
        rr_offset1, cc_offset1 = skimage.draw.line(int(pt1[0]+i), int(pt1[1]), int(pt2[0]+i), int(pt2[1]))
        rr_offset2, cc_offset2 = skimage.draw.line(int(pt1[0]-i), int(pt1[1]), int(pt2[0]-i), int(pt2[1]))
        coords.append((rr_offset1, cc_offset1))
        coords.append((rr_offset2, cc_offset2))

    # Combinare tutte le coordinate
    rr_all = np.concatenate([c[0] for c in coords])
    cc_all = np.concatenate([c[1] for c in coords])

    return rr_all, cc_all

# Funzione per controllare se due bounding boxes sono abbastanza vicini
def boxes_are_close(box1, box2, threshold):
    return (box1[2] >= box2[0] - threshold and box2[2] >= box1[0] - threshold and
            box1[3] >= box2[1] - threshold and box2[3] >= box1[1] - threshold)

# Funzione per il controllo del vicinato e ricerca KDTree
def check_and_search(region1, region2, tree1, tree2, box1, box2, threshold):
    if boxes_are_close(box1, box2, threshold):
        dist, idx2 = tree2.query(region1.coords, k=1, distance_upper_bound=threshold)
        valid_indices = np.where(dist < threshold)[0]
        if valid_indices.size > 0:
            min_dist_idx = idx2[valid_indices[0]]
            if min_dist_idx < len(region2.coords):
                closest_point_region1 = region1.coords[valid_indices[0]]
                closest_point_region2 = region2.coords[min_dist_idx]
                return (closest_point_region1, closest_point_region2)
    return None

def remove_small_regions(binary_image, area_threshold):
    """
    Rimuova zone da un'immagine binaria che hanno area inferiore al valore selezionato
    
    Parametri: 
    - binary_image: immagine binaria su cui lavorare
    - area_threshold: soglia di area sotto la quale si deve rimuovere
    
    Restituisce:
    - binary_image: immagine con zone rimosse
    """
    # Label the image
    labeled_image = measure.label(binary_image, connectivity=2)

    # Get region properties
    regions = measure.regionprops(labeled_image)

    # Create a mask to hold the regions to be removed
    mask_to_remove = np.zeros_like(binary_image, dtype=bool)

    # Iterate over the regions
    for region in regions:
        # If the region area is below the threshold, add it to the mask
        if region.area < area_threshold:
            mask_to_remove[labeled_image == region.label] = True

    # Set the regions in the mask to 0 in the original image
    binary_image[mask_to_remove] = 0

    return binary_image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def adaptive_gamma_lightening(img, block_size, gamma_min, gamma_max):
    """
    Funzione per il calcolo adattivo localizzato della funzione gamma
    Parametri: 
    - img: immagine su cui lavorare
    - block_size: dimensione dei blocchi in cui viene divisa l'immagine per il calcolo locale della funzione di gamma
    - gamma_min: valore minimo che può assumere la gamma calcolata
    - gamma_max: valore massimo che può assumere la gamma calcolata
    
    Restituisce:
    - corrected_img: immagine con gamma modificata e valori compresi tra 0 e 1
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

#def divide_into_blocks(img, block_size=(256, 256)):
def divide_into_blocks(img, block_size):
    """
    Funzione per dividere in sottoimmagine l'immagine
    
    Parametri: 
    - img: immagine su cui lavorare
    - block_size: dimensione dei blocchi in cui viene divisa l'immagine per il calcolo locale della funzione di gamma
    
    Restituisce:
    - blocks: lista contenente le sottoimmagini in cui si è divisa l'immagine originale
    """
    blocks = []
    for i in range(0, img.shape[0], block_size[0]):
        for j in range(0, img.shape[1], block_size[1]):
            block = img[i:i + block_size[0], j:j + block_size[1]]
            if block.shape[0] != block_size[0] or block.shape[1] != block_size[1]:
                block = np.pad(block, ((0, block_size[0] - block.shape[0]), (0, block_size[1] - block.shape[1])), 'constant')
            blocks.append(block)
    return blocks

def reassemble_blocks(blocks, original_shape, block_size):
    """
    Funzione per riassemblare l'immagine originale partendo dalle singole sottoimmagini
    
    Parametri: 
    - blocks: lista contenente tutte le sottoimmagini da riassemblare
    - original_shape: dimensione dell'immaigne originale
    - block_size: dimensione dei blocchi in cui viene divisa l'immagine per il calcolo locale della funzione di gamma
    
    Restituisce:
    - final_image_cropped: immagine originale, ricomposta dalle singole sottoimmagini
    """
    # Calcola il numero di blocchi per riga e per colonna
    rows = original_shape[0] // block_size[0]
    cols = original_shape[1] // block_size[1]
    
    # Se l'immagine originale non è un multiplo della dimensione del blocco, aggiungi un'altra riga/colonna
    if original_shape[0] % block_size[0] != 0:
        rows += 1
    if original_shape[1] % block_size[1] != 0:
        cols += 1
    
    # Crea un'immagine vuota per l'immagine finale riassemblata
    final_image = np.zeros((rows * block_size[0], cols * block_size[1]), dtype=np.float32)
    
    # Inserisci ogni blocco nell'immagine finale
    for idx, block in enumerate(blocks):
        row = (idx // cols) * block_size[0]
        col = (idx % cols) * block_size[1]
        final_image[row:row + block_size[0], col:col + block_size[1]] = block
    
    # Ritaglia l'immagine finale per farla corrispondere alle dimensioni originali
    final_image_cropped = final_image[:original_shape[0], :original_shape[1]]
    return final_image_cropped


def prepare_block(block):
    '''
    Funzione che permette di trasformare in tensore l'immagine ricevuta
    
    Parametri:
    - block: blocco contenente sottoimmagine da trasformare in tensore per passarla alla rete
    
    Restituisce: 
    - block: tensore dell'immagine con aggiunta la dimensione del batch (1)
    '''
    transformation = transforms.Compose([
        transforms.ToTensor(),
    ])
    block = transformation(block)
    return block.unsqueeze(0)      # Aggiunge dimensione del batch al tensore, altrimenti contenente solo [canali,altezza,larghezza] 

def infer(model, block, device, isOtsu):
    '''
    Funzione per eseguire l'inferenza del blocco
    
    Parametri:
    - model: modello della rete da usare per fare inference
    - block: blocco sul quale eseguire inferenza
    - device: dispositivo da usare per fare l'inferenza 
    
    Restitusice:
    - mask: matrice con logits di uscita della rete, non ancora passata da una funzione di attivazione o una soglia
    '''
    block = block.to(device)
    with torch.no_grad():
        output = model(block)
        if isOtsu:
            mask =  output.squeeze() # Elimina la dimensione del batch (squeeze rimuove tutte le dimensioni unitarie)
        else: 
            #mask = torch.sigmoid(output).squeeze() 
            mask = output.sigmoid().squeeze() 
    return mask

def preprocess(image_path):

    
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

    # Apply CLAHE equalisation
    image_green_clahe = exposure.equalize_adapthist(image_green_gaussian, clip_limit=0.01)
    
    # Apply gamma correction
    image_gamma = (adaptive_gamma_lightening(image_green_clahe, block_size=5, gamma_min=0.7, gamma_max=1))*eye_mask

    # Apply Black Top-Hat Transform
    radius=int(max(image_gamma.shape)*0.0045)
    radius=np.clip(radius,8,15)
    transformed_img = exposure.rescale_intensity(morphology.black_tophat(image_gamma, footprint=morphology.disk(radius)))

    image_pre = (image_gamma-transformed_img)
    image_end = np.clip(image_pre,0,1)

    desired_height = round(image_end.shape[0] / 256) * 256
    desired_width = round(image_end.shape[1] / 256) * 256
    
    original_height = image_end.shape[0]
    original_width = image_end.shape[1]

    # Verifica se l'altezza e la larghezza sono entrambe multipli di 256
    height_is_multiple = original_height % 256 == 0
    width_is_multiple = original_width % 256 == 0
     
    flag = 0 
     
    if height_is_multiple and width_is_multiple:
        # Se entrambe le dimensioni sono già multipli di 256, non fare nulla
        image_resized = image_end
    elif height_is_multiple:
        # Se solo l'altezza non è un multiplo di 256, aggiusta solo l'altezza
        image_resized = resize(image_end, (desired_height, image_end.shape[1]), anti_aliasing=True)
        flag = 1 
    elif width_is_multiple:
        # Se solo la larghezza non è un multiplo di 256, aggiusta solo la larghezza
        image_resized = resize(image_end, (image_end.shape[0], desired_width), anti_aliasing=True)
        flag = 1
    else:
        # Se entrambe le dimensioni non sono multipli di 256, aggiusta entrambe
        image_resized = resize(image_end, (desired_height, desired_width), anti_aliasing=True)
        flag = 1
        
    preprocessedImage = img_as_ubyte(image_resized)
    
    return preprocessedImage, original_height, original_width, flag

def postprocess(image, isOtsu, v_otsu, v_bin, distance_threshold, thickness, rete, block_size):

    final_mask_clipped = np.clip(image, 0, 1)
    final_mask_binary = (final_mask_clipped>v_bin)
        
    # Inizio post-process
    height, width = final_mask_binary.shape
    skeleton_image = morphology.skeletonize(final_mask_binary)
    labeled_skeleton = measure.label(skeleton_image)
    regions = measure.regionprops(labeled_skeleton)
    bounding_boxes = [region.bbox for region in regions]
    
    with ThreadPoolExecutor() as executor:
        segment_kdtrees = list(executor.map(KDTree, [region.coords for region in regions]))

    # Trova i punti più vicini in parallelo
    close_points = []
    with ThreadPoolExecutor() as executor:
        future_to_segment = {(executor.submit(check_and_search, regions[i], regions[j], segment_kdtrees[i], segment_kdtrees[j], bounding_boxes[i], bounding_boxes[j], distance_threshold)): (i, j) for i in range(len(regions)) for j in range(i+1, len(regions))}
        for future in as_completed(future_to_segment):
            result = future.result()
            if result:
                close_points.append(result)

     # Aggiornamento e salvataggio dell'immagine
    updated_mask_points = np.copy(final_mask_binary)
    for pt1, pt2 in close_points:
        rr, cc = draw_thick_line(pt1, pt2, thickness)
        
        # Accoppiare le coordinate rr e cc prima di filtrarle
        coords = np.column_stack((rr, cc))

        # Filtrare le coordinate che sono all'interno dei limiti dell'immagine
        valid_coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)]

        # Separare le coordinate filtrate rr e cc
        rr_valid, cc_valid = valid_coords[:, 0], valid_coords[:, 1]
        updated_mask_points[rr_valid, cc_valid] = 255
    
    
    height, width = updated_mask_points.shape
    soglia=np.ceil((height*width)*0.000025)
    cleaned=remove_small_regions(updated_mask_points, soglia)
    
    final_mask_uint8 = img_as_ubyte(cleaned)  # Modifica in image quando hai post-process attivo

    return final_mask_uint8

# Definizione architettura rete neurale da usare per l'inference
class UNet1(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    This model consists of an encoder (contracting path), a bottleneck, and a decoder (expansive path).
    Each step in the encoder consists of two 3x3 convolutions followed by a ReLU and a 2x2 max pooling.
    The decoder upsamples the features and concatenates them with the corresponding encoder features,
    followed by two 3x3 convolutions and a ReLU.

    It is based on the U-Net paper: https://arxiv.org/abs/1505.04597

    Args:
        input_channels (int): Number of input channels. Default is 3 for RGB images.
        out_classes (int): Number of output classes. Default is 1 for binary segmentation.
    """

    def __init__(self, input_channels=1, out_classes=1):
        super(UNet1, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Lowest resolution
        self.bottleneck = self.conv_block(512, 1024)
        
        # Expansive path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)                                          
        self.out = nn.Conv2d(64, out_classes, kernel_size=1)
       
    def conv_block(self, in_channels, out_channels):
        """
        Returns a block that performs two 3x3 convolutions followed by a ReLU. With respect to the original paper, we add batch normalization. It's a method
        that allow us to train faster and higher accuracy networks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            torch.nn.Sequential: A sequential container of two 3x3 convolutions followed by ReLUs.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),   # Modificato
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_classes, height, width).
        """
        # Contracting path
        x1 = self.enc1(x) # first double convolution
        x2 = self.enc2(self.pool(x1)) # apply pooling and second double convolution
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Lowest resolution
        x5 = self.bottleneck(self.pool(x4))

        # Expansive path, repeat upconv and concatenation as needed
        x = self.upconv4(x5)  
        x = self.dec4(torch.cat([x, x4], dim=1))  # Concatenating with conv4
        x = self.upconv3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))  # Concatenating with conv3
        x = self.upconv2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))  # Concatenating with conv2
        x = self.upconv1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))  # Concatenating with conv1


        x = self.out(x)
        return x

if single_trial:
    # Definizione cartella dove trovare le immagini, creazione elenco immagini presenti
    #image_folder = os.path.join(dataset_folder, "Complete_Original")
    image_folder = os.path.join(dataset_folder, "Original")
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]  # Vettore contenente tutti i nomi delle immagini
    image_paths = sorted(image_paths)

    for idx in tqdm(range(0,len(image_paths))):

        image_path=image_paths[idx]
        
        # Richiamo funzione di preprocess dell'immagine
        
        
        # PREPROCESS
        # Sezione commentata siccome le immagini nella cartella train sono già state processate in fase di divisione e creazione del dataset
        '''
        try:
            image_uint, original_height, original_width, flag = preprocess(image_path)

        except Exception as e:
            print(f"Errore durante il preprocess dell'immagine {image_path}: {e}")
        '''
        
        image_uint = io.imread(image_path)
            
        try:
            # INIZIO PARTE INFERENCE
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet1()  
            model.load_state_dict(torch.load(model_path+'/'+nome_trial+'.pth'))
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Errore durante l'inizializzazione della rete!: {e}")
        
        # Divisione immagini in blocchi per processarle tramite rete
        try: 
            blocks = divide_into_blocks(image_uint, block_size)
            processed_blocks = []
            for block in blocks:
                prepared_block = prepare_block(block).to(device)            
                mask = infer(model, prepared_block, device, isOtsu)         
                mask_n = mask.cpu().numpy()                                 
                processed_blocks.append(mask_n)                             
            
            final_mask = reassemble_blocks(processed_blocks, image_uint.shape, block_size)
        except Exception as e:
            print(f"Errore durante la divisione o inference dell'immagine!: {e}")
           
        
        # SEZIONE PER NON APPLICARE POSTPROCESS ALLA MASCHERA
        '''
        final_mask_clipped = np.clip(final_mask, 0, 1)
        final_mask_binary = (final_mask_clipped>v_bin)*255 
        final_mask_uint8 = img_as_ubyte(final_mask_binary)  
        results_folder = working_folder+'/Output_final_code/'+nome_trial+'/noPost'
        if not os.path.exists(results_folder):
                os.makedirs(results_folder)
        filename_image = os.path.basename(image_path)
        complete_filename_image = results_folder+'/'+filename_image
        io.imsave(complete_filename_image, final_mask_uint8)
        '''
        
        # Create a list of all parameter combinations
        all_params = list(product([image_path],[working_folder],[versione],[isOtsu], v_otsu_options, v_bin_options, distance_threshold_options, thickness_options, [rete], [final_mask], [block_size]))
        # Define the number of worker processes to use
        num_workers = 8  # Adjust based on your system's capability
        
        # Process the combinations in parallel using Pool
        with Pool(processes=num_workers) as pool:
            final_images = pool.map(apply_postprocess, all_params)
        
else:
    # Per definire invece una lista contenenti tutti i trial presenti in questo esperimenti così da avere un ciclo for:
    # Get a list of names in the directory
    all_items = os.listdir(experiment_path)
    # Filter out only directories
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(experiment_path, item))]
    
    for trial_id in folder_names:
        #folder_path = os.path.join(experiment_path, folder_name)
        nome_trial = experiment_id+'_'+trial_id
        model_path = experiment_path+'/'+trial_id
        
        # Definizione cartella dove trovare le immagini, creazione elenco immagini presenti
        image_folder = os.path.join(dataset_folder, "Original")
        image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]  # Vettore contenente tutti i nomi delle immagini
        image_paths = sorted(image_paths)

        for idx in tqdm(range(0,len(image_paths))):

            image_path=image_paths[idx]
            
            # Richiamo funzione di preprocess dell'immagine
            try:
                image_uint, original_height, original_width, flag = preprocess(image_path)
                if flag: 
                    resize(image_uint, (original_height, original_width), anti_aliasing=True)
            except Exception as e:
                print(f"Errore durante il preprocess dell'immagine {image_path}: {e}")
            
            try:
                # INIZIO PARTE INFERENCE
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = UNet1()  # Assicurati che questo corrisponda alla tua configurazione di rete    resnet50_segmentation
                #model = resnet50_segmentation()
                model.load_state_dict(torch.load(model_path+'/'+nome_trial+'.pth'))
                model = model.to(device)
                model.eval()
            except Exception as e:
                print(f"Errore durante l'inizializzazione della rete!: {e}")
            
            # Divisione immagini in blocchi per processarle tramite rete
            try: 
                blocks = divide_into_blocks(image_uint, block_size)
                processed_blocks = []
                for block in blocks:
                    prepared_block = prepare_block(block).to(device)            # Sposta il blocco preparato sul dispositivo
                    mask = infer(model, prepared_block, device, isOtsu)         # Aggiungi anche il device qui 
                    mask_n = mask.cpu().numpy()                                 # Altrimenti da qui vai direttamente al salvataggio
                    processed_blocks.append(mask_n)                             # Sposta la maschera di nuovo sulla CPU e converti in NumPy array 
                
                final_mask = reassemble_blocks(processed_blocks, image_uint.shape, block_size)
            except Exception as e:
                print(f"Errore durante la divisione o inference dell'immagine!: {e}")
                
            
            # Sezione per non applicare post-process
            '''
            final_mask_clipped = np.clip(final_mask, 0, 1)
            final_mask_binary = (final_mask_clipped>v_bin)*255 
            final_mask_uint8 = img_as_ubyte(final_mask)  
            results_folder = working_folder+'/Output_final_code/'+nome_trial+'/noPost'
            if not os.path.exists(results_folder):
                    os.makedirs(results_folder)
            filename_image = os.path.basename(image_path)
            complete_filename_image = results_folder+'/'+filename_image
            io.imsave(complete_filename_image, final_mask_uint8)
            '''
            
            # Sezione per testare vari post-process
            
            # Create a list of all parameter combinations
            all_params = list(product([image_path],[working_folder],[versione],[isOtsu], v_otsu_options, v_bin_options, distance_threshold_options, thickness_options, [rete], [final_mask], [block_size]))
            # Define the number of worker processes to use
            num_workers = 8  # Adjust based on your system's capability
            
            # Process the combinations in parallel using Pool
            with Pool(processes=num_workers) as pool:
                final_images = pool.map(apply_postprocess, all_params)
            