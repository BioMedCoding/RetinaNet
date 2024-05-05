from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nni                                 # Perch'e funzioni senzaz problemi serve versione 2.7 mi pare, vedu docker lava_experiments di franzhd su Github
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.transform import resize
import argparse
from torch.cuda.amp import autocast, GradScaler
from nni import get_trial_id
import torch.nn.functional as F
from datetime import datetime
import inspect

# Usando la combined_loss, alfa rappresenta l'importanza data alla dice_loss

# =====================================================================================================
#INFO CODICE
'''
Questo codice ha come obiettivo l'allenamento della rete neurale, sfruttando nni per la gestione automatica degli esperimenti e una ricerca ottimizzata dei valori del search_space,
contenente vari parametri relativi alla gestione della rete. L'esecuzione richiede nni versione 2.7 e 2 file (config..yml e search_space.json) e viene avviata tramita apposita comando
nel terminale, riportato per comodità qui sotto. 
Come per gli altri codici, è richiesta la presenza di una struttura con le cartelle create automaticamente dagli altri codici satellite
COMAND UTILI NNI
- Avvio: nnictl create --config ./config.yml --port 8080
- Stop: nnictl stop --all
- Risultati: nnictl experiment show [ID-esperimento]
- Elenco trial e relativi risultati: nnictl trial ls [ID-esperimento]

'''
# =====================================================================================================


# NUOVE MODIFICHE
# AGGIUNTO: , num_workers=8, pin_memory=True   AI DATALOADER                        - attivo
# AGGIUNTO: cuDNN benchmarking                                                      - attivo     riduce
# AGGIUNTO: , bias=False nella unet1 alle convoluzione 2d                           - attivo     non avvertibile come velocità, dimensione non cotnrollata

# =====================================================================================================
# DEFINIZIONE PAREMETRI E COMANDI INIZIALI INIZIALI
# =====================================================================================================
n_epochs = 20
torch.backends.cudnn.benchmark = True # Abilita cuDNN benchmarking per ottimizzazione convoluzioni (selezione automatica miglior algoritmo per calcolo convoluzioni, a seguito di benchmark). Richiede BS e risoluzione costante
torch.cuda.empty_cache()  # Comando per svuotare cache GPU prima dell'inizio del training


# =====================================================================================================
# DEFINIZIONE FUNZIONI
# =====================================================================================================

# Creazione classe per dataset
class Dataset_challenge(Dataset):

    # The __init__ function is run once when instantiating the UltrasoundDataset class, and it's used to initialize the dataset
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

         # check filenames of images and masks to make sure they match
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        assert image_path.split('/')[-1].split('.')[0] == mask_path.split('/')[-1].split('.')[0], "Filename mismatch between image and mask!"

        # open image and mask using PIL and convert to numpy array
        image = np.array(Image.open(image_path)) #Apertuta immagine indicata all'indice come np.array
        mask = np.array(Image.open(mask_path))
        
        # Condizione di controllo per la correzione della dimensione della maschera
        if mask.ndim > 2:  # controlla se la maschera ha più di due dimensioni
            mask = mask[:,:,0]  # seleziona solo il primo canale
        
        image = min_max_normalization(image)
        
        # Binarize the mask to have 0 on the background and 1 on the foreground
        mask = (mask >0).astype(np.float32)   #Intanto e` gia` binaria

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # If image tensor is only one channel concatenate image tensor to RGB in channel-first format and make it a float tensor, else just make it a float tensor
        # you can do this if your model expects RGB images but your dataset is grayscale
        #if image.shape[0] == 1:
        #    image = torch.cat((image, image, image), dim=0)                                                   #    SI È TOLTO PER TESTARE RETE SOLO SCALA DI GRIGI

        # Make sure image is tensor
        if type(image) == np.ndarray:
            #print('Entrato in if type(image) == np.ndarray:')
            image = torch.from_numpy(image)

        if type(mask) == np.ndarray:
            #print('Entrato in if type(mask) == np.ndarray:')
            mask = torch.from_numpy(mask)

        # Make sure image is a float tensor, not an integer tensor
        image = image.float()                       # Si ottiene dimensione di 3, 256, 256
        
        # Make sure mask is a float tensor
        mask = mask.float()
        
        assert image.size(-1) == mask.size(-1), f"Image ({image.size()}) and mask ({mask.size()}) have different last two dimensions!"

        # check that mask has only 0 and 1 values
        assert torch.all(torch.logical_or(mask == 0, mask == 1)), f"Mask has values other than 0 and 1!"

        # check that image and mask are float32 tensors
        assert image.dtype == torch.float32, f"Image dtype {image.dtype} is not torch.float32!"
        assert mask.dtype == torch.float32, f"Mask dtype {mask.dtype} is not torch.float32!"

        # get name of image
        file_name = image_path.split('/')[-1].split('.')[0]

        return image, mask, file_name


# Creazione funzione min_max_normalization
def min_max_normalization(matrix):   #Funzione di normalizzazione usata in seguito
                
                min_value = np.min(matrix)
                max_value = np.max(matrix)
                epsilon = 1e-6
                matrix_normalizzata = (matrix - min_value) / (max_value - min_value + epsilon)

                return matrix_normalizzata



# Definizione dell'architettura U-Net
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

#======================================
# Definizione loss functions e metriche
'''
Le funzioni seguenti sono relative alla definizione delle funzioni di loss e di valutazione della rete, si fanno 2 precisazioni relative al loro funzionamento
- Parametro alfa: tutte le funzioni contengono questo parametro, ma solo la combined_loss lo utilizza effettivamente per dare un peso al contributo di Dice_loss e BCELoss. 
    La sua presenza in tutte le funzioni risulta comunque necessaria per non causare errori quando si richiama criterion fornendo anche alfa, ed è pertanto necessaria
    a garantire la stabilità dell'esecuzione del codice vista la sua architettura
    
- Utilizzo .sigmoid(): per diminuire l'utilizzo di risorse da parte della rete si è implementata l'automatic mixed precision implementata da Pytorch. Tuttavia, questa modifica 
    è incompatibile conj l'utilizzo della loss function BCELoss. Si è quindi modificata la funzione di training in modo da ottenere i logit in uscita dalla rete, senza l'utilizzo
    di sigmoide, per poter utilizzare la funzione BCEWithLogits, compatibile con AMP. Questo ha infine reso necessaria una piccola modifiche alle funzioni contenenti il calcolo
    del dice, dove si è reso necessario andare a utuilizzare la funzione sigmoid per rendere possibile il calcolo di questa metrica. 
'''
#======================================

# Definizione funzione Dice loss 
def dice_loss(outputs, labels, alfa, eps=1e-7):  
    outputs = outputs.sigmoid()
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                 # Modifica alla funzione effettuata per evitare NaN in caso di assenza di maschere
        dice_score = 2.0 * intersection / (union + eps)
        dice_loss = torch.clamp(1.0 - dice_score, min=0.0, max=1.0)       
    else:
        dice_loss=0.0                            # Questa situazione si verifica quando né la maschera né la predizione individuano nulla, per cui la rete ha perfettamente classificato l'immagine
        
    return dice_loss

# Funzione per il calcolo della metrica dice
def dice(outputs, labels):
    outputs = outputs.sigmoid()
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                 # Modifica alla funzione effettuata per evitare NaN in caso di assenza di maschere
        dice_score = 2.0 * intersection / (union)    
    else:
        dice_score=1.0                           # Questa situazione si verifica quando né la maschera né la predizione individuano nulla, per cui la rete ha perfettamente classificato l'immagine
        
    return dice_score

# Funzione per il calcolo della funzione di loss che combina BCE e dice_loss
def combined_loss(outputs, labels, alfa, eps=1e-7):
    outputs_s = outputs.sigmoid()
    intersection = torch.sum(outputs_s * labels)
    union = torch.sum(outputs_s) + torch.sum(labels)
    dice_score = 2.0 * intersection / (union + eps)
    dice_loss = torch.clamp(1.0 - dice_score, min=0.0, max=1.0)
    BCE = nn.BCEWithLogitsLoss()
    BCE_loss = BCE(outputs, labels)
    return alfa*dice_loss+(1-alfa)*BCE_loss

# Funzione per il calcolo della Binary Cross Entropy
def BCELoss(outputs, labels, alfa):
    BCE = nn.BCEWithLogitsLoss()
    BCE_loss = BCE(outputs, labels)
    return BCE_loss

# Definizione training function
def train(args, model, device, train_loader, optimizer, epoch,val_loader,criterion, trial_dir,trial_name, file_name, best_dice): # Rivedi parametri in base a cosa varia

    train_losses = []
    val_losses = []
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args['patience'],threshold=0.5e-2) # Factor indica di quanto moltiplicare il lr se per un numero di epoche pari a patience la rete non presente un miglioramento della metrica superiore a threshold
    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{n_epochs}", unit="batch")
    for images, masks, filename in train_progress_bar:

        # Sposta immagini e maschere sul dispotivo
        images, masks = images.to(device), masks.to(device)
        
        # Azzera il gradiente
        optimizer.zero_grad()
    
        # Sezione per utilizzo AMP
        # Forward pass con autocast (AMP) 
        with autocast():
            
            #outputs = model(images).sigmoid().squeeze() # Lasciato per evidenziare cambiamento che si è reso necessario per l'uso di AMP, BCE e Dice, come spiegato in precedenza 
            outputs = model(images).squeeze()
            loss = criterion(outputs, masks, args['a'])
       
        
        # Backward pass con GradScaler              
        scaler.scale(loss).backward()
        
    
        # Modifica pesi con GradScaler
        scaler.step(optimizer)
        scaler.update()  

        # Aggiorna running loss
        running_loss += loss.item()*images.size(0)

        # Aggiorna progress bar
        train_progress_bar.set_postfix(loss=running_loss / ((train_progress_bar.n + 1) * train_loader.batch_size))
        train_loss = running_loss / len(train_loader.dataset)

        # Memorizza la training loss nella lista
        train_losses.append(train_loss)

    # Inizio valutazione per salvataggio rete migliore al termine di ogni epoca (vista la dimensione del training set)
    model.eval()
    running_loss = 0.0
    running_dice_score = 0.0       
    val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{n_epochs}", unit="batch")
    with torch.no_grad():
            for images, masks, filename in val_progress_bar:

                # Sposta immagini e maschere sul dispotivo
                images, masks = images.to(device), masks.to(device)
                #outputs = model(images).sigmoid().squeeze() # Analogamente a sopra, si lascia per evidenziare differenza richiesta da AMP, BCE e Dice
                outputs = model(images).squeeze()
                masks = masks.squeeze()               
                loss = criterion(outputs, masks, args['a'])

                # Aggiorna running loss e dice score
                running_loss += loss.item() * images.size(0)
                running_dice_score += dice(outputs, masks).item() * images.size(0)   
                val_progress_bar.set_postfix(loss=running_loss / ((val_progress_bar.n + 1) * val_loader.batch_size))
    
    val_loss = running_loss / len(val_loader.dataset)
    dice_score = running_dice_score / len(val_loader.dataset)
    val_losses.append(val_loss)
                
    scheduler.step(val_loss)     # Aggiorna scheduler con valore di loss, per la gestione del lr
    
    if dice_score > best_dice: # NB il valore di best_dice viene aggiornato dal ciclo di allenamento lungo le epoche, pertanto NON viene aggiornato qui
            torch.save(model.state_dict(), os.path.join(trial_dir,trial_name+'.pth'))   # Salvataggio della rete se presente una metrica sul validation set più alta rispetto alle precedenti

    # Scrittura valori su terminale (verrà visualizzato anche nell'apposita sezione nni)
    print('\n')
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice: {dice_score:.4f}')
    print('\n')
    
    # Scrittura valori in apposito file .txt per agevolare eventuali analisi sull'andamento della rete nel corso dell'allenamento, il file è salvato nella stessa cartella dove si salva la rete allenata
    with open(file_name, 'a') as file:
        file.write('\n'+str(epoch+1)+'/'+str(n_epochs)+' | '+str(train_loss)+' | : '+str(val_loss)+' | : '+str(dice_score)) # Questo formato agevola l'analisi successiva dei dati tramite script/Excel
    
    
    return dice_score

# Definzione funzione di testing. Viene riportata per completezza ma non viene richiamata, siccome si è preferito creare apposito script di testing da avviare al termine dell'allenamento, con caratteristiche più complete
def test_model(args,test_loader, model, criterion, device,log):
        model.eval()
        running_loss = 0.0
        running_dice_score = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images).squeeze()             
                loss = criterion(outputs, masks.squeeze(), args['a'])   
                dice = dice(outputs, masks.squeeze())

                running_loss += loss.item() * images.size(0)
                running_dice_score += dice.item() * images.size(0)
                
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_dice_score = running_dice_score / len(test_loader.dataset)
        
        # Salva il risultato nell'apposito file descrittivo della rete
        with open(log, 'a') as file:
            file.write('\n Test metrics:      Epoch loss: '+str(epoch_loss)+' | '+'Epoch Dice: '+str(epoch_dice_score))

        return epoch_dice_score

# Definizione del main usato da nni per permettere l'esecuzione dei vari trial all'interno di uno scpeficio esperimento
def main(args):
    
    criterion = globals()[args['criterion']]  # Definizione del criterio di loss convertendo la stringa ricevuta nel nome della relativa funzione
    
    working_folder = os.path.dirname(os.path.abspath(__file__))

    # =====================================================================================================
    # Definizione Data Augmentation
    # =====================================================================================================
    
    train_transforms = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        ToTensorV2()
    ])
    

    # =====================================================================================================
    # Import training e validation set
    # Sono stati precedentemente creati da apposito script
    # =====================================================================================================
    
    train_images_folder = working_folder+'/Dataset/train'+args['path']+'/Blocks_Original'
    train_masks_folder = working_folder+'/Dataset/train/Blocks_Mask'
    val_images_folder = working_folder+'/Dataset/val'+args['path']+'/Blocks_Original'
    val_masks_folder = working_folder+'/Dataset/val/Blocks_Mask'
    
    # Stampa dei valori per conferma e controllo
    print('Percorso immagini train: '+train_images_folder)
    print('Percorso maschere train: '+train_masks_folder)
    print('Percorso immagini val: '+val_images_folder)
    print('Percorso maschere val: '+val_masks_folder)
    
    # Funzione per ottenere i percorsi dei file
    def get_file_paths(images_dir, masks_dir):
        images = [os.path.join(images_dir, file) for file in sorted(os.listdir(images_dir))]
        masks = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir))]
        return images, masks

    # Caricamento dei percorsi utilizzando le variabili definite
    train_image_paths, train_mask_paths = get_file_paths(train_images_folder, train_masks_folder)
    val_image_paths, val_mask_paths = get_file_paths(val_images_folder, val_masks_folder)

    # Creazione dei dataset (si assumono definite le classi Dataset_challenge)
    train_dataset = Dataset_challenge(train_image_paths, train_mask_paths, transform=train_transforms)
    val_dataset = Dataset_challenge(val_image_paths, val_mask_paths, transform=val_transforms)

    # Dimensione del batch
    BS = 16

    # Creazione dei dataloader
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

    # =====================================================================================================
    # Creazione rete e caricamento da modelli precedentemente salvati
    # =====================================================================================================
    if args['rete'] == 1:
        model = UNet1()
        rete_string = inspect.getsource(UNet1)
        if args['criterion'] == "combined_loss":   # In base al criterio di allenamento scelto, carica la relativa rete preallenata (codice relativa all'esperimento finale, dove si è allenato per un maggior numero di epoche)
            model_path = working_folder+'/modelli_allenati/y1zdovh9/wYg8o/y1zdovh9_wYg8o.pth'  # Percorso al modello salvato
            model.load_state_dict(torch.load(model_path))
        if args['criterion'] == "dice_loss":
            model_path = working_folder+'/modelli_allenati/y1zdovh9/tPN28/y1zdovh9_tPN28.pth'  # Percorso al modello salvato
            model.load_state_dict(torch.load(model_path))
    else:
        model = UNet1()
        print("Net type error")
    
    # =====================================================================================================
    # Definizione parametri e creazione del file descrittivo del trial in corso
    # Si ricorda che per esperimento si definisce una "classe" con vari tentativi da provare,
    # un trial sarà un'istanza di questa classe e sarà quindi caratterizzato da parametri precisi scelti 
    # all'interno di quelli possibili per l'esperimento
    # =====================================================================================================
    
    # Creazione variabili con informazioni relative all'allenamento, usate in seguito nei file descrittivi della rete che accompagnano la rete salvata
    trial_id = get_trial_id()
    experiment_id = os.environ.get('NNI_EXP_ID')
    trial_name = experiment_id+'_'+trial_id
    ora_attuale = datetime.now()
    data_ora_stringa = ora_attuale.strftime('%Y-%m-%d %H:%M:%S')
    
    # Creazione cartella dove salvare la rete e le informazioni relative al training
    info_trial = experiment_id+'/'+trial_id
    trial_dir = os.path.join(working_folder,'modelli_allenati/'+info_trial)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    
    filename = trial_dir+'/'+trial_name+'.txt'
    
    # Salva in apposito file tutte le informazioni relative al trial, ai parametri scelti e alla rete
    with open(filename, 'a') as file:
        file.write('\n Data allenamento: '+data_ora_stringa+',      Experiment ID: '+experiment_id+',       Trial ID: '+trial_id)
        file.write('\n ------ Parametri allenamento ------ ')
        file.write('\n      Loss criterion: '+args['criterion'])
        file.write('\n      Alfa parameter: '+str(args['a']))                                 
        file.write('\n      Learning rate: '+str(args['lr']))
        file.write('\n      Patience: '+str(args['patience']))
        file.write('\n      B1: '+str(args['b1']))
        file.write('\n      Rete: '+str(args['rete']))
        file.write('\n      Dataset path: '+args['path'])
        file.write('\n      Training Epochs: '+str(n_epochs))
        file.write('\n\n\n ------ Architettura rete ------')
        file.write('\n'+rete_string)
        file.write('\n\n\n ------ Metriche durante epoche allenamento ------')
        file.write('\n Epoca: '+' | Training Loss: '+' | Validation Loss: '+' | Validation Dice: ')
        file.write('\n')
        
    
    
    # =====================================================================================================
    # Preparazione e avvio training rete
    # =====================================================================================================
    # Definizione funzioni per gestire l'allenamento
    optimizer = Adam(model.parameters(), lr=args['lr'], betas=(args['b1'],0.999))    # Definizione algoritmo di ottimizzazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            # Definizione dispositivo da usare
    
    # Invio modello al dispositivo
    model = model.to(device) 
    
    file_metriche = trial_dir+'/'+trial_name+'_metriche'+'.txt'  # Definizione del file dove andare a salvare le metriche relative all'allenamento
    best_dice = 0
    for epoch in range(n_epochs):
        val_dice = train(args, model, device, train_loader, optimizer, epoch,val_loader,criterion, trial_dir, trial_name, file_metriche, best_dice)
        best_dice = val_dice
        print('Report risultati intermedi: '+ str(val_dice))
        nni.report_intermediate_result(val_dice)                 # Invio a nni del valore di Dice ottenuto al termine dell'epoca di allenamento sul validation set, usato per la visualizzazione ed eventuale early stopping secondo decisione assessor
        
    print('Report risultati finali (validation_set): '+ str(best_dice))
    nni.report_final_result(best_dice)                           # Invio a nni del valore del miglior valore di Dice ottenuto nel corso delle epoche (calculato sul validation set), usato per la visualizzazione dei risultati tramite l'interfaccia di controllo

# Funzione necessaria per nni, usat per la scelta dei parametri all'avvio di un trial
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.00009)
    parser.add_argument("--b1", type = float, default = 0.9)
    parser.add_argument("--patience", type = int, default = 3)
    parser.add_argument("--rete", type = int, default = 1)
    parser.add_argument("--path", type = str, default = "/final")
    parser.add_argument("--criterion", type = str, default = "dice_loss")
    parser.add_argument("--a", type = float, default = "0.5")
    args, _ = parser.parse_known_args()
    return args


# Avvio del trial tramite richiamo della funzione main, scegliendo ogni volta dei parametri differenti che definiscono il trial
if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        print('Exception')
        raise