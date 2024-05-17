from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nni                                
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

# =====================================================================================================
# CODE INFO
'''
This code aims to train the neural network using nni for automatic experiment management and optimized search of the values in the search_space, 
containing various parameters related to network management. The execution requires nni version 2.7 and 2 files (config.yml and search_space.json) 
and is started by a specific command in the terminal, conveniently listed below.
As with other codes, the presence of a structure with folders automatically created by other satellite codes is required.

USEFUL NNI COMMANDS

- Start: nnictl create --config ./config.yml --port 8080
- Stop: nnictl stop --all
- Results: nnictl experiment show [experiment-ID]
- List of trials and their results: nnictl trial ls [experiment-ID]


Using the combined_loss, alpha represents the importance given to the dice_loss

POSSIBLE UPDATES:
line 87, float16 instead of float32 for the mask
'''
# =====================================================================================================

# =====================================================================================================
# INITIAL PARAMETERS AND COMMANDS DEFINITION
# =====================================================================================================
n_epochs = 20
torch.backends.cudnn.benchmark = True  
torch.cuda.empty_cache()  

# =====================================================================================================
# FUNCTION DEFINITION
# =====================================================================================================

# Dataset classe creation
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
        image = np.array(Image.open(image_path)) 
        mask = np.array(Image.open(mask_path))
        
        # Control condition for mask size correction
        if mask.ndim > 2:  # check if the mask has more the 2 dimension
            mask = mask[:,:,0]  # select only the first channel
        
        image = min_max_normalization(image)
        
        # Binarize the mask to have 0 on the background and 1 on the foreground
        mask = (mask >0).astype(np.float32) 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']                                                

        # Make sure image is tensor
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)

        if type(mask) == np.ndarray:
            mask = torch.from_numpy(mask)

        # Make sure image is a float tensor, not an integer tensor
        image = image.float()                       
        
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


# Min_max normalizaziont function
def min_max_normalization(matrix):   
                
                min_value = np.min(matrix)
                max_value = np.max(matrix)
                epsilon = 1e-6
                matrix_normalizzata = (matrix - min_value) / (max_value - min_value + epsilon)

                return matrix_normalizzata



# U-net architecture definition
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
The following functions are related to the definition of the network's loss and evaluation functions, with two clarifications about their operation
- Alpha parameter: all functions contain this parameter, but only the combined_loss actually uses it to weight the contribution of Dice_loss and BCELoss. 
    Its presence in all functions is still necessary to avoid errors when calling criterion while providing alpha, and it is therefore necessary
    to ensure the stability of the code execution given its architecture
    
- Use of .sigmoid(): to reduce the network's resource usage, the automatic mixed precision implemented by Pytorch was used. However, this change 
    is incompatible with the BCELoss loss function. Therefore, the training function was modified to obtain the logits output from the network without using
    the sigmoid, to be able to use the BCEWithLogits function, which is compatible with AMP. This finally required a small modification to the functions containing the 
    dice calculation, where it was necessary to use the sigmoid function to enable the calculation of this metric.
'''
#======================================

# Dice loss definition
def dice_loss(outputs, labels, alfa, eps=1e-7):  
    outputs = outputs.sigmoid()
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                 # Avoid NaN
        dice_score = 2.0 * intersection / (union + eps)
        dice_loss = torch.clamp(1.0 - dice_score, min=0.0, max=1.0)       
    else:
        dice_loss=0.0                            # Case with empty mask and prediction
        
    return dice_loss

# Dice definition
def dice(outputs, labels):
    outputs = outputs.sigmoid()
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels)
    
    if union >0:                                 # Avoid NaN
        dice_score = 2.0 * intersection / (union)    
    else:
        dice_score=1.0                           # Case with empty mask and prediction
        
    return dice_score

# Function to calculate the loss based on BCE and dice_loss
def combined_loss(outputs, labels, alfa, eps=1e-7):
    outputs_s = outputs.sigmoid()
    intersection = torch.sum(outputs_s * labels)
    union = torch.sum(outputs_s) + torch.sum(labels)
    dice_score = 2.0 * intersection / (union + eps)
    dice_loss = torch.clamp(1.0 - dice_score, min=0.0, max=1.0)
    BCE = nn.BCEWithLogitsLoss()
    BCE_loss = BCE(outputs, labels)
    return alfa*dice_loss+(1-alfa)*BCE_loss

# Function to calculate the Binary Cross Entropy
def BCELoss(outputs, labels, alfa):
    BCE = nn.BCEWithLogitsLoss()
    BCE_loss = BCE(outputs, labels)
    return BCE_loss

# Training function definition
def train(args, model, device, train_loader, optimizer, epoch,val_loader,criterion, trial_dir,trial_name, file_name, best_dice):

    train_losses = []
    val_losses = []
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=args['patience'],threshold=0.5e-2) 
    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{n_epochs}", unit="batch")
    for images, masks, filename in train_progress_bar:

        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
    
        # Section to use AMP
        # Forward pass con autocast (AMP) 
        with autocast():
            
            #outputs = model(images).sigmoid().squeeze() # Original version witout AMP
            outputs = model(images).squeeze()
            loss = criterion(outputs, masks, args['a'])
       
        # Backward pass with GradScaler              
        scaler.scale(loss).backward()
        
        # Weigth modification with GradScaler
        scaler.step(optimizer)
        scaler.update()  

        # Update running loss
        running_loss += loss.item()*images.size(0)

        # Update progress bar
        train_progress_bar.set_postfix(loss=running_loss / ((train_progress_bar.n + 1) * train_loader.batch_size))
    
    train_loss = running_loss / len(train_loader.dataset)

    # Store the training loss in the list
    train_losses.append(train_loss)

    # Start evaluation to save the best network at the end of each epoch (given the size of the training set)
    model.eval()
    running_loss = 0.0
    running_dice_score = 0.0       
    val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{n_epochs}", unit="batch")

    with torch.no_grad():
            for images, masks, filename in val_progress_bar:

                images, masks = images.to(device), masks.to(device)
                #outputs = model(images).sigmoid().squeeze() # Original version without AMP
                outputs = model(images).squeeze()
                masks = masks.squeeze()               
                loss = criterion(outputs, masks, args['a'])

                # Update running loss e dice score
                running_loss += loss.item() * images.size(0)
                running_dice_score += dice(outputs, masks).item() * images.size(0)   
                val_progress_bar.set_postfix(loss=running_loss / ((val_progress_bar.n + 1) * val_loader.batch_size))
    
    val_loss = running_loss / len(val_loader.dataset)
    dice_score = running_dice_score / len(val_loader.dataset)
    val_losses.append(val_loss)
                
    scheduler.step(val_loss)     # Update scheduler with loss value, for lr management
    
    if dice_score > best_dice: # The value of best_dice is updated by the training loop across epochs, therefore it is NOT updated here
            torch.save(model.state_dict(), os.path.join(trial_dir,trial_name+'.pth'))   # Save the net if it hase a validation set metric higher than previous one

    # Write values to the terminal (will also be displayed in the appropriate nni section)
    print('\n')
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Dice: {dice_score:.4f}')
    print('\n')
    
    # Write values to a specific .txt file to facilitate any analysis on the network's progress during training. The file is saved in the same folder where the trained network is saved
    with open(file_name, 'a') as file:
        file.write('\n'+str(epoch+1)+'/'+str(n_epochs)+' | '+str(train_loss)+' | : '+str(val_loss)+' | : '+str(dice_score)) # Specific format to help successive analysis
    
    
    return dice_score

# Define testing function. It is included for completeness but is not called, as a dedicated testing script with more comprehensive features is preferred to be run at the end of training
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
        
        # Save the result in the appropriate network description file
        with open(log, 'a') as file:
            file.write('\n Test metrics:      Epoch loss: '+str(epoch_loss)+' | '+'Epoch Dice: '+str(epoch_dice_score))

        return epoch_dice_score

# Definition of the main function used by nni to allow the execution of various trials within a specific experiment
def main(args):
    
    criterion = globals()[args['criterion']]  # Define the loss criterion by converting the received string into the corresponding function name
    
    working_folder = os.path.dirname(os.path.abspath(__file__))

    # =====================================================================================================
    # DATA AUGMENTATION STRATEGY DEFINITION
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
    # TRAINING AND VALIDATION SET IMPORT
    # They were previousy created by the dedied script
    # =====================================================================================================
    
    train_images_folder = working_folder+'/Dataset/train'+args['path']+'/Blocks_Original'
    train_masks_folder = working_folder+'/Dataset/train/Blocks_Mask'
    val_images_folder = working_folder+'/Dataset/val'+args['path']+'/Blocks_Original'
    val_masks_folder = working_folder+'/Dataset/val/Blocks_Mask'
    
    print('Percorso immagini train: '+train_images_folder)
    print('Percorso maschere train: '+train_masks_folder)
    print('Percorso immagini val: '+val_images_folder)
    print('Percorso maschere val: '+val_masks_folder)
    
    # Function to get file path
    def get_file_paths(images_dir, masks_dir):
        images = [os.path.join(images_dir, file) for file in sorted(os.listdir(images_dir))]
        masks = [os.path.join(masks_dir, file) for file in sorted(os.listdir(masks_dir))]
        return images, masks

    # Loading paths using the defined variables
    train_image_paths, train_mask_paths = get_file_paths(train_images_folder, train_masks_folder)
    val_image_paths, val_mask_paths = get_file_paths(val_images_folder, val_masks_folder)

    # Dataset creation
    train_dataset = Dataset_challenge(train_image_paths, train_mask_paths, transform=train_transforms)
    val_dataset = Dataset_challenge(val_image_paths, val_mask_paths, transform=val_transforms)

    # Batch dimension
    BS = 16

    # Dataloader creation
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

    # =====================================================================================================
    # Create network and load from previously saved models
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
    # Define parameters and create the descriptive file of the ongoing trial
    # Recall that an experiment is defined as a "class" with various attempts to try,
    # a trial will be an instance of this class and will therefore be characterized by specific parameters chosen
    # from those possible for the experiment
    # =====================================================================================================
    
    # Create variables with training-related information, later used in the descriptive files of the network that accompany the saved network
    trial_id = get_trial_id()
    experiment_id = os.environ.get('NNI_EXP_ID')
    trial_name = experiment_id+'_'+trial_id
    ora_attuale = datetime.now()
    data_ora_stringa = ora_attuale.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a folder to save the network and training-related information
    info_trial = experiment_id+'/'+trial_id
    trial_dir = os.path.join(working_folder,'modelli_allenati/'+info_trial)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    
    filename = trial_dir+'/'+trial_name+'.txt'
    
    # Save all information related to the trial, the chosen parameters, and the network in an appropriate file
    with open(filename, 'a') as file:
        file.write('\n Training Date: ' + data_ora_stringa + ',      Experiment ID: ' + experiment_id + ',       Trial ID: ' + trial_id)
        file.write('\n ------ Training Parameters ------ ')
        file.write('\n      Loss criterion: ' + args['criterion'])
        file.write('\n      Alpha parameter: ' + str(args['a']))                                 
        file.write('\n      Learning rate: ' + str(args['lr']))
        file.write('\n      Patience: ' + str(args['patience']))
        file.write('\n      B1: ' + str(args['b1']))
        file.write('\n      Network: ' + str(args['rete']))
        file.write('\n      Dataset path: ' + args['path'])
        file.write('\n      Training Epochs: ' + str(n_epochs))
        file.write('\n\n\n ------ Network Architecture ------')
        file.write('\n' + rete_string)
        file.write('\n\n\n ------ Metrics During Training Epochs ------')
        file.write('\n Epoch: ' + ' | Training Loss: ' + ' | Validation Loss: ' + ' | Validation Dice: ')
        file.write('\n')
        
    
    
    # =====================================================================================================
    # Final preparation and network training
    # =====================================================================================================
    # Function definitions to manage the training
    optimizer = Adam(model.parameters(), lr=args['lr'], betas=(args['b1'],0.999))    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            

    model = model.to(device) 
    
    file_metriche = trial_dir+'/'+trial_name+'_metriche'+'.txt'  # Define the file where training metrics will be saved
    best_dice = 0
    for epoch in range(n_epochs):
        val_dice = train(args, model, device, train_loader, optimizer, epoch,val_loader,criterion, trial_dir, trial_name, file_metriche, best_dice)
        best_dice = val_dice
        print('Report risultati intermedi: '+ str(val_dice))
        nni.report_intermediate_result(val_dice)                 # Send to nni the Dice value obtained at the end of the training epoch on the validation set, used for visualization and possible early stopping according to assessor's decision
        
    print('Report risultati finali (validation_set): '+ str(best_dice))
    nni.report_final_result(best_dice)                           # Send to nni the best Dice value obtained during the epochs (calculated on the validation set), used for result visualization through the control interface

# Function necessary for nni, used for parameter selection at the start of a trial
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


# Start the trial by calling the main function, each time choosing different parameters that define the trial
if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        print('Exception')
        raise