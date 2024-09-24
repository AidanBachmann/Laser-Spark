# Description: 
# Code for cropping interferometry data for DPP 2024 poster. Cropping is hard coded for simplicity.

# ---------- Imports ----------

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ---------- File Paths ----------

source = '../../Interferometry Data (Raw)/8_23_2024' # Directory containing raw images
target = '../Cropped Data' # Target directory to deposit cropped images
shot_info = target + '/Shot Info.txt' # Text file containing timing and assoicated shot numbers

# ---------- Parameters ----------

UX = 2793 # x and y pixels for upper left corner of crop
UY = 1935

LX = 4525 # x and y pixels for lower right corner of crop
LY = 2871

# ---------- Functions ----------

def getShotInfo(info_path=shot_info): # Get shot information from Shot Info text file
    #info = pd.read_csv(info_path,delimiter=',',engine='python',converters={'Time (ms)': str,'Initial Shot': str,'Final Shot': str}).to_numpy() # Read Shot Info text file, preserving leading zeros
    info = pd.read_csv(info_path,delimiter=',',engine='python').to_numpy() # Read Shot Info text file
    time = (info[:,0].astype(float))*1e6 # Get shot time in ns
    time = abs(time - time[0]) # Set initial time to zero
    Ni,Nf = info[:,1].astype(int),info[:,2].astype(int) # Initial and final shot number for each time setting
    #print(f'Time: {time}\nInitial Shot Numer: {Ni}\nFinal Shot Number: {Nf}') # Print shot info
    numSets = len(time) # Number of datasets taken
    return time,Ni,Nf,numSets

def cropSingle(shotNum,LX=LX,LY=LY,UX=UX,UY=UY,source_path=source,targ_path=target): # Crop single image, save in target directory
    arr = np.asarray(Image.open(f'{source_path}/{shotNum}_interferometer.jpg')) # Open image in greyscale, store in array
    croppedArr = arr[UY:LY+1,UX:LX+1,:] # Crop Image
    np.savez(f'{targ_path}/{shotNum}_interferometer',croppedArr) # Save cropped image data as a numpy array

    plt.imshow(croppedArr) # Plot cropped image
    plt.savefig(f'{targ_path}/{shotNum}_interferometer.jpg')

def cropSet(Ni,Nf): # Crop set of images for one time setting
    numShots = Nf - Ni + 1 # Determine number of shots in the set
    for i in np.linspace(0,numShots-1,numShots,dtype='int'): # Loop through, crop each image
        cropSingle(str(Ni + i).zfill(5)) # zfill pads the shot number with leading zeros

def cropAll(): # Crop all data described in Shot Info text file
    _,Ni,Nf,numSets = getShotInfo() # Get shot info
    for i in np.linspace(0,numSets-1,numSets,dtype='int'): # loop through sets, crop images
        cropSet(Ni[i],Nf[i])
    print('DONE') 

# ---------- Main ----------

cropAll() # Crop all images