# Description: 
# Code for cropping interferometry data for DPP 2024 poster. This code
# will also include a filter that flags bad data (i.e., black images) by
# intensity threshold.

# ---------- Imports ----------

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# ---------- File Paths ----------

source = '../../Interferometry Data (Raw)/8_23_2024' # Directory containing raw images
target = 'Cropped Data' # Target directory to deposit cropped images
shot_info = target + 'Shot Info.txt' # Text file containing timing and assoicated shot numbers

# ---------- Parameters ----------

UX = 2793 # x and y pixels for upper left corner of crop
UY = 1935

LX = 4525 # x and y pixels for lower right corner of crop
LY = 2871

# ---------- Functions ----------

def cropSingle(shotNum,LX=LX,LY=LY,UX=UX,UY=UY,source_path=source,targ_path=target):
    arr = np.asarray(Image.open(f'{source_path}/{shotNum}_interferometer.jpg').convert('L')) # Open image in greyscale, store in array
    croppedArr = arr[UY:LY+1,UX:LX+1] # Crop Image
    plt.imshow(croppedArr)
    plt.savefig(f'{targ_path}/{shotNum}_interferometer.jpg')


shot = '00247'
cropSingle(shot)