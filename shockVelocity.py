# Description: 
# Code for estimating expansion rate of the spark.

# ---------- Imports ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- File Paths ----------

source = '../Unwrapped Data' # Directory containing raw images
target = '../Shock' # Target directory to deposit cropped images
shot_info = target + '/Shot Info.txt' # Text file containing timing and assoicated shot numbers

# ---------- Functions ----------

def getShotInfo(info_path=shot_info): # Get shot information from Shot Info text file
    info = pd.read_csv(info_path,delimiter=',',engine='python').to_numpy() # Read Shot Info text file
    time = (info[:,0].astype(float))*1e6 # Get shot time in ns
    time = abs(time - time[0]) # Set initial time to zero
    Ni,Nf = info[:,1].astype(int),info[:,2].astype(int) # Initial and final shot number for each time setting
    numSets = len(time) # Number of datasets taken
    return time,Ni,Nf,numSets

def readSingle(shotNum,row=525,source=source): # Read unwrapped phase from a single shot, take lineout
    _shot_ = np.load(f'{source}/{str(shotNum).zfill(5)}_interferometer_pdiff.npz') # Open unwrapped phase
    shot = _shot_['arr_0'] # Retrieve data as a numpy array
    _shot_.close() # Close NpzFile object (prevents memory leak)

    lineout = shot[row,:] # Take lineout at defined row
    N = len(lineout) # Number of points in lineout
    pxl = np.linspace(0,N-1,N,dtype='int') # Pixel array

    # Make plots
    _,ax = plt.subplots(1,2,figsize=(12,8))
    ax[0].plot(pxl,np.ones(N)*row,c='black',label='Lineout at Row {row}') # Plot line indicating lineout row
    ax[0].imshow(shot) # Plot unwrapped phase
    ax[0].set_title(f'Unwrapped Phase')
    ax[0].legend()
    ax[1].plot(pxl,lineout)
    ax[1].set_xlabel('Pixel Number')
    ax[1].set_ylabel('Unwrapped Phase')
    ax[1].set_title(f'Lineout at Row {row}')
    ax[1].grid()
    plt.show()

    return lineout,pxl,N


shotNum = 215
readSingle(shotNum)
