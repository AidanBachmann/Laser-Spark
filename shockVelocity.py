# Description: 
# Code for estimating expansion rate of the spark.

# ---------- Imports ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- File Paths and Constants ----------

source = '../Unwrapped Data' # Directory containing raw images
target = '../Shock' # Target directory to deposit cropped images
shot_info = target + '/Shot Info.txt' # Text file containing timing and assoicated shot numbers

row = 575 # Index of lineout row
scale = 1 # mm per pixel scale (1 is placeholder value)

# ---------- Functions ----------

def getShotInfo(info_path=shot_info): # Get shot information from Shot Info text file
    info = pd.read_csv(info_path,delimiter=',',engine='python').to_numpy() # Read Shot Info text file
    time = (info[:,0].astype(float))*1e6 # Get shot time in ns
    time = abs(time - time[0]) # Set initial time to zero
    Ni,Nf = info[:,1].astype(int),info[:,2].astype(int) # Initial and final shot number for each time setting
    numSets = len(time) # Number of datasets taken
    return time,Ni,Nf,numSets

def readSingle(shotNum,row=row,source=source): # Read unwrapped phase from a single shot, take lineout, find width in pixels
    _shot_ = np.load(f'{source}/{str(shotNum).zfill(5)}_interferometer_pdiff.npz') # Open unwrapped phase
    shot = _shot_['arr_0'] # Retrieve data as a numpy array
    _shot_.close() # Close NpzFile object (prevents memory leak)

    lineout = shot[row,:] # Take lineout at defined row
    N = len(lineout) # Number of points in lineout
    pxl = np.linspace(0,N-1,N,dtype='int') # Pixel array

    # ***** James' Code to Find FWHM *****

    # Find the maximum value of the lineout
    max_idx = np.argmax(lineout)
    max_val = lineout[max_idx]

    # Now find the location of the half max of this value
    half_max = max_val/2

    # Find the indices of the points closest to the half max
    half_max_idx_top = np.argmin(np.abs(lineout[:max_idx]-half_max))
    half_max_idx_bottom = np.argmin(np.abs(lineout[max_idx:]-half_max)) + max_idx

    idx = np.asarray([half_max_idx_bottom,half_max_idx_top]) # Array of FWHM indices

    fwhm = np.abs(pxl[half_max_idx_top] - pxl[max_idx])+np.abs(pxl[max_idx]-pxl[half_max_idx_bottom]) # FWHM in pixels
    #width = half_max_idx_top + (half_max_idx_bottom+max_idx) # Not sure what this is...

    # ***** End of James' Code *****

    # Make plots
    _,ax = plt.subplots(1,2,figsize=(12,8))
    ax[0].plot(pxl,np.ones(N)*row,c='black',label='Lineout at Row {row}') # Plot line indicating lineout row
    ax[0].imshow(shot) # Plot unwrapped phase
    ax[0].set_title(f'Unwrapped Phase')
    ax[0].legend()
    ax[1].plot(pxl,lineout,label='Lineout')
    ax[1].scatter(pxl[idx],lineout[idx],marker='+',c='r',label=f'FWHM = {fwhm} pixels')
    ax[1].scatter(pxl[max_idx],max_val,marker='*',c='r',label='Max Value')
    ax[1].set_xlabel('Pixel Number')
    ax[1].set_ylabel('Unwrapped Phase')
    ax[1].set_title(f'Lineout at Row {row}')
    ax[1].legend()
    ax[1].grid()
    plt.show()

    return fwhm,idx,lineout,pxl,N

def readAll(scale=scale):
    time,Ni,Nf,numSets = getShotInfo() # Get shot info    


#shotNum = np.asarray([131,132,133,134,135,136,137,138,139,140])
shotNum = np.asarray([21,23,24,25,27,28])
for i in shotNum:
    readSingle(i)
