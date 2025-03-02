# Description: 
# Code for estimating expansion rate of the spark.

# ---------- Imports ----------

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ---------- File Paths and Constants ----------

source = '../Unwrapped Data' # Directory containing raw images
target = '../Shock' # Target directory to deposit cropped images
shot_info = target + '/Shot Info.txt' # Text file containing timing and assoicated shot numbers

row = 575 # Index of lineout row
scale = 0.0085 # mm per pixel scale
scale_err = 0.0002 # Uncertainty in scale in mm per pixel

ms = 9.109e-31 # Average mass of relevant particles in kg
N = 1e25 # Number density (#/m^3)
rho = ms*N # Mass density (kh/m^3)
L = 1e-2 # Average spark length in meters
abel_tn = np.asarray([45,70,90,125],dtype='int') # Time steps that were Abel inverted

# ---------- Functions ----------

def getShotInfo(info_path=shot_info): # Get shot information from Shot Info text file
    info = pd.read_csv(info_path,delimiter=',',engine='python').to_numpy() # Read Shot Info text file
    time = (info[:,0].astype(float))*1e6 # Get shot time in ns
    time = abs(time - time[0]) # Set initial time to zero
    Ni,Nf = info[:,1].astype(int),info[:,2].astype(int) # Initial and final shot number for each time setting
    numSets = len(time) # Number of datasets taken
    return time,Ni,Nf,numSets

def readSingle(shotNum,row=row,source=source,target=target): # Read unwrapped phase from a single shot, take lineout, find width in pixels
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
    half_max_idx_left = np.argmin(np.abs(lineout[:max_idx]-half_max))
    half_max_idx_right = np.argmin(np.abs(lineout[max_idx:]-half_max)) + max_idx

    #file = open(f'{target}/widths.txt','a')  # Open widths file in append mode
    #file.write(f'{str(shotNum).zfill(5)},{pxl[half_max_idx_left]},{pxl[half_max_idx_right]}\n') # Write left and write pixels of FWHM
    #file.close()

    idx = np.asarray([half_max_idx_left,half_max_idx_right]) # Array of FWHM indices

    fwhm = np.abs(pxl[half_max_idx_left] - pxl[max_idx])+np.abs(pxl[max_idx]-pxl[half_max_idx_right]) # FWHM in pixels

    # ***** End of James' Code *****

    # Make plots
    '''_,ax = plt.subplots(1,2,figsize=(12,8))
    ax[0].plot(pxl,np.ones(N)*row,c='black',label=f'Lineout at Row {row}') # Plot line indicating lineout row
    ax[0].imshow(shot) # Plot unwrapped phase
    ax[0].set_title(f'Unwrapped Phase')
    ax[0].legend()
    ax[1].plot(pxl,lineout,label='Lineout')
    ax[1].scatter(pxl[idx[0]],lineout[idx[0]],marker='+',c='r',label=f'Left')
    ax[1].scatter(pxl[idx[1]],lineout[idx[1]],marker='+',c='g',label=f'Right')
    ax[1].scatter(pxl[max_idx],max_val,marker='*',c='r',label='Max Value')
    ax[1].set_xlabel('Pixel Number')
    ax[1].set_ylabel('Unwrapped Phase')
    ax[1].set_title(f'Lineout at Row {row}')
    ax[1].legend()
    ax[1].grid()
    plt.show()'''

    return fwhm,idx,lineout,pxl,N

def readSet(Ni,Nf): # Read set of unwrapped phase for one time setting, find widths
    numShots = Nf - Ni + 1 # Determine number of shots in the set
    widths = [] # Create empty array for widths
    for i in np.linspace(0,numShots-1,numShots,dtype='int'): # Loop through, find widths
        try: # Try to find array, except statement catches missing numbers (blank images)
            width,_,_,_,_ = readSingle(Ni+i) # zfill pads the shot number with leading zeros
            widths.append(width) # Append width to array
        except:
            print(f'Unwrapped phase not found for shot {str(Ni+i).zfill(5)}, image is blank. Skipping to next shot...') # Skip images not unwrapped
    return widths

def readAll(scale=scale,scale_err=scale_err):
    time,Ni,Nf,numSets = getShotInfo() # Get shot info
    fwhm_pxl = np.empty([numSets],dtype='object') # Create array to save all widths in pixels
    avg_pxl = np.zeros(numSets) # Array for average widths
    std_pxl = np.zeros(numSets) # Array for STD of widths
    for i in np.linspace(0,numSets-1,numSets,dtype='int'): # loop through sets, unwrap images
        fwhm_pxl[i] = readSet(Ni[i],Nf[i]) # Save radius array
        avg_pxl[i] = np.average(fwhm_pxl[i])/2 # Compute average radius
        std_pxl[i] = np.std(fwhm_pxl[i])/2 # Compute standard deviation of radii
    print('DONE')

    ravg = scale*avg_pxl # Get average radius in mm
    std = np.sqrt( pow(avg_pxl*scale_err,2) + pow(scale*std_pxl,2) ) # Propogate error

    '''plt.figure(figsize=(12,8))
    for i in np.linspace(0,numSets-1,numSets,dtype='int'):
        plt.scatter(time[i]*np.ones(len(fwhm_pxl[i])),np.asarray(fwhm_pxl[i])/2)
        plt.scatter(time[i],avg_pxl[i],marker='+',c='r')
        plt.errorbar(time[i],avg_pxl[i],yerr=std_pxl[i],xerr=None,ls='none',c='black',capsize=5)
    plt.scatter(time[-1],avg_pxl[-1],marker='+',c='r',label='Average Value')
    plt.errorbar(time[-1],avg_pxl[-1],yerr=std_pxl[-1],xerr=None,ls='none',c='black',capsize=5,label='Standard Deviation')
    plt.xlabel('Time (ns)')
    plt.ylabel('Spark Radius (pixels)')
    plt.title('Radius vs Time')
    plt.legend()
    plt.grid()
    plt.show()'''

    return ravg,std,time

def estimateVel_fd(ravg,std,time): # Estimate shock velocity with finite difference
    numv = len(time) # Get number of velocity points
    vel = np.zeros([numv-1]) # Velocity array
    errv = np.zeros([numv-1]) # Error in velocity from error propagation
    for i in np.linspace(1,numv-1,numv-1,dtype='int'):
        dt = time[i] - time[i-1] # Get time step
        vel[i-1] = (ravg[i] - ravg[i-1])/dt # Compute v
        errv[i-1] = (1/dt)*np.sqrt(pow(std[i],2) + pow(std[i-1],2)) # Propogate error
    t = time[1:] # Get assoicated time values

    plt.figure(figsize=(12,8))
    plt.scatter(t,vel,label='Average Velocity')
    plt.errorbar(t,vel,yerr=errv,xerr=None,ls='none',c='black',capsize=5,label='Standard Deviation')
    plt.xlabel('Time (ns)')
    plt.ylabel('Shock Velocity (pixels/ns)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid()
    plt.show()

def r(t,A,p,C): # Power law to fit to rn,tn data
    return A*pow(t,p) + C

def rdot(t,A,p): # Time derivative of r(t)
    return A*p*pow(t,p-1)

def dvdA(t,p): # Derivative of rdot w.r.t. A
    return p*pow(t,p-1)

def dvdp(t,A,p): # Derivative of rdot w.r.t. A
    return A*( pow(t,p-1) + p*(p-1)*pow(t,p-2) )

def estimateT(v,r,t,errv,errR,tn,rho=rho,L=L): # Estimate kinetic energy
    idx = np.where((t*1e9).astype(int) == int(tn)) # Find index of time step of interest (Abel inverted shot)

    R = r[idx] # Get radius
    V = v[idx] # Get velocity
    ERR_R = errR[idx] # Get radius error
    ERR_V = errv[idx] # Get velocity error
    
    T = np.pi*rho*R*L*pow(V,2) # Estimate kinetic energy
    ERR_T = np.sqrt( pow(2*np.pi*rho*R*L*V*ERR_V,2) + pow(np.pi*rho*L*pow(V,2)*ERR_R,2) ) # Compute error

    return T,ERR_T

def estimateVel_fit(ravg,std,time,abel_tn=abel_tn): # Estimate shock velocity by fitting r(t) and computing derivative
    popt,pcov = opt.curve_fit(r,time,ravg,p0=[1,1,1],sigma=std) # Fit r(t,A,p,C) to data
    t_upsample = np.linspace(time[0],time[-1],1000) # Upsample time for r fit
    t_upsample_v = np.linspace(time[1],time[-1],1000) # Upsample time for velocity fit. Start at t = 5ns to avoid 1/t singularity at = 0.
    rfit = r(t_upsample,*popt) # Evaluate fit
    vfit = rdot(t_upsample_v,popt[0],popt[1]) # Estimate velocity from time derivative of r(t)
    errv = np.sqrt( pow(dvdA(t_upsample_v,popt[1]),2)*pcov[0,0] + pow(dvdp(t_upsample_v,popt[0],popt[1]),2)*pcov[1,1] ) # Propogate error for velocity fit

    T_abel = np.zeros([len(abel_tn)]) # Array to store energy estimates
    T_err = np.zeros([len(abel_tn)]) # Array to store energy errors

    for i in np.linspace(0,len(abel_tn)-1,len(abel_tn),dtype='int'):
        T_abel[i],T_err[i] = estimateT(rdot(time[1:],popt[0],popt[1])*1e6,ravg[1:]*1e-3,time[1:]*1e-9,
                    np.sqrt( pow(dvdA(time[1:],popt[1]),2)*pcov[0,0] + pow(dvdp(time[1:],popt[0],popt[1]),2)*pcov[1,1] )*1e6,std[1:]*1e-3,abel_tn[i])

    print(f'Kinetic Energy Estimates: {T_abel}\nErrors: {T_err}')

    # *** Unit Conversion ***

    vfit *= 1e3 # Get velocity in km/s
    errv *= 1e3

    print(f'\nFit Result: r(t) = At^p + C = {popt[0]}*t^{popt[1]} + {popt[2]}.')

    _,ax = plt.subplots(1,2,figsize=(12,8))
    
    ax[0].scatter(time,ravg,label='Raw Data',c='r')
    ax[0].errorbar(time,ravg,xerr=None,yerr=std,ls='none',capsize=5,label='Error',c='blue')
    ax[0].plot(t_upsample,rfit,label=r'$r(t) = At^{p} + C$',c='g')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Spark Radius (mm)')
    ax[0].set_title('Spark Radius Power Law Fit')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t_upsample_v,vfit,label=r'$\dot{r}(t) = Apt^{p-1}$') # Time in seconds
    #ax[1].errorbar(t_upsample_v*1e-9,vfit,yerr=errv,xerr=None,ls='none',c='black',label='Error') # Plot error bars
    ax[1].fill_between(t_upsample_v,vfit-errv,vfit+errv,alpha=0.25,color='g',label='Error') # Visualize error with shade between
    ax[1].scatter(time[1:],rdot(time[1:],popt[0],popt[1])*1e3,c='r',label='Data')
    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Shock Velocity (km/s)')
    ax[1].set_title(r'$\dot{r}(t)$ Computed from Power Law Fit')
    ax[1].legend()
    ax[1].grid()

    plt.subplots_adjust(wspace=0.3)
    plt.show()

def computeShockV(): # Main function to compute shock speed
    #estimateVel_fd(*readAll())
    estimateVel_fit(*readAll())