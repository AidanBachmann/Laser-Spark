# Description: 
# Phase unwrapping code written by James Young, adapted to fit pre-existing code and file structure.

# ---------- Imports ----------

from __future__ import print_function
import matplotlib as mpl
import Utils as ut
import numpy as np
import tracemalloc
from tempfile import TemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import time as TIME
outfile = TemporaryFile()

# ---------- Plotting Options ----------
# Use LaTeX for rendering
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

# ---------- File Paths ----------

source = '../Cropped Data' # Directory containing cropped .npz files
target = '../Unwrapped Data' # Target directory to deposit unwrapped images
shot_info = target + '/Shot Info.txt' # Text file containing timing and assoicated shot numbers

# ---------- Functions ----------

def getShotInfo(info_path=shot_info): # Get shot information from Shot Info text file
    info = pd.read_csv(info_path,delimiter=',',engine='python').to_numpy() # Read Shot Info text file
    time = (info[:,0].astype(float))*1e6 # Get shot time in ns
    time = abs(time - time[0]) # Set initial time to zero
    Ni,Nf = info[:,1].astype(int),info[:,2].astype(int) # Initial and final shot number for each time setting
    numSets = len(time) # Number of datasets taken
    return time,Ni,Nf,numSets

def checkIntensity(g,threshold=50): # Check if image is black or if it has a signal. 'g' is green channel from image.
    if np.sum(g)/(g.shape[0]*g.shape[1]) > threshold: # Check if green intensity exceeds threshold
        return 1
    else:
        return 0
    
def unwrap_bg(shotNumber_bg=15,typ='box',size=600,coords = (1330, 850), # Unwrap phase for the background 
         fftycoords=(400,500),fftxcoords=(475),not_horizontal=False,angle=0,downsamplef=1,source=source,target=target):
    f_name_ps = f'{str(shotNumber_bg).zfill(5)}_interferometer' # Get filename of preshot (ambient phase)

    # Initially Perform Wavelet Analysis to clean #
    # Preshot #
    # Split Channels #
    b_o, g_o, r_o = ut.preProcess(f_name=f'{source}/{f_name_ps}.npz', med_ksize=19)

    # Downsample
    f=downsamplef
    fftxcoords = (int(fftxcoords[0]/f), int(fftxcoords[1]/f))
    fftycoords = (int(fftycoords[0]/f), int(fftycoords[1]/f))
    if f > 1:
        r_o = ut.downsample_2d(r_o, freq=f)
        g_o = ut.downsample_2d(g_o, freq=f)
        b_o = ut.downsample_2d(b_o, freq=f)

    g_ps=g_o

    # Now For Shot, Resize and Find Mask #
    b_o = []
    r_o = []

    if not_horizontal:
        g_ps = np.rot90(g_ps, 1)
    # h_n, h_bin = ut.plot_hist_image(g_ps, 30, True)
    # ut.fit_to_dist(g_ps.flatten(), h_n, h_bin, 'gamma')

    # Pad with zeros to 2^N #
    ((F_s, F_ps), orig_size) = ut.pad_and_fft((g_ps))

    # create a mask
    # The mask is an array of 1s at the elements of the fft we want to keep.  The rest of the array is filled with zeros
    mask = ut.generate_fft_mask(fftxcoords=fftxcoords, fftycoords=fftycoords, F_ps=F_ps, F_s=F_ps, typ=typ)

    # apply mask and inverse DFT #
    # First For Preshot then shot #
    ## pss and ss are the images back in the real space, but after filtering in the fourier domain. ##
    ## F_m_ps and F_m_s are the fourier domain images, but after the mask has been applied ##
    (pss, F_m_ps) = ut.apply_mask_ifft((F_ps), mask)
    (phase_ps) = ut.resize_list_images(ut.compute_wrapped_phase((pss)), orig_size)

    # Unwrap Phase #
    (ResPS, RmaskPS) = ut.gen_res_map((phase_ps), int(10/f))
    ut.plot_one_thing(RmaskPS, "residualPS", colorbar=True)
    pps, n_iters_ps, err_ps, maskedSigPS = ut.modifiedCGSolver(phase_ps, 1E-15, RmaskPS, to_plot=True)

    # Resize Images to original size #
    (F_ps, pps) = ut.resize_list_images((F_ps, pps), orig_size)
    pps = pps + abs(pps.min())

    np.savez(f'{target}/{f_name}_ps',pdiffR)

    # Plot Everything and save them#

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14,14)
    im = ax1.imshow(np.log10(np.abs(F_s)+1), cmap="RdBu")
    plt.colorbar(im)
    plt.tight_layout()
    #plt.savefig("./Plots/8_4_2023_shot1_F_s.png", dpi=600)
    plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(np.log10(np.abs(F_ps)+1), cmap="RdBu")
    plt.colorbar(im)
    #plt.savefig("./Plots/8_4_2023_shot1_F_ps.png", dpi=600)
    plt.show()

    # ut.save_mat_to_im(pdiff, '8_4_2023_shot1_pdiff.tiff')

    # stopping the library
    tracemalloc.stop()
    return err_ps

def unwrapSingle(shotNumber=21,shotNumber_bg=15,typ='box',fftycoords=(int(790),int(840)),fftxcoords=(int(300), int(625)),
                 not_horizontal=True,angle=0.0,downsamplef=1,source=source,target=target):
    f_name = f'{str(shotNumber).zfill(5)}_interferometer' # Get filename of shot
    f_name_ps = f'{str(shotNumber_bg).zfill(5)}_interferometer' # Get filename of preshot (ambient phase)

    # Initially Perform Wavelet Analysis to clean #
    # Preshot #
    # Split Channels #
    b_o, g_o, r_o = ut.preProcess(f_name=f'{source}/{f_name_ps}.npz', med_ksize=19)

    # Shot #
    # Split Channels #
    (b, g, r) = ut.preProcess(f_name=f'{source}/{f_name}.npz', med_ksize=19)
    if checkIntensity(g): # Check if image is blank
        pass
    else:
        print(f'Image {f_name} has no signal, skipping to next image...')
        return -1,-1

    # Downsample
    f=downsamplef
    fftxcoords = (int(fftxcoords[0]/f), int(fftxcoords[1]/f))
    fftycoords = (int(fftycoords[0]/f), int(fftycoords[1]/f))
    if f > 1:
        r_o = ut.downsample_2d(r_o, freq=f)
        g_o = ut.downsample_2d(g_o, freq=f)
        b_o = ut.downsample_2d(b_o, freq=f)
        r = ut.downsample_2d(r, freq=f)
        g = ut.downsample_2d(g, freq=f)
        b = ut.downsample_2d(b, freq=f)

    g_ps=g_o
    g_s =g

    # Now For Shot, Resize and Find Mask #
    b = []
    r = []
    b_o = []
    r_o = []

    if not_horizontal:
        g_s = np.rot90(g_s, 1)
        g_ps = np.rot90(g_ps, 1)
    # h_n, h_bin = ut.plot_hist_image(g_ps, 30, True)
    # ut.fit_to_dist(g_ps.flatten(), h_n, h_bin, 'gamma')

    # Pad with zeros to 2^N #
    ((F_s, F_ps), orig_size) = ut.pad_and_fft((g_s, g_ps))

    # create a mask
    # The mask is an array of 1s at the elements of the fft we want to keep.  The rest of the array is filled with zeros
    mask = ut.generate_fft_mask(fftxcoords=fftxcoords, fftycoords=fftycoords, F_ps=F_ps, F_s=F_s, typ=typ)

    # apply mask and inverse DFT #
    # First For Preshot then shot #
    ## pss and ss are the images back in the real space, but after filtering in the fourier domain. ##
    ## F_m_ps and F_m_s are the fourier domain images, but after the mask has been applied ##
    ((pss, F_m_ps), (ss, F_m_s)) = ut.apply_mask_ifft((F_ps, F_s), mask)
    (phase_ps, phase_s) = ut.resize_list_images(ut.compute_wrapped_phase((pss, ss)), orig_size)
    #ut.plot_one_thing(phase_s, "phase_s", colorbar=True)

    # Unwrap Phase #
    ((ResS, RmaskS), (ResPS, RmaskPS)) = ut.gen_res_map((phase_s, phase_ps), int(10/f))
    #ut.plot_one_thing(RmaskS, "residualS", colorbar=True)
    #ut.plot_one_thing(RmaskPS, "residualPS", colorbar=True)
    ps, n_iters_s, err_s, maskedSigS = ut.modifiedCGSolver(phase_s, 1E-15, RmaskS, to_plot=True)
    pps, n_iters_ps, err_ps, maskedSigPS = ut.modifiedCGSolver(phase_ps, 1E-15, RmaskPS, to_plot=True)

    # Compute Difference #
    pdiff = ps-pps

    # Resize Images to original size #
    (pdiff, ps, F_ps, pps, ps) = ut.resize_list_images((pdiff, ps, F_ps, pps, ps), orig_size)
    pps = pps + abs(pps.min())
    ps = ps + abs(ps.min())

    pdiffR = ut.rotate_im(p=pdiff, angle=angle)
    #ut.plot_one_thing(pdiffR, "pdiff", colorbar=True, plot_3d=False, to_show=False, vmax=(-20, 12))

    np.savez(f'{target}/{f_name}_pdiff',pdiffR)
    np.savez(f'{target}/{f_name}_pps',pps)
    np.savez(f'{target}/{f_name}_ps',ps)

    # pdiffRLP = ut.gaussian_filter(pdiffR, sigma=6)
    # ut.plot_one_thing(pdiffRLP, "pdiffRLP", colorbar=True, plot_3d=False, vmax=(-20, 12))

    # Plot Everything #
    # ut.plot_one_thing(pdiff, "pdiff", colorbar=True, plot_3d=False)
    #ut.plot_the_things(g_ps, phase_ps, pps, F_m_ps, F_ps, g_s, phase_s, ps, F_s, F_m_s, pdiff, f_name=f_name)


    # Plot Everything and save them#

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(phase_s, cmap="RdBu", vmin=0, vmax=6.5)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'{target}/{f_name}_phase_s.jpg', dpi=600)
    #plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(pdiff, cmap="RdBu", vmin=-20, vmax=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'{target}/{f_name}_pdiff.jpg', dpi=600)
    #plt.show()

    '''fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14,14)
    im = ax1.imshow(np.log10(np.abs(F_s)+1), cmap="RdBu")
    plt.colorbar(im)
    plt.tight_layout()
    #plt.savefig("./Plots/8_4_2023_shot1_F_s.png", dpi=600)
    plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(np.log10(np.abs(F_ps)+1), cmap="RdBu")
    plt.colorbar(im)
    #plt.savefig("./Plots/8_4_2023_shot1_F_ps.png", dpi=600)
    plt.show()'''

    # ut.save_mat_to_im(pdiff, '8_4_2023_shot1_pdiff.tiff')

    # stopping the library
    tracemalloc.stop()
    return err_s, err_ps

def unwrapSet(Ni,Nf): # Unwrap set of images for one time setting
    numShots = Nf - Ni + 1 # Determine number of shots in the set
    for i in np.linspace(0,numShots-1,numShots,dtype='int'): # Loop through, unwrap each image
        unwrapSingle(str(Ni + i).zfill(5)) # zfill pads the shot number with leading zeros

def unwrapAll(): # Unwrap all data described in Shot Info text file
    _,Ni,Nf,numSets = getShotInfo() # Get shot info
    for i in np.linspace(0,numSets-1,numSets,dtype='int'): # loop through sets, unwrap images
        unwrapSet(Ni[i],Nf[i])
    print('DONE') 

if __name__ == '__main__':
    # # coords in (row, column).
      start = TIME.time()
      e_s, e_ps = unwrapSingle(shotNumber=165)
      end = TIME.time()
      print(f'Finished in {end - start} seconds.')
    