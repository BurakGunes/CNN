# -*- coding: utf-8 -*-
"""
Commonly used functions for the proccessing of data. When the algorithm is
taken from elsewhere, its source is given in the source code.
"""

import numpy as np
import mne


def get_eeg():
    path_ = "patient_records\chb06_chb06_"
    extension = ".edf"
    record_numbers = np.arange(1, 25).reshape(24,1)    
    paths = np.repeat(path_, 24).reshape((24,1))
    exts = np.repeat(extension, 24).reshape((24,1))
    p = np.concatenate((paths, record_numbers, exts), axis = 1)
    p = np.delete(p, [10, 18, 19, 20, 21, 22], axis = 0) #not available & corrupted
    return p


def load_patient_data():
    
    """
    Loads all the data in the given path to the memory.
    
    Param:
        path (string) : Path to the .edf file.
        
    Return:
        data (float64) : The continuous multichannel EEG data. 
    """
    
    p = get_eeg()
    
    data_locs = []
    for i in range(p.shape[0]):
        path = p[i,0] + str(p[i,1]) + p[i,2]
        data_locs.append(mne.io.read_raw_edf(path,
                                             preload = False, verbose = False))
    
    return data_locs


def get_as_seconds(t1, t2):
    
    """
    Allows to extract a time span between the given t1 and t2 values. 
    
    The second return value is not inclusive.
    """
    sampling_freq = 256
    
    return sampling_freq * t1, sampling_freq*t2


def normalize(x, axs = 2):
    m = np.mean(x, axis = axs, keepdims = True)
    s = np.std(x, axis = axs, keepdims = True)
    
    return (x - m) / s
 
def im2col(A, BSZ):
    """
    IM2COL related functions are inspired from the lecture notes of Stanford
    University's CS231N course. As the algorithms here are not directly related
    with the coursework we have, I decided to not to implement them from scratch.
    
    Source:
    https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python/30110497#30110497
    """
    stepsize = 1
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

 
def extract_linear_features(xt,yt,xv,yv,xte,yte):
    
    """Deriving from the results of the article 
    "Seizure prediction using cost-sensitive support vector machine"
    9 bands from the 22 channels are calculated. Features are inputted to
    the SVM layer as a 198 dimensional input. 
    
    Bands: : delta (0.5-4Hz), theta (4-8Hz), alpha(8-13Hz), beta(13-30Hz),
    four gamma(30-50Hz, 50-70Hz, 70-90Hz, 90Hz-), 
    
    Sampling rate for the MIT database: 256 Hz
    
    The inputs to this function are in frequency domain.
    """
    
    L = 256*20 #FFT length
    
    double_sided_xt =  xt / L
    double_sided_xv =  xv / L
    double_sided_xte =  xte / L

    single_sided_xt =  double_sided_xt[:, :, 0:256*10+1]
    single_sided_xv =  double_sided_xv[:, :, 0:256*10+1]
    single_sided_xte = double_sided_xte[:, :, 0:256*10+1]
    
    del double_sided_xt, double_sided_xte, double_sided_xv
    
    single_sided_xt[:,:, 1:-1] =  2*single_sided_xt[:, :, 1:-1]
    single_sided_xv[:,:, 1:-1] =  2*single_sided_xv[:, :, 1:-1]
    single_sided_xte[:,:,1:-1]  = 2*single_sided_xte[:, :,1:-1]

    xt_f = np.zeros((single_sided_xt.shape[0], 22 * 8))
    xv_f = np.zeros((single_sided_xv.shape[0], 22 * 8))
    xte_f = np.zeros((single_sided_xte.shape[0], 22 * 8))
    
    
    xt_f = np.concatenate((get_delta(single_sided_xt),
                           get_theta(single_sided_xt),
                           get_alpha(single_sided_xt),
                           get_beta(single_sided_xt),
                           get_gamma1(single_sided_xt),
                           get_gamma2(single_sided_xt),
                           get_gamma3(single_sided_xt),
                           get_gamma4(single_sided_xt)), axis = 1)
    
    xv_f = np.concatenate((get_delta(single_sided_xv),
                           get_theta(single_sided_xv),
                           get_alpha(single_sided_xv),
                           get_beta(single_sided_xv),
                           get_gamma1(single_sided_xv),
                           get_gamma2(single_sided_xv),
                           get_gamma3(single_sided_xv),
                           get_gamma4(single_sided_xv)), axis = 1)

    xte_f = np.concatenate((get_delta(single_sided_xte),
                           get_theta(single_sided_xte),
                           get_alpha(single_sided_xte),
                           get_beta(single_sided_xte),
                           get_gamma1(single_sided_xte),
                           get_gamma2(single_sided_xte),
                           get_gamma3(single_sided_xte),
                           get_gamma4(single_sided_xte)), axis = 1)

    yt_f = np.zeros((yt.shape[0]))
    yv_f = np.zeros((yv.shape[0]))
    yte_f = np.zeros((yte.shape[0]))
    
    for i in range(yt.shape[0]):
        yt_f[i] = 1 if np.argmax(yt[i,:]) == 0 else -1
        
    for i in range(yv.shape[0]):
        yv_f[i] = 1 if np.argmax(yv[i,:]) == 0 else -1
        
    for i in range(yte.shape[0]):
        yte_f[i] = 1 if np.argmax(yte[i,:]) == 0 else -1
    
        
    return xt_f,yt_f,xv_f,yv_f,xte_f,yte_f

def get_delta(x):
    
    delta = np.zeros((x.shape[0], 22))
    
    for i in np.arange(0,5*20):
        delta += x[:, :22, i]
        
    return delta

def get_theta(x):
    
    theta = np.zeros((x.shape[0], 22))
    
    for i in np.arange(4*20,9*20):
        theta += x[:, :22, i]
        
    return theta

def get_alpha(x):
    
    alpha = np.zeros((x.shape[0], 22))
    
    for i in np.arange(8*20,14*20):
        alpha += x[:, :22, i]
        
    return alpha


def get_beta(x):
    
    beta = np.zeros((x.shape[0], 22))
    
    for i in np.arange(13*20,31*20):
        beta += x[:, :22, i]
        
    return beta


def get_gamma1(x):
    
    gamma = np.zeros((x.shape[0], 22))
    
    for i in np.arange(30*20,51*20):
        gamma += x[:, :22, i]
        
    return gamma

def get_gamma2(x):
    
    gamma = np.zeros((x.shape[0], 22))
    
    for i in np.arange(50*20,70*20):
        gamma += x[:, :22, i]
        
    return gamma

def get_gamma3(x):
    
    gamma = np.zeros((x.shape[0], 22))
    
    for i in np.arange(70*20,91*20):
        gamma += x[:, :22, i]
        
    return gamma

def get_gamma4(x):
    
    gamma = np.zeros((x.shape[0], 22))
    
    for i in np.arange(90*20,128*20):
        gamma += x[:, :22, i]
        
    return gamma


def add_bias_term(x):
    
    ones = np.ones((x.shape[0], 1))
    
    x = np.append(x, ones, axis = 1)
    
    return x
