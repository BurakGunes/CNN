# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:42:24 2020

@author: ASUS
"""

import numpy as np

def pca(x, d):
    """Implements the PCA algorithm to reduce the dimension of data.
    
    Params:
        x (float64) : an array of input values
        d (int16) : the desired dimension count
        
    Return:
        z (float64) : in some sense, equal to the input vector, x
    """
    
    N = x.shape[0] #number of inputs
    sample_covariance = np.matmul(x.T, x) / N
    w, v = np.linalg.eig(sample_covariance)
    
    idx = w.argsort()
    idx = idx[:d]
    
    v = v[:, idx]
    v = np.array(v)
    
    z = x @ v
    
    return z

    