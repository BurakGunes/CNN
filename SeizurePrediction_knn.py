# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:42:39 2020

@author: ASUS
"""

from Learning.Sequential import Sequential

from Learning.NonParametric.kNN import kNN

from data_util import util, PCA

import LoadData

import numpy as np

#ignore warnings coming from the mne package for now. 
import warnings
warnings.filterwarnings("ignore")


"""This script is used to make predictions with kNN"""

###############################################################################
# # l1, l2, l3 = LoadData.collect_data_svm() # use it when data is not in the disk.
xt,yt,xv,yv,xte,yte = LoadData.read_data_svm(927, 463, 465)
xtf,ytf,xvf,yvf,xtef,ytef = util.extract_linear_features(xt,yt,xv,yv,xte,yte)
#append xvf to xtef because we don't need the validation set anymore...
xte_f = np.append(xtef, xvf, axis = 0)
yte_f = np.append(ytef, yvf, axis = 0)
#copy 
xt_f = xtf
yt_f = ytf
###############################################################################

#initialize the kNN
knn = kNN()
knn.compile_model(input_size = (927, 10), output_size = 2)
knn.set_k(k = 3)
knn.set_distance_function("M") #minkowski...

"""choose reduction parameter r"""


"""choose k """


"""choose p of minkowski"""




def pca_test():
    d = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    acc = np.zeros((9))
    sen = np.zeros((9))
    spe = np.zeros((9))

    for i, di in enumerate(d):
        print(f"Given d = {di}:")
    
        tr = util.normalize(xt_f, axs = 1)
        te = util.normalize(xte_f, axs = 1)
    
    
        tr = PCA.pca(tr, di)
        te = PCA.pca(te, di)
    
        knn.fit_model(tr, yt_f)
        acc[i], spe[i], sen[i] = knn.test(te, yte_f, verbose = True, pp=2)
        print("") # skip one line
    

def k_test():
    ks = [1,3,5,7,9]
    acc = np.zeros((5))
    sen = np.zeros((5))
    spe = np.zeros((5))

    for i, ki in enumerate(ks):
        print(f"Given k = {ki}:")
        knn.set_k(ki)
        
        tr = util.normalize(xt_f, axs = 1)
        te = util.normalize(xte_f, axs = 1)
    
    
        tr = PCA.pca(tr, 10)
        te = PCA.pca(te, 10)

        knn.fit_model(tr, yt_f)
        acc[i], spe[i], sen[i] = knn.test(te, yte_f, verbose = True, pp = 2)
        print("") # skip one line
        
def p_test():
    pii = [1.5,2, 2.5, 3,3.5]
    acc = np.zeros((6))
    sen = np.zeros((6))
    spe = np.zeros((6))

    for i, pi in enumerate(pii):
        print(f"Given p = {pi}:")
        knn.set_k(1)
        
        tr = util.normalize(xt_f, axs = 1)
        te = util.normalize(xte_f, axs = 1)
    
    
        tr = PCA.pca(tr, 10)
        te = PCA.pca(te, 10)

        knn.fit_model(tr, yt_f)
        acc[i], spe[i], sen[i] = knn.test(te, yte_f, verbose = True, pp = pi)
        print("") # skip one line



knn.set_k(1)
tr = util.normalize(xt_f, axs = 1)
te = util.normalize(xte_f, axs = 1)
    
tr = PCA.pca(tr, 10)
te = PCA.pca(te, 10)

knn.fit_model(tr, yt_f)
knn.test(te, yte_f, verbose = True, pp = 3)


# import matplotlib.pyplot as plt

# plt.plot(p,acc, label="acc")
# plt.plot(p,spe, label="spe")
# plt.plot(p,sen, label="sen")
# plt.legend(loc = "upper right")
