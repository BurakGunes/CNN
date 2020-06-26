# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 01:50:23 2020

@author: ASUS
"""

from Learning.Layers.Conv2D import Conv2D
from Learning.Layers.Pool import Pool
from Learning.Layers.FullyConnected import FullyConnected
from Learning.Layers.SVM import SVM

from Learning.Sequential import Sequential

from Learning.NonParametric.kNN import kNN

from data_util import util, PCA

import LoadData

import numpy as np

#ignore warnings coming from the mne package for now. 
import warnings
warnings.filterwarnings("ignore")


# # # l1, l2, l3 = LoadData.collect_data_cnn() # use it when data is not in the disk.
# xt,yt,xv,yv,xte,yte = LoadData.read_data_cnn(19017, 9508, 9510)
    

# """Create a Sequential layer. This layer will eventually host layers such as
# a convolutional and fully connected one. Sequential layer can also be used to
# build a fully functioning neural network but this has not been tried before."""
# s = Sequential() 


# s.add(Conv2D(filter_size = (3,3,10),
#               input_size = (22, 128),
#               initialization_technique = "He Normal",
#               nonlinearity = "ReLU"))



# s.add(Pool(poolsize = (2,2),
#             technique = "Mean"))  

    
# s.add(FullyConnected(activation = "SoftMax",
#                       output_size = 2,
#                       initialization_technique = "He Normal"))


# s.set_model_parameters(learning_rate = 0.000005,
#                         rho = 0.975,
#                         optimizer = "SGD+Momentum",
#                         loss = "CrossEntropy",
#                         initial_epoch = 1) 

# s.load_weights(645)


# # # s.fit_model(xt, yt,     #for testing
# # #             xv, yv, #for validation
# # #             epoch = 200,
# # #             verbose = True,
# # #             perEpochVerbose = 1,
# # #             weightSavePerEpoch = 5)  

# acc, spe, sen = s.test(xte[:,:,:128], yte, verbose = True)      


"""Next, classification by SVM is demonstrated"""

# l1, l2, l3 = LoadData.collect_data_svm() # use it when data is not in the disk.
xt,yt,xv,yv,xte,yte = LoadData.read_data_svm(927, 463, 465)


ss = Sequential()

svm = SVM(input_size = (177), output_size =  (1),
          initialization_technique = "Xavier", 
          C = 320, #10
          N = 927)

ss.add(svm)

ss.set_model_parameters(learning_rate = 0.00001, #0.001
                        rho = 0.975,
                        optimizer = "SGD+Momentum",
                        loss = "SoftMargin") 



xtf,ytf,xvf,yvf,xtef,ytef = util.extract_linear_features(xt,yt,xv,yv,xte,yte)

#normalize...
xtf = util.normalize(xtf, axs = 1)
xvf = util.normalize(xvf, axs = 1)
xtef = util.normalize(xtef, axs = 1)

xtf = util.add_bias_term(xtf)
xvf = util.add_bias_term(xvf)
xtef = util.add_bias_term(xtef)

ss.fit_model(xtf, ytf,     #for training
            xvf, yvf, #for validation
            epoch = 1000,
            verbose = True,
            perEpochVerbose = 100,
            weightSavePerEpoch = 201)  

# ss.load_weights(9801)

acc, spe, sen = ss.test(xtef, ytef, verbose = True)      


