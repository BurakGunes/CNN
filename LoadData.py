# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:46:44 2020

@author: ASUS
"""

import numpy as np

from data_util import util


def load_data_first_record():
    i, p = first_record()    
    
    #arbitrarily assign 1,0 to preictal, 0,1 to preictal
    pre = np.array((1,0)).reshape(1,2)
    inter = np.array((0,1)).reshape(1,2)
    #we have 630 interictal, 431 preictal
    prem = np.repeat(pre, len(p), axis = 0)
    interm = np.repeat(inter, len(i), axis = 0)
    
    labels = np.concatenate((prem, interm), axis = 0)
    data = np.concatenate((p, i), axis = 0)
    
    
    idx = np.random.permutation(data.shape[0])
    x,y = data[idx, :22, :], labels[idx, :]
    
    x = np.abs(np.fft.fftn(x, axes = (1,2)))
    x = util.normalize(x)
    
    x_training = x[:len(x) // 2, :, :]
    y_training = y[:len(y) // 2, :]
    
    x_val = x[len(x) // 2 : len(x) // 2 + len(x) // 4 , :, :]
    y_val = y[len(y) // 2 : len(y) // 2 + len(y) // 4 , :]
    
    x_test = x[len(x) // 2 + len(x) // 4 : len(x), :, :]
    y_test = y[len(y) // 2 + len(y) // 4 : len(y), :]

    
    return x_training, y_training, x_val, y_val, x_test, y_test


def load_data_fourth_record():
    i, p = fourth_record()#i, interictal;p, preictal
    
    #arbitrarily assign 1,0 to preictal, 0,1 to interictal
    pre = np.array((1,0)).reshape(1,2)
    inter = np.array((0,1)).reshape(1,2)
    #we have 630 interictal, 431 preictal
    prem = np.repeat(pre, len(p), axis = 0)
    interm = np.repeat(inter, len(i), axis = 0)
    
    labels = np.concatenate((prem, interm), axis = 0)
    data = np.concatenate((p, i), axis = 0)
    
    
    idx = np.random.permutation(data.shape[0])
    x,y = data[idx, :22, :], labels[idx, :]
    
    # x = util.normalize(x)
    x = np.abs(np.fft.fftn(x, axes = (1,2)))
    x = util.normalize(x)
    
    x_training = x[:len(x) // 2, :, :]
    y_training = y[:len(y) // 2, :]
    
    x_val = x[len(x) // 2 : len(x) // 2 + len(x) // 4 , :, :]
    y_val = y[len(y) // 2 : len(y) // 2 + len(y) // 4 , :]
    
    x_test = x[len(x) // 2 + len(x) // 4 : len(x), :, :]
    y_test = y[len(y) // 2 + len(y) // 4 : len(y), :]

    
    return x_training, y_training, x_val, y_val, x_test, y_test




def first_record():
    
    data = util.load_patient_data()

    i = 0#first record
    raw = data[i].get_data()

    start, end = util.get_as_seconds(0, 1000)#1440
    preictal_1 = raw[:, start:end]
    preictal_windows_1 = preictal_1.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(1520, 1720)
    interictal_1 = raw[:, start:end]
    interictal_windows_1 = interictal_1.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(2200, 6000) #1740,7160
    preictal_2 = raw[:, start:end]
    preictal_windows_2 = preictal_2.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(7260, 7460)
    interictal_2 = raw[:, start:end]
    interictal_windows_2 = interictal_2.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(8000, 12200) #7480 13220
    preictal_3 = raw[:, start:end]
    preictal_windows_3 = preictal_3.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(13320, 13520)
    interictal_3 = raw[:, start:end]
    interictal_windows_3 = interictal_3.reshape((-1, 23, 256))
    
    preictal = np.concatenate((preictal_windows_1,
                               preictal_windows_2,
                               preictal_windows_3), axis = 0)
    
    interictal = np.concatenate((interictal_windows_1,
                                 interictal_windows_2,
                                 interictal_windows_3), axis = 0)
    
    interictal = np.repeat(interictal, 14, axis = 0)
    
    return interictal, preictal


def fourth_record():
       
    data = util.load_patient_data()

    i = 3#fourth record
    raw = data[i].get_data()

    start, end = util.get_as_seconds(100, 327)
    interictal_1 = raw[:, start:end]
    interictal_windows_1 = interictal_1.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(700, 5000)#347-5900
    preictal_1 = raw[:, start:end]
    preictal_windows_1 = preictal_1.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(6000, 6211)
    interictal_2 = raw[:, start:end]
    interictal_windows_2 = interictal_2.reshape((-1, 23, 256))
    
    start, end = util.get_as_seconds(7000, 39052) #6211
    preictal_2 = raw[:, start:end]
    preictal_windows_2 = preictal_2.reshape((-1, 23, 256))
        
    
    preictal = np.concatenate((preictal_windows_1,
                               preictal_windows_2), axis = 0)
    
    interictal = np.concatenate((interictal_windows_1,
                                 interictal_windows_2), axis = 0)
    
    interictal = np.repeat(interictal, 23, axis = 0)
    
    return interictal, preictal




def collect_data_cnn():
    
    xt,yt,xv,yv,xte,yte = load_data_first_record()
        
    xx_t = np.memmap('data/cnn/xt.npy', dtype='float64', mode='w+', shape=(len(xt),22,256))
    yy_t = np.memmap('data/cnn/yt.npy', dtype='float64', mode='w+', shape=(len(yt),2))
    
    xx_v = np.memmap('data/cnn/xv.npy', dtype='float64', mode='w+', shape=(len(xv),22,256))
    yy_v = np.memmap('data/cnn/yv.npy', dtype='float64', mode='w+', shape=(len(yv),2))
    
    xx_te = np.memmap('data/cnn/xte.npy', dtype='float64', mode='w+', shape=(len(xte),22,256))
    yy_te = np.memmap('data/cnn/yte.npy', dtype='float64', mode='w+', shape=(len(yte),2))


    
    xx_t[:,:,:] = xt[:, :, :]
    yy_t[:,:] = yt[:, :]
    xx_v[:,:,:] = xv[:, :, :]
    yy_v[:,:] = yv[:, :]
    xx_te[:,:,:] = xte[:, :, :]
    yy_te[:,:] = yte[:, :]

    
    lt = len(xt)
    lv = len(xv)
    lte = len(xte)

    xt,yt,xv,yv,xte,yte = load_data_fourth_record()
        
    print("First record collected...")
    
    xx_tt = np.memmap('data/cnn/xt.npy', dtype='float64', mode='r+', shape=(lt+len(xt),22,256))
    yy_tt = np.memmap('data/cnn/yt.npy', dtype='float64', mode='r+', shape=(lt+len(yt),2))
    
    xx_vv = np.memmap('data/cnn/xv.npy', dtype='float64', mode='r+', shape=(lv+len(xv),22,256))
    yy_vv = np.memmap('data/cnn/yv.npy', dtype='float64', mode='r+', shape=(lv+len(yv),2))
    
    xx_tete = np.memmap('data/cnn/xte.npy', dtype='float64', mode='r+', shape=(lte+len(xte),22,256))
    yy_tete = np.memmap('data/cnn/yte.npy', dtype='float64', mode='r+', shape=(lte+len(yte),2))

    
    xx_tt[lt:lt+len(xt),:,:] = xt[:, :, :]
    yy_tt[lt:lt+len(xt),:] = yt[:, :]
    xx_vv[lv:lv+len(xv),:,:] = xv[:, :, :]
    yy_vv[lv:lv+len(xv),:] = yv[:, :]
    xx_tete[lte:lte+len(xte),:,:] = xte[:, :, :]
    yy_tete[lte:lte+len(xte),:] = yte[:, :]

    
    idx = np.random.permutation(xx_tt.shape[0])
    xx_tt,yy_tt = xx_tt[idx, :22, :], yy_tt[idx, :]
    
    idx = np.random.permutation(xx_vv.shape[0])
    xx_vv,yy_vv = xx_vv[idx, :22, :], yy_vv[idx, :]
    
    idx = np.random.permutation(xx_tete.shape[0])
    xx_tete,yy_tete = xx_tete[idx, :22, :], yy_tete[idx, :]

    print("Second record collected...")
    
    print()
    print()
    print()
    
    return lt+len(xt), lv+len(xv), lte+len(xte)


def read_data_cnn(l1, l2, l3):
    
    print("Started reading the data")
    
    xt = np.memmap('data/cnn/xt.npy', dtype='float64', mode='r', shape=(l1,22,256), order='C')
    yt = np.memmap('data/cnn/yt.npy', dtype='float64', mode='r', shape=(l1,2), order='C')
    
    xv = np.memmap('data/cnn/xv.npy', dtype='float64', mode='r', shape=(l2,22,256), order='C')
    yv = np.memmap('data/cnn/yv.npy', dtype='float64', mode='r', shape=(l2,2), order='C')
    
    xte = np.memmap('data/cnn/xte.npy', dtype='float64', mode='r', shape=(l3,22,256), order='C')
    yte = np.memmap('data/cnn/yte.npy', dtype='float64', mode='r', shape=(l3,2), order='C')

    
    return xt,yt,xv,yv,xte,yte



def load_data_first_record_svm():
    i, p = first_record_svm()    
    
    #arbitrarily assign 1,0 to preictal, 0,1 to interictal
    pre = np.array((1,0)).reshape(1,2)
    inter = np.array((0,1)).reshape(1,2)
    #we have 630 interictal, 431 preictal
    prem = np.repeat(pre, len(p), axis = 0)
    interm = np.repeat(inter, len(i), axis = 0)
    
    labels = np.concatenate((prem, interm), axis = 0)
    data = np.concatenate((p, i), axis = 0)
    
    
    idx = np.random.permutation(data.shape[0])
    x,y = data[idx, :22, :], labels[idx, :]
    
    x = np.abs(np.fft.fftn(x, axes = (1,2)))
    x = util.normalize(x)
    
    x_training = x[:len(x) // 2, :, :]
    y_training = y[:len(y) // 2, :]
    
    x_val = x[len(x) // 2 : len(x) // 2 + len(x) // 4 , :, :]
    y_val = y[len(y) // 2 : len(y) // 2 + len(y) // 4 , :]
    
    x_test = x[len(x) // 2 + len(x) // 4 : len(x), :, :]
    y_test = y[len(y) // 2 + len(y) // 4 : len(y), :]

    
    return x_training, y_training, x_val, y_val, x_test, y_test


def load_data_fourth_record_svm():
    i, p = fourth_record_svm()#i, interictal;p, preictal
    
    #arbitrarily assign 1,0 to preictal, 0,1 to interictal
    pre = np.array((1,0)).reshape(1,2)
    inter = np.array((0,1)).reshape(1,2)
    #we have 630 interictal, 431 preictal
    prem = np.repeat(pre, len(p), axis = 0)
    interm = np.repeat(inter, len(i), axis = 0)
    
    labels = np.concatenate((prem, interm), axis = 0)
    data = np.concatenate((p, i), axis = 0)
    
    
    idx = np.random.permutation(data.shape[0])
    x,y = data[idx, :22, :], labels[idx, :]
    
    # x = util.normalize(x)
    x = np.abs(np.fft.fftn(x, axes = (1,2)))
    x = util.normalize(x)
    
    x_training = x[:len(x) // 2, :, :]
    y_training = y[:len(y) // 2, :]
    
    x_val = x[len(x) // 2 : len(x) // 2 + len(x) // 4 , :, :]
    y_val = y[len(y) // 2 : len(y) // 2 + len(y) // 4 , :]
    
    x_test = x[len(x) // 2 + len(x) // 4 : len(x), :, :]
    y_test = y[len(y) // 2 + len(y) // 4 : len(y), :]

    
    return x_training, y_training, x_val, y_val, x_test, y_test




def first_record_svm():
    
    data = util.load_patient_data()

    i = 0#first record
    raw = data[i].get_data()

    start, end = util.get_as_seconds(0, 1000)#1440
    preictal_1 = raw[:, start:end]
    preictal_windows_1 = preictal_1.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(1520, 1720)
    interictal_1 = raw[:, start:end]
    interictal_windows_1 = interictal_1.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(2200, 6000) #1740,7160
    preictal_2 = raw[:, start:end]
    preictal_windows_2 = preictal_2.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(7260, 7460)
    interictal_2 = raw[:, start:end]
    interictal_windows_2 = interictal_2.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(8000, 12200) #7480 13220
    preictal_3 = raw[:, start:end]
    preictal_windows_3 = preictal_3.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(13320, 13520)
    interictal_3 = raw[:, start:end]
    interictal_windows_3 = interictal_3.reshape((-1, 23, 256*20))
    
    preictal = np.concatenate((preictal_windows_1,
                               preictal_windows_2,
                               preictal_windows_3), axis = 0)
    
    interictal = np.concatenate((interictal_windows_1,
                                 interictal_windows_2,
                                 interictal_windows_3), axis = 0)
    
    interictal = np.repeat(interictal, 14, axis = 0)
    
    return interictal, preictal


def fourth_record_svm():
       
    data = util.load_patient_data()

    i = 3#fourth record
    raw = data[i].get_data()

    start, end = util.get_as_seconds(127, 327)
    interictal_1 = raw[:, start:end]
    interictal_windows_1 = interictal_1.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(700, 5000)#347-5900
    preictal_1 = raw[:, start:end]
    preictal_windows_1 = preictal_1.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(6011, 6211)
    interictal_2 = raw[:, start:end]
    interictal_windows_2 = interictal_2.reshape((-1, 23, 256*20))
    
    start, end = util.get_as_seconds(7000, 13200) #6211
    preictal_2 = raw[:, start:end]
    preictal_windows_2 = preictal_2.reshape((-1, 23, 256*20))
        
    
    preictal = np.concatenate((preictal_windows_1,
                               preictal_windows_2), axis = 0)
    
    interictal = np.concatenate((interictal_windows_1,
                                 interictal_windows_2), axis = 0)
    
    interictal = np.repeat(interictal, 23, axis = 0)
    
    return interictal, preictal




def collect_data_svm():
    
    xt,yt,xv,yv,xte,yte = load_data_first_record_svm()
        
    xx_t = np.memmap('data/svm/xt.npy', dtype='float64', mode='w+', shape=(len(xt),22,256*20))
    yy_t = np.memmap('data/svm/yt.npy', dtype='float64', mode='w+', shape=(len(yt),2))
    
    xx_v = np.memmap('data/svm/xv.npy', dtype='float64', mode='w+', shape=(len(xv),22,256*20))
    yy_v = np.memmap('data/svm/yv.npy', dtype='float64', mode='w+', shape=(len(yv),2))
    
    xx_te = np.memmap('data/svm/xte.npy', dtype='float64', mode='w+', shape=(len(xte),22,256*20))
    yy_te = np.memmap('data/svm/yte.npy', dtype='float64', mode='w+', shape=(len(yte),2))


    
    xx_t[:,:,:] = xt[:, :, :]
    yy_t[:,:] = yt[:, :]
    xx_v[:,:,:] = xv[:, :, :]
    yy_v[:,:] = yv[:, :]
    xx_te[:,:,:] = xte[:, :, :]
    yy_te[:,:] = yte[:, :]

    
    lt = len(xt)
    lv = len(xv)
    lte = len(xte)

    xt,yt,xv,yv,xte,yte = load_data_fourth_record_svm()
        
    print("First record collected...")
    
    xx_tt = np.memmap('data/svm/xt.npy', dtype='float64', mode='r+', shape=(lt+len(xt),22,256*20))
    yy_tt = np.memmap('data/svm/yt.npy', dtype='float64', mode='r+', shape=(lt+len(yt),2))
    
    xx_vv = np.memmap('data/svm/xv.npy', dtype='float64', mode='r+', shape=(lv+len(xv),22,256*20))
    yy_vv = np.memmap('data/svm/yv.npy', dtype='float64', mode='r+', shape=(lv+len(yv),2))
    
    xx_tete = np.memmap('data/svm/xte.npy', dtype='float64', mode='r+', shape=(lte+len(xte),22,256*20))
    yy_tete = np.memmap('data/svm/yte.npy', dtype='float64', mode='r+', shape=(lte+len(yte),2))

    
    xx_tt[lt:lt+len(xt),:,:] = xt[:, :, :]
    yy_tt[lt:lt+len(xt),:] = yt[:, :]
    xx_vv[lv:lv+len(xv),:,:] = xv[:, :, :]
    yy_vv[lv:lv+len(xv),:] = yv[:, :]
    xx_tete[lte:lte+len(xte),:,:] = xte[:, :, :]
    yy_tete[lte:lte+len(xte),:] = yte[:, :]

    
    idx = np.random.permutation(xx_tt.shape[0])
    xx_tt,yy_tt = xx_tt[idx, :22, :], yy_tt[idx, :]
    
    idx = np.random.permutation(xx_vv.shape[0])
    xx_vv,yy_vv = xx_vv[idx, :22, :], yy_vv[idx, :]
    
    idx = np.random.permutation(xx_tete.shape[0])
    xx_tete,yy_tete = xx_tete[idx, :22, :], yy_tete[idx, :]

    print("Second record collected...")
    
    return lt+len(xt), lv+len(xv), lte+len(xte)


def read_data_svm(l1, l2, l3):
    
    print("Started reading the data")
    
    xt = np.memmap('data/svm/xt.npy', dtype='float64', mode='r', shape=(l1,22,256*20), order='C')
    yt = np.memmap('data/svm/yt.npy', dtype='float64', mode='r', shape=(l1,2), order='C')
    
    xv = np.memmap('data/svm/xv.npy', dtype='float64', mode='r', shape=(l2,22,256*20), order='C')
    yv = np.memmap('data/svm/yv.npy', dtype='float64', mode='r', shape=(l2,2), order='C')
    
    xte = np.memmap('data/svm/xte.npy', dtype='float64', mode='r', shape=(l3,22,256*20), order='C')
    yte = np.memmap('data/svm/yte.npy', dtype='float64', mode='r', shape=(l3,2), order='C')

    
    return xt,yt,xv,yv,xte,yte
