#!/usr/bin/env python

#import hyperParamChooseLib
import numpy as np
import subprocess
import os
#from svmutil import *
from six.moves import cPickle as pickle
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import re
import time
#import coordinateDescent
#import combined-path-test-real-data.py

time.strftime("%d/%m||%H:%M:%S")
currentdate = time.strftime("%m_%d")
commentForFiles = 'selection_best_m16A_'
dirName = '/home/peters/Work/Tasks/may-9-stft/1000models/'
saveRangeName = 'proposedHyperParamRanges_'+currentdate +'_' + commentForFiles + '.txt'
saveRangeName_SQL_UPDATE = 'proposedHyperParam_'+currentdate +'_' + commentForFiles + '.sql'

allparams = [];
levels = 16
params = pickle.load(open(dirName + 'populatedRandomSearch.pickle','rb'));

#model = np.asarray(params['model'+str(0)])
#plt.figure()
#sortedError = np.sort(model[:,5])
#plt.plot(sortedError)
#model = np.asarray(params['model'+str(0)])
#bestval = min(model[:,5,:])
#model_bests = model[model[:,5] <= 0.000028]
#print("for model: " + str(0) + '\n\n')
#print(model_bests)
### VALIDATEE VALUES BY CALLING LIBSVM ON SELECTED BEST VALUES
                    
#trainandvalidate_continiousmodel(gamma_rbf=rbf_gamma, Lambda=Lambda, eps=eps, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)

#def extractAverages(model_all_iterations):
#    model = model_all_iterations[:,2:]
#    return model
           
outputs = np.zeros(shape=[levels, 3, 3])
for modelIdx in np.arange(0,levels):
#    model_all_iterations = np.asarray(params['model'+str(modelIdx)])
#    model = extractAverages(model_all_iterations)
    model = np.asarray(params['model'+str(modelIdx)])
#    model = model2[300:,:]
#    model[:,3] = abs(model[:,3])
#    target_acc = np.percentile(model[:,3],5)
#    model_bests = model[model[:,3] <= target_acc]
#    acrossSlides = (model[:,5])
#    avgAcrossSlides = np.mean(acrossSlides)
    bestval = min(model[:,3])
    model_bests = model[model[:,3] <= bestval]
#    print("for model: " + str(modelIdx) + '\n\n')
#    print(model_bests)
    # do a 5-percent trimmed mean for upper and lower
    # do a median for best
    for hyper_param in [0,1,2]:
        p_hi = np.percentile(model_bests[:,hyper_param], 90)
        p_lo = np.percentile(model_bests[:,hyper_param], 10)
        p_med = np.median(model_bests[:,hyper_param])
        outputs[modelIdx, hyper_param, 2] = p_hi
        outputs[modelIdx, hyper_param, 1] = p_lo
        outputs[modelIdx, hyper_param, 0] = p_med

    
#bestsPerLevel = np.asarray(bestsPerLevel)
#newBestsPerLevel = np.ndarray(shape=[bestsPerLevel.shape[0],bestsPerLevel.shape[2]])
#for idx, item in enumerate(bestsPerLevel):
#    for feat in range(0,5):
#        mn = np.mean(np.log(bestsPerLevel[idx,:,feat]))
#        newBestsPerLevel[idx,feat] = np.exp(mn)
#    
#print(newBestsPerLevel)
with open(dirName + saveRangeName_SQL_UPDATE,'wt') as f:
    for levelidx in range(0,levels):
        f.write('UPDATE public.svr_parameters ')
        f.write(' SET  ')        
        f.write('svr_c=%.15f' %(outputs[levelidx, 2, 0]))
        f.write(', svr_epsilon=%.15f' %(outputs[levelidx, 1, 0]))
        f.write(', svr_kernel_param=%.15f' %(outputs[levelidx, 0, 0]))
        f.write('\n   WHERE decon_level = ' + str(levelidx))
        f.write(' and dataset_id = 10000;\n\n') 



#UPDATE public.svr_parameters
#   SET svr_c=97.109437733532999, svr_epsilon=0.000018515436145, svr_kernel_param=0.001381519086749
# WHERE decon_level = 0 and dataset_id = 10000;

with open(dirName + saveRangeName,'wt') as f:
    for levelidx in range(0,levels):
        f.write('Level '+str(levelidx)+'\n')
        hparamnames = ['Gamma: ', 'Eps: ','Cost: ']
#        hparamnames = ['Eps: ']        
        for hyper_param in np.arange(0,3):
            f.write(hparamnames[hyper_param])
            f.write('(%.15f) %.15f - %.15f' % ((outputs[levelidx, hyper_param, 0], outputs[levelidx, hyper_param, 1], outputs[levelidx, hyper_param, 2])))
#            f.write('(%.15f) ' % ((outputs[levelidx, hyper_param, 1])))
            f.write('\n')