#!/usr/bin/env python

#import hyperParamChooseLib
import numpy as np
#import subprocess
#import os
#from svmutil import *
from six.moves import cPickle as pickle
#import math
import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
#from mpl_toolkits.mplot3d import Axes3D
#import re
#import coordinateDescent
import sys
#import os.path
sys.path.insert(0, '/home/peters/Work/libsvm-3.22/python/')
#import os
#os.chdir('/home/peters/Work/libsvm-3.22/python/')
#import numpy as np
import svmutil

allparams = [];
levels = 16
params = pickle.load(open('/home/peters/Work/Tasks/apr20-2017-stft-untime/populatedRandomSearch.pickle','rb'));

lengthOfValidationSet = 15
lengthOfSlidingWindow = 1440
lengthOfTrainingWindow = 6600  + lengthOfValidationSet  #lengthOfValidationSet is included 
slidingWindowNumber = 0
slidingOffset = slidingWindowNumber * lengthOfSlidingWindow

dirNameModelToLoad = "/home/peters/Work/Tasks/apr20-2017-stft-untime/"



def trainlibmsvm_here(gamma_rbf, Cost, eps, svm_x, svm_y, lengthOfValidationSet):
        svm_y_train = svm_y[:-lengthOfValidationSet];
        svm_x_train = svm_x[:-lengthOfValidationSet];
        cost = Cost
        base_svm_options = '-s 3 -t 2';
        gamma_opt = '-g ' + str(gamma_rbf);
        eps_opt = '-p ' + str(eps);
        cost_opt = '-c ' + str(cost);
        svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
        m = svmutil.svm_train(svm_y_train, svm_x_train, svm_params);
        return m
    
def trainandvalidate_continiousmodel(gamma_rbf, Cost, eps, fileNameModelToLoadFull, lengthOfValidationSet):
    res_validation_over_window = {'gamma_rbf' : gamma_rbf, 'Cost' : Cost, 'eps' : eps, 
                                  'lengthOfValidationSet' : lengthOfValidationSet, 
                                  'lengthOfTrainingWindow' : lengthOfTrainingWindow, 'lengthOfSlidingWindow' : lengthOfSlidingWindow,
                                  'slidingOffset' : [], 'ValidationMAEError' : [], 'avgVMAES' : 0.}
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    slidingWindowNumber = 0
    sumValidationPredictionErrors = 0
    slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
    while lengthOfTrainingWindow + slidingOffset < len(svm_y_init):
        print ('\nValidation: sliding window: ', slidingWindowNumber, ' sliding offset:', slidingOffset, ' length of training window:',lengthOfTrainingWindow)
        svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        slidingWindowNumber += 1
        slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
        m = trainlibmsvm_here(gamma_rbf, Cost, eps, svm_x, svm_y, lengthOfValidationSet)
        ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
        ValidStartIdx = (int(len(svm_x) - lengthOfValidationSet));
        yValidNew = svm_y[ValidStartIdx:];
        xValidNew = svm_x[ValidStartIdx:];
        _label, p_acc, p_val = svmutil.svm_predict(yValidNew, xValidNew, m)
        res_validation_over_window['slidingOffset'].append(slidingOffset)
        res_validation_over_window['ValidationMAEError'].append(p_acc[1])
        sumValidationPredictionErrors += p_acc[1]
        
    res_validation_over_window['avgVMAES'] = sumValidationPredictionErrors / (slidingWindowNumber+1)
    print ('\nValidation: Continious Model Results, sliding over : ', 
           slidingWindowNumber, ' \nwindows with TP lenght:', lengthOfTrainingWindow, 
           '\n Average validation-predict MAE error of libsvm: ', res_validation_over_window['avgVMAES']
           )
    plt.figure(); 
    plt.title(fileNameModelToLoadFull + ':' + str(lengthOfTrainingWindow) + ':' + str(lengthOfSlidingWindow) + ':' + ("%.3E" % res_validation_over_window['avgVMAES']) + '\nAt hyperparameters: ' + 'Gamma-RBF:' + str(gamma_rbf) +' eps:' +  str(eps) + ' Cost:' + str(Cost));    
    plt.xlabel('Offset');
    plt.ylabel('Validation-RMSE-libsvm');
    xSlidingOffset = res_validation_over_window['slidingOffset']
    yValidationError = res_validation_over_window['ValidationMAEError']
    plt.plot(xSlidingOffset, yValidationError, 'x')
                    
### VALIDATEE VALUES BY CALLING LIBSVM ON SELECTED BEST VALUES
outputs = np.zeros(shape=[levels, 3, 3])
for modelIdx in np.arange(0,levels):
    model = np.asarray(params['model'+str(modelIdx)])
    target_acc = np.percentile(model[:,3],2)
    model_bests = model[model[:,3] <= target_acc]
#    print(model_bests)
    medianGamma = np.median(model_bests[modelIdx,0])
    medianEPS = np.median(model_bests[modelIdx,1])
    medianCost = np.median(model_bests[modelIdx,2])
    fileModelIterationName = str(modelIdx)
    fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
    fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
    trainandvalidate_continiousmodel(gamma_rbf=medianGamma, Cost=medianCost, eps=medianEPS, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)

    # do a 5-percent trimmed mean for upper and lower
    # do a median for best
#    for hyper_param in [0,1,2]:
#        p_hi = np.percentile(model_bests[:,hyper_param], 90)
#        p_lo = np.percentile(model_bests[:,hyper_param], 10)
#        p_med = np.median(model_bests[:,hyper_param])
#        outputs[modelIdx, hyper_param, 2] = p_hi
#        outputs[modelIdx, hyper_param, 1] = p_lo
#        outputs[modelIdx, hyper_param, 0] = p_med                    
                    
#outputs = np.zeros(shape=[levels, 3, 3])
#for modelIdx in np.arange(0,levels):
#    model = np.asarray(params['model'+str(modelIdx)])
#    target_acc = np.percentile(model[:,3],0.5)
#    model_bests = model[model[:,3] <= target_acc]
#    # do a 5-percent trimmed mean for upper and lower
#    # do a median for best
#    for hyper_param in [0,1,2]:
#        p_hi = np.percentile(model_bests[:,hyper_param], 90)
#        p_lo = np.percentile(model_bests[:,hyper_param], 10)
#        p_med = np.median(model_bests[:,hyper_param])
#        outputs[modelIdx, hyper_param, 2] = p_hi
#        outputs[modelIdx, hyper_param, 1] = p_lo
#        outputs[modelIdx, hyper_param, 0] = p_med

    
#bestsPerLevel = np.asarray(bestsPerLevel)
#newBestsPerLevel = np.ndarray(shape=[bestsPerLevel.shape[0],bestsPerLevel.shape[2]])
#for idx, item in enumerate(bestsPerLevel):
#    for feat in range(0,5):
#        mn = np.mean(np.log(bestsPerLevel[idx,:,feat]))
#        newBestsPerLevel[idx,feat] = np.exp(mn)
#    
#print(newBestsPerLevel)
#with open('proposedHyperParamRanges_local_5daysrun_specRBF_2.txt','wt') as f:
#    for levelidx in range(0,levels):
#        f.write('Level '+str(levelidx)+'\n')
#        hparamnames = ['Gamma: ', 'Eps: ','Cost: ']
#        for hyper_param in np.arange(0,3):
#            f.write(hparamnames[hyper_param])
#            f.write('(%.15f) %.15f - %.15f' % ((outputs[levelidx, hyper_param, 0], outputs[levelidx, hyper_param, 1], outputs[levelidx, hyper_param, 2])))
#            f.write('\n')



