#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:42:44 2017

@author: boyko
"""
print('Hello world..')
import sys
import os.path
sys.path.insert(0, '/home/peters/Work/libsvm-3.22/python/')
import os
import numpy as np
import svmutil
os.chdir('/home/peters/Work/Tasks/FindOptimalParameters/svrpath_research/combined-path-python/')
import comb_epspath
import svrpath_initialcopy
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

dirNameModelToLoad = "/home/peters/Work/Tasks/apr21-2017-stft-2/stft_data_100k/decon_queue/"

lengthOfValidationSet = 15
lengthOfSlidingWindow = 1440
lengthOfTrainingWindow = 6500  + lengthOfValidationSet  #lengthOfValidationSet is included 
slidingWindowNumber = 0
slidingOffset = slidingWindowNumber * lengthOfSlidingWindow

rbf_gamma = 0.001254873649928
eps = 0.000000001480642
Lambda = 1880.702058888577312

maxIterations = 10000
lambdamin = 1e-4
rho = 1e-9

def trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet):
        svm_y_train = svm_y[:-lengthOfValidationSet];
        svm_x_train = svm_x[:-lengthOfValidationSet];
#        gamma_rbf = 10e-3
        cost = 1/Lambda
        base_svm_options = '-s 3 -t 2';
        gamma_opt = '-g ' + str(gamma_rbf);
        eps_opt = '-p ' + str(eps);
        cost_opt = '-c ' + str(cost);
        svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
        m = svmutil.svm_train(svm_y_train, svm_x_train, svm_params);
        return m
    
def trainandvalidate_continiousvalidation(gamma_rbf, Lambda, eps, fileNameModelToLoadFull, lengthOfValidationSet):
#    m = comb_epspath.trainlibmsvm(gamma_rbf=1e-3, Lambda = 4, eps = 8e-6, fileNameModelToLoadFull=fileNameModelToLoadFull)    
    m = comb_epspath.trainlibmsvm(gamma_rbf, Lambda, eps, fileNameModelToLoadFull,lengthOfValidationSet)     
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    slidingWindowNumber = 0
    slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
    while lengthOfTrainingWindow + lengthOfValidationSet + slidingOffset < len(svm_y_init):
#    while True:
        print ('\nValidation: sliding window: ', slidingWindowNumber, ' sliding offset:', slidingOffset, ' length of training window:',lengthOfTrainingWindow)
        svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        slidingWindowNumber += 1
        slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
        ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
        ValidStartIdx = (int(len(svm_x) - lengthOfValidationSet));
        yValidNew = svm_y[ValidStartIdx:];
        xValidNew = svm_x[ValidStartIdx:];
        _label, p_acc, p_val = svmutil.svm_predict(yValidNew, xValidNew, m);

def trainandvalidate_continiousmodel(gamma_rbf, Lambda, eps, fileNameModelToLoadFull, lengthOfValidationSet):
    res_validation_over_window = {'gamma_rbf' : gamma_rbf, 'Lambda' : Lambda, 'eps' : eps, 
                                  'lengthOfValidationSet' : lengthOfValidationSet, 
                                  'lengthOfTrainingWindow' : lengthOfTrainingWindow, 'lengthOfSlidingWindow' : lengthOfSlidingWindow,
                                  'slidingOffset' : [], 'ValidationMAEError' : [], 'avgVMAES' : 0.}
#    res = { 'alphas' : [], 'gammas' : [], 'beta0' : [], 'maes' : [], 'errors' : [], 
#       'eps' : [], 'ElbowLeft' : [], 'ElbowRight' : [], 'Center' : [], 'RightRegion' : [], 'LeftRegion' : [], 
#       'TimeTaken' : float('Inf'), 'fl' : [],     'LastEventPoint' : None, 'LastEvent' : None }
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
        m = trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet)
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
    plt.title(fileNameModelToLoad + ':' + str(lengthOfTrainingWindow) + ':' + str(lengthOfSlidingWindow) + ':' + ("%.3E" % res_validation_over_window['avgVMAES']) + '\nAt hyperparameters: ' + 'Gamma-RBF:' + str(gamma_rbf) +' eps:' +  str(eps) + ' Lambda:' + str(Lambda));    
    plt.xlabel('Offset');
    plt.ylabel('Validation-RMSE-libsvm');
    xSlidingOffset = res_validation_over_window['slidingOffset']
    yValidationError = res_validation_over_window['ValidationMAEError']
    plt.plot(xSlidingOffset, yValidationError, 'x')
    
def plotlambdaeps_1(RBFGamma, Lambda, res):
    plt.figure(); 
    plt.title('At hyperparameters: ' + 'Gamma-RBF:' + str(RBFGamma) + ' Lambda:' + str(Lambda))   
    plt.xlabel('Epsilon');
    plt.ylabel('RMSE');
    x = res['eps']
    y = res['maes']
    plt.plot(x, y, 'x')
    
def plotlambdaeps_2(RBFGamma, eps, res):
    plt.figure(); 
    plt.title('At hyperparameters: ' + 'Gamma-RBF:' + str(RBFGamma) + ' Lambda:' + str(eps))    
    plt.xlabel('Lambda');
    plt.ylabel('RMSE');
    x = res['lambdas']
    y = res['maes']
    plt.plot(x, y, 'x')    

def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

listBestMAE = []
listBestRMSE = []
#resBestLambda = { 'listBestLambdaMAE' : [], 'listBestLambdaRMSE' : []}
 
for i in range (1,8):
    
    fileModelIterationName = str(i)
    fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
    fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
    gram_matrix_filename = fileNameModelToLoadFull + '.gram_matrix' + '.pickle'
    
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)

    svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
    svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
    ValidStartIdx = int(len(svm_x) - lengthOfValidationSet)
    xFull = get_numpy_x_from_svm(svm_x)
    xFull2 = np.asarray(xFull)
    xFull3 = xFull2[0:ValidStartIdx]
    yFull = np.asarray(svm_y)
    x = svm_x[0:ValidStartIdx];
    y = svm_y[0:ValidStartIdx];
    K = np.ndarray(shape=[len(xFull3),len(xFull3)])

    xValidation = xFull2[-lengthOfValidationSet:]
    yValidation = yFull[-lengthOfValidationSet:]
#    print('Looking for previous Gram matrix...')
#    K= pickle.load(open(gram_matrix_filename,'rb')) #gramMatrix
    print('Calculating Gram matrix...')
    for idx1, item1 in enumerate(xFull3):
        for idx2, item2 in enumerate(xFull3):
            K[idx1,idx2] = np.exp(-rbf_gamma * np.dot(item1-item2,item1-item2))
    print('Done calculating Gram matrix.')
#    pickle.dump(K,open(gram_matrix_filename,'wb'),pickle.HIGHEST_PROTOCOL);
#    print('Save calculated Gram matrix.')
                                        
    print('Testing svrpath..')
    length = int(15000) #+(i-1)*250)
    res_initial = svrpath_initialcopy.svrpath(xFull3, y, xValidation, yValidation, K, eps, RBF_Gamma = rbf_gamma, maxIterations = maxIterations, lambdamin = lambdamin, rho = 1e-8, loadPreviousComputed = False);
    print('------',len(res_initial['alphas']),'iterations,')
    bestIteration = np.argmin(res_initial['maes'])
    print('Output lambda:',res_initial['lambdas'][bestIteration])
    print('Output eps-lambda, Lambda run#'+ (str(i)) + ' Eps:',eps,'Lambda:', res_initial['lambdas'][bestIteration], ' at iteration:', bestIteration)                                 
    Lambda = res_initial['lambdas'][bestIteration]
    bestIteration_vMAES = np.argmin(res_initial['v_maes'])
    print('Output lambda V-MAES:',res_initial['lambdas'][bestIteration_vMAES])
    bestIteration_vRMSE = np.argmin(res_initial['v_rmse'])
    print('Output lambda V-RMSE:',res_initial['lambdas'][bestIteration_vRMSE])
    listBestMAE.append(res_initial['lambdas'][bestIteration_vMAES])
    listBestRMSE.append(res_initial['lambdas'][bestIteration_vRMSE])
                               