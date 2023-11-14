#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:42:44 2017

@author: boyko
"""
print('Hello world..')
import sys
import os.path
sys.path.insert(0, '/home/peters/Work/gitfolder/libsvm-mod/libsvm-modified/python/')
import os
import numpy as np
import svmutil
import time
os.chdir('/home/peters/Work/Tasks/FindOptimalParameters/svrpath_research/combined-path-python/')

import comb_epspath
#import svrpath_initialcopy
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
dirNameModelToLoad = "/home/peters/Work/Tasks/may-9-stft/15models/"
#fileModelIterationName = '4'
#fileNameModelToLoad = "modelCheck" + fileModelIterationName + ".close" + ".txt"
#fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
#gram_matrix_filename = fileNameModelToLoadFull + '.gram_matrix' + '.pickle'

lengthOfValidationSet = 15
lengthOfSlidingWindow = int(6500/4)
lengthOfTrainingWindow = int(6500) #+ lengthOfValidationSet  #lengthOfValidationSet is included 
maxSlidingWindowNumberGlobal = 15 
slidingWindowNumberGlobal = 0
slidingOffsetGlobal = slidingWindowNumberGlobal * lengthOfSlidingWindow + int(40000)

#maxIterations = 65000
#lambdamin = 1E-6
#rho = 1e-11

def visualizeAllModels():
    plt.figure(1)
    maxnumbermodel = 16
    for i in range(0,maxnumbermodel):
        fileModelIterationName = str(i)
        fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
        svm_filename = dirNameModelToLoad + fileNameModelToLoad
        print ('filename : ' , svm_filename)
        plt.subplot(4,4,i+1)
        svm_y_0_init, svm_x_init = svmutil.svm_read_problem(svm_filename)
        plt.plot(svm_y_0_init)

def svr_train_here_p(_svr_params, x, y):
    gamma_rbf =  _svr_params[2]
    Lambda =  1/(_svr_params[0])
    eps =  _svr_params[2]
    m = trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x = x, svm_y = y, lengthOfValidationSet = 15)
    ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
    ValidStartIdx = (int(len(svm_x) - lengthOfValidationSet));
    yValidNew = svm_y[ValidStartIdx:];
    xValidNew = svm_x[ValidStartIdx:];
    _label, p_acc, p_val = svmutil.svm_predict(yValidNew, xValidNew, m)
    diffMAESValidation = 0
    p_val_array_1 = np.array(p_val)
    for i in range(0,15):
        diffMAESValidation += abs(p_val_array_1[i] - yValidNew[i])
        
    _prev_best_mse  = diffMAESValidation / 15
    return _prev_best_mse

def trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet):
        svm_y_train = svm_y[:-lengthOfValidationSet];
        svm_x_train = svm_x[:-lengthOfValidationSet];
        cost = 1/Lambda
        base_svm_options = '-s 3 -t 2 -h 0 -e 0.001';
#        if(addH0 == True):
#            base_svm_options += ' -h 0 -e 0.001'
        gamma_opt = '-g ' + str(gamma_rbf)
        eps_opt = '-p ' + str(eps)
        cost_opt = '-c ' + str(cost)

        svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
        m = svmutil.svm_train(svm_y_train, svm_x_train, svm_params);
        return m
    
def trainlibmsvm_here_nusvr(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet, Level):
        svm_y_train = svm_y[:-lengthOfValidationSet];
        svm_x_train = svm_x[:-lengthOfValidationSet];
        cost = 1/Lambda
        base_svm_options = '-s 4 -t 2 -n 0.5 -h 0 -e 0.001';
        gamma_opt = '-g ' + str(gamma_rbf);
        cost_opt = '-c ' + str(cost);
        svm_params = base_svm_options+' '+ gamma_opt +' '+cost_opt;
        m = svmutil.svm_train(svm_y_train, svm_x_train, svm_params)
        return m    
    
def trainandvalidate_continiousvalidation(gamma_rbf, Lambda, eps, fileNameModelToLoadFull, lengthOfValidationSet):
    m = comb_epspath.trainlibmsvm(gamma_rbf, Lambda, eps, fileNameModelToLoadFull,lengthOfValidationSet)     
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    slidingWindowNumber = 0
    slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
    while lengthOfTrainingWindow + lengthOfValidationSet + slidingOffset < len(svm_y_init):
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
        visualizeresiduals(yValidNew, p_val)
                                                  
def visualizeresiduals(yValid_GroundTruth, p_val):
    plt.figure(201)
    plt.plot(yValid_GroundTruth, 'r')
    plt.plot(p_val, 'b')    
    

def trainandvalidate_continiousmodel(gamma_rbf, Lambda, eps, fileNameModelToLoadFull, lengthOfValidationSet, train_eps_svr, Level):
    res_validation_over_window = {'gamma_rbf' : gamma_rbf, 'Lambda' : Lambda, 'eps' : eps, 
                                  'lengthOfValidationSet' : lengthOfValidationSet, 
                                  'lengthOfTrainingWindow' : lengthOfTrainingWindow, 'lengthOfSlidingWindow' : lengthOfSlidingWindow,
                                  'slidingOffset' : [], 'ValidationMAEError' : [], 'avgVMAES' : 0.}
    listTimeTakenStringPerlevel = []
#    listTimeTakenTime = []    
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    slidingWindowNumber = 0
    sumValidationPredictionErrors = 0
    sumValidation_p_accrmse = 0
    sumValidation_p_acc1 = 0
    slidingOffset = int(0/2)+slidingWindowNumber * lengthOfSlidingWindow
    while (lengthOfTrainingWindow + slidingOffset < len(svm_y_init)) and (slidingWindowNumber < maxSlidingWindowNumberGlobal):
        print ('\nValidation: sliding window: ', slidingWindowNumber, ' sliding offset:', slidingOffset, ' length of training window:',lengthOfTrainingWindow)
        svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
        slidingWindowNumber += 1
        slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
        print('Length of vector:', len(svm_y),'\n')
#       end = time.time()
#    m, s = divmod(end-start, 60)
#    h, m = divmod(m, 60)
#    timeTaken = ("%d:%02d:%02d" % (h, m, s))
        startTrainingTime = time.time()
        if(train_eps_svr == True):
            m = trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet)
            endTrainingTime = time.time()
            minutes, s = divmod(endTrainingTime-startTrainingTime, 60)
            h, minutes = divmod(minutes, 60)
            timeTaken = ("%d:%02d:%02d" % (h, minutes, s))
            listTimeTakenStringPerlevel.append(timeTaken)
#            listTimeTakenTime.append()
            print("Time to train EPS-SVR with enabled heuristics is",timeTaken)
#        else:
#            m = trainlibmsvm_here(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet, True)
#            endTrainingTime = time.time()
#            minutes, s = divmod(endTrainingTime-startTrainingTime, 60)
#            h, minutes = divmod(minutes, 60)
#            timeTaken = ("%d:%02d:%02d" % (h, minutes, s))
#            listTimeTakenStringPerlevel.append(timeTaken)
##            listTimeTakenTime.append()
#            print("Time to train EPS-SVR with disabled heuristics is",timeTaken)            
        else:
            m = trainlibmsvm_here_nusvr(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet, Level = Level)
            endTrainingTime = time.time()
            minutes, s = divmod(endTrainingTime-startTrainingTime, 60)
            h, minutes = divmod(minutes, 60)
            timeTaken = ("%d:%02d:%02d" % (h, minutes, s))
            print("Time to train NU-SVR is",timeTaken)        

        ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
        ValidStartIdx = (int(len(svm_x) - lengthOfValidationSet));
        yValidNew = svm_y[ValidStartIdx:];
        xValidNew = svm_x[ValidStartIdx:];
        _label, p_acc, p_val = svmutil.svm_predict(yValidNew, xValidNew, m)
        res_validation_over_window['slidingOffset'].append(slidingOffset)
        diffMAESValidation = 0
        p_val_array_1 = np.array(p_val)
        for i in range(0,15):
            diffMAESValidation += abs(p_val_array_1[i] - yValidNew[i])

        res_validation_over_window['ValidationMAEError'].append(diffMAESValidation)
        sumValidationPredictionErrors += diffMAESValidation
        sumValidation_p_accrmse += np.sqrt(p_acc[1])
        sumValidation_p_acc1 += p_acc[1]        
        number = maxSlidingWindowNumberGlobal
        cmap = plt.get_cmap('gnuplot')
        colorsReal = [cmap(i) for i in np.linspace(0, 0.9, number)] 
        colorsPredicted = [cmap(i) for i in np.linspace(0.1, 0.99, number)] 
        
        plt.figure(10)
        plt.plot(yValidNew, color=colorsReal[slidingWindowNumber-1], label='Real')
        plt.plot(p_val, color=colorsPredicted[slidingWindowNumber-1], label='Predicted')
        red_patch = mpatches.Patch(color=colorsReal[0], label='Real')
        blue_patch = mpatches.Patch(color=colorsPredicted[0], label='Predicted')
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel('Time Tick Value')
        plt.ylabel('Time series Value')
        plt.title(fileNameModelToLoadFull + ':' + str(lengthOfTrainingWindow) + ':' + str(lengthOfSlidingWindow) + ':' + ("%.3E" % res_validation_over_window['avgVMAES']) + '\nAt hyperparameters: ' + 'Gamma-RBF:' + str(gamma_rbf) +' eps:' +  str(eps) + ' Lambda:' + str(Lambda));    
       
        
    res_validation_over_window['avgVMAES'] = sumValidationPredictionErrors / (slidingWindowNumber+1)
    print ('\nValidation: Continious Model Results, sliding over : ', 
           slidingWindowNumber, ' \nwindows with TP lenght:', lengthOfTrainingWindow, 
           '\n Average validation-predict MAE error of libsvm: ', res_validation_over_window['avgVMAES']
           )
    return [(sumValidationPredictionErrors / (slidingWindowNumber+1)), sumValidation_p_accrmse/((slidingWindowNumber+1)), sumValidation_p_acc1 / ((slidingWindowNumber+1)), listTimeTakenStringPerlevel]
#    plt.figure(); 
#    plt.title(fileNameModelToLoadFull + ':' + str(lengthOfTrainingWindow) + ':' + str(lengthOfSlidingWindow) + ':' + ("%.3E" % res_validation_over_window['avgVMAES']) + '\nAt hyperparameters: ' + 'Gamma-RBF:' + str(gamma_rbf) +' eps:' +  str(eps) + ' Lambda:' + str(Lambda));    
#    plt.xlabel('Offset');
#    plt.ylabel('Validation-RMSE-libsvm');
#    xSlidingOffset = res_validation_over_window['slidingOffset']
#    yValidationError = res_validation_over_window['ValidationMAEError']
#    plt.plot(xSlidingOffset, yValidationError, 'x')
    
def plotlambdaeps_1(RBFGamma, Lambda, res):
    plt.figure(); 
    plt.title('At hyperparameters: ' + 'Gamma-RBF:' + str(RBFGamma) + ' Lambda:' + str(Lambda))   
    plt.xlabel('Epsilon');
    plt.ylabel('MAE');
    x = res['lambdas']
    y = res['maes']
    plt.plot(x, y, 'x')
    
def plotlambdaeps_3(RBFGamma, eps, res):
    plt.figure(); 
    plt.title('At hyperparameters: ' + 'Gamma-RBF:' + str(RBFGamma) + ' Lambda:' + str(eps))    
    plt.xlabel('Lambda');
    plt.ylabel('V-Error, v-maes and v-rmse');
    x = res['lambdas']
    y_maes = res['v_maes']
    y_rmse = res['v_rmse']    
    plt.plot(x, y_maes, 'x')    
    plt.plot(x, y_rmse, 'x') 
    
def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

#svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
#
#svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
#svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
#ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
#xFull = get_numpy_x_from_svm(svm_x)
#xFull2 = np.asarray(xFull)
#xFull3 = xFull2[0:ValidStartIdx];
#yFull = np.asarray(svm_y)
#x = svm_x[0:ValidStartIdx];
#y = svm_y[0:ValidStartIdx];
#K = np.ndarray(shape=[len(xFull3),len(xFull3)])
#
#xValidation = xFull2[-lengthOfValidationSet:]
#yValidation = yFull[-lengthOfValidationSet:]


#Kvalid = calculate_Kline(xValid, x)
#def calculate_Kline(validationVectors, trainingVectors):
#    Kline = np.ndarray(shape=[len(validationVectors),len(trainingVectors)])
#    for idx1, item1 in enumerate(validationVectors):
#        for idx2, item2 in enumerate(trainingVectors):
#        Kline[idx1,idx2] = np.exp(-rbf_gamma * np.dot(item1-item2,item1-item2))
#
#def validate(validationVectors, validationGroundTruth, trainingVectors, beta0, alphas, gammas, Lambda):
#    K = calculate_Kline(validationVectors, trainingVectors)
#    forecasts = np.ndarray(len(validationVectors))
#    for i in np.arange(0,len(forecasts)):
#        forecasts[i] = beta0[this_iteration] + 1/Lambda * np.dot(K[i,:], alphas - gammas)
#    residuals = validationGroundTruth - forecasts
#    mrse = np.mean(np.sqrt(np.sum(residuals * residuals)))
#    mae = np.mean(np.abs(np.sum(residuals)))
#    return mae
#validate(xValid, yValid, xTrain, beta0[this_iteration], alphas[this_iteration],. gammas[this_iteration], Lambda[this_iteration])
#        
#print('Looking for previous Gram matrix...')
#K= pickle.load(open(gram_matrix_filename,'rb')) #gramMatrix
#K = gramMatrix
#print('Calculating Gram matrix...')
#for idx1, item1 in enumerate(xFull3):
#    for idx2, item2 in enumerate(xFull3):
#        K[idx1,idx2] = np.exp(-rbf_gamma * np.dot(item1-item2,item1-item2))
#print('Done calculating Gram matrix.')
#pickle.dump(K,open(gram_matrix_filename,'wb'),pickle.HIGHEST_PROTOCOL);
#print('Save calculated Gram matrix.')
##        
#for i in range (1):
#print('Testing combined svrpath..')
##print('\n#########         COMBINED ITERATION. ', i, ' Eps:', eps, '  Lambda:', Lambda)
#res = comb_epspath.combined_svrpath(x=xFull3, y=y, xValidation=xValidation, yValidation=yValidation, K=K, Lambda=Lambda, maxIterations = maxIterations, 
#    lambdamin=lambdamin, rho = 1e-8, RBFGamma = rbf_gamma, loadPreviousComputed = False, fileNameModelToLoadFull = fileNameModelToLoadFull);
#print('------',len(res['alphas']),'iterations')
#print('Time taken:',res['TimeTaken'])
#print('Filename of model:',fileNameModelToLoadFull)
#bestIteration = np.argmin(res['maes'])
#print('best iteration maes:',bestIteration,'out of',len(res['maes']))
#print('Output eps, eps run:',res['eps'][bestIteration])
#print('Output eps-lambda, eps run:',res['eps'][bestIteration],' Lambda:', Lambda, ' at iteration:', bestIteration)
#bestIteration_vRMSE = np.argmin(res['v_rmse'])
#print('Output eps, v-rmse-eps run:',res['eps'][bestIteration_vRMSE])
#bestIteration_vMAES = np.argmin(res['v_maes'])
#print('best iteration vmaes:',bestIteration_vMAES,'out of',len(res['v_maes']))
#print('Output eps, eps run:',res['eps'][bestIteration_vMAES])
#print('Support vectors(elbows) sizes left,right:',len(res['ElbowLeft'][bestIteration]),len(res['ElbowRight'][bestIteration]))
#print('Epsilon-insensitive region:',len(res['Center'][bestIteration]))
#print('Out-of-margin left,right:',len(res['LeftRegion'][bestIteration]),len(res['RightRegion'][bestIteration]))
#plotlambdaeps_1(rbf_gamma, Lambda, res)

##                                         
#print('Testing svrpath..')
#res_initial = svrpath_initialcopy.svrpath(xFull3, y, xValidation, yValidation, K, eps, RBF_Gamma = rbf_gamma, maxIterations = maxIterations, lambdamin = lambdamin, rho = 1e-8, loadPreviousComputed = False);
#print('------',len(res_initial['alphas']),'iterations,')
#bestIteration = np.argmin(res_initial['maes'])
#print('Output lambda:',res_initial['lambdas'][bestIteration])
#print('Output eps-lambda, Lambda run. Eps:',eps,'Lambda:', res_initial['lambdas'][bestIteration], ' at iteration:', bestIteration)
#print('Support vectors(elbows) sizes left,right:',len(res_initial['ElbowLeft'][bestIteration]),len(res_initial['ElbowRight'][bestIteration]))
##print('Epsilon-insensitive region:',len(res_initial['Center'][bestIteration]))
##print('Out-of-margin left,right:',len(res_initial['LeftRegion'][bestIteration]),len(res_initial['RightRegion'][bestIteration]))                                        
#Lambda = res_initial['lambdas'][bestIteration]
#bestIteration_vMAES = np.argmin(res_initial['v_maes'])
#print('Output lambda V-MAES / cost:',res_initial['lambdas'][bestIteration_vMAES], 1. / (res_initial['lambdas'][bestIteration_vMAES]))
#bestIteration_vRMSE = np.argmin(res_initial['v_rmse'])
#print('Output lambda V-RMSE / cost :',res_initial['lambdas'][bestIteration_vRMSE], 1. / (res_initial['lambdas'][bestIteration_vRMSE]))
#plotlambdaeps_3(rbf_gamma, eps, res_initial)

def create_exp_scaled_array(min, max, nItemsInGrid):
    base = np.exp(np.log(max/min)/nItemsInGrid);
    return min * np.power(base,np.arange(0,nItemsInGrid));

#TRAIN AND VALIDATE CALL
#trainandvalidate(gamma_rbf=1e-3, Lambda=2, eps=1e-6, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)
#trainandvalidate_continiousmodel(gamma_rbf=rbf_gamma, Lambda=Lambda, eps=eps, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)                                   


#listBestMAE = []
listBestOptimizedGrid = []
listInitGrid = []
params = pickle.load(open(dirNameModelToLoad + 'populatedRandomSearch.pickle','rb'))
listBestErrorDifferenceValidation = []
listEPSSVR_ValidationError = []
listNUSVR_ValidationError = []
maxMinAcross = []
differenceListGammas = []
listBestErrorValidationScaled = []
listTimeStringAllWithH1 = []
listTimeStringAllWithH0 = []

listSVR1_ValidationError = []
listSVR2_ValidationError = []
listSVR1_ValidationError_0 = []
listSVR2_ValidationError_0 = []
listSVR1_ValidationError_MAE = []
listSVR2_ValidationError_MAE = []

gamma_rbf_chosen = []
gamma_rbf_chosen.append(2.0374485156840983e-05) #model 0+
gamma_rbf_chosen.append(0.000159771752906)   #model 1+
gamma_rbf_chosen.append(0.000027422055154) #model 2+ 
gamma_rbf_chosen.append(0.000146892627764) #model 3+
gamma_rbf_chosen.append(0.000027478941531) #model 4+
gamma_rbf_chosen.append(1.5009557653946903e-05) #model 5+
gamma_rbf_chosen.append(0.002814491879807) #model 6+
gamma_rbf_chosen.append(0.000044442655434) #model 7+ 
gamma_rbf_chosen.append(0.000031915378551) #model 8+
gamma_rbf_chosen.append(0.000024779925708) #model 9+
gamma_rbf_chosen.append(2.1412371434196555e-05) #model 10+
gamma_rbf_chosen.append(2.1412371434196555e-05) #model 11+
gamma_rbf_chosen.append(2.199564204282985e-05)  #model 12+
gamma_rbf_chosen.append(0.000113370483810) #model 13
gamma_rbf_chosen.append(0.003914713648474) #model 14
gamma_rbf_chosen.append(2.4446215921711825e-05) #model 15

#                       
#eps_nusvr_chosen = []
#eps_nusvr_chosen.append(0.000378) #model 0
#eps_nusvr_chosen.append(0.051589) #model 1
#eps_nusvr_chosen.append(0.018241) #model 2
#eps_nusvr_chosen.append(0.004356) #model 3
#
#eps_nusvr_chosen.append(0.022390) #model 4
#eps_nusvr_chosen.append(0.015602) #model 5
#eps_nusvr_chosen.append(0.014542) #model 6
#eps_nusvr_chosen.append(0.030373) #model 7
#
#eps_nusvr_chosen.append(0.000073) #model 8
#eps_nusvr_chosen.append(0.055236) #model 9
#eps_nusvr_chosen.append(0.060461) #model 10
#eps_nusvr_chosen.append(0.066617) #model 11
#
#eps_nusvr_chosen.append(0.062800) #model 12
#eps_nusvr_chosen.append(0.063763) #model 13
#eps_nusvr_chosen.append(0.064431) #model 14
#eps_nusvr_chosen.append(0.040577) #model 15

#eps_nusvr_chosen = []
#eps_nusvr_chosen.append(0.000056) #model 0
#eps_nusvr_chosen.append(0.027021) #model 1x
#eps_nusvr_chosen.append(0.009346) #model 2x
#eps_nusvr_chosen.append(0.002249) #model 3x
#
#eps_nusvr_chosen.append(0.011440) #model 4x
#eps_nusvr_chosen.append(0.007949) #model 5x
#eps_nusvr_chosen.append(0.007525) #model 6x
#eps_nusvr_chosen.append(0.015730) #model 7x
#
#eps_nusvr_chosen.append(0.000027) #model 8x
#eps_nusvr_chosen.append(0.030648) #model 9x
#eps_nusvr_chosen.append(0.033450) #model 10x
#eps_nusvr_chosen.append(0.036904) #model 11x
#
#eps_nusvr_chosen.append(0.035107) #model 12x
#eps_nusvr_chosen.append(0.034502) #model 13x
#eps_nusvr_chosen.append(0.035571) #model 14x
#eps_nusvr_chosen.append(0.022124) #model 15x

eps_nusvr_chosen = []
eps_nusvr_chosen.append(0.000098) #model 0y1
eps_nusvr_chosen.append(0.019067) #model 1y1
eps_nusvr_chosen.append(0.006584) #model 2y1
eps_nusvr_chosen.append(0.001542) #model 3y1

eps_nusvr_chosen.append(0.007942) #model 4y
eps_nusvr_chosen.append(0.005528) #model 5y
eps_nusvr_chosen.append(0.005330) #model 6y
eps_nusvr_chosen.append(0.011088) #model 7y

eps_nusvr_chosen.append(0.000015) #model 8y
eps_nusvr_chosen.append(0.016192) #model 9y
eps_nusvr_chosen.append(0.006752) #model 10y
eps_nusvr_chosen.append(0.019524) #model 11y

eps_nusvr_chosen.append(0.013784) #model 12y
eps_nusvr_chosen.append(0.019844) #model 13y
eps_nusvr_chosen.append(0.019414) #model 14y
eps_nusvr_chosen.append(0.013267) #model 15y

cost_ch_chosen = []
#cost_ch_chosen.append(907.820530178198851) #model 0y1
#cost_ch_chosen.append(907.820530178198851) #model 0y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
#cost_ch_chosen.append(571.478636671874710) #model 1y1
nu_svr_eps_selected = np.zeros((16,maxSlidingWindowNumberGlobal))

for iterationLevel in range (1,16):

    fileModelIterationName = str(iterationLevel)
    fileNameModelToLoad = "scaledmodel" + fileModelIterationName + ".close" + ".txt"
    fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
#    gram_matrix_filename = fileNameModelToLoadFull + '.gram_matrix' + '.pickle'
#    
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
#
#    svm_x = svm_x_init[slidingOffsetGlobal:slidingOffsetGlobal+lengthOfSlidingWindow*maxSlidingWindowNumberGlobal+lengthOfValidationSet]
#    svm_y = svm_y_init[slidingOffsetGlobal:slidingOffsetGlobal+lengthOfSlidingWindow*maxSlidingWindowNumberGlobal+lengthOfValidationSet]
#    ValidStartIdx = int(len(svm_x) - lengthOfValidationSet)
##    fileNameModelToSaveSlice = "model" + fileModelIterationName + ".close" + ".txt"
##    with open('/home/peters/Work/Tasks/apr-28-stft/datasets3/' + fileNameModelToSaveSlice,'wt') as f:
##        for rowidx in range(0,6515):
##            f.write(str(svm_y_init[rowidx]) + ' ')
##            for xidx in range(0,len(svm_x_init[rowidx])):
##                f.write(str(xidx) +':' + str(svm_x_init[rowidx][xidx]) + ' ')
##            f.write('\n')            
#    xFull = get_numpy_x_from_svm(svm_x)
#    xFull2 = np.asarray(xFull)
#    xFull3 = xFull2[0:ValidStartIdx]
#    yFull = np.asarray(svm_y)
#    x = svm_x[0:ValidStartIdx];
#    y = svm_y[0:ValidStartIdx];
#    K = np.ndarray(shape=[len(xFull3),len(xFull3)])
##    print ('Length:', len(svm_y_init))
#    xValidation = xFull2[-lengthOfValidationSet:]
#    yValidation = yFull[-lengthOfValidationSet:]
# 
    model = np.asarray(params['model'+str(iterationLevel)])
    
    print("Associated file below is: ", fileNameModelToLoadFull)    
    bestval = min(model[:,5])
    model_bests = model[model[:,5] <= bestval]   
    gamma_rbf_best = model_bests[0][0]
    Lambda_best = 1/(model_bests[0][2])
    eps_best = model_bests[0][1]

#    eps_best = 0.000965032927880
#    gamma_rbf_best = 0.000020000000000
#    Lambda_best = 1/571.478636671874710
    [errorValidation_eps_svr, errorValidation_eps_svr_libsvm_rmse,
             errorValidation_eps_svr_libsvmmse1, listTimeStringPerLevel] = trainandvalidate_continiousmodel(
            gamma_rbf = gamma_rbf_best, Lambda = Lambda_best, 
            eps = eps_best, fileNameModelToLoadFull = fileNameModelToLoadFull, 
            lengthOfValidationSet = lengthOfValidationSet,  train_eps_svr= True, Level = iterationLevel)
#    listTimeStringAllWithH1.append(listTimeStringPerLevel)
#    [errorValidation_eps_svr_eps_2,  errorValidation_eps_svr_libsvmmse_nueps_0, 
#         errorValidation_eps_svr_libsvmmse_nueps_1, 
#         listTimeStringPerLevel] = trainandvalidate_continiousmodel(gamma_rbf = gamma_rbf_chosen[iterationLevel], 
#         Lambda = Lambda_best, 
#         eps = eps_nusvr_chosen[iterationLevel], fileNameModelToLoadFull = fileNameModelToLoadFull, 
#         lengthOfValidationSet = lengthOfValidationSet,  train_eps_svr= True, Level = iterationLevel)
#    listTimeStringAllWithH0.append(listTimeStringPerLevel)
#    differenceError = errorValidation_eps_svr_libsvmmse1 - errorValidation_eps_svr_libsvmmse_nueps_1
#    print('\ndifferenceError eps-H1 vs eps-H0:', differenceError,'\n')
#    listBestErrorDifferenceValidation.append(differenceError)
    listSVR1_ValidationError.append(errorValidation_eps_svr_libsvmmse1)
#    listSVR2_ValidationError.append(errorValidation_eps_svr_libsvmmse_nueps_1)
    listSVR1_ValidationError_MAE.append(errorValidation_eps_svr)
#    listSVR2_ValidationError_MAE.append(errorValidation_eps_svr_eps_2)
    rmse = errorValidation_eps_svr_libsvm_rmse
    maxY = max(svm_y_init)
    minY = min(svm_y_init)
    scale = maxY-minY
    rmse_scaled = rmse / scale
    listSVR1_ValidationError_0.append(rmse)
    listSVR2_ValidationError_0.append(rmse_scaled)
    
#    [errorValidation_eps_svr_eps_2,  
#     errorValidation_eps_svr_libsvmmse_nueps_0, 
#     errorValidation_eps_svr_libsvmmse_nueps_1, 
#     listTimeStringPerLevel] = trainandvalidate_continiousmodel(gamma_rbf = gamma_rbf_best, 
#                                 Lambda = Lambda_best, eps = eps_best, 
#                                 fileNameModelToLoadFull = fileNameModelToLoadFull, 
#                              lengthOfValidationSet = lengthOfValidationSet, train_eps_svr= False, Level = iterationLevel)    
#
###    modelSlice = model[model[:,0] <= 10*gamma_rbf_derived_script[iterationLevel]]
###    bestvalSlice = min(modelSlice[:,5])
###    model_bests_slice = modelSlice[modelSlice[:,5] <= bestval]
####    gamma_rbf_best_slice = model_bests_slice[0][0]
###    Lambda_best_slice = 1/(model_bests_slice[0][2])
###    eps_best_slice = model_bests_slice[0][1]
###    errorValidation_gamma_est = trainandvalidate_continiousmodel(gamma_rbf = gamma_rbf_chosen[iterationLevel], Lambda = Lambda_best, eps = eps_best, fileNameModelToLoadFull = fileNameModelToLoadFull, lengthOfValidationSet = lengthOfValidationSet)
##
#    errorValidation_nu_svr = trainandvalidate_continiousmodel(gamma_rbf = gamma_rbf_chosen[iterationLevel], 
#                             Lambda = Lambda_best, eps = eps_best, 
#                             fileNameModelToLoadFull = fileNameModelToLoadFull, 
#                             lengthOfValidationSet = lengthOfValidationSet,  train_eps_svr= False, Level = iterationLevel )

#    differenceError = errorValidation_eps_svr - errorValidation_nu_svr
#    print('\ndifferenceError EPS-SVR-vs-NU-SVR is:', differenceError,'\n\')
#    differencegamma = gamma_rbf_chosen[iterationLevel] - gamma_rbf_best
#    differenceListGammas.append(differencegamma)
#    listBestErrorDifferenceValidation.append(differenceError)
#    listNUSVR_ValidationError.append(errorValidation_nu_svr)
#    listEPSSVR_ValidationError.append(errorValidation_eps_svr)
#    
#    minValueForModel = min(y)
#    maxValueForModel = max(y)
#    maxMinAcross.append([minValueForModel, maxValueForModel])
#    differenceModel = maxValueForModel - minValueForModel
#    scaledError = errorValidation / differenceModel
#    listBestErrorValidationScaled.append(scaledError)    
#    print(model_bests)
#       
#    listBestOptimizedGrid.append(returnGridOptimized)