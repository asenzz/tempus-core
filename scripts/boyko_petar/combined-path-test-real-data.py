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
#os.chdir('/home/peters/Work/libsvm-3.22/python/')
import numpy as np
import svmutil
os.chdir('/home/peters/Work/Tasks/FindOptimalParameters/svrpath_research/combined-path-python/')
import comb_epspath
#import lambda_path
#import svrpath_progresscopy
import svrpath_initialcopy
#import lambda_path.py
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dirNameModelToLoad = "/home/peters/Work/Tasks/apr-28-stft/datasets/"
fileModelIterationName = '4'
fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad

gram_matrix_filename = fileNameModelToLoadFull + '.gram_matrix' + '.pickle'

#sizeOfValidationSet = 0.005;
lengthOfValidationSet = 15
lengthOfSlidingWindow = 1440
lengthOfTrainingWindow = 6500  + lengthOfValidationSet  #lengthOfValidationSet is included 
maxSlidingWindowNumber = 4
slidingWindowNumber = 0
slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
#Eps-Lambda-MAES value after running Lambda-2D-path iteration.  Eps: 1e-07  Lambda: 1.292379238 MAES: 0.00160435064098
#4.77859253e-03   3.26783410e-10   9.94122721e+00
rbf_gamma = 0.000105140779396
eps = 0.001456180154023
Cost = 24.165267810050079
Lambda = 1/Cost
#Lambda = 9.96480589118e-07
#0.000600195051263
maxIterations = 65000
lambdamin = 1E-6
rho = 1e-11


## diapason for values
## C, epsilon, kernel_param, kernel_param2
#low = (1, 1e-6, 0.0000105140779396, 1)
#up =  (50, 1e-9, 0.00105140779396, 1000)
# uncoditionally multiply steps count every recursion
Default_step_degree = 0.8

# number of steps per training a range of parameter values
Default_num_steps = 5
Default_num_cycles = 2
# uncoditionally multiply steps count every recursion
Default_step_coef = 0.25

# default data chunk lenght in number of instances
#Default_chunk_len = 128

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
        visualizeresiduals(yValidNew, p_val)
                                                  
def visualizeresiduals(yValid_GroundTruth, p_val):
    plt.figure(201)
    plt.plot(yValid_GroundTruth, 'r')
    plt.plot(p_val, 'b')    
    

def trainandvalidate_continiousmodel(gamma_rbf, Lambda, eps, fileNameModelToLoadFull, lengthOfValidationSet):
    res_validation_over_window = {'gamma_rbf' : gamma_rbf, 'Lambda' : Lambda, 'eps' : eps, 
                                  'lengthOfValidationSet' : lengthOfValidationSet, 
                                  'lengthOfTrainingWindow' : lengthOfTrainingWindow, 'lengthOfSlidingWindow' : lengthOfSlidingWindow,
                                  'slidingOffset' : [], 'ValidationMAEError' : [], 'avgVMAES' : 0.}
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    slidingWindowNumber = 0
    sumValidationPredictionErrors = 0
    slidingOffset = slidingWindowNumber * lengthOfSlidingWindow
    while (lengthOfTrainingWindow + slidingOffset < len(svm_y_init)) and (slidingWindowNumber < maxSlidingWindowNumber):
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
        diffMAESValidation = 0
        p_val_array_1 = np.array(p_val)
        for i in range(0,15):
            diffMAESValidation += abs(p_val_array_1[i] - yValidNew[i])

        res_validation_over_window['ValidationMAEError'].append(diffMAESValidation)
        sumValidationPredictionErrors += diffMAESValidation
        plt.figure(10)
        plt.plot(yValidNew, 'r', label='Real')
        plt.plot(p_val, 'b', label='Predicted')
        red_patch = mpatches.Patch(color='red', label='Real')
        blue_patch = mpatches.Patch(color='blue', label='Predicted')
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel('Time Tick Value')
        plt.ylabel('Time series Value')
        plt.title(fileNameModelToLoadFull + ':' + str(lengthOfTrainingWindow) + ':' + str(lengthOfSlidingWindow) + ':' + ("%.3E" % res_validation_over_window['avgVMAES']) + '\nAt hyperparameters: ' + 'Gamma-RBF:' + str(gamma_rbf) +' eps:' +  str(eps) + ' Lambda:' + str(Lambda));    
       
        
    res_validation_over_window['avgVMAES'] = sumValidationPredictionErrors / (slidingWindowNumber+1)
    print ('\nValidation: Continious Model Results, sliding over : ', 
           slidingWindowNumber, ' \nwindows with TP lenght:', lengthOfTrainingWindow, 
           '\n Average validation-predict MAE error of libsvm: ', res_validation_over_window['avgVMAES']
           )
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

svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)

svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
ValidStartIdx = int(len(svm_x) - lengthOfValidationSet);
xFull = get_numpy_x_from_svm(svm_x)
xFull2 = np.asarray(xFull)
xFull3 = xFull2[0:ValidStartIdx];
yFull = np.asarray(svm_y)
x = svm_x[0:ValidStartIdx];
y = svm_y[0:ValidStartIdx];
K = np.ndarray(shape=[len(xFull3),len(xFull3)])

xValidation = xFull2[-lengthOfValidationSet:]
yValidation = yFull[-lengthOfValidationSet:]

with open(dirNameModelToLoad + 'results_2xB.txt','wt') as f:
    iterationAlongColumn = len(x[0])
    iterationAlongRow = len(x)    
    for xidRowx in range(0,iterationAlongRow):
        for xidColx in range(0,iterationAlongColumn):        
            f.write(str(x[xidRowx][xidColx]) + ' ')
        f.write('\n')

#print('Hello world..')


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
print('Testing svrpath..')
res_initial = svrpath_initialcopy.svrpath(xFull3, y, xValidation, yValidation, K, eps, RBF_Gamma = rbf_gamma, maxIterations = maxIterations, lambdamin = lambdamin, rho = 1e-8, loadPreviousComputed = False);
print('------',len(res_initial['alphas']),'iterations,')
bestIteration = np.argmin(res_initial['maes'])
print('Output lambda:',res_initial['lambdas'][bestIteration])
print('Output eps-lambda, Lambda run. Eps:',eps,'Lambda:', res_initial['lambdas'][bestIteration], ' at iteration:', bestIteration)
print('Support vectors(elbows) sizes left,right:',len(res_initial['ElbowLeft'][bestIteration]),len(res_initial['ElbowRight'][bestIteration]))
#print('Epsilon-insensitive region:',len(res_initial['Center'][bestIteration]))
#print('Out-of-margin left,right:',len(res_initial['LeftRegion'][bestIteration]),len(res_initial['RightRegion'][bestIteration]))                                        
Lambda = res_initial['lambdas'][bestIteration]
bestIteration_vMAES = np.argmin(res_initial['v_maes'])
print('Output lambda V-MAES / cost:',res_initial['lambdas'][bestIteration_vMAES], 1. / (res_initial['lambdas'][bestIteration_vMAES]))
bestIteration_vRMSE = np.argmin(res_initial['v_rmse'])
print('Output lambda V-RMSE / cost :',res_initial['lambdas'][bestIteration_vRMSE], 1. / (res_initial['lambdas'][bestIteration_vRMSE]))
plotlambdaeps_3(rbf_gamma, eps, res_initial)

def create_exp_scaled_array(min, max, nItemsInGrid):
    base = np.exp(np.log(max/min)/nItemsInGrid);
    return min * np.power(base,np.arange(0,nItemsInGrid));

def trainrep(    
    x, y,
    _svr_params,    # list of svr parameters: C, epsilon, kernel_param, kernel_param2
    _begin,
    _end,
    _num_steps,
    _prev_best_mse=0,
    _train_param=None,
    ):

    # stop optimizing when number of sub steps is less than one
    if _num_steps < 1: return _prev_best_mse
    print('run')
    # # some sanity checking
    if _end < low[_train_param]:
        _end == low[_train_param]
    if _begin < low[_train_param]:
        _begin = low[_train_param]


    # absolute counter of svm_train calls
    global train_cycles
    # # put a dot on show and flush immediately
    # sys.stdout.write(".")
    # sys.stdout.flush()

    # if previous MSE is 0 do one train and get mean squared error
    if _prev_best_mse == 0:
        _prev_best_mse = svr_train_here_p(_svr_params, x, y)
#        LOG("initial " + str(_svr_params))
####        _prev_best_mse = svr_train(_svr_params)
        #_prev_best_mse = svr_train(_svr_params)
       # list of svr parameters: C, epsilon, kernel_param (gamma-rbf), kernel_param2
   
#        LOG("mse = " + str(_prev_best_mse))

    # save initial values
    best_mse = _prev_best_mse
    best_parval = _svr_params[_train_param]
#    step_size = float(_end - _begin) / _num_steps
    logScaledCoefficients = create_exp_scaled_array( _begin, _end, _num_steps);
    
    # number of steps to be used for self calls
    num_sub_steps = _num_steps * Default_step_coef

#    LOG(str(num_sub_steps) + " in (" + str(_begin) + ", " + str(_end) + ")")

    # do a grid search in the range starting from _being to _end in _num_steps count
#    for parval in np.linspace(_begin, _end, _num_steps):
    for parval in logScaledCoefficients:
        # bump up counter
        train_cycles += 1

        # set optimized parameter to value step
        _svr_params[_train_param] = parval

#        LOG(str(train_cycles) + " " + str(_svr_params))

        # get MSE for current parameters
        mse = svr_train_here_p(_svr_params, x, y)
#        LOG("mse = " + str(mse))

        # see if we got something better and save it for return
        # if we got zero MSE then break out of the loop immediately
        if best_mse > mse:
            best_mse = mse
            best_parval = _svr_params[_train_param]
#            LOG("changing best_svr_params " + str(_svr_params))
            if mse == 0: break
        
        step_size_logarithmic = np.log(parval) / np.log(10)
        # call self with current best MSE and same optimized parameter as inputs
        # range is of tweaking is set to the begin of previous step and end on the next one.
        # number of steps to be used set by multiplying ratio of this MSE to current best MSE
        mse = trainrep(
            x, y,
            _svr_params,
            _begin=parval - step_size_logarithmic,
            _end=parval + step_size_logarithmic,
            _num_steps=num_sub_steps * ((best_mse / mse) ** Default_step_degree),
            _prev_best_mse=best_mse,
            _train_param=_train_param)

    # set best parameter value and best MSE
    _svr_params[_train_param] = best_parval
    return best_mse

#    svr_param = [10, 0.01, 17, 30]
## list of svr parameters: C, epsilon, kernel_param, kernel_param2
#svr_param = [10, 1e-4, 0.01, 30]
#svr_param = [24.165267810050079, 0.001456180154023,0.000105140779396, 0.108544]
def optimize_svr_params(
    x, y,
    _svr_param, 				# tweak this svm_parameter object or create new one
    _num_cycles, 			# number of repetitions to do on all tweaked parameters
    _num_steps,# starting number of steps per training cycle, decreases by __steps_coef every level
    low, up
    ):
    print("\n\n\nOptimized to " + str(_svr_param))
    print(_svr_param)
    # check input sanity
    if _num_steps < 1 or _num_cycles < 1: return -1

    # init locals and include global variable train_cycles for updating and print
    global train_cycles
    mse = 0
    train_cycles = 0

    # # set epsilon based on LibSVM user guide recommendation and a bit lower
    # # WARNING: going further will probably make training take eternally
    # if _param.svm_type in [NU_SVC, NU_SVR]: _param.eps = 1e-10
    # else: _param.eps = 1e-5

    # start num_cycles optimizations of all used parameters
    # for the specified SVM type and kernel
    for ix in range(_num_cycles):
        print ("Optimization run",ix+1)

        print("Optimizing parameter C ")
        mse = trainrep(x, y, _svr_param, low[0], up[0], _num_steps, mse, 0 )
        print("\n\n\nOptimized to " + str(_svr_param))

        print("Optimizing parameter Epsilon")
        mse = trainrep(x, y,_svr_param, low[0], up[1], _num_steps, mse, 1 )
        print("\n\n\nOptimized to " + str(_svr_param))

        print("Optimizing parameter kernel_param ")
        mse = trainrep(x, y,_svr_param, low[2], up[2], _num_steps, mse, 2  )
        print("\n\n\nOptimized to " + str(_svr_param))

#        print("Optimizing parameter kernel_param2 ")
#        mse = trainrep(x, y, _svr_param, low[3], up[3], _num_steps, mse, 3)
#        print("\n\n\nOptimized to " + str(_svr_param))
#
#
#
    # # set epsilon based on LibSVM user guide recommendation and a bit lower
    # # WARNING: going further will probably make training take eternal time
    # if _param.svm_type in [NU_SVC, NU_SVR]: _param.eps = 1e-12
    # else: _param.eps = 1e-7


    # Print results
    print ("Parameter optimization done in ",train_cycles," train cycles. ")
    print ("Mean squared error ",mse)
    print ("SVM parameters follow (C, epsilon, kernel_param, kernel_param2) :")
    print (_svr_param)
    print ('\n')
    return _svr_param                               


#optimize_svr_params(
#    x, y, 
#    _svr_param = svr_param, 				# tweak this svm_parameter object or create new one
#    _num_cycles=Default_num_cycles, 			# number of repetitions to do on all tweaked parameters
#    _num_steps=Default_num_steps	# starting number of steps per training cycle, decreases by __steps_coef every level
#    )
#def ranAllModels():
#    dirNameModelToLoad = "/home/peters/Work/Tasks/12apr-2017-stfl-3weeks/"
#    for i in range(9,16):
#        fileModelIterationName = str(i)
#        fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
#        fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
#        svm_filename = fileNameModelToLoadFull
#        print ('filename : ' , svm_filename)
##        plt.figure()
#        ans = trainandvalidate_continiousmodel(gamma_rbf=rbf_gamma, Lambda=Lambda, eps=eps, fileNameModelToLoadFull=svm_filename, lengthOfValidationSet=lengthOfValidationSet)
#        print('RBF-gamma estimated for level ' + str(i) + ' :' + str(ans) + '\n')
#TRAIN AND VALIDATE CALL
#trainandvalidate(gamma_rbf=1e-3, Lambda=2, eps=1e-6, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)
#trainandvalidate_continiousmodel
#trainandvalidate_continiousmodel(gamma_rbf=rbf_gamma, Lambda=Lambda, eps=eps, fileNameModelToLoadFull=fileNameModelToLoadFull, lengthOfValidationSet=lengthOfValidationSet)                                   


#listBestMAE = []
listBestOptimizedGrid = []
listInitGrid = []
params = pickle.load(open(dirNameModelToLoad + 'populatedRandomSearch.pickle','rb'))

for iterationLevel in range (0,16):
    
    fileModelIterationName = str(iterationLevel)
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

#    xValidation = xFull2[-lengthOfValidationSet:]
#    yValidation = yFull[-lengthOfValidationSet:]
#    svr_param = [24.165267810050079, 0.001456180154023,0.000105140779396, 0.108544]

# diapason for values
# C, epsilon, kernel_param, kernel_param2
#    low = (1, 1e-6, 0.0000105140779396, 1)
#    up =  (50, 1e-9, 0.00105140779396, 1000)
    
    model = np.asarray(params['model'+str(iterationLevel)])
    bestval = min(model[:,5])
    model_bests = model[model[:,5] <= bestval]
    print(model_bests)

    svr_param = [model_bests[0][2],model_bests[0][1], model_bests[0][0],0.108544]
    low = (1, 1e-6, 0.0000105140779396, 1)
    up =  (50, 1e-9, 0.00105140779396, 1000)
    
    low = (svr_param[0] / 100, svr_param[1] / 100, svr_param[2] / 100)
    up = (svr_param[0] * 100, svr_param[1] * 100, svr_param[2] * 100)
       
    listInitGrid.append(svr_param)
    returnGridOptimized = optimize_svr_params(
    x, y, 
    _svr_param = svr_param, 				# tweak this svm_parameter object or create new one
    _num_cycles=Default_num_cycles, 			# number of repetitions to do on all tweaked parameters
    _num_steps=Default_num_steps,	# starting number of steps per training cycle, decreases by __steps_coef every level
    low = low, up = up
    )
    listBestOptimizedGrid.append(returnGridOptimized)