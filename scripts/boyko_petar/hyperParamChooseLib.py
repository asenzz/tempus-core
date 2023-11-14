#!/usr/bin/env python
import numpy as np
import time
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import griddata
import functools;
from scipy.optimize import *
import scipy
import time;
from functools import partial
import scipy.optimize
import os
import sys

def generateGramMatrix(x, gamma):
    K = np.ndarray(shape=[len(x),len(x)])
    for idx1, item1 in enumerate(x):
        for idx2, item2 in enumerate(x):
            K[idx1,idx2] = np.exp(-gamma * np.dot(item1-item2,item1-item2))
    return K
    
def trainpredict(y, x, yValid, xValid, gamma, eps, cost):
    base_svm_options = '-s 3 -t 2';
    gamma_opt = '-g ' + str(gamma);
    eps_opt = '-p ' + str(eps);
    cost_opt = '-c ' + str(cost);
    svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
    print(svm_params);
    os.chdir('/home/boyko/misc-projects/libsvm/python')
    import svmutil
    m = svmutil.svm_train(y, x, svm_params);
    _label, p_acc, p_val = svmutil.svm_predict(yValid, xValid, m);
    vec = [gamma,eps,cost,p_acc[1],p_acc[2]];
    return vec;

def generateHyperParameters(basedir, eps=1e-11, column='close', levels=np.arange(0,13), sizeOfValidationSet = 15):
    os.chdir('/home/boyko/work-projects/gamma-estimation')
    import findGamma
    os.chdir('/home/boyko/work-projects/svrpath-python')
    import svrpath
    hyperParams = {}
    for l in levels:
        modelname = 'model'+str(l)
        modelfile = basedir+'/'+modelname+'.'+column+'.txt'
        hyperParams.update({ modelname : { 'modelfile' : modelfile,'results': [float('Inf'),float('Inf'),float('Inf'),float('Inf'),float('Inf')]} })
    # Check for pregenerated parameters
    resultfilename = basedir + '/' + 'hyperParameters.pickle'
    try:
        hyperParams = pickle.load(open(resultfilename,'rb'))
    except FileNotFoundError:
        pass;
    for model in hyperParams:
        print(model)
        modelDictItem = hyperParams[model]
        resVector = modelDictItem['results']
        # First, eval gamma if needed (findGamma)
        gamma = resVector[0]
        anyParamChanged = False;
        gammaChanged = False;
        if gamma == float('Inf'):
           print('Gamma..',end='')
           sys.stdout.flush()
           gammaChanged = True
           anyParamChanged = True
           gamma = findGamma.findGamma(modelDictItem['modelfile']) 
           print(gamma)
           resVector[0] = gamma
           modelDictItem.update({'results' : resVector});
           hyperParams.update({model : modelDictItem})
           pickle.dump(hyperParams,open(resultfilename,'wb'))
        # Then, eval cost if needed (svrpath)
        cost = resVector[2]
        costError = False;
        if cost == float('Inf') or gammaChanged or eps != resVector[1]:
            print('Cost..',end='')
            sys.stdout.flush()
            anyParamChanged = True
            os.chdir('/home/boyko/misc-projects/libsvm/python')
            import svmutil
            y, svm_x = svmutil.svm_read_problem(modelDictItem['modelfile']);
            x = findGamma.get_numpy_x_from_svm(svm_x);
            # generate Gram matrix
            K = generateGramMatrix(x, gamma)
            try:
                res = svrpath.svrpath(x=x,y=y,K=K,eps=eps,maxIterations=1000000, lambdamin=1e-10, rho=1e-15)
                print('Time taken:',res['TimeTaken'])
                bestIteration = np.argmin(res['maes'])
                print('best iteration:',bestIteration,'out of',len(res['maes']))
                print('Output lambda:',res['lambdas'][bestIteration])
                cost = 1.0 / res['lambdas'][bestIteration];
                print(cost)
                
                resVector[2] = cost
                resVector[1] = eps
                
            except Exception as e:
                print(e);
                resVector[2] = float('-Inf')
                resVector[1] = eps
                costError = True
            modelDictItem.update({'results' : resVector})
            hyperParams.update({model : modelDictItem})
            pickle.dump(hyperParams,open(resultfilename,'wb'))
            
        # finally, do training and validation (libsvm) if needed
        if (anyParamChanged or resVector[3] == float('Inf')) and not costError and cost != float('-Inf') and cost != float('Inf'):
            print('Validation..',end='')
            sys.stdout.flush()
            try:
                ValidStartIdx = int(len(x) - sizeOfValidationSet);
            except UnboundLocalError:
                y, svm_x = svm_read_problem(modelDictItem['modelfile']);
                x = findGamma.get_numpy_x_from_svm(svm_x);
                ValidStartIdx = int(len(x) - sizeOfValidationSet);
            #svm_x = get_numpy_x_from_svm(svm_x);
            yValid = y[ValidStartIdx:];
            xValid = svm_x[ValidStartIdx:];
            y_train = y[0:ValidStartIdx];
            x_train = svm_x[0:ValidStartIdx];

            res = trainpredict(y=y_train, x=x_train, yValid=yValid, xValid=xValid, gamma=gamma, eps=eps, cost=cost);
            resVector = res
            print(resVector)
            modelDictItem.update({'results' : resVector})
            hyperParams.update({model : modelDictItem})
            pickle.dump(hyperParams,open(resultfilename,'wb'))