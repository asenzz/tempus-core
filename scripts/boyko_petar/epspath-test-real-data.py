#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:42:44 2017

@author: boyko
"""
import os
os.chdir('/home/boyko/misc-projects/libsvm/python')
import numpy as np
import svmutil
os.chdir('/home/boyko/work-projects/epspath-python/')
import epspath

def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

y, svm_x = svmutil.svm_read_problem('/home/boyko/datasets/sliding-datasets/1/model1.close.txt')
x = get_numpy_x_from_svm(svm_x)
y = np.asarray(y)
K = np.ndarray(shape=[len(x),len(x)])
gamma = 1e-3
#eps = 1e-9
Lambda = 1e-5
maxIterations = 10000
lambdamin = 1e-10
rho = 1e-15
print('Calculating Gram matrix...')
for idx1, item1 in enumerate(x):
    for idx2, item2 in enumerate(x):
        K[idx1,idx2] = np.exp(-gamma * np.dot(item1-item2,item1-item2))
print('Done calculating Gram matrix.')
        
print('Testing svrpath..')
res = epspath.svrpath(x=x, y=y, K=K, Lambda=Lambda, maxIterations = 500000, lambdamin=lambdamin, rho = 1e-12, RBFGamma = gamma);
print('------',len(res['alphas']),'iterations')
print('Time taken:',res['TimeTaken'])
bestIteration = np.argmin(res['maes'])
print('best iteration:',bestIteration,'out of',len(res['maes']))
print('Output eps:',res['eps'][bestIteration])
print('Support vectors(elbows) sizes left,right:',len(res['ElbowLeft'][bestIteration]),len(res['ElbowRight'][bestIteration]))
print('Epsilon-insensitive region:',len(res['Center'][bestIteration]))
print('Out-of-margin left,right:',len(res['LeftRegion'][bestIteration]),len(res['RightRegion'][bestIteration]))