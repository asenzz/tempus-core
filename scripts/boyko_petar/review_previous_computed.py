#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:57:10 2017

@author: peters
"""
import numpy as np
from six.moves import cPickle as pickle


resultsMatrixToLoad = pickle.load(open('results_ongoing_at10k.pickle','rb'))
alphasAll = resultsMatrixToLoad['alphas']
numRows = len(alphasAll)
numRows = numRows - 1
    #pick last row; not the lowest RMSE. 
    #TODO pick another row
    #TODO: eps not loaded?? 
alphas = alphasAll[numRows]
eps =  resultsMatrixToLoad['eps'][numRows]
ElbowLeft =  resultsMatrixToLoad['ElbowLeft'][numRows]
ElbowRight = resultsMatrixToLoad['ElbowRight'][numRows]
Center =  resultsMatrixToLoad['Center'][numRows]
LeftRegion =  resultsMatrixToLoad['LeftRegion'][numRows]
RightRegion = resultsMatrixToLoad['RightRegion'][numRows]
gammas =  resultsMatrixToLoad['gammas'][numRows]
beta0 =  resultsMatrixToLoad['beta0'][numRows]
mae = resultsMatrixToLoad['maes'][numRows]       

#results_ongoing_at10k.pickle
#Kstar = np.zeros(shape=[1,1], dtype=globaltype)
#Kstar = updateKstar(K, Kstar, Lambda, [], [], ElbowRight, ElbowLeft);
#f0 = np.repeat(beta0,len(K))


#threshold for alphas/gammas: 
threshold_alphas = 0.0001
if ((gammas[ElbowLeft] < threshold_alphas).any() or (alphas[ElbowRight] < threshold_alphas).any()):
    print('Some points have lower value than the threshold')
else:
    print('All values at the elbow poitns have higher alphas/gammas higher than the threshold')    