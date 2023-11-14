#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:33:40 2017

@author: peters
"""

print('Hello world..')
import os
os.chdir('/home/peters/Work/Tasks/FindOptimalParameters/svrpath_research/combined-path-python/')
import numpy as np
from six.moves import cPickle as pickle
from decimal import Decimal
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

dirNameModelToLoad = "/home/peters/Work/Tasks/may-9-stft/1000models/"
fileNamePickleToLoad = dirNameModelToLoad + 'populatedRandomSearch.pickle'

paramPickle = pickle.load(open(fileNamePickleToLoad,'rb'))
model = np.asarray(paramPickle['model'+str(0)])
print('Lenght of data is: ' + str(len(paramPickle)))
print('Lenght of data in array is: ' + str(len(model)))



params = pickle.load(open(fileNamePickleToLoad,'rb'));
levels = 16
outputs = np.zeros(shape=[levels, 3, 3])
for modelIdx in np.arange(0,levels):
    model = np.asarray(params['model'+str(modelIdx)])
    bestval = min(model[:,3])
    model_bests = model[model[:,3] <= bestval]
    print("for model: " + str(modelIdx))
    print("%e" % bestval)
    #("%.3E" % res_
    #("%.3E" % res_validation_over_window['avgVMAES'])
    #print(model_bests)