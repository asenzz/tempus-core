#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:33:59 2017

@author: peters
"""
print('Hello world..')
import sys
import os.path
#sys.path.insert(0, '/home/peters/Work/libsvm-3.22/python/')
sys.path.insert(0, '/home/peters/Work/gitfolder/libsvm-mod/libsvm-modified/python/')
import os
#os.chdir('/home/peters/Work/libsvm-3.22/python/')
#import numpy as np
import svmutil
os.chdir('/home/peters/Work/Tasks/FindOptimalParameters/svrpath_research/combined-path-python/')

#import comb_epspath
#import lambda_path
#import svrpath_progresscopy
#import svrpath_initialcopy
#import lambda_path.py
#from six.moves import cPickle as pickle
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#/home/peters/Work/Tasks/apr21-2017-stft-2/stft_data_100k/decon_queue
#"/home/peters/Work/Tasks/apr-28-stft/datasets/"
dirNameModelToLoad = "/home/peters/Work/Tasks/apr-28-stft/datasets3/"



def trainlibmsvm_here_svr(gamma_rbf, Lambda, eps, svm_x, svm_y, lengthOfValidationSet):
        svm_y_train = svm_y[:-lengthOfValidationSet];
        svm_x_train = svm_x[:-lengthOfValidationSet];
        cost = 1/Lambda
        base_svm_options = '-s 3 -t 2';
        gamma_opt = '-g ' + str(gamma_rbf);
        eps_opt = '-p ' + str(eps);
        cost_opt = '-c ' + str(cost);

        svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
        m = svmutil.svm_train(svm_y_train, svm_x_train, svm_params);
        return m

for iterationLevel in range (0,16):
    
    fileModelIterationName = str(iterationLevel)
    fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
    fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
    m = trainlibmsvm_here_svr(gamma_rbf=0.000368044218572, Lambda=1/(335.737614243270968), eps = 0.000000013879058, svm_x= svm_x_init, svm_y= svm_y_init, lengthOfValidationSet=15)