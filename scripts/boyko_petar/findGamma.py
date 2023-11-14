import os
import numpy as np
#import time
#from six.moves import cPickle as pickle
#import matplotlib.pyplot as plt
#import datetime
#from scipy.interpolate import griddata
import functools;
#from scipy.optimize import *
import scipy
#import time;
#from functools import partial
import scipy.optimize

def ranAllModels():
    dirNameModelToLoad = "/home/peters/Work/Tasks/may-9-stft/"
    for i in range(0,16):
        fileModelIterationName = str(i)
        fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
        fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
        svm_filename = fileNameModelToLoadFull
        print ('filename : ' , svm_filename)
        ans = findGamma(svm_filename)
        print('RBF-gamma estimated for level ' + str(i) + ' :' + str(ans) + '\n')

def mean_dist(xes, gamma, trimmed_ratio, meanOrMedian):
    
    RBF_dists = np.zeros(shape=[len(xes),len(xes)]);
    mean_dist = 0;
    for idx1, item1 in enumerate(xes):
        for idx2 in range(idx1,len(xes)):
            RBF_dists[idx1,idx2] = np.dot(item1-xes[idx2],item1-xes[idx2])
            RBF_dists[idx2,idx1] = RBF_dists[idx1,idx2];
    
#    RBF_dists = 2 - 2*np.exp(-gamma * RBF_dists);
    RBF_dists = 1 - np.exp(-gamma * RBF_dists);
    mask = np.ones(RBF_dists.shape, dtype=bool);
    np.fill_diagonal(mask,0);
    if trimmed_ratio == 0:
        if meanOrMedian == 'median':
            mean_dist = np.median(RBF_dists[mask])
        else:
            mean_dist = np.sum(RBF_dists[mask]) / ((len(xes)-1)*len(xes));
    else:
        a = RBF_dists[mask].flatten()
        cutoffdown = np.percentile(a,trimmed_ratio * 100)
        cutoffup = np.percentile(a,100 - trimmed_ratio * 100)
        a = a[a <= cutoffup]
#        a = a[a >= cutoffdown]
        if meanOrMedian == 'median':
            mean_dist = np.median(a)
        else:
            mean_dist = np.sum(a) / len(a);
    return mean_dist
    
def mean_dist_diff_from_1(gammaLog, x, trimmed_ratio, meanOrMedian):
    gamma = np.power(10,gammaLog)
    dist = mean_dist(x, gamma, trimmed_ratio, meanOrMedian);
    return np.abs(dist*len(x)-1)
    
def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

def findGamma(svm_filename, trimmed_ratio = 0, meanOrMedian = 'mean'):
    os.chdir('/home/peters/Work/gitfolder/tempus-core/libs/libsvm-modified/python/')
    import svmutil
    y_init, x_init = svmutil.svm_read_problem(svm_filename);
    x = x_init[0:200]
    y = y_init[0:200]    
    y = np.asarray(y)
    x = get_numpy_x_from_svm(x);
    boundFunc = functools.partial(mean_dist_diff_from_1, x=x, trimmed_ratio=trimmed_ratio, meanOrMedian=meanOrMedian)
    res = scipy.optimize.minimize(boundFunc, np.log(1.0/len(x))/np.log(10), method='COBYLA', tol=1e-3)
    return float(np.power(10,res['x']))
    #return mean_dist_diff_from_1(x, 1e-5, 0)
