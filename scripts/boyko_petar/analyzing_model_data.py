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
from sklearn import preprocessing

dirNameModelToLoad = "/home/peters/Work/gitfolder/tempus-core/build-Release/model-s1-run7-nonorm-dbrestart/"

lengthOfValidationSet = 15
lengthOfSlidingWindow = 1440
lengthOfTrainingWindow = 6500  + lengthOfValidationSet  #lengthOfValidationSet is included 
slidingWindowNumber = 0
slidingOffset = slidingWindowNumber * lengthOfSlidingWindow


def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

 
for i in range (0,16):
    
    fileModelIterationName = str(i)
    fileNameModelToLoad = "model" + fileModelIterationName + ".close" + ".txt"
    fileNameModelToLoadFull = dirNameModelToLoad + fileNameModelToLoad
    gram_matrix_filename = fileNameModelToLoadFull + '.gram_matrix' + '.pickle'
    
    svm_y_init, svm_x_init = svmutil.svm_read_problem(fileNameModelToLoadFull)
#    plt.figure(i)
#    plt.plot(svm_y_init)
    print('\nLength whole set', len(svm_x_init),' for level ', i, '\n')
    print('Maximum value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(max(svm_y_init)))
    print('Minimum value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(min(svm_y_init)))
    print('Mean value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(np.mean(svm_y_init)))  
    print('Median value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(np.median(svm_y_init))) 
    print('0.1-percentile value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(np.percentile(svm_y_init,0.01)))
    print('99.9-percentile value for svm-y level ' + str(i) + ' svm_y_init is: ' + str(np.percentile(svm_y_init,99.99)))
#    arr_svm_x_init = np.array(svm_x_init)
#    print('Min value for svm-x level ' + str(i) + ' svm_x_init is: ' + str(np.min(arr_svm_x_init)))  
#    print('Max value for svm-x level ' + str(i) + ' svm_x_init is: ' + str(np.max(arr_svm_x_init)))  
    
#    arr_svm_y_init = np.array(svm_y_init)
#    svm_y_scaled_manual_4A = arr_svm_y_init - np.median(arr_svm_y_init)
#    svm_y_scaled_manual_4B = svm_y_scaled_manual_4A.reshape(-1,1)
#    max_abs_scaler = preprocessing.MaxAbsScaler()
#    svm_y_scaled_manual_4C = max_abs_scaler.fit_transform(svm_y_scaled_manual_4B)
#
#
#    arr_svm_y_init = np.array(svm_y_init)
#    svm_y_init_res = arr_svm_y_init.reshape(-1,1)
#    min_max_scaler = preprocessing.MinMaxScaler()
#    svm_y_scaled_minmax = min_max_scaler.fit_transform(svm_y_init_res)    
    
#    arr_svm_y_init = np.array(svm_y_init)
#    svm_y_init_res = arr_svm_y_init.reshape(-1,1)
#    svm_y_scaled_robustscale1 = preprocessing.robust_scale(svm_y_init_res, axis=0, with_centering=True, with_scaling=True, quantile_range=(0.01, 99.99))
#    svm_y_scaled_manual_4C_to_insert_x = np.zeros(shape=[len(svm_y_scaled_manual_4C)+26,1])
#    for iterat in range(0,55502):
#        svm_y_scaled_manual_4C_to_insert_x[iterat+26] = svm_y_scaled_manual_4C[iterat][0]

    
##robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True
#    svm_y_scaled_manual_1 = svm_y_init - np.mean(svm_y_init)  
#    lowerpercentile01 = np.percentile(svm_y_scaled_manual_2,0.1)
#    upperpercentile99 = np.percentile(svm_y_scaled_manual_2,99.9)
#    scale_percentile = upperpercentile99 - lowerpercentile01
#    svm_y_scaled_manual_2[svm_y_scaled_manual_2 >=  upperpercentile99] /= scale_percentile
#    svm_y_scaled_manual_2[svm_y_scaled_manual_2 <=  lowerpercentile01] *= scale_percentile                         
##    svm_y_scaled_manual /= scale_percentile
#    sumVal = 0
#    for iterdiff in range(0,len(svm_y_scaled_manual_2)):
#        sumVal += abs(svm_y_scaled_manual_2[iterdiff] - svm_y_scaled_robustscale1[iterdiff][0])
#    
#    print('The difference is::', sumVal)

#    svm_y_scaled_manual = svm_y_init - np.median(svm_y_init)  
#    lowerpercentile01 = np.percentile(svm_y_scaled_manual,0.01)
#    upperpercentile99 = np.percentile(svm_y_scaled_manual,99.99)
#    scale_percentile = upperpercentile99 - lowerpercentile01
#    svm_y_scaled_manual /= scale_percentile
#    sumVal = 0
#    for iterdiff in range(0,len(svm_y_scaled_manual)):
#        sumVal += abs(svm_y_scaled_manual[iterdiff] - svm_y_scaled_robustscale1[iterdiff][0])

##    svm_y_scaled = np.clip(svm_y_init, lowerpercentile01, upperpercentile99)
##    svm_y_scaled = svm_y_scaled - np.mean(svm_y_scaled)    
#    fileNameModelToSaveSlice = "scaledmodel" + fileModelIterationName + ".close" + ".txt"
#    with open('/home/peters/Work/Tasks/may-9-stft/15models/' + fileNameModelToSaveSlice,'wt') as f:
#        for rowidx in range(0,55502):
#            f.write(str(svm_y_scaled_manual_4C[rowidx][0]) + ' ')
#            for xidx in range(0,len(svm_x_init[rowidx])):
#                f.write(str(xidx) +':' + str(svm_x_init[rowidx][xidx]) + ' ')           
#            f.write('\n') 
#            
#    xWholeSet = get_numpy_x_from_svm(svm_x_init)
#    print('Maximum value for svm-x level ' + str(i) + 'svm_x_init is: ' + str(np.max(xWholeSet)))
#    print('Minimum value for svm-x level ' + str(i) + 'svm_x_init is: ' + str(np.min(xWholeSet)))          
    #PRINT MAXIMUM / MINIMUM VALUES FOR THE WHOLE SET
#    yWholeSet = get_numpy_x_from_svm(svm_y_init)    
#    print('Maximum value for svm_y_init is: ' + np.max(yWholeSet))
#    print('Minimum value for svm_y_init is: ' + np.min(yWholeSet))    
    #PRINT MAXIMUM / MINIMUM VALUES FOR THE SELECTED WINDOW SET
    
#    svm_x = svm_x_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
#    svm_y = svm_y_init[slidingOffset:slidingOffset+lengthOfTrainingWindow]
#    ValidStartIdx = int(len(svm_x) - lengthOfValidationSet)
#    xFull = get_numpy_x_from_svm(svm_x)
#    xFull2 = np.asarray(xFull)
#    xFull3 = xFull2[0:ValidStartIdx]
#    yFull = np.asarray(svm_y)
#    x = svm_x[0:ValidStartIdx];
#    y = svm_y[0:ValidStartIdx];
#    K = np.ndarray(shape=[len(xFull3),len(xFull3)])