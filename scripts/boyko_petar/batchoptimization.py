#!/usr/bin/env python

#import hyperParamChooseLib
import sys
import numpy as np
import subprocess
import os
import svmutil
#from svmutil import *
from six.moves import cPickle as pickle
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import re
#import coordinateDescent

column_name = 'close';
levelsPlusOne = 9

def get_numpy_x_from_svm(svm_x):
    numpy_x = np.zeros(shape=[len(svm_x),len(svm_x[1])]);
    for vecidx, vec in enumerate(svm_x):
        for label in vec:
            numpy_x[vecidx,label-1] = vec[label];
    return numpy_x;

def trainpredict(y, x, yValid, xValid, gamma, eps, cost):
    base_svm_options = '-s 3 -t 2';
#    y, x = svm_read_problem(filename);
    #yValid, xValid = svm_read_problem('valid');
    gamma_opt = '-g ' + str(gamma);
    eps_opt = '-p ' + str(eps);
    cost_opt = '-c ' + str(cost);
    svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
    print(svm_params);
    m = svmutil.svm_train(y, x, svm_params);
    _label, p_acc, p_val = svmutil.svm_predict(yValid, xValid, m);
    vec = [gamma,eps,cost,p_acc[1],p_acc[2]];
    print('Result: ',vec);
    return vec;

def GenerateHyperParams(eps = 1e-5, basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    hyperParams = {}
    for modelidx in range(0,levelsPlusOne):
        modelname = 'model'+str(modelidx);
        y, svm_x = svmutil.svm_read_problem(basedir+modelname+'.'+column_name+'.txt');
        modelhparams = { 'name' : modelname };
        modelhparams.update({'filename' : basedir+modelname+'.'+column_name+'.txt'});
        x_large = get_numpy_x_from_svm(svm_x);
        svm_x = get_numpy_x_from_svm(svm_x);
        yValid = np.asarray(y)[-15:];
        xValid = np.asarray(svm_x)[-15:];
        y = np.asarray(y)[0:-15];
        x = np.asarray(svm_x)[0:-15];
        opt_gamma = hyperParamChooseLib.findSeedGamma(x, stepmul=1.3,step_initial=0.5,tolerance=1e-2,gamma_initial=1e-3, max_iterations = 30, max_num_feat_vectors = 1500)
        modelhparams.update({ 'gamma-seed' : opt_gamma })
    
        command = 'Rscript'
#        pypath = '.'
        pypath = '/home/peters/Downloads/libsvm-3.22/python'
#        rpath = '.'
        rpath = '/home/boyko/work-projects/issue-3269-auto-lambda/r-implementation'
        os.chdir(rpath)
        script = 'findLambda.r'
        nametopass = modelname+'.'+column_name+'.large.train'
        datapath = basedir + 'subdatasets';
#        if modelidx == 0:
        args = [datapath+'/'+nametopass, str(modelhparams['gamma-seed'][0]),str(eps)];
#        else:
#            args = [datapath+'/'+nametopass, str(modelhparams['gamma-seed'][0]),str(eps * 1e-4)];
                        
        
        cmd = [command,rpath+'/'+script]+args;
        try:
            x = subprocess.check_output(cmd, universal_newlines=True);
            try:
                pass;
            except ValueError:
                pass;
        except subprocess.CalledProcessError:
            x = 'the R script crashed';
        print('Lambda:',str(x));
        modelhparams.update({'optimum lambda' : x});
        os.chdir(pypath);
        
        print(modelhparams);
        hyperParams.update({ modelname : modelhparams } );
        pickle.dump(hyperParams,open('hyperparams.pickle','wb'),pickle.HIGHEST_PROTOCOL);

def validate(eps = 1e-5, basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    hyperParams = pickle.load(open(basedir+'hyperparams.pickle','rb'));
    for model in hyperParams:
        print();
        print(model);
        modelHyperParams = hyperParams[model];
        try:
            gamma = modelHyperParams['gamma-seed'][0];
            cost = 1 / float(modelHyperParams['optimum lambda']);
            y, svm_x = svmutil.svm_read_problem(basedir+model+'.'+column_name+'.txt');
#            xValid = svm_x[684:684+16];
#            yValid = y[684:684+16];
#            x = svm_x[0:683];
#            y = y[0:683];

            sizeOfValidationSet = 0.005;
            ValidStartIdx = int(len(svm_x) - len(svm_x)*sizeOfValidationSet);
            #svm_x = get_numpy_x_from_svm(svm_x);
            yValid = y[ValidStartIdx:];
            xValid = svm_x[ValidStartIdx:];
            y = y[0:ValidStartIdx];
            x = svm_x[0:ValidStartIdx];

            res = trainpredict(y=y, x=x, yValid=yValid, xValid=xValid, gamma=gamma, eps=eps, cost=cost);
            modelHyperParams.update({'results' : res});
            hyperParams.update({model : modelHyperParams})
        except ValueError:
            modelHyperParams.update({'results' : 'failed'})
            hyperParams.update({model : modelHyperParams})
            print('Invalid hyperparameter.');
    pickle.dump(hyperParams,open(basedir+'hyperparams.pickle','wb'),pickle.HIGHEST_PROTOCOL);
    return hyperParams;
            
def printRMSReport(basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    hyperParams = pickle.load(open(basedir+'hyperparams.pickle','rb'));
    for model in hyperParams:
        resvec = hyperParams[model]['results'];
        print(model,resvec);
        
def create_exp_scaled_array(min, max, nItemsInGrid):
    base = np.exp(np.log(max/min)/nItemsInGrid);
    return min * np.power(base,np.arange(0,nItemsInGrid));
        
def populateRandomSearch(models, basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    gammas = create_exp_scaled_array(1e-5, 1e5, 100000);
#    gammas = create_exp_scaled_array(1e-25, 1e10, 100000);
#    epsilons = create_exp_scaled_array(1/np.power(10,10), 1, 1000);
#    epsilons = create_exp_scaled_array(1/np.power(10,3),1/np.power(10,3),1);
    epsilons = []
    epsilons.append(1e-11);
    costs = create_exp_scaled_array(1e-5, 1e10, 100000);
#    costs = create_exp_scaled_array(1e-25, 1e5, 100000);
    
    d = {0:gammas,1:epsilons,2:costs};#,3:models};
#    r = pickle.load(open('populatedRandomSearch.pickle','rb'));
#    for item in r:
#        results.append(item);
    last_gamma = gammas[np.random.randint(len(gammas))];
    last_eps = epsilons[np.random.randint(len(epsilons))];
    last_cost = costs[np.random.randint(len(costs))];
    base_svm_options = '-s 3 -t 2';
#    try:
#        results = pickle.load(open('populatedRandomSearch.pickle','rb'));
#    except FileNotFoundError:
    results = {};
    for model in models:
        results.update({model['name'] : []});    
    i = 0;
    while True:
        for model in models:
            i += 1;
            typeparam = np.random.randint(len(d));
            idx = np.random.randint(len(d[typeparam]));
            if typeparam == 0:
                last_gamma = d[typeparam][idx];
            elif typeparam == 1:
                last_eps = d[typeparam][idx];
#                i -= 1; continue; # because eps is fixed
            elif typeparam == 2:
                last_cost = d[typeparam][idx];
            elif typeparam == 3:
                model = d[typeparam][idx];
            gamma_opt = '-g ' + str(last_gamma);
            eps_opt = '-p ' + str(last_eps);
            cost_opt = '-c ' + str(last_cost);
            svm_params = base_svm_options+' '+gamma_opt+' '+eps_opt+' '+cost_opt;
            print(svm_params);
            
            y = model['y'];
            x = model['x'];
            yValid = model['yValid'];
            xValid = model['xValid'];
            
            m = svmutil.svm_train(y, x, svm_params);
            _label, p_acc, p_val = svmutil.svm_predict(yValid, xValid, m);
            vec = [last_gamma,last_eps,last_cost,p_acc[1],p_acc[2]];
            print(i,model['name'],vec);
            if not math.isnan(vec[3]):
                results[model['name']].append(vec);
            if i % 20 == 0:
                print('Saved.')
                try:
                    r = pickle.load(open(basedir+'populatedRandomSearch.pickle','rb'));
                except FileNotFoundError:
                    r = {};
                try:
                    for m in results:
                        r[m] += results[m];
                except KeyError:
                    r = results;
                    
                pickle.dump(r,open(basedir+'populatedRandomSearch.pickle','wb'));
                
                for model in models:
                    results.update({model['name'] : []});
#    pickle.dump(results,open('populatedRandomSearch.pickle','wb'));
#    pickle.dump(results,open('populatedRandomSearch.pickle','wb'));
    return results;

def doRandomSearch(basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    models = [];
    for modelidx in range(0,levelsPlusOne):
        modelname = 'model'+str(modelidx);
        y, svm_x = svmutil.svm_read_problem(basedir+modelname+'.'+column_name+'.txt');
#        xValid = svm_x[684:684+16];
#        yValid = y[684:684+16];
#        x = svm_x[0:683];
#        y = y[0:683];
        
        sizeOfValidationSet = 0.005;
        ValidStartIdx = int(len(svm_x) - len(svm_x)*sizeOfValidationSet);
        #svm_x = get_numpy_x_from_svm(svm_x);
        yValid = y[ValidStartIdx:];
        xValid = svm_x[ValidStartIdx:];
        y = y[0:ValidStartIdx];
        x = svm_x[0:ValidStartIdx];
        
        model = { 'name' : modelname };
        model.update({ 'x':x,'y':y,'xValid':xValid,'yValid':yValid } );
        models.append(model)
    return populateRandomSearch(models, basedir);
    
def getMedianBestHyperParams(data, percentile=1):
    targetRMS = np.percentile(data[:,4],percentile);
    r = data[data[:,4] <= targetRMS]
    median_eps = np.median(r[:,1] );
    median_cost = np.median(r[:,2]);
    median_gamma = np.median(r[:,0]);
    return median_gamma, median_eps, median_cost;

def pruneByBestEpsilon(res_vector, percentile=1):
    median_gamma, median_eps, median_cost = getMedianBestHyperParams(res_vector, percentile);
    pruned_by_eps = [];
    for i in res_vector:
        if i[1] > median_eps * 0.5 and i[1] < median_eps * 2:
            pruned_by_eps.append(i);
    pruned_by_eps = np.asarray(pruned_by_eps);
    return pruned_by_eps, median_eps;
    
def plotCostAndGamma(vecs_to_plot):
    sc = plt.scatter(np.log(vecs_to_plot[:,2])/np.log(10),np.log(vecs_to_plot[:,0])/np.log(10),c=np.log(vecs_to_plot[:,3])/np.log(10),s=3,lw=0);
    plt.colorbar(sc);
    plt.xlabel('Cost');
    plt.ylabel('Gamma');
    
def plotInterpCostAndGamma(vecs_to_plot):
    gammas = create_exp_scaled_array(1/1e17, 1e10, 1000);
    costs = create_exp_scaled_array(1e-8, 1e10, 1000);
    gx = np.log(costs) / np.log(10);
    gy = np.log(gammas) / np.log(10);
    gyy = np.ndarray(shape=[len(gx),len(gy)]);
    gxx = np.ndarray(shape=[len(gx),len(gy)]);
    for idx in range(0,len(gx)):
        for idx2 in range(0,len(gy)):
            gxx[idx][idx2] = gx[idx];
            gyy[idx][idx2] = gy[idx2];
    
    xgamma = np.log(vecs_to_plot[:,0])/np.log(10);
    xcost = np.log(vecs_to_plot[:,2])/np.log(10);
    a = np.ndarray(shape=[len(xgamma),2]);
    for idx in range(0,len(a)):
        a[idx][1] = xgamma[idx];
        a[idx][0] = xcost[idx];
    #grid_nn = griddata(a, np.exp(vecs_to_plot[:,4]), (gxx, gyy),method='nearest'); #tointerp[0:100],method='nearest');
    grid_nn = griddata(a, np.log(vecs_to_plot[:,3])/np.log(10), (gxx, -gyy),method='nearest'); #tointerp[0:100],method='nearest');
    plt.ylabel('RMSE:'+str('%.2e' % np.min(vecs_to_plot[:,3]))+'-'+str('%.2e' % np.max(vecs_to_plot[:,3])));
    plt.xlabel('eps ~= %.2e' % np.median(vecs_to_plot[:,1]));
    plt.imshow(grid_nn.T)

def plotRMSTopology3d(percentileToObtain = 100,basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/'):
    results = pickle.load(open(basedir+'populatedRandomSearch.pickle','rb'));
    i = 0;
    for model in results:
        results[model] = np.asarray(results[model]);
    for model in results:
        if model == 'model0':
            continue;
        fig = plt.figure();  
        i += 1;
#        plt.subplot(4,4,i);
        pruned_by_eps = results[model];
        a = pruned_by_eps[pruned_by_eps[:,3] < np.percentile(pruned_by_eps[:,3]*1.05,percentileToObtain)]
        g = a[:,0];
        c = a[:,2];
        rms = a[:,3];
        print(g.shape);
        print(c.shape);
        print(rms.shape);
        ax = fig.add_subplot(111, projection='3d');
        plt.title(model);        
        ax.plot_wireframe(np.log(c)/np.log(10),np.log(g)/np.log(10), 1/rms)#, zdir='z', s=2, c='b')   
    
def plotRMSTopology(basedir='/home/peters/Desktop/Tasks/7mar-2017-6k/', SystematicParamOverlay = True, EmphasizeBestRMS = True, plotSysGamma = True, percentileToObtain = 100):
    results = pickle.load(open(basedir+'populatedRandomSearch.pickle','rb'));
    i = 0;
    optGammas = np.zeros(len(results));
    optCosts = np.zeros(len(results));
    optRMSErrors = np.zeros(len(results));
    plt.figure(1);  
    for model in results:
        results[model] = np.asarray(results[model]);
    for model in results:
        r = re.compile("([a-zA-Z]+)([0-9]+)")
        i = int(r.match(model).groups()[1]);
        i += 1;
        plt.subplot(4,4,i);

        m = np.asarray(results[model]);
        print(model);
        idx = np.argmin(m[:,3]);
        optGammas[i-1] = (m[idx,0]);
        optCosts[i-1] = (m[idx,2]);
        optRMSErrors[i-1] = (m[idx,3]);
        
        pruned_by_eps = results[model];
        plt.title(model);        
        a = pruned_by_eps[pruned_by_eps[:,3] < np.percentile(pruned_by_eps[:,3]*1.05,percentileToObtain)]

        plotInterpCostAndGamma(a);
        
        
    fig2 = plt.figure(2,figsize=(20,10));
    
    if SystematicParamOverlay or plotSysGamma:
        hyperParams = pickle.load(open(basedir+'hyperParameters.pickle','rb'));
        for model in hyperParams:
            resvec = hyperParams[model]['results'];
            print(model,resvec);
    
    for model in results:
#        if model == 'model0':
#            continue;
        # use regex to get the plot position.. not pretty but meh.
        r = re.compile("([a-zA-Z]+)([0-9]+)")
        i = int(r.match(model).groups()[1]);
        plt.subplot(4,4,i+1);      
        pruned_by_eps = results[model];
        plt.title(model);        
        a = pruned_by_eps[pruned_by_eps[:,3] < np.percentile(pruned_by_eps[:,3]*1.05,percentileToObtain)];
        plotCostAndGamma(a);
        if EmphasizeBestRMS:
            bestRmsIdx = np.argmin(pruned_by_eps[:,3]);
            plt.scatter(np.log(pruned_by_eps[bestRmsIdx,2])/np.log(10),np.log(pruned_by_eps[bestRmsIdx,0])/np.log(10),marker='o', facecolors='none', edgecolors='blue', s=20);
            print('Best RMS for',model,':',str(pruned_by_eps[bestRmsIdx]));
        if plotSysGamma:
            local_g = np.log(hyperParams[model]['results'][0])/np.log(10);
            plt.plot([-6, 9], [local_g, local_g],c='blue')
            # now plot a slope=-1 logline from the intersect [gamma_sys, cost=0].
#            plt.plot([0, 5], [local_g, local_g - 5],c='blue');
#            plt.plot([0, 5], [0, -5],c='purple');
#            try:
#                sys_gamma = hyperParams[model]['results'][0]
#                sys_cost = hyperParams[model]['results'][2]
#                if sys_cost != float('Inf') and sys_cost != float('-Inf'):
#                    sys_cost = np.log(sys_cost)/np.log(10);
#                    plt,plt(sys_gamma)
#                    
#                    plt.plot([opt_cost-5, opt_cost+5], [np.log(gammas_scale[deltaidx])/np.log(10), np.log(gammas_scale[deltaidx])/np.log(10) - 5]);
#                    plt.plot([opt_cost-5, opt_cost+5], [np.log(gammas_scale[deltaidxRight])/np.log(10), np.log(gammas_scale[deltaidxRight])/np.log(10) - 5],c='red');
#                    plt.plot([opt_cost-5, opt_cost+5], [local_g, local_g - 5],c='green');
#            except TypeError:
#                pass;
        if SystematicParamOverlay:
            try:
                opt_cost = hyperParams[model]['results'][2];
                opt_gamma = hyperParams[model]['results'][0];
                opt_cost = float(opt_cost);
                opt_gamma = float(opt_gamma);
                opt_rms = hyperParams[model]['results'][3];
                plt.scatter(np.log(opt_cost)/np.log(10), np.log(opt_gamma)/np.log(10), marker='x', c='red', s = 50);
                #diff gamma to region orders of mag.
                highg = np.max(a[:,0]);
                lowg =  np.min(a[:,0]);
                highc = np.max(a[:,2]);
                lowc = np.min(a[:,2]);
                print('\n'+model+': ');
                idx = np.argmin(pruned_by_eps[:,3]);
                globalOptGamma = pruned_by_eps[idx,0];
                globalOptCost = pruned_by_eps[idx,2];
                globalOptRMS = pruned_by_eps[idx,3];
                gammaError = np.log(np.abs(opt_gamma/globalOptGamma))/np.log(10);
                costError = np.log(np.abs(opt_cost/globalOptCost))/np.log(10)
                RMSError = np.log(np.abs(opt_rms/globalOptRMS))/np.log(10)
                print('Gamma off from global optimum by ',str(gammaError),'orders of magnitude');
                print('Cost off from global optimum by ',str(costError),'orders of magnitude');
                print('RMS off from global optimum by ',str(optRMSErrors),'orders of magnitude');
                if lowg < opt_gamma and opt_gamma < highg:
                    print('Gamma ',opt_gamma,'in range ',str(lowg),'-',str(highg));
                else:
                    if opt_gamma > highg:
                        print('Gamma off by ',str((np.log(opt_gamma)-np.log(highg))/np.log(10)),'orders of magnitude.');
                    else:
                        print('Gamma off by ',str((np.log(opt_gamma)-np.log(lowg))/np.log(10)),'orders of magnitude.');
                if lowc < opt_cost and opt_cost < highc:
                    print('Cost ',opt_cost,'in range ',str(lowc),'-',str(highc));
                else:
                    if opt_cost > highc:
                        print('Cost off by ',str((np.log(opt_cost)-np.log(highc))/np.log(10)),'orders of magnitude.');
                    else:
                        print('Cost off by ',str((np.log(opt_cost)-np.log(lowc))/np.log(10)),'orders of magnitude.');
            except ValueError:
                pass;
            except TypeError:
                pass;
    fig2.savefig(basedir+'/graphed-topology.png', dpi=800, facecolor='w', edgecolor='w',
                 orientation='portrait', papertype=None, format=None,
                 transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None)
    plt.show();
    
    return optGammas, optCosts, optRMSErrors;        
    
def printHelp():
    print(sys.argv[0],'[command] [parameters]')
    print('command: populate-random-search, generate-hyperparameters, graph-topology')
    print(sys.argv[0],'populate-random-search [dirpath=\'./\'] [eps=1e-11]');
    print(sys.argv[0],'generate-hyperparameters [svmfilepath=\'./\'] [eps=1e-11] [optional \'calccost\'=\'\' or \'cost\']');
    print(sys.argv[0],'graph-topology [dirpath=\'./\'] [optional \'[systematic]\'');
    exit(1)
try:
    if len(sys.argv) < 2:
        printHelp();
    if sys.argv[1] == 'populate-random-search':
        if len(sys.argv) < 3:
            basedir = './'
        else:
            basedir = sys.argv[2]
        if len(sys.argv) < 4:
            eps=1e-11;
        else:
            eps=float(sys.argv[3])
        doRandomSearch(basedir)
    elif sys.argv[1] == 'generate-hyperparameters':
        if len(sys.argv) < 3:
            printHelp();
        else:
            svmfilepath=sys.argv[2]
        if len(sys.argv) < 4:
            eps = 1e-11;
        else:
            eps = float(sys.argv[3])
        if len(sys.argv) < 5:
            calcCost=''
        else:
            calcCost=sys.argv[4]
        args = [svmfilepath, str(eps), calcCost]
        command='python findGammaSimple.py'
        cmd = [command]+args;
        process_std_out = subprocess.check_output(cmd, universal_newlines=True);
        print(process_std_out);
    elif sys.argv[1] == 'graph-topology':
        if len(sys.argv) < 3:
            basedir = './'
        else:
            basedir = sys.argv[2]
        systematic = False;
        if len(sys.argv) >= 4:
            if sys.argv[3] == 'systematic' or sys.argv[3] == 'sys':
                systematic = True
        plotRMSTopology(basedir=basedir, percentileToObtain = 100, SystematicParamOverlay = systematic, plotSysGamma = systematic, EmphasizeBestRMS = systematic)
    else:
        printHelp();
        
except TypeError as e:
    print('exception:',e.with_traceback())
    printHelp()
except IndexError as e:
    print('exception:',e.with_traceback())
    printHelp()

        
        
            
    my_name = sys.argv[0];
    command = sys.argv[1];
    eps = float(sys.argv[2]);
#eps=1e-15

#basedir = './';
#basedir = '/home/boyko/datasets/sliding-datasets/2/'
#GenerateHyperParams(eps = eps, basedir=basedir);
#h = validate(eps=eps, basedir=basedir);
#printRMSReport();
#a = doRandomSearch(basedir=basedir);
#plotRMSTopology3d(percentileToObtain=20);
#optGammas, optCosts, optRMSErrors = plotRMSTopology(basedir=basedir, percentileToObtain = 100, SystematicParamOverlay = True, plotSysGamma = True, EmphasizeBestRMS = True);
#g, c = produceGammasCosts();
