import epspath
import numpy as np
import time

#plot 2*sin(x/10)+x^(0.5) from 0 to 100
#real function to be regressed

# Converges
def real_objective(x):
    return 2*np.sin(x/10)+np.power(x,0.5);
    
# Converges
def real_objective2(x):
    return x + 1

# Converges
def real_objective4(x):
    return np.sinh(x/10 - 5)
    
# Converges
def real_objective5(x):
    return np.power(x/100,2) + 2*x + 2
    
# Converges
def real_objective6(x):
    return np.sin(x/20)+2
    
def extractFeatures(base_x):
    x = [];
    for item in base_x:
        feat_0 = item;
        feat_1 = np.sin(item/20);
        feat_2 = np.sin(item/5);
        x.append([feat_0, feat_1, feat_2]);
    return np.asarray(x);
    
def RBF(x, gamma):
    gram_matrix = np.ndarray(shape=[len(x),len(x)]);
    for idx1, x1 in enumerate(x):
        for idx2, x2 in enumerate(x):
            gram_matrix[idx1,idx2] = np.exp(- gamma * np.dot(x1-x2,x1-x2));
    return gram_matrix;

base_x = np.arange(0,100,2);
y = real_objective(base_x)
base_xValid = np.arange(100,115);
yValid = real_objective(base_xValid);
x = extractFeatures(base_x);
xValid = extractFeatures(base_xValid);

gamma = 1e-3;
eps = 1e-9;
Lambda = 1e-3#1e-1
K = RBF(x,gamma);
Kvalid = RBF(xValid,gamma);

lambdamin = 1e-5;

print('Testing svrpath..')
start = time.time()
res = epspath.svrpath(x=x, y=y, K=K, Lambda=Lambda, maxIterations = 10000, lambdamin=lambdamin, rho = 1e-12, RBFGamma = gamma);
end = time.time()
#print('------',len(res['alphas']),'iterations,',(end-start),' s ------')
#bestIteration = np.argmin(res['maes'])
#print('Output lambda:',res['lambdas'][bestIteration])
#print('Support vectors(elbows) sizes left,right:',len(res['ElbowLeft'][bestIteration]),len(res['ElbowRight'][bestIteration]))
#print('Epsilon-insensitive region:',len(res['Center'][bestIteration]))
#print('Out-of-margin left,right:',len(res['LeftRegion'][bestIteration]),len(res['RightRegion'][bestIteration]))