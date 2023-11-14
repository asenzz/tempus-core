import svrpath
import numpy as np
import time
import pyarma
import sys
from numba import jit
# TODO Port to C++

gamma = float(sys.argv[3])
eps = 0. # 1e-9
lambdamin = 1e-6
maxiter = 25000

K = pyarma.mat()
y = pyarma.mat()
K.load(str(sys.argv[1])) #, filetype=hdf5_binary)
y.load(str(sys.argv[2])) #, filetype=pyarma.auto_detect)

@jit(parallel=True)
def conv_mat_to_ndarray(mat):
    res = np.ndarray(shape=[mat.n_rows, mat.n_cols])
    for i in range(mat.n_rows):
        for j in range(mat.n_cols):
            res[i, j] = mat[i, j]
    print(str(res))
    return res

@jit
def conv_vec_to_ndarray(colvec):
    res = []
    for i in range(colvec.n_rows):
        res.append(colvec[i,0])
    ar_res = np.asarray(res)
    print(str(ar_res))
    return ar_res

print('Testing SVRPath..')
start = time.time()
res = svrpath.svrpath(conv_vec_to_ndarray(y), conv_mat_to_ndarray(K), eps, maxiter, lambdamin, rho=1e-12)
end = time.time()
print('------',len(res['alphas']),'iterations,',(end-start),' s ------')
bestIteration = np.argmin(res['maes'])
print('Output lambda:',res['lambdas'][bestIteration])
print('Support vectors(elbows) sizes left,right:',len(res['ElbowLeft'][bestIteration]),len(res['ElbowRight'][bestIteration]))
print('Epsilon-insensitive region:',len(res['Center'][bestIteration]))
print('Out-of-margin left,right:',len(res['LeftRegion'][bestIteration]),len(res['RightRegion'][bestIteration]))