#!/usr/bin/python3

import matplotlib.pyplot as plt
import pyarma
import sys

if sys.argv[1] == 'arma':
    mat = pyarma.mat()
    mat.load(sys.argv[2])
    mat.shed_rows(0, mat.n_rows - 206)
    mat.shed_rows(mat.n_rows - 6, mat.n_rows - 1)
    lmat = list(pyarma.vectorise(mat))
else:
    file1 = open(sys.argv[2])
    lines = file1.readlines()
    lmat = []
    for l in lines[2:]:
        if len(l) > 1: lmat.append(float(l))
    mat = pyarma.mat(lmat)

shifted_mat = pyarma.shift(mat, 1)
abs_diff_mat = pyarma.abs(shifted_mat - mat)
abs_diff_mat.shed_row(0)
diff = pyarma.mean(abs_diff_mat)

print("MAPE " + str(100 * diff / pyarma.mean(pyarma.abs(mat))) + " percent")
plt.plot(abs_diff_mat)
plt.show()
