#!/usr/bin/python3

import matplotlib.pyplot as plt
import pyarma
import sys

if sys.argv[1] == 'arma':
    mat = pyarma.mat()
    mat.load(sys.argv[2])
    lmat = list(pyarma.vectorise(mat))
else:
    file1 = open(sys.argv[2])
    lines = file1.readlines()
    lmat = []
    for l in lines[2:]:
        if len(l) > 1: lmat.append(float(l))

print(str(lmat))
plt.plot(lmat)
plt.show()
