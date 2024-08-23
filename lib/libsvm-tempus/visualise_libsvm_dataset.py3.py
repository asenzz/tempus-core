import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import subprocess

print("Usage: "+sys.argv[0]+" <dataset_path>.libsvm.txt\n\n")


orig_dataset_file = sys.argv[1]
dataset_file = orig_dataset_file+".clean.txt"
dir_name = os.path.dirname(dataset_file)
file_name = os.path.basename(dataset_file)

def get_lag_count(file_name):
    lag_count_start = str.find(file_name, beg = 0, end = len(file_name), sub = "_lag_") + 5
    lag_count_end = str.find(file_name, beg = lag_count_start, end=len(file_name), sub = "_")
    return int(file_name[lag_count_start:lag_count_end])

lag_count = get_lag_count(file_name)

os.system("cut -d ' ' -f 3- " + orig_dataset_file + " > " + dataset_file)
with open(dataset_file) as f:
    ncols = len(f.readline().split(' '))
labels = np.loadtxt(dataset_file, dtype=np.float64, usecols=0)
# features = np.loadtxt(dataset_file, dtype=np.float64, usecols=range(1,ncols))


fig1 = plt.figure()
plt1 = fig1.add_subplot(111)
plt1.plot(labels)

for i in range(0..len(features)):
    plt2 = fig1.add_subplot(111)
    plt2.plot(features[i])

fig1.show()
 
input("Press any key to continue...")
