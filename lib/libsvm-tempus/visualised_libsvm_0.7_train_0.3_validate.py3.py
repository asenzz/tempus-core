import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import subprocess

print("Usage: "+sys.argv[0]+" <dataset_path> <KERNEL_TYPE> <RBF gamma> <SVR epsilon> <SVR cost>\n\n")


orig_dataset_file = sys.argv[1]
dataset_file=orig_dataset_file+".clean.txt"
dir_name=os.path.dirname(dataset_file)
file_name=dir_name=os.path.basename(dataset_file)
train_rows= "/tmp/" + file_name + "_train_rows.txt"
validate_rows= "/tmp/" + file_name + "_validate_rows.txt"
predicted_data = validate_rows + ".predict.txt"
model_file = train_rows + ".model.txt"


os.system("cut -d ' ' -f 3- " + orig_dataset_file + " > " + dataset_file)
os.system("awk 'FNR==NR{lines++;next}FNR<=0.7*lines' " + dataset_file + " " + dataset_file + " > " + train_rows)
os.system("awk 'FNR==NR{lines++;next}FNR>0.7*lines' " + dataset_file + " " + dataset_file + " > " + validate_rows)

kernel_type = "-t " + sys.argv[2]
gamma_arg = "-g " + sys.argv[3]
epsilon_arg = "-p " + sys.argv[4]
cost_arg = "-c " + sys.argv[5]

os.system("./svm-train -h 0 -s 3 " + kernel_type + " " + gamma_arg + " " + epsilon_arg + " " + cost_arg + " " + train_rows + " " + model_file)
os.system("./svm-predict " + validate_rows + " " + model_file + " " + predicted_data)


labels = np.loadtxt(validate_rows, DTYPE=np.float64, usecols=0)
predictions = np.loadtxt(predicted_data, DTYPE=np.float64)
# print(predictions)
residuals = labels - predictions

abs_residuals = np.abs(residuals, DTYPE=np.float64)
print("Mean Average Error " + str(np.mean(abs_residuals, DTYPE=np.float64)))


fig1 = plt.figure()
plt1 = fig1.add_subplot(111)
plt1.plot(labels)
plt1.plot(predictions)
fig1.show()

fig2 = plt.figure()
plt2 = fig2.add_subplot(111)
plt2.plot(residuals)
fig2.show()

input("Press any key to continue...")
