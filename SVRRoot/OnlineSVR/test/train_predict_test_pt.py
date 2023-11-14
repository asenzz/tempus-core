import torch.nn as nn
import torch
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
#import lstm
#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy, random
import time
import custom_lstms
from adabelief_pytorch import AdaBelief


torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

#flight_data = sns.load_dataset("flights")
#flight_data.head()

lag = 12
test_data_size = lag
#all_data = flight_data['passengers'].values.astype(float)
all_data = []
file1 = open("/mnt/faststore/labels_dataset_100_q_svrwave_eurusd_avg_14400_bid_level_0_adjacent_levels_1_lag_400_call_7.out", 'r')
lines = file1.readlines()
for line in lines[1:250]:
    all_data.append(float(line))
all_data = numpy.array(all_data)
file1.close()

train_data = all_data[: - test_data_size - lag]
test_data = all_data[- lag - 1 - test_data_size:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = scaler.fit_transform(test_data.reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)

def create_inout_sequences(input_data):
    inout_seq = []
    L = len(input_data)
    for i in range(L - lag - 1):
        train_seq = input_data[i:i+lag].clone().detach().cuda()
        train_label = input_data[i+1:i+lag+1].clone().detach().cuda()
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized)
test_inout_seq = create_inout_sequences(test_data_normalized)
print("All data len " + str(len(all_data)) + " train data len " + str(len(train_data)) + " test data len " + str(len(test_data)))
print("Test Inout " + str(test_inout_seq))
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=24, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 3
        self.input_size = input_size # batch size
        self.output_size = output_size # batch size
        self.hidden_layer_size = hidden_layer_size

        #self.lstm = custom_lstms.script_lnlstm(input_size, hidden_layer_size, num_layers=self.num_layers).cuda()
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers).cuda()

        self.linear = torch.nn.Linear(hidden_layer_size, output_size).cuda()

        self.reset_hidden()

    def forward(self, input_seq):
        input_seq_view = input_seq.view(len(input_seq), -1, len(input_seq[0])).cuda()
        lstm_out, self.hidden_cell = self.lstm(input_seq_view, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions

    def reset_hidden(self):
        #self.hidden_cell = [custom_lstms.LSTMState(torch.zeros(self.input_size, self.hidden_layer_size).cuda(),
        #                    torch.zeros(self.input_size, self.hidden_layer_size).cuda())
        #          for _ in range(self.num_layers)]
        self.hidden_cell = (torch.zeros(self.num_layers, self.output_size, self.hidden_layer_size).cuda(),
                            torch.zeros(self.num_layers, self.output_size, self.hidden_layer_size).cuda())

    def set_in_size(self, new_size):
        self.input_size = new_size
        self.lstm.input_size = new_size

    def set_out_size(self, new_size):
        self.output_size = new_size
        self.linear.out_features = new_size


model = LSTM()
model.set_in_size(len(train_inout_seq))
model.set_out_size(len(train_inout_seq[0][0]))

loss_function = nn.MSELoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
n_losses = 20
epochs = 1000
avg_losses = []
prev_avg_loss = sys.float_info.max

all_seq = []
all_labels = []
for seq, labels in train_inout_seq:
    all_seq.append(seq)
    all_labels.append(labels)

all_seq = torch.cat(all_seq).view(len(all_seq), 1, len(all_seq[0])).cuda()
all_labels = torch.cat(all_labels).view(len(all_labels), 1, len(all_labels[0])).cuda()
all_seq = torch.transpose(all_seq, 1, 0).cuda()
all_labels = torch.transpose(all_labels, 1, 0).cuda()
print(all_seq)
exit(1)
time_start = time.time()
for i in range(epochs):
    model.reset_hidden()
    y_pred = model(all_seq)
    optimizer.zero_grad()
    all_loss = loss_function(y_pred, all_labels)
    avg_loss = all_loss.item()
    all_loss.backward()
    optimizer.step()

    avg_losses.append(avg_loss)
    # Moving Average of loss
    if len(avg_losses) > n_losses:
        curr_avg_loss = 0
        for n in range(len(avg_losses) - n_losses, len(avg_losses)):
            curr_avg_loss += avg_losses[n]
        curr_avg_loss = curr_avg_loss / n_losses
        if prev_avg_loss < curr_avg_loss:
            break
        else:
            prev_avg_loss = curr_avg_loss

"""
for i in range(epochs):
    avg_loss = 0.
    item_ct = 0
    for seq, labels in train_inout_seq:
        model.reset_hidden()

        y_pred = model(seq)

        optimizer.zero_grad()
        all_loss = loss_function(y_pred.view(len(y_pred), -1).cuda(), labels.view(len(labels), -1).cuda())
        #all_loss = loss_function(y_pred[-1], labels[-1].view(1, -1))
        avg_loss += all_loss.item()
        item_ct += 1
        all_loss.backward()
        optimizer.step()

    avg_loss = avg_loss / item_ct

    print(f'epoch: {i:3} loss: {avg_loss:10.8f}')
    avg_losses.append(avg_loss)
    # Moving Average of loss
    if len(avg_losses) > n_losses:
        curr_avg_loss = 0
        for n in range(len(avg_losses) - n_losses, len(avg_losses)):
            curr_avg_loss += avg_losses[n]
        curr_avg_loss = curr_avg_loss / n_losses
        if prev_avg_loss < curr_avg_loss:
            break
        else:
            prev_avg_loss = curr_avg_loss
"""

print("Took " + str((time.time() - time_start)/60.) + " minutes")
model.set_inout_size(1)
fut_pred = lag
if i == epochs - 1: print("Warning epochs reached limit!")
#model.hidden_cell = best_model
predictions = []
orig_labels = []
model.lstm.eval()
"""
for seq, labels in test_inout_seq:
    with torch.no_grad():
        pred = model(seq)

        #print("first_seq " + str(first_seq))
        #print("labels " + str(labels))
        #print("seq " + str(seq))

        predictions = pred.view(len(pred), -1).tolist()
        orig_labels = labels.tolist()
"""

first_seq = None
for seq, labels in test_inout_seq:
    if first_seq == None: first_seq = seq.clone().detach().cuda().view(len(seq), -1)
    orig_labels.append(labels[-1])

for seq, labels in test_inout_seq:
    with torch.no_grad():
        pred = model(seq)

        #print("first_seq " + str(first_seq))
        #print("labels " + str(labels))
        #print("seq " + str(seq))

        predictions.append(pred[-1].item())
        first_seq = torch.cat([first_seq[1:], pred[-1]])

maeloss = torch.nn.L1Loss()
predictions = predictions[1:]
orig_labels = orig_labels[:-1]
print("MAE: "+str(maeloss(torch.tensor(predictions).view(len(predictions), -1), torch.tensor(orig_labels).view(len(orig_labels), -1))))

actual_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#print(actual_predictions)

#x = np.arange(len(flight_data['passengers']) - len(actual_predictions), len(flight_data['passengers']), 1)
#print(x)
plt.title('EURUSD Level 0 data')
plt.ylabel('Price component 0')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#plt.plot(flight_data['passengers'])
#plt.plot(x, actual_predictions)
plt.plot(predictions)
plt.plot(orig_labels)
plt.show()

