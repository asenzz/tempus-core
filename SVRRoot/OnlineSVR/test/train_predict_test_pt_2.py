#!/usr/bin/python3
import torch
#from torch.nn import Parameter
#import torch.jit as jit
#import warnings
#from collections import namedtuple
#from typing import List, Tuple
#from torch import Tensor
#import lstm
#import seaborn as sns
#import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy, random
import time
from sklearn.preprocessing import MinMaxScaler

#import custom_lstms
#from adabelief_pytorch import AdaBelief
from numba import jit


# SINGLE STEP BATCH

#torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

lag = 100
test_data_size = 150
all_data = []
file1 = open("/mnt/faststore/repo/tempus-core/SVRRoot/OnlineSVR/test/test_data/train_predict_test_eurusd_twap_1h.tsv", 'r')
lines = file1.readlines()
for line in lines[-1000:]:
    all_data.append(float(line))
all_data = numpy.array(all_data)
file1.close()


scaler = MinMaxScaler(feature_range=(-1, 1))
all_data = scaler.fit_transform(all_data.reshape(-1, 1))

train_data = all_data[: - test_data_size - 4]
test_data = all_data[- lag - 1 - test_data_size - 4: - 4]

train_data_normalized = torch.FloatTensor(train_data).view(-1)
#test_data_normalized = scaler.fit_transform(test_data.reshape(-1, 1))
test_data_normalized = torch.FloatTensor(test_data).view(-1)


def create_inout_sequences(input_data):
    inout_seq = []
    L = len(input_data)
    for i in range(L - lag - 1):
        train_seq = input_data[i:i+lag].clone().detach().cuda()
        train_label = abs((input_data[i+lag+1] - input_data[i+lag]).clone().detach().cuda())
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized)
test_inout_seq = create_inout_sequences(test_data_normalized)
print("All data len " + str(len(all_data)) + " train data len " + str(len(train_data)) + " test data len " + str(len(test_data)))
#print("Train Inout 0 " + str(train_inout_seq[0]))

class LSTM(torch.nn.Module):
    def __init__(self, input_size=lag, hidden_layer_size=1.5*lag, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 2
        self.input_size = input_size # batch size
        self.output_size = output_size

        #self.lstm = custom_lstms.script_lnlstm(input_size, hidden_layer_size, num_layers=self.num_layers).cuda()
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers).cuda()

        self.linear = torch.nn.Linear(hidden_layer_size, output_size).cuda()

        self.reset_hidden()

    def forward(self, input_seq):
        # front_size = len(input_seq) if input_seq[0].dim() == 0 else len(input_seq[0])
        # print("front_size " + str(front_size) + " len " + str(len(input_seq)))
        input_seq_view = input_seq.view(1, 1, 100).cuda()
        lstm_out, self.hidden_cell = self.lstm(input_seq_view, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions

    def reset_hidden(self):
        #self.hidden_cell = [custom_lstms.LSTMState(torch.zeros(self.input_size, self.hidden_layer_size).cuda(),
        #                    torch.zeros(self.input_size, self.hidden_layer_size).cuda())
        #          for _ in range(self.num_layers)]
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_layer_size).cuda(),
                            torch.zeros(self.num_layers, 1, self.hidden_layer_size).cuda())

    def set_inout_size(self, new_size):
        self.input_size = new_size
        self.lstm.input_size = new_size
        self.output_size = new_size
        self.linear.out_features = new_size

time_start = time.time()
model = LSTM()
#model.set_inout_size(1)
loss_function = torch.nn.MSELoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
n_losses = 200
epochs = 500
avg_losses = []
prev_avg_loss = sys.float_info.max

# Train
for i in range(epochs):
    avg_loss = 0.
    item_ct = 0
    for seq, labels in train_inout_seq:
        model.reset_hidden()

        y_pred = model(seq)

        optimizer.zero_grad()
        #all_loss = loss_function(y_pred.view(len(y_pred), -1).cuda(), labels.view(len(labels), -1).cuda())
        all_loss = loss_function(y_pred.view(1, 1, 1).cuda(), labels.view(1, 1, 1).cuda())
        #print(str(y_pred[-1]) + " " + str(labels.view(1, 1)))
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
        if curr_avg_loss >= prev_avg_loss:
            break
        else:
            prev_avg_loss = curr_avg_loss

if i == epochs - 1: print("Warning epochs reached limit!")
#model.hidden_cell = best_model
predictions = []
orig_labels = []
model.lstm.eval()

for seq, labels in test_inout_seq:
    orig_labels.append(labels.item())
    with torch.no_grad():
        pred = model(seq)
        predictions.append(pred.item())
print("Took " + str(time.time() - time_start) + " seconds")

maeloss = torch.nn.L1Loss()
#predictions = predictions[1:]
#orig_labels = orig_labels[:-1]

mae = 0
for i in range(len(predictions)):
    mae += numpy.abs(predictions[i] - orig_labels[i])
mae = mae / len(predictions)

#print("MAE: "+str(maeloss(torch.tensor(predictions).view(len(predictions), -1), torch.tensor(orig_labels).view(len(orig_labels), -1))))
print("my MAE: " + str(mae))

actual_predictions = scaler.inverse_transform(numpy.array(predictions).reshape(-1, 1))
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

