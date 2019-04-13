# Imports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt

# We are going to implement Jason Brownlee basic LSTM code using PyTorch instead of Keras.
# See https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/ for more
# details. The data preparation part comes directly from his example

# Parameters
# n_pred must be equal to output_size
n_steps = 5
n_pred = 1
bs = 64
max_epochs = 100
hidden_size = 256
output_size = 1
input_size = 1
num_layers = 5
lr = 0.0001

# Let's define the device we are going to work on

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Data preparation 

def split_sequence(sequence, n_steps, n_pred):
    #X, y = [], []
    X, y = list(), list()
    for i in range(len(sequence)):
        # look for the end of this pattern
        end_idx = i + n_steps
        # check if we are beyon the end of the sequence
        if end_idx > len(sequence) - n_pred - 1:
            break
        # Prepare inputs and outputs of the sequence
        seq_X, seq_y = sequence[i:end_idx], sequence[end_idx+1:end_idx+n_pred+1]
        X.append(seq_X)
        y.append(seq_y)
    return torch.tensor(X).float(), torch.tensor(y).float()

seq = list(range(1000))

X, y = split_sequence(seq, n_steps, n_pred)

# Let's now define our dataset and dataloader using PyTorch tools

idx = int(len(X) * 0.8)
train = TensorDataset(X[:idx], y[:idx])
valid = TensorDataset(X[idx:], y[idx:])

train_loader = DataLoader(train, batch_size=bs, shuffle=False)
valid_loader = DataLoader(valid, batch_size=2, shuffle=False)

# Let's now define our LSTM Model

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # input_size is equal to the number of features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Output of LSTM layers will be [batch_size, seq_length, input_size]
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Initialize initial state for h0 and c0
        # input.size(0) actually corresponds to the batch size here
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)

        # print('h0 shape: ', h0.shape)
        # print('c0 shape: ', c0.shape)
        # print('input shape: ', input.shape)

        # Forward propagate the input into the LSTM
        result, _ = self.lstm(input, (c0, h0))
        result = result[:,-1,:]
        # print('result shape: ', result.shape)
        result = self.linear(result)
        return result

model = MyLSTM(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optim = optim.Adam(model.parameters(), lr=lr)

total_steps = len(train_loader)

lossdata = []
for epoch in range(max_epochs):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X = torch.reshape(X,(X.size(0), X.size(1), input_size)).to(device)
        out = model(X)
        y = y.to(device)
        # print(out)
        # print('out shape: ', out.shape)
        # print('y shape: ', y.shape)
        loss = criterion(out, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 100 == 0:
            lossdata.append([epoch+1,loss.item()])
            print('Epoch = {}/{}, Loss: {:.4f}'.format(epoch+1, max_epochs, loss.item()))

plt.plot(lossdata)
plt.show()

model.eval()
with torch.no_grad():
    for i, (X, y) in enumerate(valid_loader):
        X = torch.reshape(X, (X.size(0), X.size(1), input_size)).to(device)
        out = model(X)
        if i % 50 == 0:
            print('X = ', X)
            print('Prediction = ', out)
            print('Ground Truth = ', y)