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
n_steps = 3
n_pred = 1
bs = 4
max_epochs = 300
hidden_size = 128
output_size = 1
input_size = 1
num_layers = 1
lr = 1e-1
bs_valid = 2

# Let's define the device we are going to work on

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation 

def split_sequence(sequence, n_steps, n_pred):
    X, y = [], []
    for i in range(len(sequence)):
        # look for the end of this pattern
        end_idx = i + n_steps
        # check if we are beyon the end of the sequence
        if end_idx > len(sequence) - n_pred - 1:
            break
        # Prepare inputs and outputs of the sequence
        seq_X, seq_y = sequence[i:end_idx], sequence[end_idx:end_idx+n_pred]
        X.append(seq_X)
        y.append(seq_y)
    return torch.tensor(X).float(), torch.tensor(y).float()

# seq = list(range(1000))

seq = [10,20,30,40,50,60,70,80,90,100]

X, y = split_sequence(seq, n_steps, n_pred)

# Let's now define our dataset and dataloader using PyTorch tools

idx = int(len(X) * 0.8)
train = TensorDataset(X[:idx], y[:idx])
valid = TensorDataset(X[idx:], y[idx:])

train_loader = DataLoader(train, batch_size=bs, shuffle=False)
valid_loader = DataLoader(valid, batch_size=bs_valid, shuffle=False)

# Let's now define our LSTM Model

class MyLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # input_size is equal to the number of features
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, dropout = 0.4, batch_first=True)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        # Output of LSTM layers will be [batch_size, seq_length, input_size]
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Initialize initial state for h0 and c0
        # input.size(0) actually corresponds to the batch size here
        # h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)

        # Forward propagate the input into the LSTM
        out, _ = self.lstm(input)

        # We only need the last output of the sequence
        result = out[:,-1,:]

        # Apply a linear transformation to get the output we need
        result = self.linear(self.bn(result))
        return result

model = MyLSTM(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
# optim = optim.Adam(model.parameters(), weight_decay=0.4, lr=lr)
optim = optim.Adagrad(model.parameters(), lr = lr, lr_decay = 0.1, weight_decay=0.1)

lossdata = []
for epoch in range(max_epochs):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        if X.size(0) == bs:
            X = torch.reshape(X,(X.size(0), X.size(1), input_size)).to(device)
            out = model(X)
            y = y.to(device)
            loss = criterion(out, y)

            if i % 100 == 0:
                print('Epoch = {}/{}, Training Loss: {:.4f}'.format(epoch+1, max_epochs, loss.item()))
                lossdata.append([loss.item()])

            optim.zero_grad()
            loss.backward()
            optim.step()
        else:
            continue

plt.plot(lossdata)
plt.show()

lossvalid, validitems = [], 0.0
with torch.no_grad():
    model.eval()
    for i, (X, y) in enumerate(valid_loader):
        if X.size(0) == bs_valid:
            validitems += len(X)
            y = y.to(device)
            X = torch.reshape(X, (X.size(0), X.size(1), input_size)).to(device)
            prediction = model(X).detach()
            loss = criterion(prediction, y)
            lossvalid.append(loss.item())
            # print('Validation loss = ', loss.item())
        else:
            continue

print('Overall loss: {:.4f}'.format(sum(lossvalid)/validitems))

def check_models_are_equal(model1, model2):
    """ Check whether two models have identical parameters or not """
    for mp1, mp2 in zip(model1.parameters(), model2.parameters()):
        if mp1.data.ne(mp2.data).sum() >0:
            return False
    return True