# =========================================================================================
# 1. These are all the modules we'll be using later. 
#    Make sure you can import them before proceeding further.
from __future__ import print_function
import numpy as np
import torch
from six.moves import cPickle as pickle
from six.moves import range
import os

pickle_file = 'data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)   # Training set (200000, 28, 28) (200000,)
    print('Validation set', valid_dataset.shape, valid_labels.shape) # Validation set (10000, 28, 28) (10000,)
    print('Test set', test_dataset.shape, test_labels.shape)         # Test set (10000, 28, 28) (10000,)

# =========================================================================================   
# 2. Reformat into a shape that's more adapted to the models we're going to train:
"""
1) unnormalize data
2) data as a flat matrix
"""

image_size = 28
num_labels = 10

def reformat(dataset):
    dataset = dataset * 255.0 + 255.0/2
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset

train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset = reformat(test_dataset)

print('Training set', train_dataset.shape, train_labels.shape)    # Training set (200000, 784) (200000,)
print('Validation set', valid_dataset.shape, valid_labels.shape)  # Validation set (10000, 784) (10000,)
print('Test set', test_dataset.shape, test_labels.shape)          # Test set (10000, 784) (10000,)

# =========================================================================================
# 3. We're first going to train a fully connected network 
#    with 1 hidden layer with 1024 units using stochastic gradient descent (SGD).

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch.optim as optim

class NotMNIST(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_idx = self.data[idx]
        label_idx = self.label[idx]
        return data_idx, label_idx

    
notmnist_train = NotMNIST(train_dataset, train_labels)
notmnist_valid = NotMNIST(valid_dataset, valid_labels)
notmnist_test = NotMNIST(test_dataset, test_labels)

print('Training set size: ', len(notmnist_train))   # Training set size:  200000
print('Validation set size: ', len(notmnist_valid)) # Validation set size:  10000
print('Test set size: ', len(notmnist_test))        # Test set size:  10000

# =========================================================================================
# 4. using dataLoader

batch_size = 64

train_loader = DataLoader(dataset=notmnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=notmnist_valid, batch_size=len(notmnist_valid), shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=notmnist_test, batch_size=len(notmnist_test), shuffle=False, drop_last=False)

from collections.abc import Iterable
print(issubclass(DataLoader, Iterable)) # True 
print()

inputs, labels = next(iter(train_loader))
print(f'Type of inputs: {type(inputs)}\tshape: {inputs.shape}') # Type of inputs: <class 'torch.Tensor'>	shape: torch.Size([64, 784])
print(f'Type of labels: {type(labels)}\tshape: {labels.shape}') # Type of labels: <class 'torch.Tensor'>	shape: torch.Size([64])
print()

print('Train loader size: ', len(train_loader)) # same as len(dataset) // batch_size, # Train loader size:  3125
print('Valid loader size: ', len(valid_loader))
print('Test loader size: ', len(test_loader))

# =========================================================================================
# 5. Naive Linear model

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class NaiveLinear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(NaiveLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        torch.nn.init.uniform_(self.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias



