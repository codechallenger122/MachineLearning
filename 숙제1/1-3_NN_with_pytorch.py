#0. These are all the modules we'll be using later. Make sure you can import them before proceeding further.
from __future__ import print_function
import numpy as np
import torch
from six.moves import cPickle as pickle
from six.moves import range
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

#1. load dataset
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
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

"""
Training set (200000, 28, 28) (200000,)
Validation set (10000, 28, 28) (10000,)
Test set (10000, 28, 28) (10000,)
"""
    
#2. Reformat into a shape that's more adapted to the models we're going to train:
#   unnormalize data, data as a flat matrix
image_size = 28
num_labels = 10

def reformat(dataset):
    dataset = dataset * 255.0 + 255.0/2
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset

train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset = reformat(test_dataset)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

"""
Training set (200000, 784) (200000,)
Validation set (10000, 784) (10000,)
Test set (10000, 784) (10000,)
"""

# 3. PyTorch tutorial: Fully Connected Network
#  We're first going to train a fully connected network with 1 hidden layer with 1024 units using stochastic gradient descent (SGD).

# 3-1. first, define NotMNIST dataset class.
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

print('Training set size: ', len(notmnist_train))
print('Validation set size: ', len(notmnist_valid))
print('Test set size: ', len(notmnist_test))

"""
Training set size:  200000
Validation set size:  10000
Test set size:  10000
"""

# 3-2. Then, make dataloader using NotMNIST dataset objects
batch_size = 64

train_loader = DataLoader(dataset=notmnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=notmnist_valid, batch_size=len(notmnist_valid), shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=notmnist_test, batch_size=len(notmnist_test), shuffle=False, drop_last=False)

from collections.abc import Iterable
print(issubclass(DataLoader, Iterable))
print()

inputs, labels = next(iter(train_loader))
print(f'Type of inputs: {type(inputs)}\tshape: {inputs.shape}')
print(f'Type of labels: {type(labels)}\tshape: {labels.shape}')
print()

print('Train loader size: ', len(train_loader)) # same as len(dataset) // batch_size
print('Valid loader size: ', len(valid_loader))
print('Test loader size: ', len(test_loader))

"""
True

Type of inputs: <class 'torch.Tensor'>	shape: torch.Size([64, 784])
Type of labels: <class 'torch.Tensor'>	shape: torch.Size([64])

Train loader size:  3125
Valid loader size:  1
Test loader size:  1
"""

# 3.3 Define Naive Linear model
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
      
class Model(nn.Module):
    
    def __init__(self, in_features, nn_hidden, num_labels):
        super(Model, self).__init__()
        self.fc1 = NaiveLinear(in_features, nn_hidden)
        self.fc2 = NaiveLinear(nn_hidden, num_labels)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
nn_hidden = 1024

model = Model(image_size*image_size, nn_hidden, num_labels)

# move model to GPU
if torch.cuda.is_available():
    print("Cuda is available. Move model to GPU")
    device = 'cuda:0'
    model.to(device)
else:
    device = 'cpu'
    
# print model, initialized weight, grad buffer
print(model)
print(model.fc1.weight.data)
print(model.fc1.bias.grad)
"""
Cuda is available. Move model to GPU
Model(
  (fc1): NaiveLinear()
  (fc2): NaiveLinear()
)
tensor([[ 0.5153, -0.4414, -0.1939,  ..., -0.0334, -0.3184, -0.6335],
        [ 0.1658,  0.0407,  0.1526,  ..., -0.9122, -0.9774, -0.8611],
        [-0.5218,  0.6270,  0.8866,  ..., -0.1884,  0.3789, -0.5059],
        ...,
        [ 0.0949, -0.5369,  0.1770,  ...,  0.7149,  0.9331, -0.7348],
        [-0.7956,  0.9141,  0.3562,  ..., -0.1767, -0.8524,  0.0262],
        [-0.8096, -0.3422, -0.3679,  ...,  0.1363,  0.2865,  0.6339]],
       device='cuda:0')
None
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.005)
epochs = 10
log_step = 1000

def accuracy(logits, labels):
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    return (100.0 * np.sum(np.equal(np.argmax(logits, 1), labels)) / logits.shape[0])


# train_model
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for idx, data in enumerate(train_loader):
        images_flatten, labels = data[0].to(device), data[1].long().to(device)
        logits = model(images_flatten)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if (idx % log_step) == log_step-1:
            print(f'epoch: {epoch+1} [{idx + 1} / {len(train_loader)}]\t train_loss: {loss.item():.3f}\t train_accuracy: {accuracy(logits, labels):.1f}')


# evaluate model
def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            test_images_flatten, test_labels = data[0].to(device), data[1].to(device)
            test_logits = model(test_images_flatten)

        print(f'accuracy: {accuracy(test_logits, test_labels):.1f}\n')
for epoch in range(epochs):
    train(model, train_loader, optimizer, criterion, epoch, device)
    print('-------- validation --------')
    evaluate(model, valid_loader, device)

            
print('-------- test ---------')
evaluate(model, test_loader, device)

    
# save model
torch.save(model.state_dict(), './model_checkpoints/naive_model_final.pt')
print('naive model saved')


# 4. So far, you have built the model in a naive way. However, PyTorch provides a linear module named nn.Linear for your convenience.
model_layer = nn.Sequential(
            # neural network using nn.Linear
            nn.Linear(image_size * image_size, nn_hidden),
            nn.Tanh(),
            nn.Linear(nn_hidden, num_labels)
            )

model_layer.to(device)
"""
Sequential(
  (0): Linear(in_features=784, out_features=1024, bias=True)
  (1): Tanh()
  (2): Linear(in_features=1024, out_features=10, bias=True)
)
"""

criterion_layer = nn.CrossEntropyLoss()
optimizer_layer = optim.SGD(model_layer.parameters(), lr=0.005)
for epoch in range(epochs):
    train(model_layer, train_loader, optimizer_layer, criterion_layer, epoch, device)
    print('-------- validation --------')
    evaluate(model_layer, valid_loader, device)

            
print('-------- test ---------')
evaluate(model_layer, test_loader, device)

    
# save model
torch.save(model_layer.state_dict(), './model_checkpoints/layer_model_final.pt')
print('layer_model saved')

"""
naive 한 방법은 initial weight/bias 를 setting 하는 method, regularization methods 등이 포함되어 있지 않으나 
torch 에서 제공하는 nn.Linear 모델은 이러한 기능을 제공하기에 naive 한 방법의 성능이 떨어진다.
"""

learning_rate = 0.0005
epochs = 10
nn_hidden = 512
nn_hidden_2 = 256

""" TODO """
model_layer = nn.Sequential(
            # neural network using nn.Linear
            nn.Linear(image_size * image_size, nn_hidden),
            nn.ReLU(),
            nn.Linear(nn_hidden, nn_hidden_2),
            nn.ReLU(),
            nn.Linear(nn_hidden_2, num_labels)
            )

model_layer.to(device)

criterion_layer = nn.CrossEntropyLoss()
#optimizer_layer = optim.SGD(model_layer.parameters(), lr=learning_rate)
optimizer_layer = optim.Adam(model_layer.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train(model_layer, train_loader, optimizer_layer, criterion_layer, epoch, device)
    print('-------- validation --------')
    evaluate(model_layer, valid_loader, device)

            
print('-------- test ---------')
evaluate(model_layer, test_loader, device)

    
# save model
torch.save(model_layer.state_dict(), './model_checkpoints/problem2_2022-23717.pt')
print('layer_model saved')
