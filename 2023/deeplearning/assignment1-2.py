"""
     Neural network from scratch
"""

#1. first reload the data we generated in assignment 1-1.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

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
    
#2. Reformat into a shape that's more adapted to the models we're going to train:
#   - unnormalize data
#   - data as a flat matrix,
#   - labels as float 1-hot encodings.
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset * 255.0 + 255.0/2
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # one-hot encoding, Map the label 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

data_size = 5000
train_dataset = train_dataset[0:data_size]
train_labels = train_labels[0:data_size]

#3. implementation
num_examples = len(train_dataset) # training set size
input_dim = 784 # input layer dimensionality
hidden_dim = 100 # hidden layer dimensionality
output_dim = 10 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent

def init_model(input_dim, hidden_dim, output_dim):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    model = {}
    model['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    model['b1'] = np.zeros((1, hidden_dim))
    model['W2'] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    model['b2'] = np.zeros((1, output_dim))
    
    return model
 
# 4. 문제 - forward, backward 함수 implement 하기.
def forward(model, x):
    # Forward propagation
    # model: a model which contains keys 'W1', 'b1', 'W2', 'b2' as defined above
    # x: a batch (or mini-batch) of flattened input images
    # z1, a1, z2, a2: intermediate values (z1, a1, z2) and output probability distribution (a2) of the model
    # Try not to use "for loops"
    
    ########## TO DO ##########
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    score = np.exp(z2)
    a2 = score / np.sum(score, axis=1, keepdims=True)
    ###########################
    
    return z1, a1, z2, a2

  
def backward(model, x, z1, a1, z2, a2, labels):
    # Backward propagation
    # model: a model which contains keys 'W1', 'b1', 'W2', 'b2' as defined above
    # x: a batch (or mini-batch) of flattened input images
    # z1, a1, z2, a2: intermediate values (z1, a1, z2) and output probability distribution (a2) of the model
    # labels: target labels for the given input images x
    # dW1, db1, dW2, db2: calculated gradients with respect to W1, b1, W2, and b2, respectively
    
    # Some intermediate values (z1, a1, z2) may not be used
    # You can define additional variables (e.g., delta3) if you want
    # You can define additional function if you want
    # Note that gradients should be averaged over a given batch (or mini-batch)
    # Try not to use "for loops"
    
    ########## TO DO ##########
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    delta3 = (a2 - labels) / data_size
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(np.tanh(z1), 2))
    dW1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)
    
    ###########################
    
    return dW1, db1, dW2, db2

# Helper function to evaluate the total loss on the dataset <-- loss function
def calculate_loss(model, x):
    # Forward propagation to calculate predicted probabilities
    _, _, _, probs = forward(model, x)
    # Calculating the loss
    corect_logprobs = -np.log([probs[i,np.nonzero(train_labels)[(1)][i].astype('int64')] for i in range(num_examples)])
    data_loss = np.sum(corect_logprobs)
    
    return 1./num_examples * data_loss
  
# Helper function to predict an output (0 or 1)
def predict(model, x):
    # Forward propagation to calculate predicted probabilities
    _, _, _, probs = forward(model, x)
    
    return np.argmax(probs, axis=1)
  
def accuracy(model, x, labels):
    return (np.sum(np.argmax(labels, axis=1) == predict(model, x))) / len(labels)
  
# This function learns parameters for the neural network and returns the model.
# - hidden_dim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
# - print_val: If True, print the validation accuracy every 2500 iterations
def train_model(hidden_dim, num_passes=10000, print_loss=False, print_val=False):

    print('Start model training')
    
    model = init_model(input_dim, hidden_dim, output_dim)
    print('Model initialized')
    
    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1, a1, z2, a2 = forward(model, train_dataset)

        # Backpropagation
        dW1, db1, dW2, db2 = backward(model, train_dataset, z1, a1, z2, a2, train_labels)

        # Gradient descent parameter update
        model['W1'] += -epsilon * dW1
        model['b1'] += -epsilon * db1
        model['W2'] += -epsilon * dW2
        model['b2'] += -epsilon * db2
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and (i + 1) % 1000 == 0:
            print(f"Train loss after iteration {i + 1}: {calculate_loss(model, train_dataset)}")
        
        # Optionally print the validation loss and the validation classification accuracy.
        if print_val and (i + 1) % 2500 == 0:
            print(f"Validation accuracy: {accuracy(model, valid_dataset, valid_labels):.3f}")
    
    print('Training ends\n')
            
    return model

# Build a model with a 10-dimensional hidden layer
model = train_model(10, print_loss=True, print_val=True)

hidden_layer_dimensions = [50, 200]
for hidden_dim in hidden_layer_dimensions:
    print(f'Hidden layer dimension: {hidden_dim}')
    model = train_model(hidden_dim, print_loss=True, print_val=True)
