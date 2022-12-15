#1. load datasets
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# print size of single image
print(images[1].shape)

#2. Training CNN models

# Define a CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, bias=True)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 100, bias=True)
        self.fc2 = nn.Linear(100, 80, bias=True)
        self.fc3 = nn.Linear(80, 10, bias=True)       
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, x):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)     
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return x
      
# Function to train the network
def train(net, trainloader, max_epoch, crit, opt, model_path='./cifar_net.pth'):

    for epoch in range(max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
        
            # Training on GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), model_path)
    print('Saved Trained Model')
    
PATH = './cifar_net.pth'
epoch = 2

# initialize model
net = Net()

# Training on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train(net, trainloader, epoch, criterion, optimizer, PATH)

# function to calculate accuracy
def print_accuracy(net, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # Inference on GPU
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the %d test images: %d %%' % (total,
        100 * correct / total))

# load trained model then test
net.load_state_dict(torch.load(PATH))
print_accuracy(net, testloader)

#3. Train inception module
'''
About parameter
in_planes : # of input channel
n1xn1 : # of output channel for first branch
n3xn3_blue : # of output channel for second branch's 1x1 conv layer
n3xn3 : # of output channel for second branch
n5xn5_blue : # of output channel for third branch's 1x1 conv layer
n5xn5 : # of output channel for third branch
pool_planes : # of output channel for fourth branch

'''
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3_blue, n3x3, n5x5_blue, n5x5, pool_planes):
        super(Inception, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n1x1, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=n1x1),
            nn.ReLU()
        ) 
        
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n3x3_blue, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=n3x3_blue),
            nn.ReLU(),
            nn.Conv2d(in_channels=n3x3_blue, out_channels=n3x3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=n3x3),
            nn.ReLU()
        ) 

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n5x5_blue, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=n5x5_blue),
            nn.ReLU(),
            nn.Conv2d(in_channels=n5x5_blue, out_channels=n5x5, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(num_features=n5x5),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_planes, out_channels=pool_planes, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=pool_planes),
            nn.ReLU()
        ) 
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)
  

#4. design a better net
## Define a CNN model
class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # conv - pool - conv - conv - pool 구조.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, stride=1, bias=True), # 8 filters, 9x9 conv layer
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # 3x3 max pooling
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=1, stride=1, padding=1, bias=True),   #  32 filters, 1x1 conv layer 
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True), # 128 filters, 3x3 conv layer
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 3x3 max pooling
        )
        self.inception1 = Inception(in_planes=128, n1x1=64, n3x3_blue=32, n3x3=64, n5x5_blue=32, n5x5=64, pool_planes=64)    # 1 : 1 : 1 : 1 비율 output
        self.inception2 = Inception(in_planes=256, n1x1=64, n3x3_blue=32, n3x3=64, n5x5_blue=32, n5x5=64, pool_planes=64)    # 1 : 1 : 1 : 1 비율 output
        self.inception3 = Inception(in_planes=256, n1x1=128, n3x3_blue=64, n3x3=96, n5x5_blue=64, n5x5=96, pool_planes=128)   # 4 : 3 : 3 : 4 비율 output
        self.inception4= Inception(in_planes=448, n1x1=128, n3x3_blue=64, n3x3=128, n5x5_blue=64, n5x5=128, pool_planes=128) # 1 : 1 : 1 : 1 비율 output
        self.inception5= Inception(in_planes=512, n1x1=128, n3x3_blue=64, n3x3=128, n5x5_blue=64, n5x5=128, pool_planes=128) # 1 : 1 : 1 : 1 비율 output
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=-1)        
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, x):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        out = self.softmax(x) 
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return out
# initialize model
betternet = BetterNet()
betternet = betternet.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(betternet.parameters(), lr=0.001, momentum=0.9)

PATH = './better_net.pth'
# Train
train(betternet, trainloader, 20, criterion, optimizer, PATH)
# Test
betternet.load_state_dict(torch.load(PATH))
print_accuracy(betternet, testloader)
