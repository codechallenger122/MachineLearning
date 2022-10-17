# 0. 용어
"""
1) batch size ? 
   전체 트레이닝 데이터 셋을 여러 작은 그룹을 나누었을 때, 
   batch size는 하나의 소그룹에 속하는 데이터 수를 의미합니다.

2) epoch ?
   epoch는 전체 트레이닝 셋이 신경망을 통과한 횟수 의미합니다. 
   예를 들어, 1-epoch는 전체 트레이닝 셋이 하나의 신경망에 적용되어 
   순전파와 역전파를 통해 신경망을 한 번 통과했다는 것을 의미합니다.
   --> 전체 training set 이 NN 을 한번 통과한 횟수.

3) iteration ?
  iteration은 1-epoch를 마치는데 필요한 미니배치 갯수를 의미합니다. 
  다른 말로, 1-epoch를 마치는데 필요한 파라미터 업데이트 횟수 이기도 합니다.
  예를 들어, 700개의 데이터를 100개씩 7개의 미니배치로 나누었을때, 
  1-epoch를 위해서는 7-iteration이 필요하며 7번의 파라미터 업데이트가 진행됩니다.
"""
# =================================================
# 1. 데이터 로딩 & 인풋 포맷을 원하는 형태로 변경하기.
for X, Y in data_loader:
  # 입력 이미지를 [batch_size × 784]의 크기로 reshape
  # 레이블은 원-핫 인코딩
  X = X.view(-1, 28*28)
  # batch_size * channel * x_size * y_size 인 input 을 
  # barch_size * (x_size*y*size) 로 resizing 한다.
  
  # 위의 코드에서 X는 for문에서 호출될 때는 (배치 크기 × 1 × 28 × 28)의 크기를 가지지만, 
  # view를 통해서 (배치 크기 × 784)의 크기로 변환됩니다.

# =================================================
# 2. Cuda 사용하기.
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# =================================================
# 3. seed 고정 하기 -- 재현해보기 위함.

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# hyperparameters
training_epochs = 15
batch_size = 100

# =================================================
# 4. dataSet load 하기

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# =================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. Network 정의.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()    # (1) constructor
        self.linear = nn.Linear(10, 5) # (2) self.linear 정의 --> nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x) # (2) 에서 정의한 self.linear 호출.
        return x
   
# 2. Network initiate  
net = Net() # net 생성.
# 여기서부터 additional info.
params = list(net.parameters()) # net에 포함된 parameter 들 보여줌.


inputs = torch.randn(1, 10)  # input 이 10행 1열.
target = torch.randn(5)      # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output --> 5행 1열의 output 생성.

# 3. Criterian = Loss function 정의.
criterion = nn.MSELoss()

# 4. Optimizer 정의.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 5. in your training loop:
optimizer.zero_grad()   # zero the gradient buffers, training 시 gradient buffer 를 항상 zero-grad로 해주지 않으면 gradient 가 stacking 된다.
output = net(inputs)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update   
