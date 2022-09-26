import torch

x = torch.empty(5, 3)   # initialized 되지 않은 5 x 3 tensor 생성.
y = torch.ones(5, 3)    # 0 으로 initialize 된  5 x 3 tensor 생성.
z = torch.zeros(5, 3)   # 1 으로 initialize 된  5 x 3 tensor 생성.

x = torch.rand(5, 3)    # 0 ~ 1 사이 값으로 random 하게 initialize 된 5 x 3 tensor 생성. 
y = torch.randn(5, 3)   # 정규분포를 따르는 값으로 random 하게 initialize 된 5 x 3 tensor 생성. 

