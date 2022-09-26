import torch

# 0. cuda
  torch.cuda.is_available() # cuda 가 사용가능한지 확인.

# 1-1. tensor 만들기 - automatic
  x = torch.ones(5, 3)    # 0 으로 initialize 된  5 x 3 tensor 생성.
  y = torch.zeros(5, 3)   # 1 으로 initialize 된  5 x 3 tensor 생성.
  z = torch.empty(5, 3)   # initialized 되지 않은 5 x 3 tensor 생성.
  
  x = torch.rand(5, 3)    # 0 ~ 1 사이 값으로 random 하게 initialize 된 5 x 3 tensor 생성. 
  y = torch.randn(5, 3)   # 정규분포를 따르는 값으로 random 하게 initialize 된 5 x 3 tensor 생성. 

# 1-2. tensor 만들기 - from data
  x = torch.tensor([4, 5, 6]) 
  
# 1-3. tensor 만들기 - 만들어진 tensor 와 같은 사이즈 만들기.
  y = torch.ones_like(x)
  z = torch.zeros_like(x)
  w = torch.empty_like(x)
  
  k = torch.rand_like(x, dtype=double)   # rand_like 는 dtype 정의 필수.
  h = torch.randn_like(x, dtype=double)  # randn_like 는 dtype 정의 필수.
  
  x.size()
  x.dtype
  
# 2. 
