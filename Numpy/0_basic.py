import numpy as np

# ===========================================================================================
# 1. Axis 개념
"""
 axis 개념이 중요.
 np.ndarray 의 제일 바깥이 axis-0, 안쪽으로 들어갈 수록 axis-1, axis-2, ... 된다고 보면 됨.
"""

a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2]) # <class 'numpy.ndarray'>    (3,)     1 2 3
a[0] = 5    # Change an element of the array
print(a)    # [5 2 3]         

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array  
print(type(b), b.shape) # <class 'numpy.ndarray'> (2, 3)
print(b[0, 0], b[0, 1], b[1, 0]) # 1, 2, 4

"""
참고로, shape 기준으로 보면
1차원 : (n, )      <-- (axis-0) <-- 열
2차원 : (n, m)     <-- (axis-0, axis-1) <-- 행, 열
3차원 : (n, m, k)  <-- (axis-0, axis-1, axis-2) <-- 높이, 행, 열
n차원 : (axis-0, axis-1, ... , axis-(n-1))
"""

# ===========================================================================================
# 2. ones, zeros, full, eye, random.random
a = np.zeros((2,2))  # Create an array of all zeros
[[0. 0.]
 [0. 0.]]

b = np.ones((1,2))   # Create an array of all ones
[[1. 1.]]

c = np.full((2,2), 7) # Create a constant array
[[7 7]
 [7 7]]

d = np.eye(3)        # Create a 2x2 identity matrix
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

e = np.random.random((2,2)) # Create an array filled with random values, 0과 1사이 값.
[[0.75669026 0.08726535]
 [0.27874284 0.67490899]]

# ===========================================================================================
# 3. array indexing

"""
  1. 콜론  = ":"  indexing 을 사용하면 차원이 유지된다.
     예를 들어 아래 a 를 example 로 두면,
    a[:], a[:2], a[:,:2], a[:][:2] 모두 차원이 유지된다.

  2. 숫자 indexing 을 사용하면 차원이 하나 감소된다.
    a : 2차원
    a[1] : 1차원
    a[1][2] : 0차원 = scalar.

  3. 위의 원리에 따라    --- case 1
    a[:2,:1] != a[:2][:1], a[:1,1] != a[:1][1] 이지만
    a[1][1] == a[1][1].
    즉 a[scalar][scalar] = a[scalar, scalar] 이고
    나머지 케이스는 a[..,..] != a[..][..]
"""

# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
[[ 1  2  3  4]
 [ 5  6  7  8]  
 [ 9 10 11 12]]

# case 1.
print(a[:2,:1])   
[[1]
 [5]]

print(a[:2][:1])
[[1 2 3 4]]

print(a[:2][1])
[5 6 7 8]

print(a[:2,1])
[2 6]

# case 2.
print(a[:], "\n") 
""" a[:] 는  
1) a의 axis-0 의 모든 element 를 포함. 
2) : operator 를 사용함으로서 차원 유지.

따라서 a[:] 는 a 와 같은 결과.
"""
print(a[:2], "\n")
""" a[:2] 는  
1) a의 axis-0 의 0, 1 번째 element 포함.
2) : operator 를 사용함으로서 차원 유지.
따라서 
[[1 2 3 4]
 [5 6 7 8]] 
"""
print(a[:,:2], "\n")
""" a[:,:2] 는
1) a의 axis-0 의 모든 element, axis-1 의 0,1 element 포함.
2) : operator 를 사용함으로서 차원 유지.

[[ 1  2]
 [ 5  6]
 [ 9 10]] 
"""
print(a[:][:2], "\n")
"""
1) a의 axis-0 의 모든 element 먼저 구함.
2) : operator 를 사용함으로서 차원 유지.
따라서 a[:] = a

1) a의 axis-0 의 0, 1 element 구함.
2) : operator 를 사용함으로서 차원 유지.
따라서 
[[1 2 3 4]
 [5 6 7 8]] 

"""
print(a[:1,:1], "\n")
"""
1) a의 axis-0 의 0 번째 element, axis-1 의 0 번째 element 구함
2) : operator 를 사용함으로서 차원 유지.
따라서

[[1]]

"""

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):

b = a[:2, 1:3] # 해석하면, axis-0 에서 0,1 element, axis-1 에서 1,2, element
print(b)
[[2 3]
[6 7]]

print(a.shape, b.shape) # (3, 4) (2, 2) 

c = a[:1] # :(콜론) 을 1개 사용하면, 차원 유지 됨
print(c) # [[1 2 3 4]]
print(a.shape, c.shape) # (3, 4) (1, 4)

d = a[0]  # 그냥 indexing 하면 차원 감소.
print(d) # [1 2 3 4]
print(a.shape, d.shape)

# example
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

row_r1 = a[1, :]    # Rank 1 view of the second row of a   [5  6  7  8]
row_r3 = a[[1], :]  # Rank 2 view of the second row of a   [[ 5  6  7  8]]
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a   [[ 5  6  7  8]]

# 즉 a[1:2] 와 a[[1]] 은 같은 표현. 
# index 위치에 : 또는 [] 가 있으면 차원이 유지된다고 보면 된다.

row_r1 = a[1, :]    # Rank 1 view of the second row of a   [ 5  6  7  8]
row_r2 = a[1:3, :]  # Rank 2 view of the second row of a   [ [ 5  6  7  8]]
row_r3 = a[[1,2], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape) 
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)

[5 6 7 8] (4,)
[[ 5  6  7  8]
 [ 9 10 11 12]] (2, 4)
[[ 5  6  7  8]
 [ 9 10 11 12]] (2, 4)
















