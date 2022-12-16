1. CONV layer
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

- in_channels  : input image 의 channel 수 
- out_channels : output 의 channel 수 = kernel 의 filter 개수
- kernel_size  : 커널 사이즈.
  
# With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)  # input_channel = 16, 3x3 필터 33개

# non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)) # input channel = 16, 3x5 필터 33개

# non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)  

2. transposed conv layer
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                         stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
