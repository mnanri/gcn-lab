
import gzip
import random
from matplotlib import pyplot as plt

import numpy as np

def show_data(dataset):
  data = 0
  with gzip.open(dataset, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)

    k = 0
    k = random.randint(0, len(data)-1)
    plt.imsave('./mnist/original_data.png', data[k], cmap='gray')

    p_data = np.zeros((32, 32), dtype=np.complex128)
    for l in range(len(data[k])):
      _data = np.array(data[k][l], dtype=np.complex128)
      _data = np.append(0,_data)
      _data = np.append(0,_data)
      _data = np.append(_data,0)
      _data = np.append(_data,0)
      p_data[l+2] = _data

    plt.imsave('./mnist/zero_padding.png', np.real(p_data), cmap='gray')
    tmp_data = np.where(p_data < 102, 0, 216)
    plt.imsave('./mnist/black_white.png', np.real(tmp_data), cmap='gray')

    # make mask for high pass filter
    high_mask = np.zeros((32,32))
    # center = 16
    # r = 1
    for x in range(32):
      for y in range(32):
        if not((x in [15,16]) and (y in [15,16])): # (x-center)**2 + (y-center)**2 > r**2:
          high_mask[x][y] = 1

    # print(high_mask[16])

    # 2D FFT
    p_data = np.fft.fft2(p_data)
    # shift
    p_data = np.fft.fftshift(p_data)
    # show FFT image
    tmpimg = np.abs(p_data)
    plt.imsave('./mnist/original_fft.png', tmpimg, cmap='gray')
    # high pass filter
    p_data = p_data * high_mask
    tmpimg = np.abs(p_data)
    plt.imsave('./mnist/filtered_fft.png', tmpimg, cmap='gray')
    # inverse shift
    p_data = np.fft.ifftshift(p_data)
    # inverse 2D FFT
    p_data = np.fft.ifft2(p_data)
    # get real part
    f_data = np.zeros((32, 32), dtype=np.uint8)
    f_data = np.abs(p_data)
    # print(f_data.max())
    f_data = f_data.clip(0,255).astype(np.uint8)

    plt.imsave('./mnist/high_pass_filter.png', f_data, cmap='gray')

show_data('./mnist/train-images-idx3-ubyte.gz')
