import gzip
import os
from matplotlib import pyplot as plt
import numpy as np

dataset_dir = os.getcwd() + "/mnist" # this file wants to be executed in gnn-exp directory.

def transform_to_fourier(dataset, graphs_dir, node_features_dir):
  data = 0
  with gzip.open(dataset, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape([-1,28,28])

    _dataset = np.zeros((len(data), 32, 32), dtype=np.complex128)
    for i in range(len(data)):
      _dataset[i] = np.pad(data[i],[(2,2),(2,2)],"constant",constant_values=(0))

    # make mask for high pass filter
    high_mask = np.zeros((32,32))
    for x in range(32):
      for y in range(32):
        if not((x in [15,16]) and (y in [15,16])):
          high_mask[x][y] = 1

    h_dataset = np.zeros((len(data), 32, 32), dtype=np.uint8)

    for i in range(len(_dataset)):
      # 2D FFT
      _dataset[i] = np.fft.fft2(_dataset[i])
      # shift
      _dataset[i] = np.fft.fftshift(_dataset[i])
      # high pass filter
      _dataset[i] = _dataset[i] * high_mask
      # inverse shift
      _dataset[i] = np.fft.ifftshift(_dataset[i])
      # inverse 2D FFT
      _dataset[i] = np.fft.ifft2(_dataset[i])
      # get real part
      h_dataset[i] = np.abs(_dataset[i])
      h_dataset[i] = h_dataset[i].clip(0,255).astype(np.uint8)

  threshold = 102
  # data = np.where(h_dataset < threshold, 0, 1)
  data = h_dataset

  for e,imgtmp in enumerate(data):

    img = np.pad(imgtmp,[(2,2),(2,2)],"constant",constant_values=(0))

    # calculate the center of gravity
    deno = 0
    x_nume = 0
    y_nume = 0
    for i in range(34):
      for j in range(34):
        deno = deno + img[i][j]
        x_nume = x_nume + j * img[i][j]
        y_nume = y_nume + i * img[i][j]
    cx = x_nume / deno
    cy = y_nume / deno

    cnt = 0
    img = np.where(img < threshold, 0, 1)
    for i in range(2,34):
      for j in range(2,34):
        if img[i][j] == 1:
          img[i][j] = cnt
          cnt+=1

    edges = []
    # make fearures of nodes
    features = np.zeros((cnt,2))

    for i in range(2,34):
      for j in range(2,34):
        if img[i][j] == 0:
          continue

        # 8 neighbors
        filter = img[i-2:i+3,j-2:j+3].flatten()
        filter1 = filter[[6,7,8,11,13,16,17,18]]

        features[filter[12]][0] = np.sqrt((i-cy)**2 + (j-cx)**2)
        features[filter[12]][1] = np.arctan2(i-cy,j-cx)

        for tmp in filter1:
          if not tmp == 0:
            edges.append([filter[12],tmp])

    np.save(dataset_dir + '/' + graphs_dir + '/' + str(e),edges)
    np.save(dataset_dir + '/' + node_features_dir + '/' +str(e),features)
