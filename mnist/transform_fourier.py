import gzip
import os
import numpy as np

dataset_dir = os.getcwd() + "/mnist" # this file wants to be executed in gnn-exp directory.

def transform_to_fourier(dataset, graphs_dir, node_features_dir):
  data = 0
  with gzip.open(dataset, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape([-1,28,28])
    _dataset = np.zeros((len(data), 32, 32), dtype=np.complex128)
    for i in range(len(data)):
      for j in range(len(data[i])):
        _data = np.array(data[i][j], dtype=np.complex128)
        _data = np.append(0,_data)
        _data = np.append(0,_data)
        _data = np.append(_data,0)
        _data = np.append(_data,0)
        _dataset[i][j] = _data

    # make mask for high pass filter
    high_mask = np.zeros((32,32))
    center = 16
    r = 8
    for x in range(32):
      for y in range(32):
        if (x-center)**2 + (y-center)**2 > r**2:
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
      h_dataset[i] = np.real(_dataset[i])
      h_dataset[i] = h_dataset[i].clip(0,255).astype(np.uint8)

  '''
  for i in range(32):
    print(max(h_dataset[0][i]))
  '''

  data = np.where(h_dataset < 102, 0, 1)

  for e,imgtmp in enumerate(data):
    img = np.pad(imgtmp,[(2,2),(2,2)],"constant",constant_values=(0))
    cnt = 0

    for i in range(2,34):
      for j in range(2,34):
        if img[i][j] == 1:
          img[i][j] = cnt
          cnt+=1

    edges = []
    # make coordinate of (x,y)
    np_coordinate = np.zeros((cnt,2))

    for i in range(2,34):
      for j in range(2,34):
        if img[i][j] == 0:
          continue

        # 8 neighbors
        filter = img[i-2:i+3,j-2:j+3].flatten()
        filter1 = filter[[6,7,8,11,13,16,17,18]]

        np_coordinate[filter[12]][0] = i-2
        np_coordinate[filter[12]][1] = j-2

        for tmp in filter1:
          if not tmp == -1:
            edges.append([filter[12],tmp])

    np.save(dataset_dir + '/' + graphs_dir + '/' + str(e),edges)
    np.save(dataset_dir + '/' + node_features_dir + '/' +str(e),np_coordinate)

transform_to_fourier(dataset_dir + '/train-images-idx3-ubyte.gz', 'train_fourier_graphs', 'train_fourier_node_features')
transform_to_fourier(dataset_dir + '/t10k-images-idx3-ubyte.gz', 'test_fourier_graphs', 'test_fourier_node_features')
