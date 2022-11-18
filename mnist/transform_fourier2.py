import gzip
import os

import numpy as np

dataset_dir = os.getcwd() + '/mnist'

def transform_to_fourier2(dataset, graphs_dir, node_features_dir):
  data = 0
  with gzip.open(dataset, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape([-1,28,28])

    _dataset = np.zeros((len(data), 32, 32), dtype=np.complex128)
    for i,_data in enumerate(data):
      _dataset[i] = np.pad(_data,[(2,2),(2,2)],"constant",constant_values=(0))
      # 2D FFT
      _dataset[i] = np.fft.fft2(_dataset[i])
      # shift
      _dataset[i] = np.fft.fftshift(_dataset[i])

    conjugate_dataset = np.conjugate(_dataset)
    fourier_spectrum = np.abs(_dataset * conjugate_dataset)
    fourier_spectrum = np.log1p(fourier_spectrum)
    # print(fourier_spectrum.shape)

    for i in range(len(fourier_spectrum)):
      v = []
      for j in range(len(fourier_spectrum[i])):
        for k in range(len(fourier_spectrum[i][j])):
          if fourier_spectrum[i][j][k] > 13:
            v.append([j,k])

      cos = np.zeros((len(v),len(v)), dtype=np.float32)
      for j in range(len(v)):
        for k in range(j+1,len(v)):
          nume = (_dataset[i][v[j][0]][v[j][1]]*conjugate_dataset[i][v[k][0]][v[k][1]] + _dataset[i][v[k][0]][v[k][1]]*conjugate_dataset[i][v[j][0]][v[j][1]]).real
          deno = 2*np.abs(_dataset[i][v[j][0]][v[j][1]])*np.abs(_dataset[i][v[k][0]][v[k][1]])
          if deno == 0:
            continue
          if nume/deno < 1/np.sqrt(2):
            continue
          cos[j][k] = nume/deno
          cos[k][j] = cos[j][k]

      edges = []
      featuers = np.zeros((len(v),2), dtype=np.float32)

      for j in range(len(cos)):
        for k in range(len(cos[j])):
          edges.append([j,k])

      for j in range(len(v)):
        featuers[j][0] = fourier_spectrum[i][v[j][0]][v[j][1]]
        featuers[j][1] = np.angle(_dataset[i][v[j][0]][v[j][1]])

      np.save(dataset_dir + '/' + graphs_dir + '/' + str(i), edges)
      np.save(dataset_dir + '/' + node_features_dir + '/' + str(i), featuers)

transform_to_fourier2(dataset_dir + '/train-images-idx3-ubyte.gz', 'train_fourier_spectrum_graphs', 'train_fourier_spectrum_node_features')
transform_to_fourier2(dataset_dir + '/t10k-images-idx3-ubyte.gz', 'test_fourier_spectrum_graphs', 'test_fourier_spectrum_node_features')
