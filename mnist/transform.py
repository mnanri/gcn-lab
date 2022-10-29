import gzip
import numpy as np
import os

dataset_dir = os.getcwd() + "/mnist" # this file wants to be executed in gnn-exp directory.

# load mnist train image dataset from gz file and return matrix.
def transform_mnist_data(dataset, graphs_dir, node_features_dir):
  data = 0
  with gzip.open(dataset, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape([-1,28,28])
  data = np.where(data < 102, -1, 1000)

  for e,imgtmp in enumerate(data):
    img = np.pad(imgtmp,[(2,2),(2,2)],"constant",constant_values=(-1))
    cnt = 0

    for i in range(2,30):
        for j in range(2,30):
            if img[i][j] == 1000:
                img[i][j] = cnt
                cnt+=1

    edges = []
    # make coordinate of (x,y)
    np_coordinate = np.zeros((cnt,2))

    for i in range(2,30):
        for j in range(2,30):
            if img[i][j] == -1:
                continue

            #8近傍に該当する部分を抜き取る。
            filter = img[i-2:i+3,j-2:j+3].flatten()
            filter1 = filter[[6,7,8,11,13,16,17,18]]

            np_coordinate[filter[12]][0] = i-2
            np_coordinate[filter[12]][1] = j-2

            for tmp in filter1:
                if not tmp == -1:
                    edges.append([filter[12],tmp])

    np.save(dataset_dir + '/' + graphs_dir + '/' + str(e),edges)
    np.save(dataset_dir + '/' + node_features_dir + '/' +str(e),np_coordinate)

transform_mnist_data(dataset_dir + '/train-images-idx3-ubyte.gz', 'train_graphs', 'train_node_features')
transform_mnist_data(dataset_dir + '/t10k-images-idx3-ubyte.gz', 'test_graphs', 'test_node_features')
