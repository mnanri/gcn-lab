import urllib.request
import os

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
  'train_img':'train-images-idx3-ubyte.gz',
  'train_label':'train-labels-idx1-ubyte.gz',
  'test_img':'t10k-images-idx3-ubyte.gz',
  'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.getcwd() + "/mnist" # this file wants to be executed in gnn-exp directory.

for v in key_file.values():
  file_path = dataset_dir + '/' + v
  urllib.request.urlretrieve(url_base + v, file_path)
