# GNN experimental inspection
process of install library for Mac
```
$ brew install python # install python for using pip3 (or pip).
$ pip3 --version # check whether pip is installed, or not. if you cannot use pip3, try to use pip.
$ pip3 install torch==1.12.0 # install PyTorch, pay attention of dependency for Pytorch-Geometric version.

# To install Pytorch-Geometric, visit following site.
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# e.g. If you install Pytorch-Geometric under the following env;
# PyTorch: PyTorch 1.12*,
# Your OS: Mac,
# Package: Pip,
# CUDA: CPU,
# you can run following command.
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

$ pip3 install networkx # install networkx to handle visualized figures.
$ pip3 install matplotlib # install matplotlib to draw figures
```
## GCN karate club classification task
The graph contains 34 nodes, connected by 156 (undirected and unweighted) edges. Every node is labeled by one of 4 classes obtained via modularity-based clustering.
```
$ python3 ./karate_club/karate_club_gcn.py

train data: {
  nodes: 34
  edges: 156
  features: 34
  classes: 4
}
convolution layer: 2
epoch count: 100
average of train accuracy: 95.65%
```
## GCN MNIST classification task
```
$ python3 ./mnist/mnist_gcn.py
```
images are not fourier transformed
```
train dataset: 60000
test dataset: 10000
convolution layer: 6
affine layer: 2

epoch count: 100
test accuracy: 97.78%

epoch count: 20
test accuracy: 94.57%
total time: 4665.24 sec
```
images are fourier transformed with high pass filter
```
train dataset: 60000
test dataset: 10000
convolution layer: 6
affine layer: 2

filter radius: 2x2
epoch count: 20
test accuracy: 94.18%
total time: 3518.64 sec
```
images are fourier transfirmed and make graphs with fourier spectrum
```
train dataset: 60000
test dataset: 10000
convolution layer: 6
affine layer: 2

epoch: 1 loss: 2.302  Test Accuracy: 11.35 %%  Test Loss: 2.301
Time for 1 epoch: 2491.03 sec # too slow
```

## CNN MNIST classification task
```
$ python3 ./mnist/mnist_cnn.py

train dataset: 60000
test dataset: 10000
convolution layer: 2
linear layer: 2
epoch count: 20
test accuracy: 98.42%
total time: 1601.62 sec
```

## GCN Amazon Computers classification task
Nodes represent products and edges represent co-purchasing relations. The task is to map products to their respective product category based on the product’s co-purchasing network.
```
$ python3 ./amazon_computers/amazon_computers.py

train dataset: {
  nodes: 13752
  egdes: 491722
  features: 767
  classes: 10
}
convolution layer: 6
epoch count: 1000
train accuracy: 87.51%
```

## GCN GitHub network classification task
Nodes represent users and edges represent following relations. The task is to predict the user’s primary programming language based on the user’s following network.
```
$ python3 ./github/github_gcn.py

train dataset: {
  nodes: 37700
  egdes: 578006
  features: 128
  classes: 2
}
convolution layer: 6
epoch count: 200
train accuracy: 87.28%
```
