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
```
$ python3 ./karate_club/karate_club2.py

train dataset: 1
test dataset: 34x10 # changed about 10% of edges in train data.
convolution layer: 2
epoch count: 100
average of accuracy: 93.40%
```
## GCN MNIST classification task
```
$ python3 ./mnist/mnist.py

train dataset: 60000
test dataset: 10000
convolution layer: 6
affine layer: 2
epoch count: 100
final accuracy: 97.78%
```

## CNN MNIST classification task
```
$ python3 ./mnist/mnist2.py

train dataset: 60000
test dataset: 10000
convolution layer: 2
linear layer: 2
epoch count: 20
final accuravy: 98.32%
```
detail data
```
epoch: 20 loss: 0.029  Test Accuracy: 98.44 %%  Test Loss: 0.063
```
