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
## GCN MNIST classification task
```
train dataset: 60000
test dataset: 10000
convolution layer: 6
affine layer: 2
epoch count: 100
final accuracy: 97.78%
```

## CNN MNIST classification task
```
train dataset: 60000
test dataset: 10000
convolution layer: 2
linear layer: 2
epoch count: 20
final accuravy: 98.32%
```
detail data
```
epoch: 1 loss: 0.311  Test Accuracy: 95.83 %%  Test Loss: 0.136
epoch: 2 loss: 0.122  Test Accuracy: 96.57 %%  Test Loss: 0.112
epoch: 3 loss: 0.094  Test Accuracy: 97.33 %%  Test Loss: 0.085
epoch: 4 loss: 0.078  Test Accuracy: 97.38 %%  Test Loss: 0.086
epoch: 5 loss: 0.071  Test Accuracy: 98.01 %%  Test Loss: 0.075
epoch: 6 loss: 0.064  Test Accuracy: 98.01 %%  Test Loss: 0.071
epoch: 7 loss: 0.059  Test Accuracy: 97.95 %%  Test Loss: 0.067
epoch: 8 loss: 0.053  Test Accuracy: 97.86 %%  Test Loss: 0.072
epoch: 9 loss: 0.048  Test Accuracy: 97.94 %%  Test Loss: 0.071
epoch: 10 loss: 0.047  Test Accuracy: 98.08 %%  Test Loss: 0.068
epoch: 11 loss: 0.043  Test Accuracy: 98.21 %%  Test Loss: 0.060
epoch: 12 loss: 0.040  Test Accuracy: 98.22 %%  Test Loss: 0.063
epoch: 13 loss: 0.039  Test Accuracy: 98.11 %%  Test Loss: 0.068
epoch: 14 loss: 0.038  Test Accuracy: 98.14 %%  Test Loss: 0.073
epoch: 15 loss: 0.036  Test Accuracy: 98.19 %%  Test Loss: 0.062
epoch: 16 loss: 0.033  Test Accuracy: 98.22 %%  Test Loss: 0.061
epoch: 17 loss: 0.032  Test Accuracy: 98.22 %%  Test Loss: 0.066
epoch: 18 loss: 0.031  Test Accuracy: 98.31 %%  Test Loss: 0.073
epoch: 19 loss: 0.029  Test Accuracy: 98.45 %%  Test Loss: 0.068
epoch: 20 loss: 0.029  Test Accuracy: 98.44 %%  Test Loss: 0.063
```
