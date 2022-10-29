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
