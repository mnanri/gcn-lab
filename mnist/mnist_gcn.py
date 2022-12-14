import time
from matplotlib import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import matplotlib.pyplot as plt
import gzip

train_size = 60000
test_size = 10000
batch_size = 100
epoch_num = 20

# model by GCN
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = GCNConv(2, 16)
    self.conv2 = GCNConv(16, 32)
    self.conv3 = GCNConv(32, 48)
    self.conv4 = GCNConv(48, 64)
    self.conv5 = GCNConv(64, 96)
    self.conv6 = GCNConv(96, 128)

    self.linear1 = torch.nn.Linear(128,64)
    self.linear2 = torch.nn.Linear(64,10)

  def forward(self, data):
    # print("data.x.size(): ",data.size())
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)
    # print("after conv1: ",x.size()) # torch.Size([-1, 16])
    x = F.relu(x)
    x = self.conv2(x, edge_index)
    # print("after conv2: ",x.size()) # torch.Size([-1, 32])
    x = F.relu(x)
    x = self.conv3(x, edge_index)
    x = F.relu(x)
    x = self.conv4(x, edge_index)
    x = F.relu(x)
    x = self.conv5(x, edge_index)
    x = F.relu(x)
    x = self.conv6(x, edge_index)
    x = F.relu(x)
    x, _ = scatter_max(x, data.batch, dim=0)
    # print("after scatter_max: ",x.size()) # torch.Size([100, 128])
    x = self.linear1(x)
    # print("after linear1: ",x.size()) # torch.Size([100, 64])
    x = F.relu(x)
    x = self.linear2(x)
    # print("after linear2: ",x.size()) # torch.Size([100, 10])
    return x

def load_mnist_graph(data_size, dataset, graphs_dir, node_features_dir):
  data_list = []
  labels = 0
  with gzip.open(dataset, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

  for i in range(data_size):
    edge = torch.tensor(np.load(graphs_dir+str(i)+'.npy').T,dtype=torch.long)
    x = torch.tensor(np.load(node_features_dir+str(i)+'.npy')/28,dtype=torch.float)

    d = Data(x=x, edge_index=edge.contiguous(),t=int(labels[i]))
    data_list.append(d)
    if i%1000 == 999:
      print("\rData loaded "+ str(i+1), end="  ")

  print("Complete!")
  return data_list

# learning part
def main():
  # load data for first mnist dataset
  # train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_graphs/', node_features_dir='./mnist/train_node_features/')
  # test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_graphs/', node_features_dir='./mnist/test_node_features/')

  # load data for fourier transformed mnist dataset
  # train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_fourier_graphs/', node_features_dir='./mnist/train_fourier_node_features/')
  # test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_fourier_graphs/', node_features_dir='./mnist/test_fourier_node_features/')

  # load data for fourier transformed mnist dataset and polar based features
  # train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_fourier_graphs/', node_features_dir='./mnist/train_fourier_polar_node_features/')
  # test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_fourier_graphs/', node_features_dir='./mnist/test_fourier_polar_node_features/')

  # load data for rotated forier transformed mnist dataset and polar based features
  train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_rotate_fourier_graphs/', node_features_dir='./mnist/train_rotate_fourier_node_features/')
  test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_rotate_fourier_graphs/', node_features_dir='./mnist/test_rotate_fourier_node_features/')

  # load data for fourier transformed fourier spectlum mnist dataset
  # train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_fourier_spectrum_graphs/', node_features_dir='./mnist/train_fourier_spectrum_node_features/')
  # test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_fourier_spectrum_graphs/', node_features_dir='./mnist/test_fourier_spectrum_node_features/')

  print("train set size: >>> ", len(train_set))
  print("test set size: >>>", len(test_set))

  # set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # set model
  model = Net().to(device)

  # set model as train mode
  model.train()

  # get graph data
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=batch_size)

  # set optimizer
  optimizer = torch.optim.Adam(model.parameters())

  # set loss function
  criterion = nn.CrossEntropyLoss()

  # define struct for history of loss and accuracy
  history = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
  }

  print("==========Start Training==========")
  # learning loop part by epoch
  start = time.time()
  for epoch in range(epoch_num):
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
      batch = batch.to(device)
      optimizer.zero_grad()
      outputs = model(batch)
      loss = criterion(outputs,batch.t)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      if i % 10 == 9:
        progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
        print('\repoch: {:d} loss: {:.3f}  {}'.format(epoch + 1, loss.item(), progress_bar), end="  ")

    print('\repoch: {:d} loss: {:.3f}'.format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
    history["epoch"].append(epoch+1)
    history["train_loss"].append(train_loss / (train_size / batch_size))

    correct = 0
    total = 0
    batch_num = 0
    loss = 0
    with torch.no_grad():
      for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        loss += criterion(outputs,data.t)
        _, predicted = torch.max(outputs, 1)
        total += data.t.size(0)
        batch_num += 1
        correct += (predicted == data.t).sum().cpu().item()

    history["test_acc"].append(correct/total)
    history["test_loss"].append(loss.item()/batch_num)
    endstr = ' '*max(1,(train_size//1000-39))+"\n"
    print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
    print(f'Test Loss: {loss.item()/batch_num:.3f}',end=endstr)

    if epoch == 0:
      print("Time for 1 epoch: {:.2f} sec".format(time.time()-start))

  end = time.time()
  print("==========Finish Training==========")
  print("Total time: {:.2f} sec".format(end-start))

  fig1, ax1 = plt.subplots()
  ax1.set_title('Loss [train=blue, test=orange]')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')
  ax1.grid()
  ax1.plot(history['epoch'], history["train_loss"], color="orange",label="train")
  ax1.plot(history['epoch'], history["test_loss"], color="blue",label="test")
  fig1.tight_layout()
  fig1.savefig('./mnist/loss_gcn.png')
  fig1.show()

  fig2, ax2 = plt.subplots()
  ax2.set_title('Accuracy')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('accuracy')
  ax2.grid()
  ax2.plot(history['epoch'], history["test_acc"], color="black")
  fig2.tight_layout()
  fig2.savefig('./mnist/accuracy_gcn.png')
  fig2.show()

  # output final result
  correct = 0
  total = 0

  with torch.no_grad():
    for data in test_loader:
      data = data.to(device)
      outputs = model(data)
      _, predicted = torch.max(outputs, 1)
      total += data.t.size(0)
      correct += (predicted == data.t).sum().cpu().item()
  print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

main()
