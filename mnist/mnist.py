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
epoch_num = 100

# model by GCN
class Net(torch.nn.Module):
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
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = self.conv2(x, edge_index)
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
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    return x

def load_mnist_graph(data_size, dataset, graphs_dir, node_feattures_dir):
  data_list = []
  labels = 0
  with gzip.open(dataset, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

  for i in range(data_size):
    edge = torch.tensor(np.load(graphs_dir+str(i)+'.npy').T,dtype=torch.long)
    x = torch.tensor(np.load(node_feattures_dir+str(i)+'.npy')/28,dtype=torch.float)

    d = Data(x=x, edge_index=edge.contiguous(),t=int(labels[i]))
    data_list.append(d)
    if i%1000 == 999:
      print("\rData loaded "+ str(i+1), end="  ")

  print("Complete!")
  return data_list

# learning part
def main():
  # load data
  train_set = load_mnist_graph(train_size, dataset='./mnist/train-labels-idx1-ubyte.gz', graphs_dir='./mnist/train_graphs/', node_feattures_dir='./mnist/train_node_features/')
  test_set = load_mnist_graph(test_size, dataset='./mnist/t10k-labels-idx1-ubyte.gz', graphs_dir='./mnist/test_graphs/', node_feattures_dir='./mnist/test_node_features/')

  print("train set size: >>> ", len(train_set))
  print("test set size: >>>", len(test_set))

  # set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # set model
  model = Net().to(device)
  # print(model)

  # set model as train mode
  model.train()

  # get graph data
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=batch_size)

  # set optimizer
  optimizer = torch.optim.Adam(model.parameters())

  # set loss function
  criterion = nn.CrossEntropyLoss()

  history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
  }

  print("==========Start Training==========")
  # learning loop part by epoch
  for epoch in range(epoch_num):
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
      batch = batch.to('cpu')
      optimizer.zero_grad()
      outputs = model(batch)
      loss = criterion(outputs,batch.t)
      loss.backward()
      optimizer.step()

      train_loss += loss.cpu().item()
      if i % 10 == 9:
        progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
        print('\repoch: {:d} loss: {:.3f}  {}'.format(epoch + 1, loss.cpu().item(), progress_bar), end="  ")

    print('\repoch: {:d} loss: {:.3f}'.format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
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
    history["test_loss"].append(loss.cpu().item()/batch_num)
    endstr = ' '*max(1,(train_size//1000-39))+"\n"
    print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
    print(f'Test Loss: {loss.cpu().item()/batch_num:.3f}',end=endstr)

  print("==========Finish Training==========")

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
