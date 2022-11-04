import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GitHub
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

def check_large_graph(data):
  '''Show Large Graph Information'''
  print("Count of Nodes:\n>>>", data.num_nodes)
  print("Count of Edges:\n>>>", data.num_edges)
  print("Count of Features in a Node:\n>>>", data.num_node_features)
  print("Is There Isorated Nodes?:\n>>>", data.has_isolated_nodes())
  print("Is There Self-loops?:\n>>>", data.has_self_loops())

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = GCNConv(dataset.num_node_features, 96)
    self.conv2 = GCNConv(96, 64)
    self.conv3 = GCNConv(64, 48)
    self.conv4 = GCNConv(48, 32)
    self.conv5 = GCNConv(32, 16)
    self.conv6 = GCNConv(16, 2)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index

    x = self.conv1(x, edge_index)
    x = x.relu()
    x = self.conv2(x, edge_index)
    x = x.relu()
    x = self.conv3(x, edge_index)
    x = x.relu()
    x = self.conv4(x, edge_index)
    x = x.relu()
    x = self.conv5(x, edge_index)
    x = x.relu()
    x = self.conv6(x, edge_index)

    return F.log_softmax(x, dim=1)

dataset = GitHub(root='./github')
print("Count of Graphs:\n>>>", len(dataset))  # 1
print("Count of Classes:\n>>>",dataset.num_classes) # 2

data = dataset[0]
check_large_graph(data)
print("=====Success: GitHub Dataset Download=====")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
  optimizer.zero_grad()
  out = model(data)
  loss = F.nll_loss(out, data.y)
  loss.backward()
  optimizer.step()
  print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

model.eval()

_, pred = model(data).max(dim=1)
err = 0.0
for i, p in enumerate(pred):
  if p != data.y[i]:
    err += 1
print(f"Accuracy: {(1 - err / len(pred)) * 100:.2f}%")
