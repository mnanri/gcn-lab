import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon

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

    hidden_size = 640
    hidden_size_diff = 128
    hidden_size_list = [hidden_size]
    for _ in range(4):
      hidden_size -= hidden_size_diff
      hidden_size_list.append(hidden_size)

    self.conv1 = GCNConv(dataset.num_node_features, hidden_size_list[0])
    self.conv2 = GCNConv(hidden_size_list[0], hidden_size_list[1]) # 512
    self.conv3 = GCNConv(hidden_size_list[1], hidden_size_list[2]) # 384
    self.conv4 = GCNConv(hidden_size_list[2], hidden_size_list[3]) # 256
    self.conv5 = GCNConv(hidden_size_list[3], hidden_size_list[4]) # 128
    self.conv6 = GCNConv(hidden_size_list[4], dataset.num_classes)

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

    return F.log_softmax(x, dim=1)

dataset = Amazon(root='./amazon', name='Computers')
print("Count of Graphs:\n>>>", len(dataset))  # 1
print("Count of Classes:\n>>>",dataset.num_classes)  # 10

data = dataset[0]
check_large_graph(data)
print("=====Success: Amazon Computers Dataset Download=====")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

history = {
  "epoch": [],
  "loss": [],
}
epoch_num = 1000
for epoch in range(epoch_num):
  optimizer.zero_grad()
  out = model(data)
  loss = F.nll_loss(out, data.y)
  loss.backward()
  optimizer.step()

  history["epoch"].append(epoch+1)
  history["loss"].append(loss.item())
  print("Epoch: %d, Loss: %.4f" % (epoch+1, loss.item()))

fig, ax = plt.subplots()
ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid()
ax.plot(history["epoch"], history["loss"], color="red")
fig.tight_layout()
fig.savefig("./amazon/amazon_computers_loss.png")

model.eval()

_, pred = model(data).max(dim=1)
err = 0.0
for i, p in enumerate(pred):
  if p != data.y[i]:
    err += 1
print("Accuracy: {:.2f}%".format((1 - err / data.num_nodes) * 100))
