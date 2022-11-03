import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

def check_graph(data):
    '''Show Graph Information'''
    print("Structure of Graph:\n>>>", data)
    print("Key of Graph:\n>>>", data.keys)
    print("Count of Nodes:\n>>>", data.num_nodes)
    print("Count of Edges:\n>>>", data.num_edges)
    print("Count of Features in a Node:\n>>>", data.num_node_features)
    print("Is There Isorated Nodes?:\n>>>", data.has_isolated_nodes())
    print("Is There Self-loops?:\n>>>", data.has_self_loops())
    print("=== Features of Nodes: x ===\n", data['x'])
    print("=== Class of Nodes: y ======\n", data['y'])
    print("=== Type of Edge ===========\n", data['edge_index'])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_size = 5
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# read dataset
dataset = KarateClub()

# print("Count of Graphs:\n>>>", len(dataset))  # 1
# print("Count of Classes:\n>>>",dataset.num_classes)  # 4; each member belongs to a group

# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# produce instance of model
model = Net()

# set model as train mode
model.train()

# get 1st graph
data = dataset[0]
# check_graph(data)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# learning loop part
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    # print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

# set model as evaluation mode
model.eval()

# prediction part
# _, pred = model(data).max(dim=1)

# print("Result:\n", pred)
# print("Truth:\n", data['y'])

'''
# produce test graph data
test_dataset = KarateClub()
test_data = test_dataset[0]

x = test_data["x"]
edge_index = test_data['edge_index']

# change some edges of graph
for j in range(int(data.num_edges/10)):
    a = random.randint(0, data.num_edges-1)
    b = random.randint(0, data.num_nodes-1)
    if edge_index[0][a] == b:
        continue
    edge_index[1][a] = b

t_data = Data(x=x, edge_index=edge_index)

# prediction with test graph data
_, pred = model(t_data).max(dim=1)

print(" === Label of Former Graph =========== ")
print(data['y'])
print(" === Result of Prediction of Label === ")
print(pred)
'''

accuracy_list = []
for _ in range(data.num_nodes*10):
    test_dataset = KarateClub()
    test_data = test_dataset[0]

    x = test_data["x"]
    edge_index = test_data['edge_index']

    for _ in range(int(data.num_edges/10)):
        a = random.randint(0, data.num_edges-1)
        b = random.randint(0, data.num_nodes-1)
        if edge_index[0][a] == b:
            continue
        edge_index[1][a] = b

    t_data = Data(x=x, edge_index=edge_index)
    _, pred = model(t_data).max(dim=1)

    err = 0.0
    for j, p in enumerate(pred):
        if p != data['y'][j]:
            err += 1

    accuracy_list.append(1 - err/data.num_nodes)

print("Max Accuaracy: ", max(accuracy_list))
print("Min Accuaracy: ", min(accuracy_list))
print("Average of Accuaracy: %d%", np.mean(accuracy_list)*100)
