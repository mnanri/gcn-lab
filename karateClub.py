import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
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

# reed dataset
dataset = KarateClub()

print("Count of Graphs:\n>>>", len(dataset))  # 1
print("Count of Classes:\n>>>",dataset.num_classes)  # 4; each member belongs to a group

# get 1st graph
data = dataset[0]
check_graph(data)

# visualize 1st graph
nxg = to_networkx(data)

# calculate pagerank for visualization
pr = nx.pagerank(nxg)
pr_max = np.array(list(pr.values())).max()

# calculate node position for visualization
draw_pos = nx.spring_layout(nxg, seed=0)

# set color for each node
cmap = plt.get_cmap('tab10')
labels = data.y.numpy()
colors = [cmap(l) for l in labels]

# set size of figure
plt.figure(figsize=(10, 10))

# draw graph
nx.draw_networkx_nodes(nxg,
                       draw_pos,
                       node_size=[v / pr_max * 1000 for v in pr.values()],
                       node_color=colors, alpha=0.5)
nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

plt.title('KarateClub')
plt.show()
