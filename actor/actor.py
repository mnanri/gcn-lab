from torch_geometric.datasets import Actor

def check_large_graph(data):
  '''Show Large Graph Information'''
  print("Count of Nodes:\n>>>", data.num_nodes) # 7600
  print("Count of Edges:\n>>>", data.num_edges) # 30019
  print("Count of Features in a Node:\n>>>", data.num_node_features) # 932
  print("Is There Isorated Nodes?:\n>>>", data.has_isolated_nodes()) # False
  print("Is There Self-loops?:\n>>>", data.has_self_loops()) # True

dataset = Actor(root='./actor')
print("Count of Graphs:\n>>>", len(dataset))  # 1
print("Count of Classes:\n>>>",dataset.num_classes) # 5

data = dataset[0]
check_large_graph(data)
print("=====Success: Actor Dataset Download=====")
