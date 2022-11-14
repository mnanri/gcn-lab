from torch_geometric.datasets import AmazonProducts

def check_large_graph(data):
  '''Show Large Graph Information'''
  print("Count of Nodes:\n>>>", data.num_nodes) # 1,569,960
  print("Count of Edges:\n>>>", data.num_edges) # 264,339,468
  print("Count of Features in a Node:\n>>>", data.num_node_features) # 200
  print("Is There Isorated Nodes?:\n>>>", data.has_isolated_nodes()) # False
  print("Is There Self-loops?:\n>>>", data.has_self_loops()) # True

dataset = AmazonProducts(root='./amazon_products')
print("Count of Graphs:\n>>>", len(dataset))  # 1
print("Count of Classes:\n>>>",dataset.num_classes) # 107

data = dataset[0]
check_large_graph(data)
print("=====Success: Amazon Products Dataset Download=====")
