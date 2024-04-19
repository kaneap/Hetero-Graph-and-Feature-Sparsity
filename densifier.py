import torch
from heteroGNN import HeteroGNN
import torch.nn.functional as F
from torch_geometric.datasets import DBLP

import torch_geometric.transforms as T


dataset = DBLP('./data/dblp', transform=T.Constant(node_types='conference'))
data = dataset[0]

node_type = 'author'

print(torch.bincount(torch.sum(data[node_type].x, 1).to(torch.int)))


# find the sparse nodes
sparse_threshold = 3
is_sparse = torch.sum(data[node_type].x, 1).to(torch.int) <= sparse_threshold
print(is_sparse)