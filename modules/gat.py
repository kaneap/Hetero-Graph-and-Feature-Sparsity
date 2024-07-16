import torch
from torch_geometric import nn

class GAT(torch.nn.Module):
    def __init__(self, metadata, classif_node, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()

        self.model = nn.GAT(in_channels=(-1,-1), hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, dropout=dropout, add_self_loops=False)
        self.model = nn.to_hetero(self.model, metadata)

        self.classif_node = classif_node


    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)[self.classif_node]