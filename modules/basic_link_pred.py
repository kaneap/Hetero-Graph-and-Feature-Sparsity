from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_from: Tensor, x_to: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_from[edge_label_index[0]]
        edge_feat_movie = x_to[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, metadata, hidden_channels , num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)
        self.classifier = Classifier()

    def get_embedding(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict
    
    def forward(self, data, edge_type) -> Tensor:
        from_type, to_type = edge_type[0], edge_type[2]
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        x_dict = self.get_embedding(x_dict, edge_index_dict)
        pred = self.classifier(
            x_dict[from_type],
            x_dict[to_type],
            data[edge_type].edge_label_index,
        )
        return pred