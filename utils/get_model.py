import torch
from modules.heteroGNN import HeteroGNN
from modules.hgt import HGT
from modules.gat import GAT
from torch_geometric import nn

def get_model(model_name, data, device, target_node_type, lr=0.005, weight_decay=0.001, out_channels=4):
    num_classes = len(torch.unique(data[target_node_type].y))
    if model_name == 'hetero_conv':
        model = HeteroGNN(data.metadata(), hidden_channels=50, out_channels=num_classes, num_layers=2, target_node_type=target_node_type)
    elif model_name == 'gat':
        model = GAT(data.metadata(), classif_node=target_node_type, hidden_channels=10, num_layers=4, out_channels=num_classes, dropout=0.5)
    elif model_name == 'hgt':
        model = HGT(hidden_channels=64, out_channels=num_classes, num_heads=2, num_layers=3, data=data, target_node_type=target_node_type)
    else:
        raise ValueError(f'Unsupported model name: {model_name}')
    
    data, model = data.to(device), model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer