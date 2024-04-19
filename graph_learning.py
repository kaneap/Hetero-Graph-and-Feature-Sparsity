import torch
import random
import numpy as np
from heteroGNN import HeteroGNN
import torch.nn.functional as F

def set_seed(seed=42):
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

def init_parameters(data, device):
    model = HeteroGNN(data.metadata(), hidden_channels=50, out_channels=4, num_layers=2)
    #model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3, data=data)
    data, model = data.to(device), model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    return model, optimizer


def train_epoch(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_epoch(data, model):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

def train(data, model, epochs):
    return