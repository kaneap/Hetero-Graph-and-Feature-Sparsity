import torch
from heteroGNN import HeteroGNN
import torch.nn.functional as F
from torch_geometric.datasets import DBLP

import torch_geometric.transforms as T




# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def init_parameters(data):
    model = HeteroGNN(data.metadata(), hidden_channels=10, out_channels=2, num_layers=2)
    #model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=3, data=data)
    data, model = data.to(device), model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    return model, optimizer

# %%
def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], torch.where(is_sparse,0,1)[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data, model):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == torch.where(is_sparse,0,1)[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


dataset = DBLP('./data/dblp', transform=T.Constant(node_types='conference'))
data = dataset[0]

node_type = 'author'
probability = 0.5
# Choose which features will be sparsified
is_sparse = torch.rand_like(data[node_type].y, dtype=float) > 0.5
#node_features = data[node_type].x[is_sparse]
#data[node_type].x[is_sparse] = torch.where(torch.rand_like(node_features) < probability, torch.zeros_like(node_features), node_features)




is_sparse = is_sparse.to(device)

dataset_copy = dataset.copy()
data_copy = dataset_copy[0]
data_copy.to(device)
model, optimizer = init_parameters(data_copy)
train_accs, val_accs, test_accs = [],[],[]
for epoch in range(100):
    loss = train(data=data_copy, model=model, optimizer=optimizer)
    train_acc, val_acc, test_acc = test(data = data_copy, model=model)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
best_epoch = max(enumerate(val_accs),key=lambda x: x[1])[0]
print(f'Baseline, Train: {train_accs[best_epoch]:.4f}, '
        f'Val: {val_accs[best_epoch]:.4f}, Test: {test_accs[best_epoch]:.4f}')