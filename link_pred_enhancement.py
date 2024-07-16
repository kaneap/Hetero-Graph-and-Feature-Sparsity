import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import tqdm
import os

from modules.basic_link_pred import Model
from utils.save_model import save_model



def main():
    # We initialize conference node features with a single one-vector as feature:
    dataset = DBLP('./data/dblp', transform=T.Constant(node_types='conference'))
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_type = ('author', 'to', 'paper')
    rev_edge_type = ("paper", "to", "author")
    predict_links(data, device, 0.9, edge_type, rev_edge_type)
    
    

def predict_links(data, device, threshold, edge_type, rev_edge_type):
    print(data)
    model = Model(data.metadata(), hidden_channels=50, num_layers=10)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=True,
        edge_types=edge_type,
        rev_edge_types=rev_edge_type
    )

    train_data, val_data, test_data = transform(data)
    train_link_pred(train_data, val_data, model, edge_type)
    with torch.no_grad():
        pred = model(train_data, edge_type)
    pred_min, pred_max = pred.min(), pred.max()

    train_and_val = torch.cat([train_data[edge_type].edge_index, val_data[edge_type].edge_index], dim=1)
    sizes = train_data[edge_type[0]].x.shape[0], train_data[edge_type[2]].x.shape[0]
    potential_links = negative_sampling(train_and_val, num_nodes=sizes, num_neg_samples=5000)

    #check which data to use as base
    train_data[edge_type].edge_label_index = potential_links
    
    #decide yes/no for random potential links
    with torch.no_grad():
        pred = model(train_data, edge_type)
        pred = scale_prediction(pred, pred_max, pred_min).detach().cpu().numpy()
    new_links = potential_links[:,pred > threshold]
    data=data.to(device)
    data[edge_type].edge_index = torch.cat([data[edge_type].edge_index, new_links],dim=1)
    rev_new_links = torch.roll(new_links,1,0)
    data[rev_edge_type].edge_index = torch.cat([data[rev_edge_type].edge_index, rev_new_links],dim=1)
    return data.cpu()

def scale_prediction(prediction, max, min):
    pred_norm = (prediction - min ) / ( max - min)
    return pred_norm

def train_epoch(data, model, optimizer, edge_type):
    optimizer.zero_grad()
    model.train()
    pred = model(data, edge_type)
    ground_truth = data[edge_type].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad
def eval_epoch(data, model, edge_type):
    model.eval()
    pred = model(data, edge_type)
    ground_truth = data[edge_type].edge_label
    score = roc_auc_score(ground_truth.cpu().numpy(), pred.cpu().numpy())
    return score


def train_link_pred(train_data, val_data, model, edge_type, epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        _ = model(train_data, edge_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    train_loss_history, val_loss_history = [],[]
    best_val_score = 0
    directory = 'graph-learning/saved_models/link_predictor'
    model_name = 'link_prediction_model'

    for epoch in tqdm.trange(epochs):
        train_loss_history.append(train_epoch(train_data, model, optimizer, edge_type))
        val_score = eval_epoch(val_data, model, edge_type)
        if val_score > best_val_score:
            
            save_model(model,  directory, file_name=model_name)
            best_val_score = val_score
        val_loss_history.append(val_score)
    path = os.path.join(directory, model_name+'.pt')
    model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    main()