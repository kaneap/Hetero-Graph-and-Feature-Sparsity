import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from utils.save_model import save_model
from tqdm import trange
import os
from enhancements import new_edges
from enhancements.feature_prop_2 import propagate_features
from modules.heteroGNN_new_edges import HeteroGNN

def train_epoch(data, model, optimizer, target_node_type):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type].train_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def valid_epoch(data, model, target_node_type):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type].val_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    return float(loss)


@torch.no_grad()
def get_results(split, data, model, target_node_type):
    masks = {'train': 'train_mask', 'val': 'val_mask', 'test':'test_mask'}
    mask_name = masks[split]
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    mask = data[target_node_type][mask_name]
    accuracy = accuracy_score(data[target_node_type].y[mask].cpu(), pred[mask].cpu())

    # TODO: parametarize average
    # TODO: should zero division be something else?
    prec, recall, f1, support = precision_recall_fscore_support(data[target_node_type].y[mask].cpu(), pred[mask].cpu(), average='weighted', zero_division=0.0)
    #print(prec, recall, f1, support)
    cur_results = {f'{split}_acc': accuracy,f'{split}_prec': prec,f'{split}_recall': recall ,f'{split}_f1':f1 ,f'{split}_sup': support}
    return cur_results


def train(data, device, model, optimizer, epochs, experiment_name, target_node_type):
    directory = 'graph-learning/saved_models'
    train_results, val_results = {},{}
    train_losses, val_losses = [],[]
    progress_bar = trange(epochs)
    for epoch in progress_bar:
        train_loss = train_epoch(data=data, model=model, optimizer=optimizer, target_node_type=target_node_type)
        train_losses.append(train_loss)
        val_loss = valid_epoch(data, model, target_node_type)
        val_losses.append(val_loss)
        if val_loss <= np.min(val_losses):
            # save the model
            save_model(model, directory=directory, file_name=experiment_name)
            best_epoch = epoch
            train_results = get_results(split='train', data=data, model=model, target_node_type=target_node_type) | {'train_loss': train_loss}
            val_results = get_results(split='val', data=data, model=model, target_node_type=target_node_type) | {'val_loss': val_loss}
        progress_bar.set_description(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')
    
    path = os.path.join(directory, experiment_name +'.pt')
    model.load_state_dict(torch.load(path))
    test_results = get_results(split='test', data=data, model=model, target_node_type=target_node_type)
    return train_results | val_results | test_results | {'best_epoch': best_epoch}

def train_2(data, device, args, experiment_name, target_node_type):
    '''
    Alternate Training Loop for continuous enhancement of graph
    '''
    num_classes = len(torch.unique(data[target_node_type].y))
    path = [('author','to','paper'),('paper','to','conference')]
    model = HeteroGNN(data.metadata(), 
                      hidden_channels=50, 
                      out_channels=num_classes, 
                      num_layers=2, 
                      target_node_type=target_node_type, 
                      struct_emb_in_size=20,
                      struct_emb_out_size=5,
                      path=path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = model.to(device)

    directory = 'graph-learning/saved_models'
    train_results, val_results = {},{}
    train_losses, val_losses = [],[]
    progress_bar = trange(args.number_of_epochs)
    for epoch in progress_bar:
        train_loss = train_epoch(data=data, model=model, optimizer=optimizer, target_node_type=target_node_type)
        train_losses.append(train_loss)
        val_loss = valid_epoch(data, model, target_node_type)
        val_losses.append(val_loss)
        if val_loss <= np.min(val_losses):
            # save the model
            save_model(model, directory=directory, file_name=experiment_name)
            best_epoch = epoch
            train_results = get_results(split='train', data=data, model=model, target_node_type=target_node_type) | {'train_loss': train_loss}
            val_results = get_results(split='val', data=data, model=model, target_node_type=target_node_type) | {'val_loss': val_loss}
        progress_bar.set_description(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')
    
    path = os.path.join(directory, experiment_name +'.pt')
    model.load_state_dict(torch.load(path))
    test_results = get_results(split='test', data=data, model=model, target_node_type=target_node_type)
    return train_results | val_results | test_results | {'best_epoch': best_epoch}