import torch
import torch_geometric.transforms as T

def remove_features(data, probability, node_types=None):
    """
    randomly removes features from dataset with given probability
    """
    if node_types is None:
        node_types = ['author', 'paper', 'term']
    masks = {}
    for node_type in node_types:
        node_features = data[node_type].x
        mask = torch.rand_like(node_features) < probability
        data[node_type].x = torch.where(mask, torch.zeros_like(node_features), node_features)
        masks[node_type] = mask
    return data, masks


# %%
def randomize_features(data, probability, node_types=None):
    """
    randomly replaces features with random noise with given probability
    """
    if node_types is None:
        node_types = ['author', 'paper']
    for node_type in node_types:
        feature_prob = data[node_type].x.sum().item() / data[node_type].x.flatten().size(dim=0)
        node_features = data[node_type].x
        random_ones = torch.where(torch.rand_like(node_features) < feature_prob, torch.ones_like(node_features), torch.zeros_like(node_features))
        data[node_type].x = torch.where(torch.rand_like(node_features) < probability, random_ones, node_features)
    return data


def remove_edges(data, probability, edge_types=None, rev_edge_types=None):
    if edge_types is None:
        edge_types = [("paper", "to", "author"), ("paper", "to", "term"), ("paper", "to", "conference")]
    if rev_edge_types is None:
        rev_edge_types = [("author", "to", "paper"), ("term", "to", "paper"), ("conference", "to", "paper")]
    split_transform = T.RandomLinkSplit(num_val = probability, num_test=0, add_negative_train_samples=False, edge_types=edge_types, rev_edge_types=rev_edge_types)
    split_data, _, _ = split_transform(data)    
    for edge_type in edge_types:
        del split_data[edge_type].edge_label
        del split_data[edge_type].edge_label_index                   
    return split_data