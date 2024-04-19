import torch
import torch_geometric.transforms as T

def remove_features(data, probability, node_types=None):
    """
    randomly removes features from dataset with given probability
    """
    if node_types is None:
        node_types = ['author', 'paper', 'term']
    for node_type in node_types:
        node_features = data[node_type].x
        data[node_type].x = torch.where(torch.rand_like(node_features) < probability, torch.zeros_like(node_features), node_features)
    return data


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


edge_types = [("paper", "to", "author"), ("paper", "to", "term"), ("paper", "to", "conference")]
rev_edge_types = [("author", "to", "paper"), ("term", "to", "paper"), ("conference", "to", "paper")]


def remove_edges(data, probability):
    split_transform = T.RandomLinkSplit(num_val = probability, num_test=0, add_negative_train_samples=False, edge_types=edge_types, rev_edge_types=rev_edge_types)
    split_data, _, _ = split_transform(data)                       
    return split_data