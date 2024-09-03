from torch_geometric import utils
import torch



def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.

    Method adapted from https://github.com/twitter-research/feature-propagation/blob/main/src/utils.py
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = utils.degree(row, num_nodes=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    prop_weights = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, prop_weights

def propagate_features(data, masks, node_type, edge_type, rev_edge_type, iters):

    num_nodes_1 = data[node_type].x.shape[0]
    num_nodes_2 = data[edge_type[2]].x.shape[0]
    features = data[node_type].x
    mask = masks[node_type] == False

    ei_0 = utils.to_torch_sparse_tensor(data[edge_type].edge_index, size=(num_nodes_1,num_nodes_2))
    ei_1 = utils.to_torch_sparse_tensor(data[rev_edge_type].edge_index, size=(num_nodes_2, num_nodes_1))
    edge_index, _ = utils.to_edge_index(torch.sparse.mm(ei_0, ei_1))


    out = features
    if mask is not None:
        out = torch.zeros_like(features)
        out[mask] = features[mask]

    edge_index, prop_weights = get_symmetrically_normalized_adjacency(edge_index, num_nodes_1)
    adj = torch.sparse.FloatTensor(edge_index, values=prop_weights, size=(num_nodes_1, num_nodes_1)).to(edge_index.device)

    for _ in range(iters):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = features[mask]
    
    data[node_type].x = out

def propagate_features_homogeneous(data, masks, node_type, edge_type, iters, device):

    num_nodes = data[node_type].x.shape[0]
    features = data[node_type].x.to(device)
    mask = masks[node_type] == False
    mask = mask.to(device)

    edge_index = data[edge_type].edge_index


    out = features
    if mask is not None:
        out = torch.zeros_like(features).to(device)
        out[mask] = features[mask]

    edge_index, prop_weights = get_symmetrically_normalized_adjacency(edge_index, num_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=prop_weights, size=(num_nodes, num_nodes)).to(device)

    for _ in range(iters):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = features[mask]
    
    data[node_type].x = out