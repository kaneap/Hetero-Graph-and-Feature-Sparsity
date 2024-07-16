from modules.pca import PCA
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F

def compress_add_feats(data, node_types, method=1):
    #_pca_compress_features(data, device, node_types)
    data = _add_artificial_features(data, method, node_types)
    return data

def _pca_compress_features(data, device, node_types):
    for node_type in node_types:
        data[node_type].x = PCA(18).to(device).fit_transform(data[node_type].x)
        data[node_type].x = F.normalize


def _add_artificial_features(data, method, node_types):
    if method == 1:
        add_feats = _get_pageranks(data)
        #need to unsqueeze to get the pageranks in the same format as the features
        for type, rank in add_feats.items():
            add_feats[type] = rank.unsqueeze(-1)
    elif method == 2:
        add_feats = get_degrees_one_hot(data)
    elif method == 3:
        add_feats = get_degrees_one_hot(data, bucketize=True)

    for node_type in node_types:
        data[node_type].x = torch.cat([data[node_type].x, add_feats[node_type]], dim=1)

    return data

def _get_pageranks(data):
    nx_data = to_networkx(data)
    ranks = list(nx.pagerank(nx_data, alpha=0.9, tol=1.e-08).values())

    node_types = data.node_offsets.keys()
    lower_bounds = list(data.node_offsets.values())
    upper_bounds = [i for i in lower_bounds[1:]]+ [len(ranks)]
    
    ranks = {node_type:torch.as_tensor(ranks[lower:upper]) for (node_type,lower,upper) in zip(node_types, lower_bounds, upper_bounds)}
    for rank in ranks.values():
        rank = F.normalize(rank, dim=0)
    return ranks



def get_degrees_one_hot(data, bucketize=False):
    degrees = {}
    for node_type in data.node_types:
        degrees[node_type] = torch.zeros(data[node_type].x.shape[0], device=data[node_type].x.device)

    for edge_type, edge_storage in data.edge_items():
        out_node_type = edge_type[0]
        edge_index = edge_storage['edge_index']
        degrees[out_node_type] += degree(edge_index[0], num_nodes=degrees[out_node_type].shape[0])
    max_degree = int(max([max(degrees[node_type]) for node_type in data.node_types]).item())
    buckets = torch.tensor([0,1,2,3,5,10] + list(range(20,max_degree+1,10)))
    if bucketize:
        for node_type in data.node_types:
            degrees[node_type] = torch.bucketize(degrees[node_type], boundaries=buckets)
    encodings = {}
    for node_type in data.node_types:
        encoded = F.one_hot(degrees[node_type].to(torch.int64))
        non_empty_mask = encoded.abs().sum(dim=0).bool()
        encodings[node_type] = encoded[:,non_empty_mask]

    return encodings