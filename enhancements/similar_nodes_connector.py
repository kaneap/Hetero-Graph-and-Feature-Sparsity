from torch_geometric import utils
import torch

def connect_similar_nodes(data, node_type, edge_type, threshold):
    '''
    connects nodes of type node_type if they are similar enough with an edge of edge_type
    '''
    i1s, i2s = [], []
    for i in range(data[node_type].x.shape[0]):
        similarities = torch.cosine_similarity(data[node_type].x[1], data[node_type].x)
        indices = torch.arange(data[node_type].x.shape[0])

        i2 = indices[similarities > threshold]
        if(i2.shape[0] > 1):
            i1 = torch.ones_like(i2)
            i1s.append(i1)
            i2s.append(i2)

    if len(i1s):
        i1s = torch.cat(i1s)
        i2s = torch.cat(i2s)
        new_edge_index = torch.stack([i1s, i2s])
        new_edge_index, _ = utils.remove_self_loops(new_edge_index)
        data[edge_type].edge_index = new_edge_index
