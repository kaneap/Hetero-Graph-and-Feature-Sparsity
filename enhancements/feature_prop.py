from torch_geometric import utils
import torch

def propigate_features(data, node_type, edge_type, rev_edge_type):
    num_nodes = data[node_type].x.shape[0]
    for i in range(num_nodes):
        _, first_neighbors, _, _ = utils.k_hop_subgraph(i,1,data[edge_type].edge_index, flow='target_to_source')
        _, second_neighbors, _, _ = utils.k_hop_subgraph(first_neighbors[1],1,data[rev_edge_type].edge_index, flow='target_to_source')
        second_neighbors = second_neighbors[1]
        second_neighbors = second_neighbors[second_neighbors != i]
        for j in second_neighbors:
            both = torch.stack([data[node_type].x[i], data[node_type].x[j]])
            data[node_type].x[j] = torch.clamp(torch.sum(both, dim=0),min=0.,max=1.)