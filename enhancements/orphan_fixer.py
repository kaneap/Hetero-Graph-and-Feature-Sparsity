from torch_geometric import utils
import torch

def add_edge_to_orphans(data, edge_type, rev_edge_type=None):
    '''
    connects nodes (from the head of the edge type) which have a degree of zero (i.e. they are orphans)
    to the node in the tail of edge type with the most similar average features
    '''
    print(f'before fixing orphans: {data[edge_type].edge_index.shape[1]}')
    head_type, _, tail_type = edge_type

    # find head_type nodes with degree zero
    degrees = utils.degree(data[edge_type].edge_index[0], num_nodes=data[head_type].x.shape[0])
    is_orphan_mask = degrees == 0
    print(f'expected afterwards: {data[edge_type].edge_index.shape[1] + degrees[is_orphan_mask].shape[0]}')

    # get mean features of all head features connected to a tail feature
    mean_connected_features = {}
    for i in range(data[tail_type].x.shape[0]):
        # for tail at index i get all edges connecting to it
        _, edge_index, _, _ = utils.k_hop_subgraph(i, 1, data[edge_type].edge_index, directed=True, flow='source_to_target')
        to_indices = edge_index[0,:]
        if(to_indices.shape[0] != 0):
            mean_connected_features[i] = (torch.mean(data[head_type].x[to_indices], dim=0))
    

    new_head_indices, new_tail_indices = [],[]
    for i in range(data[head_type].x.shape[0]):
        # skip non-orphans
        if not is_orphan_mask[i]:
            continue

        # if this is slow, this could be better parallelized
        best_index, best_similarity = -1, 0
        for index, mean_feats in mean_connected_features.items():
            similarity = torch.cosine_similarity(data[head_type].x[i], mean_feats, dim=0).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        # connect head node i to tail node best_index
        new_head_indices.append(i)
        new_tail_indices.append(best_index)

    data[edge_type].edge_index = torch.cat(
        [data[edge_type].edge_index, torch.tensor([new_head_indices, new_tail_indices])],
        dim=1)
    if(rev_edge_type is not None):
        data[rev_edge_type].edge_index = torch.cat(
            [data[rev_edge_type].edge_index, torch.tensor([new_tail_indices, new_head_indices])],
            dim=1)
    print(f'after fixing orphans: {data[edge_type].edge_index.shape}')
    print(f'after fixing orphans: {data[rev_edge_type].edge_index.shape}')


