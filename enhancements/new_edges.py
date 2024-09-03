import torch
import torch.utils
from torch_geometric.utils import to_torch_sparse_tensor


def add_new_edges(data,device,from_type, search_levels=2, num_paths=2):
    paths = choose_edge_types(data, from_type, search_levels, num_paths)
    new_edge_types = []
    for i, path in enumerate(paths):
        new_edge_type = (from_type, f'add_{i}', from_type)
        new_edge_types.append(new_edge_type)
        _add_path_shortcut(data, device, path, new_edge_type)
    return new_edge_types

def _add_path_shortcut(data, device, path, new_edge_type):
    multiplied_tensor = get_ext_adjacency_matrix(data.edge_index_dict, data.x_dict, path)
    
    new_edges = make_similar_edges(multiplied_tensor, min_similarity=0.99, device=device)
    data[new_edge_type].edge_index = new_edges
    return data

def get_ext_adjacency_matrix(edge_index_dict, x_dict, path):
    device = x_dict[path[0][0]].get_device()
    adjacency_tensors = []
    for edge in path:
        sizes = x_dict[edge[0]].shape[0], x_dict[edge[2]].shape[0]
        adjacency_tensor = to_torch_sparse_tensor(edge_index_dict[edge], size=sizes).detach().clone()
        adjacency_tensors.append(adjacency_tensor)

    multiplied_tensor = torch.eye(adjacency_tensors[0].shape[0], device=device)

    for adjacency_tensor in adjacency_tensors:
        multiplied_tensor = torch.mm(multiplied_tensor, adjacency_tensor)
  
    multiplied_tensor = multiplied_tensor.to_dense()
    return multiplied_tensor

def make_similar_edges(adjacency_matrix, min_similarity, device):
    adjacency_matrix = adjacency_matrix.to(device)
    
    new_edges = []
    size = adjacency_matrix.shape[0]
    for i in torch.arange(size, device=device):
        index_2 = torch.arange(i+1, size, device=device)
        index_1 = i * torch.ones_like(index_2, device=device)
        first_embeddings = adjacency_matrix[index_1]
        second_embeddings = adjacency_matrix[index_2]
        similarities = torch.nn.functional.cosine_similarity(first_embeddings, second_embeddings, dim=1)
        indices = torch.stack([index_1, index_2])
        new_edges.append(indices[:,similarities > min_similarity])
    new_edges = torch.concat(new_edges, 1)
    return new_edges
    '''
    indices = torch.combinations(torch.arange(size)).to(device)
    first_embeddings = adjacency_matrix[indices[:,0]]
    second_embeddings = adjacency_matrix[indices[:,1]]
    similarities = torch.nn.functional.cosine_similarity(first_embeddings, second_embeddings, dim=1)
    #return indices[similarities > min_similarity].T
    '''

def make_similar_edges_diff(adjacency_matrix, min_similarity, device):
    adjacency_matrix = adjacency_matrix.to(device)
    
    new_edges = []
    size = adjacency_matrix.shape[0]
    for i in torch.arange(size, device=device):
        index_2 = torch.arange(i+1, size, device=device)
        index_1 = i * torch.ones_like(index_2, device=device)
        first_embeddings = adjacency_matrix[index_1]
        second_embeddings = adjacency_matrix[index_2]
        similarities = torch.nn.functional.cosine_similarity(first_embeddings, second_embeddings, dim=1) - min_similarity
        indices = torch.stack([index_1, index_2])
        new_edges.append(indices[:,similarities > min_similarity])
    new_edges = torch.concat(new_edges, 1)
    return new_edges

def _get_edges_from(edge_types, from_type):
    return [edge_type for edge_type in edge_types if edge_type[0] == from_type]


def _find_potential_nodes(data, from_type, search_levels):
    if search_levels == 0:
        return [list()]

    edge_types = _get_edges_from(data.edge_types, from_type)
    path_list= list()
    for edge_type in edge_types:
        next_edges = _find_potential_nodes(data, edge_type[2], search_levels-1)
        next_edges = [[edge_type] + e for e in next_edges]
        path_list.extend(next_edges)
    return path_list

def choose_edge_types(data, from_type, search_levels=2, num_paths=2):
    '''
    Choose the edge types to use for the method.
    Leverage the edge types with the lowest number of instances.
    '''
    path_list = _find_potential_nodes(data, from_type, search_levels=search_levels)
    #filter to top num_paths items
    path_list = sorted(path_list, key=lambda item: data[item[-1][2]].x.shape[0], reverse=False)[:num_paths]
    return list(path_list)