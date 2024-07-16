from torch_geometric import datasets
import torch_geometric.transforms as T


def get_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        dataset = datasets.DBLP('./data/dblp', transform=T.Constant(node_types='conference'))[0]
    elif dataset_name == 'flikr':
        dataset = datasets.Flickr('./data/flikr')[0].to_heterogeneous()
    elif dataset_name == 'imdb':
        dataset = datasets.IMDB('./data/imdb')[0]
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return dataset

def get_target_node_type(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        node_type = 'author'
    elif dataset_name == 'flikr':
        node_type = '0'
    elif dataset_name == 'imdb':
        node_type = 'movie'
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return node_type

def get_lp_edge_types(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        edge_type=('author', 'to', 'paper')
        rev_edge_type=('paper', 'to', 'author')
    elif dataset_name == 'flikr':
        edge_type=rev_edge_type=('0', '0', '0')
    elif dataset_name == 'imdb':
        edge_type = ('movie', 'to', 'actor')
        rev_edge_type = ('actor', 'to', 'movie')
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return edge_type, rev_edge_type

def get_edge_types_to_remove(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        edge_types = [("paper", "to", "author"), ("paper", "to", "term"), ("paper", "to", "conference")]
        rev_edge_types = [("author", "to", "paper"), ("term", "to", "paper"), ("conference", "to", "paper")]
    elif dataset_name == 'flikr':
        edge_types=rev_edge_types=('0', '0', '0')
    elif dataset_name == 'imdb':
        edge_types = [('movie', 'to', 'actor'), ('movie', 'to', 'director')]
        rev_edge_types = [('actor', 'to', 'movie'), ('director', 'to', 'movie')]
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return edge_types, rev_edge_types


def get_node_types_to_remove(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        node_types = ['author', 'paper', 'term']
    elif dataset_name == 'flikr':
        node_types = ['0']
    elif dataset_name == 'imdb':
        node_types = ['movie', 'actor', 'director']
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return node_types


def get_node_types_to_enhance(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'dblp':
        node_types = ['author', 'paper', 'term', 'conference']
    elif dataset_name == 'flikr':
        node_types = ['0']
    elif dataset_name == 'imdb':
        node_types = ['movie', 'actor', 'director']
    else:
        raise ValueError(f'Unsupported dataset name: {dataset_name}')
    return node_types
    