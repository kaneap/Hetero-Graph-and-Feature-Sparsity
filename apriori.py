import torch
from apyori import apriori


def find_frequent_sets(data, node_type, min_support=0.1, min_confidence = 0.0, min_lift=0, max_length=None):
    # Might be able to make this faster for huge datasets
    collists = [torch.nonzero(t).squeeze(dim=1).tolist() for t in data[node_type]['x']]
    results = list(apriori(collists, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, max_length=max_length))
    #return[list(result.items) for result in results if len(result.items) > 1]
    return[list(result.items) for result in results if len(result.items) > 2]

def conflate_item_sets(data, node_type, item_sets):
    for item_set in item_sets:
        for item1 in item_set:
            indices = torch.nonzero(data[node_type].x[:,item1]).squeeze()
            for item2 in item_set:
                if item2 != item1:
                    data[node_type].x[indices, item2] = 1
