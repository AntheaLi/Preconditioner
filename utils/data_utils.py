import numpy as np

import torch

# Define the data type we plan to use.
np_float_dtype = np.float64
torch_float_dtype = torch.float64
np_int_dtype = np.int64
torch_int_dtype = torch.int64
torch_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def to_np_float(x):
    return np.asarray(x, dtype=np_float_dtype).copy()

def to_np_int(x):
    return np.asarray(x, dtype=np_int_dtype).copy()

def to_torch_float(x, requires_grad):
    return torch.tensor(to_np_float(x), dtype=torch_float_dtype, requires_grad=requires_grad)

def to_torch_bool(x):
    return torch.tensor(x, dtype=torch.bool, requires_grad=False)

def to_torch_int(x, requires_grad):
    return torch.tensor(to_np_int(x), dtype=torch_int_dtype, requires_grad=requires_grad)

def torch_to_np_float(x):
    return to_np_float(x.clone().cpu().detach().numpy())

def torch_to_np_int(x):
    return to_np_int(x.clone().cpu().detach().numpy())

from torch_geometric.utils import sort_edge_index, to_undirected
from itertools import combinations

def twohop(edge_index):
    # edge_index = edge_index
    sorted_edge_index = sort_edge_index(edge_index)
    sorted_edge_index_list = list(sorted_edge_index.T)
    all_edges = [(x.item(), y.item()) for x, y in sorted_edge_index_list]
    dct = dict((x.item(), []) for x, y in sorted_edge_index_list)
    
    for x, y in all_edges:
        dct[x].append(y)

    two_hop_edges = []

    for x in dct:
        y_list = dct[x]
        all_hops = list(combinations(y_list, 2))
        for hop in all_hops:
            if hop[0] < x and hop[1] < x and hop not in all_edges:
                two_hop_edges.append(hop)

    two_hop_edges = torch.tensor(two_hop_edges, device=edge_index.device).long()
    return two_hop_edges.T



    
