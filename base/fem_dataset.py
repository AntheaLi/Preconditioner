
import os
import sys
import numpy as np
from pathlib import Path

import torch 
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RadiusGraph, Cartesian, Distance, Compose, KNNGraph, Delaunay, ToUndirected
from torch_geometric.utils import to_networkx

from warnings import warn

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from utils.data_utils import *
import numpy as np

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../python/example/phys_gnn/')
sys.path.append('../python/py_phys_sim/')
sys.path.append('../python/')

from py_phys_sim.common.common import print_error
from py_phys_sim.common.project_path import root_path
from finite_element_graph import FiniteElementGraph


class FEMDataset(Dataset):
    ''' Base Dataset that deals only with mesh related 
    '''
    def __init__(self, domain_files_path=None, name=None, data_features=None):
        
        self.domain_files = []    
        if domain_files_path is None and name is None:
            self.domain_files.append(str(Path(root_path) / 'asset' / 'mesh' / '2d' / '{}.obj'.format('circle_low_res')))
        elif domain_files_path is not None: 
            if domain_files_path.endswith('txt'):
                with open(domain_files_path, 'r') as f:
                    for line in f:
                        l = line.strip()
                        self.domain_files.append(str(Path(root_path) / 'asset' / 'mesh' / '2d' / '{}.obj'.format(l)))
            else:
                raise Exception(f'data set type {domain_files_path} is not defined')
        elif name is not None:
            self.domain_files.append(str(Path(root_path) / 'asset' / 'mesh' / '2d' / '{}.obj'.format(name)))

        self.data_features = ['node_pos',
                            'edge_len',
                            'node_interior_mask',
                            'edge_index']
                            
        if data_features is not None:
            self.data_features = data_features

        self.node_pos = []
        self.edge_index = []
        self.interior_node_mask = []
        self.edge_len = []
        start_counter = 0
        for domain_file in self.domain_files:
            # nodes, edge_index, interior_node_mask, edge_len, boundary_nodes, faces
            nodes, edge_index, interior_node_mask, edge_len, _, _ = self.load_finite_elements(domain_file, start_counter=start_counter)
            self.node_pos.extend(nodes)
            self.edge_index.extend(edge_index)
            self.interior_node_mask.extend(interior_node_mask)
            self.edge_len.extend(edge_len)
            start_counter = len(self.node_pos)

        self.node_pos = np.array(self.node_pos)
        self.edge_index = np.array(self.edge_index)
        self.interior_node_mask = np.array(self.interior_node_mask)[..., np.newaxis]
        self.edge_len = np.array(self.edge_len)

    def to(self, int_dtype, float_dtype, device):
        for g in self.graphs:
            g.to(int_dtype, float_dtype, device)
    

    def load_finite_elements(self, obj_file_name, start_counter=0):
        # read obj file
        with open(obj_file_name, 'r') as f:
            lines = f.readlines()

        v = []
        f = []
        boundary_edges = set()
        for l in lines:
            l = l.strip()
            if l.startswith('v '):
                words = l.split()
                if len(words) != 4:
                    print_error('[load_finite_elements]: invalid vertex line.')
                vx = float(words[1])
                vy = float(words[2])
                v.append([vx, vy])
            if l.startswith('f '):
                words = l.split()
                if len(words) != 4:
                    print_error('[load_finite_elements]: invalid face line.')
                nodes = [int(n.split('/')[0]) - 1 for n in words[1:]]
                f.append(nodes)
                for i in range(3):
                    ni = nodes[i]
                    nj = nodes[(i + 1) % 3]
                    pi, pj = ni, nj 
                    if ni > nj:
                        pi, pj = nj, ni 
                    if (pi, pj) in boundary_edges:
                        boundary_edges.discard((pi, pj))
                    else:
                        boundary_edges.add((pi, pj))
        nodes = to_np_float(v)
        faces = to_np_int(f)

        boundary_nodes = set()
        for pi, pj in boundary_edges:
            boundary_nodes.add(pi)
            boundary_nodes.add(pj)
        boundary_nodes = to_np_int(sorted(list(boundary_nodes)))
        
        interior_node_mask = np.array([True] * nodes.shape[0])
        interior_node_mask[boundary_nodes] = False

        edge_index = set()
        for e in faces:
            for i in range(faces.shape[1]):
                for j in range(i + 1, faces.shape[1]):
                    ei, ej = e[i], e[j]
                    edge_index.add((ei , ej ))
                    edge_index.add((ej , ei ))
        for i in range(nodes.shape[0]):
            edge_index.add((i , i ))

        edge_index = np.array(list(edge_index))
        e0 = nodes[edge_index[:, 0] ]
        e1 = nodes[edge_index[:, 1] ]
        edge_len = np.sqrt(np.sum((e0 - e1) ** 2, axis=1))[..., np.newaxis]

        edge_index += [start_counter, start_counter]
        return nodes, edge_index, interior_node_mask, boundary_nodes, edge_len, faces

    def get_nodes(self, nodes_input, idx):
        edges = torch.from_numpy(self.edge_index[idx]).view(-1).long()
        out = torch.from_numpy(nodes_input)[edges]
        return out

    def get_edges(self, edge_input, idx):
        out = torch.from_numpy(edge_input)[idx]
        return out

    def save(self, save_dir, filename='data.npy'):
        data = np.array([x.__dict__ for x in self.graphs])
        np.save(save_dir+f'/{filename}', data)        

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, idx):
        
        data_feats = ()
        for feat in self.data_features:
            
            if feat == 'node_pos':
                out = self.get_nodes(self.node_pos, idx)
                data_feats = data_feats + (out,)

            elif feat == 'edge_len':
                out = self.get_edges(self.edge_len, idx)
                data_feats = data_feats + (out,)

            elif feat == 'node_interior_mask':
                out = self.get_nodes(self.interior_node_mask, idx)
                data_feats = data_feats + (out,)

            elif feat == 'edge_index':
                out = self.get_edges(self.edge_index, idx)
                data_feats = data_feats + (out,)

            else:
                raise Exception(f' data feature {feat} is not defined! ')


        return data_feats


