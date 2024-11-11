import os
import sys
import time
import torch
import pickle
import numpy as np
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from copy import deepcopy
import matplotlib.pyplot as plt
import trimesh
import torch_geometric 
from torch_geometric.transforms import TwoHop
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce, spspmm


sys.path.append('../')
from utils.data_utils import *
from utils.data_config import poisson3d_train_config, poisson3d_test_config
from base.base_dataset import FEMDataset

class SyntheticDataset(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None,  use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        total_data_num=config['total_data_num'] if config is not None else 5000
        file_data_num=config['num_time_steps'] * config['num_inits'] if config is not None else 50
        name=config['name'] if config is not None else 'circle_low_res'
        split=config['split'] if config is not None else 'train'
        
        self.graphs = []
        ind = 0
        if split == 'train':
            base_dir = '/home/yichenl/source/gnn/phys_sim/torch/dataset/poisson_3d/' + name + '/'
        else:
            base_dir = '/home/yichenl/source/gnn/phys_sim/torch/dataset/poisson_3d/' + name + '_test/'

        num_data_to_load_per_domain = total_data_num //  file_data_num
        cur_speed_domain_files = os.listdir(base_dir)
        selected_domain_files = np.random.choice(cur_speed_domain_files, num_data_to_load_per_domain)
            
        if config['split'] == 'test': print(selected_domain_files)

        for cur_domain_file in selected_domain_files:
            data = np.load(base_dir + cur_domain_file, allow_pickle=True)

            for d in data:

                x = d['x']
                edge_attr = d['edge_attr']
                edge_index = d['edge_index']

                graph_data = Data(x=x.float(), # d['x'], \
                                edge_attr=edge_attr.float(), 
                                edge_index=edge_index,
                                y=d['rhs'].float(),
                                u=d['u'].float(),
                                diag=d['diag'].reshape(-1, 1).float(),
                                r=d['r'].float(),
                                rhs=d['rhs'].float(),
                                u_next=d['u_next'].float())
                # if augment_edge: graph_data = self.transform(graph_data
                self.graphs.append(graph_data)
                
                ind += 1

       
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1
        self.b_dim = self.graphs[0].x.shape[0]
        self.dirichlet_idx = 4

        self.graphs = self.graphs[:total_data_num]

        
    def get_data(self):
        return self.graphs

    def save(self, file_name='./data2d.npy'):
        data = np.array([x.to_dict() for x in self.graphs])
        # data = np.array([x.__dict__ for x in self.graphs])
        np.save(file_name, data)        

        with open(f'{file_name}.pickle', 'wb') as handle:
            pickle.dump(self.meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def to(self, int_dtype, float_dtype, device):
        for g in self.graphs:
            g.to(int_dtype, float_dtype, device)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
        # return self.graphs[idx] if self.transform is None else self.transform(self.graphs[idx])

if __name__ == "__main__":
    train_dataset = SyntheticDataset(name="circle_low_res", config=heat_train_config)
    test_dataset = SyntheticDataset(name="circle_low_res", config=heat_test_config)
