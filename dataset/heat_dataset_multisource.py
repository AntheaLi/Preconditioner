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
from utils.data_config import heat_train_config, heat_test_config
from base.base_dataset import FEMDataset

class HeatDatasetMultiSource(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None, use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        diffusivities=config['diffusivities'] if config is not None else [100.0]
        total_data_num=config['total_data_num'] if config is not None else 5000
        file_data_num=config['num_time_steps'] * config['num_inits'] if config is not None else 50
        name=config['name'] if config is not None else 'circle_low_res'
        split=config['split'] if config is not None else 'train'
        epsilon=0.5 # percentage of high frequency data for augmenting with high frequency data
        # base_dir = '/data/yichenl/diffusivity_{}/' + name + '/'

        if split == 'train':
            base_dir = './dataset/diffusivity_{}/' + name + '/'
        else:
            base_dir = './dataset/diffusivity_{}/' + name + '_test/'



        self.graphs = []
        domain_files_paths = []
        num_data_to_load_per_domain = total_data_num // len(diffusivities) // file_data_num
        # to form training data from the training and testing config
        for diffusivity in diffusivities:
            cur_diffusivity_dir = base_dir.format(diffusivity)
            cur_diffu_domain_files = os.listdir(cur_diffusivity_dir)
            selected_domain_files = np.random.choice(cur_diffu_domain_files, num_data_to_load_per_domain)
            
            if config['split'] == 'test': print(selected_domain_files)

            for cur_domain_file in selected_domain_files:

                data = np.load(cur_diffusivity_dir + cur_domain_file, allow_pickle=True)
                ind = 0
                data_shape = data[0]['x'].shape[0]
                
                for d in data:
                    if use_pred_x:
                        x = torch.cat([d['x'][:,:2], d['u'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                    else:
                        x = d['x']
                    edge_attr = d['edge_attr']
                    edge_index = d['edge_index']

                    graph_data = Data(x=x, # d['x'], \
                                    edge_attr=edge_attr, 
                                    edge_index=edge_index,
                                    y=d['rhs'],
                                    u=d['u'],
                                    diag=d['diag'].reshape(-1, 1),
                                    # diag = d['A_diag'],
                                    r=d['r'],
                                    rhs=d['rhs'],
                                    u_next=d['u_next'])
                    # if augment_edge: graph_data = self.transform(graph_data)
                    self.graphs.append(graph_data)
                    
                    ind += 1

       
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1
        self.dirichlet_idx = 3
        self.b_dim = self.graphs[0].x.shape[0]
        if split == "test":
            self.graphs = self.graphs[:use_data_num]
        else:
            self.graphs = self.graphs[:use_data_num]
            # self.graphs = self.graphs[:total_data_num]

        # try on only one data
        # self.graphs = [self.graphs[0]]
        
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
    train_dataset = HeatDatasetMultiSource(name="circle_low_res", config=heat_train_config)
    test_dataset = HeatDatasetMultiSource(name="circle_low_res", config=heat_test_config)
