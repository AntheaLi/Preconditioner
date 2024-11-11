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
from utils.data_config import wave_train_config, wave_test_config
from base.base_dataset import FEMDataset



class WaveDatasetMultiSource(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None, use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        speed=config['speed'] if config is not None else [0.1]
        total_data_num=config['total_data_num'] if config is not None else 5000
        file_data_num=config['num_time_steps'] * config['num_inits'] if config is not None else 50
        name=config['name'] if config is not None else 'circle_low_res'
        split=config['split'] if config is not None else 'train'
        
        # base_dir = '/data/yichenl/diffusivity_{}/' + name + '/'

        if split == 'train':
            base_dir = '/home/yichenl/source/gnn/phys_sim/torch/dataset/speed_{}/' + name + '/'
        else:
            base_dir = '/home/yichenl/source/gnn/phys_sim/torch/dataset/speed_{}/' + name + '_test/'


        self.graphs = []
        domain_files_paths = []
        num_data_to_load_per_domain = total_data_num // len(speed) // file_data_num
        # to form training data from the training and testing config
        for C in speed:
            cur_speed_dir = base_dir.format(C)
            cur_speed_domain_files = os.listdir(cur_speed_dir)
            selected_domain_files = np.random.choice(cur_speed_domain_files, num_data_to_load_per_domain)
            
            if config['split'] == 'test': print(selected_domain_files)

            for cur_domain_file in selected_domain_files:

                data = np.load(cur_speed_dir + cur_domain_file, allow_pickle=True)
                ind = 0
                data_shape = data[0]['x'].shape[0]
                
                for d in data:
                    if use_pred_x: # replace rhs with x 
                        rhs_dirichlet = d['rhs'] * d['x'][:,5].unsqueeze(-1)
                        x = torch.cat([d['x'][:,:7], rhs_dirichlet], dim=-1)
                    else:
                        x = d['x']
                
                    dirichlet_mask = d['x'][:,5]
                    dirichlet_node = torch.nonzero(dirichlet_mask)
                

                    # edge_attr[:, 2] = edge_attr[:, 2] * 100
                    rhs = d['rhs'] 
                    rhs[dirichlet_node] = 0.0
                    diag = d['diag']
                    diag[dirichlet_node] = 1.0
                    lhs = d['lhs'] 
                    lhs[dirichlet_node] = 0.0
                    edge_index = d['edge_index']
                    zero_mask = torch.logical_or(dirichlet_mask[edge_index[0, :]], dirichlet_mask[edge_index[1, :]])
                    one_mask = torch.logical_and(edge_index[0, :] == edge_index[1,:], zero_mask)
                    edge_attr = d['edge_attr']
                    edge_attr[:, 1] = edge_attr[:, 1] 
                    edge_attr[:, 1][zero_mask] = 0
                    edge_attr[:, 1][one_mask] = 1
                    # graph_data = Data(x=x.double(), # d['x'], \ 
                    #                 edge_attr=edge_attr[:,:2].double(), 
                    #                 edge_index=edge_index.long(),
                    #                 y=rhs.double(),
                    #                 diag = diag.double(),
                    #                 r=d['r'].double(),
                    #                 rhs=rhs.double(),
                    #                 prev=d['prev'].double(),
                    #                 u_dot_next=lhs.double(),
                    #                 x_next=lhs.double(),
                    #                 u_value_next=d['u_next'].double(),
                    #                 u_next=lhs.double()) # the u_next here is actually u_dot next which is the x in Ax=b


                    graph_data = Data(x=x, # d['x'], \ 
                                    edge_attr=edge_attr[:, :2], 
                                    edge_index=edge_index.long(),
                                    y=rhs,
                                    diag = diag,
                                    r=d['r'],
                                    rhs=rhs,
                                    prev=d['prev'],
                                    u_dot_next=lhs,
                                    x_next=lhs,
                                    u_value_next=d['u_next'],
                                    u_next=lhs) # the u_next here is actually u_dot next which is the x in Ax=b
                    self.graphs.append(graph_data)

       
        # prepare data related model parameters
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1 # u_dot
        self.dirichlet_idx = 5
        self.b_dim = self.graphs[0].x.shape[0]
        if split == "test":
            self.graphs = self.graphs[:use_data_num]
        else:
            self.graphs = self.graphs[:total_data_num]


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
    train_dataset = WaveDatasetMultiSource(name="circle_low_res", config=wave_train_config)
    test_dataset = WaveDatasetMultiSource(name="circle_low_res", config=wave_test_config)
