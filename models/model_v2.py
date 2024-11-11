''' Predict diagonal element and lower triangular matrix L with I on the diagonal entries separately
    This makes: A = L D L.T where L is lower triangular matrix with I on diagonal entries
'''
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
import torch_geometric
from torch_geometric.utils import segregate_self_loops, sort_edge_index
import torch_scatter
import math

from .network_module import MLP, MeshMP, MP, MLPTanh

class Net(nn.Module):
    def __init__(self, 
            args,
            # data attributes:
            in_dim_node, 
            in_dim_edge, 
            out_dim,
            b_dim,
            num_edges=31260, 
            # encoding attributes:
            out_dim_node=128, out_dim_edge=128, 
            hidden_dim_node=128, hidden_dim_edge=128,
            hidden_layers_node=2, hidden_layers_edge=2,
            # graph processor attributes:
            num_iterations=30,
            hidden_dim_processor_node=128, hidden_dim_processor_edge=128, 
            hidden_layers_processor_node=2, hidden_layers_processor_edge=2,
            norm_type='LayerNorm',
            # decoder attributes:
            hidden_dim_decoder=128, hidden_layers_decoder=2,
            dirichlet_idx=3,
            global_pool=False,
            #other:
            **kwargs):
        '''
        MLP
        # input data:
        in_dim_node: mesh node attribute u_t
        in_dim_edge: mesh edge length
        out_dim: dim for u_t, 
        b_dim: dim for x_t+1
        
        # encoder:
        out_dim_node out_dim_edge:
        hidden_dim_node, hidden_dim_edge: latent dimension
        hidden_layers_node, hidden_layers_edge: number of hidden layers

        # processor: 
        num_iterations, number of message passing iterations
        hidden_dim_processor_node=128, hidden_dim_processor_edge: latent dimension
        hidden_layers_processor_node=2, hidden_layers_processor_edge: number of hidden layers

        # decoder:
        hidden_dim_decoder: latent dimension
        hidden_layers_decoder: number of hidden layers
        '''
        
        super(Net, self).__init__()

        if 'heat' in args.dataset: 
            self.pde = 'heat' 
        elif 'flow' in args.dataset:
            self.pde = 'flow'
        elif 'wave' in args.dataset:
            self.pde = 'wave'
        else:
            self.pde = 'syn'
        self.dirichlet_idx = dirichlet_idx
        self.node_encoder = MLPTanh(in_dim_node, out_dim_node, 
            hidden_dim_node, hidden_layers_node, norm_type)
        self.edge_encoder = MLPTanh(in_dim_edge, out_dim_edge, 
            hidden_dim_edge, hidden_layers_edge, norm_type)
        # self.attention = nn.Linear(out_dim_node, 1)
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(MeshMP(out_dim_node, out_dim_edge,
                out_dim_node, out_dim_edge,
                hidden_dim_processor_node, hidden_dim_processor_edge, 
                hidden_layers_processor_node, hidden_layers_processor_edge, norm_type))
        for i in range(num_iterations - 1):
            self.mp_layers.append(MeshMP(out_dim_node, out_dim_edge,
                out_dim_node, out_dim_edge,
                hidden_dim_processor_node, hidden_dim_processor_edge, 
                hidden_layers_processor_node, hidden_layers_processor_edge, norm_type))

        # decode graph.node_feature to b --> supervision
        # decode graph.edge_feature to A^-1, matrix multiplication and aggregation
        self.node_decoder_x = MLP(hidden_dim_processor_node, out_dim, hidden_dim_decoder, hidden_layers_decoder, norm_type=None) 
        self.node_decoder_r = MLP(hidden_dim_processor_node, 1, hidden_dim_decoder, hidden_layers_decoder, norm_type=None) 
        # TODO: will need to refill these numbers here depending on training data
        edge_decoder_dim = hidden_dim_processor_edge 
        if args.use_global:        
            self.avgp = torch.nn.AvgPool1d(num_edges, num_edges)
            edge_decoder_dim += hidden_dim_processor_edge 
        self.edge_decoder_L = MLPTanh(edge_decoder_dim, 1, hidden_dim_decoder, hidden_layers_decoder, norm_type=None) # must be None
        self.edge_decoder_D = MLPTanh(edge_decoder_dim, 1, hidden_dim_decoder, hidden_layers_decoder, norm_type=None) # must be None
        self.postprocess = MP() # non parameter
        
    def forward(self, node_attr, edge_attr, edge_index, diag=None, input_r=None, input_x=None, batch_idx=None, \
                include_r=False, use_global=False, diagonalize=False, use_pred_x=False):
        node_encoder_feature = self.node_encoder(node_attr)
        edge_encoder_feature = self.edge_encoder(edge_attr)
        dirichlet_mask = node_attr[:, self.dirichlet_idx] # [bs*num_nodes, ]
        dirichlet_mask = dirichlet_mask.to(torch.bool)
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr=edge_attr)
        zero_mask = torch.logical_or(dirichlet_mask[edge_index[0, :]], dirichlet_mask[edge_index[1, :]])

        x = node_encoder_feature
        edge_feature = edge_encoder_feature
        
        for mp_l in self.mp_layers:
            x, edge_feature = mp_l(x, edge_index, edge_feature)
        decoded_x = self.node_decoder_x(x)

        if include_r:
            decoded_r = self.node_decoder_r(x)
            r_x = input_x * decoded_r
            r_x[dirichlet_mask] = 0.0
        else:
            decoded_r = torch.zeros_like(input_r, device=input_r.device)
            r_x = 0.0

        if use_global:
            edge_avg = self.avgp(edge_feature.permute(1,0)) # 16, batchsize
            batch_size = len(torch.unique(batch_idx))
            num_edges = edge_feature.shape[0] // batch_size
            edge_avg_pad = torch.repeat_interleave(edge_avg, torch.tensor([num_edges] * batch_size, device=edge_attr.device), dim=-1)
            global_padded_edge_feature = torch.cat([edge_feature, edge_avg_pad.permute(1, 0)], dim=-1)
            decoded_L = self.edge_decoder_L(global_padded_edge_feature) 
        else:
            decoded_L = self.edge_decoder_L(edge_feature) # [E, 1]


        factor = 1.0
        if self.pde == 'heat':
            diag_ele = edge_attr[:, -1] + edge_attr[:, -2]
            diag_ele = diag_ele[(edge_index[0,:] == edge_index[1, :]).T]
        elif self.pde == 'flow':
            diag_ele = edge_attr[:, -1]
            diag_ele = diag_ele[(edge_index[0,:] == edge_index[1, :]).T]
        elif self.pde == 'wave':
            diag_ele = edge_attr[:, 1]
            diag_ele = diag_ele[(edge_index[0,:] == edge_index[1, :]).T]
            factor = 1
        else:
            diag_ele = edge_attr[:,-1]
            diag_ele = diag_ele[(edge_index[0,:] == edge_index[1, :])]
        diag_ele = diag_ele.reshape(-1,1)


        mean_edge_index, decoded_L_mean = torch_geometric.utils.to_undirected(edge_index, decoded_L, reduce='mean' )
        # to enforce a diagonal validity of the matrix
        if self.pde == 'wave':
            decoded_L_mean = torch.abs(decoded_L_mean)
        decoded_L_mean[mean_edge_index.T[:, 0] < mean_edge_index.T[:, 1]] = 0
        decoded_L_mean[mean_edge_index.T[:, 0] == (mean_edge_index.T[:, 1] )] = torch.sqrt(diag_ele) 
        decoded_edge_indices = mean_edge_index
        decoded_L = decoded_L_mean
        
        if not use_pred_x:
            decoded_x = input_x
        LTx = self.postprocess(decoded_x, decoded_L, decoded_edge_indices)

        swap_mapping = torch.tensor([[0, 1], [1, 0]]).to(edge_index.device)
        trans_edge_index = decoded_edge_indices.clone()
        trans_edge_index[swap_mapping[:, 0]] = decoded_edge_indices[swap_mapping[:, 1]]
        LLTx = self.postprocess(LTx, decoded_L, trans_edge_index)


        b_pred_flattened = LLTx 

        if self.pde == 'wave':
            b_pred_flattened[dirichlet_mask] = 0
        else:    
            b_pred_flattened[dirichlet_mask] = input_x[dirichlet_mask]

        if not use_pred_x:
            output_x = torch.zeros_like(input_x)
        else:
            output_x = decoded_x

        reverse_factor=1
        return b_pred_flattened, ((decoded_L, diag_ele, reverse_factor), decoded_edge_indices), output_x
