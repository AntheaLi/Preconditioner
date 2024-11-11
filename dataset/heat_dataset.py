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

#from finite_element_graph import FiniteElementGraph


def get_finite_element_graphs(domain_file, diffusivity, time_step=1e-3, num_time_steps=50, visualize=False, output_dir_name="heat"):
    visualize=False
    mesh = trimesh.load(domain_file, process=False)
    nodes = mesh.vertices
    faces = mesh.faces
    nodes = to_np_float(nodes)
    faces = to_np_int(faces)
    all_boundary_nodes = get_boundary_vertices(nodes, faces)
    nodes = nodes[:, :2]

    np.save('eight_mid_res_nodes.npy', nodes)
    np.save('eight_mid_res_elements.npy', faces)

    # Simulation options.
    sim_opt = {
        'solver': 'pcg',    # Alternative: 'pardiso'.
        'preconditioner': 'incomplete_cholesky',    # Can delete this key if using Pardiso.
        'solver_abs_tol': '1e-9',
        'solver_rel_tol': '1e-9',
        'verbose': '0',
    }

    print('building dataset ..... ')
    graphs = []
    u_list = []
    edge_attr = []
    node_attr = []
    gt_rhs = []
    input_r = []
    bound_idx = 0
    graph_len = []
    precond = []
    diag = []

    while len(graphs) < num_time_steps:
        print('[HeatEquationDataset::__init__]: {}/{} ready.'.format(len(graphs), num_time_steps))
        graph_len.append(len(graphs))
        # Randomly generate an initial u.
        num_points = 1024
        x = to_np_float(np.linspace(-1, 1, num_points))
        u = to_np_float(np.random.uniform(-1, 1, size=(num_points, num_points)))
        # Smooth out u by Gaussian kernels.
        u = to_np_float(gaussian_filter(u, sigma=num_points * 0.05))
        # Rescale it to [-1, 1]
        u_max = np.max(u[:])
        u_min = np.min(u[:])
        u = (u - u_min) / (u_max - u_min) * 2 - 1
        # Now u is the initial value at the lattice grid [-1, 1] x [-1, 1] where along each dimension
        # we have num_points points. We now want to interpolate initial values based on node positions.
        f = interpolate.interp2d(x, x, u, kind='cubic')
        initial_values = to_np_float([f(x, y) for x, y in nodes]).ravel()

        dirichlet = []
        
        for bn in all_boundary_nodes:
            # Select a random ratio of free and neumann boundaries.
            bn_num = len(bn)
            dirichlet_node_num = int(np.random.uniform(low=0.3, high=0.5) * bn_num)
            # Randomly select a starting point.
            dirichlet_node_begin = np.random.randint(bn_num)
            bn2 = bn + bn
            dirichlet += bn2[dirichlet_node_begin:dirichlet_node_begin + dirichlet_node_num]
        dirichlet = [int(i) for i in dirichlet]
        
        # Simulation.
        sim = HeatEquation2d()
        # dirichlet_node = [int(i) for i in boundary_nodes]
        dirichlet_node = dirichlet
        sim.Initialize(diffusivity, time_step, domain_file, initial_values, dirichlet_node)
        interior_mask = to_np_float(sim.interior_nodes())
        boundary_mask = to_np_float(sim.boundary_nodes())
        dirichlet_mask = to_np_float(sim.dirichlet_nodes())
        non_dirichlet_mask = 1.0 - to_np_float(sim.dirichlet_nodes())
        init_boundary = initial_values * dirichlet_mask
        
        # In the case of heat equations, we use all boundary nodes as the dirichlet nodes, but
        # other PDEs may have different dirichlet and neumann nodes.
        graph = FiniteElementGraph(nodes, faces,
            dirichlet_node=dirichlet, neumann_node=[],
            node_attr_name=['u', 'u_next', 'rhs'],
            edge_attr_name=['mass', 'stiffness'],
            element_attr_name=[])

        # Specify mass and stiffness.
        edge_idx = to_np_int(graph.edge_index)
        e = StdIntMatrixX2d(edge_idx.shape[0])
        for i, (e0, e1) in enumerate(edge_idx):
            e[i] = [int(e0), int(e1)]

        e0 = nodes[graph.edge_index[:, 0] ]
        e1 = nodes[graph.edge_index[:, 1] ]
        edge_len = np.sqrt(np.sum((e0 - e1) ** 2, axis=1))[..., np.newaxis]
        # edge_attr [edge_len, M, K, r]
        M_idx = graph.edge_attr_name_index['mass']
        graph.edge_attr[:, M_idx] = to_np_float(sim.GetMassElements(e))
        K_idx = graph.edge_attr_name_index['stiffness']
        graph.edge_attr[:, K_idx] = to_np_float(sim.GetStiffnessElements(e))
        u_idx = graph.node_attr_name_index['u']
        u_next_idx = graph.node_attr_name_index['u_next']

        # Simulation until the results no longer change or until we have collected enough data.
        values = initial_values.copy()
        frame_idx = 0
        while len(graphs) < num_time_steps:
            results = sim.Forward(values, sim_opt)
            new_values = to_np_float(results[0])
            # iterations = results[1]
            # CT = np.array(sim.upper())
            # C = np.array(sim.lower())
            
            # Check if we need to break.
            if np.max(np.abs(values - new_values)) < 1e-2:
                break

            # Create the data item that characterize the simuation from values[-1] to new_values.
            graph_t = deepcopy(graph)
            # Specify u and u_next.
            graph_t.node_attr[:, u_idx] = values.copy()
            graph_t.node_attr[:, u_next_idx] = new_values.copy()

            # Specify rhs.
            rhs_left = to_np_float(sim.ComputeMassVectorProduct(values)) * non_dirichlet_mask
            rhs_right = (to_np_float(sim.ComputeMassVectorProduct(init_boundary)) + \
                to_np_float(sim.ComputeStiffnessVectorProduct(init_boundary))) * non_dirichlet_mask
            rhs = rhs_left - rhs_right + init_boundary
            graph_t.node_attr[:, graph_t.node_attr_name_index['rhs']] = rhs

            r = (graph.edge_attr[:, M_idx] + graph.edge_attr[:, K_idx])**2
            # boundary_elements = [True if (e0 in dirichlet or e1 in dirichlet) else False for e0, e1 in graph.edge_index]
            boundary_elements = np.logical_or(dirichlet_mask[graph_t.edge_index[:, 0]], dirichlet_mask[graph_t.edge_index[:, 1]])
            r[boundary_elements] = 0 # set edges with boundary nodes to 0
            r = 1.0 / np.max([1.0, np.sqrt(r.sum())])

            edge_identity_mask = (graph_t.edge_index[:, 0] == graph.edge_index[:, 1])
            r = edge_identity_mask * r
            r = r[..., np.newaxis]

            # r = ( graph.edge_attr[:, K_idx])**2
            # dirichlet_elements = [True if (graph_t.node_dirichlet_mask[e0] or graph_t.node_dirichlet_mask[e1]) else False for e0, e1 in graph.edge_index]
            # r[dirichlet_elements] = 0 # set edges with boundary nodes to 0
            # r = 1.0 / np.max([1.0, np.sqrt(r.sum())])
            # r = np.array([[r]]*graph.edge_attr.shape[0])
            
            node_attr.append(np.hstack([nodes, \
                                        # graph_t.node_attr[:, u_idx][..., np.newaxis], \
                                        graph_t.node_attr[:, u_next_idx][..., np.newaxis], \
                                        dirichlet_mask[..., np.newaxis]]))
            edge_attr.append(np.hstack([edge_len, \
                                        graph.edge_attr[:, M_idx][..., np.newaxis], \
                                        graph.edge_attr[:, K_idx][..., np.newaxis], \
                                        # graph_t.edge_dual_index, \
                                        # edge_identity_mask[..., np.newaxis]
                                        ]))
            # gt.append(np.hstack([graph_t.node_attr[:, u_next_idx][..., np.newaxis]]))
            u_list.append(np.hstack([graph_t.node_attr[:, u_idx][..., np.newaxis]]))
            # precond.append(np.vstack( (CT[np.newaxis, ...], C[np.newaxis, ...]) ))
            # precond.append(np.array(sim.upper()))
            # precond.append(np.array(load_sparse_matrix('upper.txt')[-1])) # upper is actually lower
            A = graph.edge_attr[:, M_idx] + graph.edge_attr[:, K_idx]
            from torch_geometric.utils import to_dense_adj
            A = to_dense_adj(torch.from_numpy(graph.edge_index).T.long(),   edge_attr=torch.from_numpy(A)).squeeze(0)
            # print(A.shape)
            A[dirichlet_node, :] = 0.0
            A[:, dirichlet_node] = 0.0
            dirichlet_pair = [(x,x) for x in dirichlet_node]
            for x in dirichlet_pair: A[x] = 1.0

            # A[dirichlet_node] = 0
            # A[:, dirichlet_node] = 0
            # dirichlet_pair = [(x,x) for x in dirichlet_node]
            # for x in dirichlet_pair: A[x] = 1            
            # print(torch.diag(A).shape)
            diag.append(torch.diag(A).reshape(-1,  1))
            

            gt_rhs.append(graph_t.node_attr[:, graph_t.node_attr_name_index['rhs']][..., np.newaxis])
            input_r.append(r)
            if len(graphs) == 0:
                vmax = values.max()
                vmin = values.min()

            graphs.append(graph_t)

            
            if visualize:
                dot_size = 20.0
                fig = plt.figure(figsize=(30,15))
                u = graph_t.node_attr[:, u_idx] = values.copy()
                u_next = graph_t.node_attr[:, u_next_idx] = new_values.copy()
                u_diff = (u_next - u).max()
                ax = fig.add_subplot()
                ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, new_values.copy(), vmin=vmin, vmax=vmax, cmap='coolwarm')
                ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], color='black', marker='o', s=dot_size, label='Dirichlet (few)')
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                # ax.set_title(name)

                # for i, (value, name) in enumerate([(u, 'before'), (u_next, 'after'), (u_next - u, 'diff: {}'.format(u_diff))]):
                #     ax = fig.add_subplot(131 + i)
                #     ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, value, vmin=-1, vmax=1, cmap='coolwarm')
                #     # ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], color='black', marker='o', s=dot_size, label='Dirichlet (few)')
                #     ax.set_aspect('equal')
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                #     ax.set_title(name)

                vis_path = Path(Path(root_path) / "torch" / "dataset" / output_dir_name / "render3" )
                vis_path.mkdir(parents=True, exist_ok=True)
                # fig.savefig(vis_path / '{:04d}_{:04d}.png'.format(bound_idx, frame_idx))
                fig.savefig(vis_path / '{:04d}_{:04d}.png'.format(bound_idx, frame_idx))
                plt.close()

            # Update states.
            values = new_values
            frame_idx += 1
        bound_idx += 1
    
    graph_len = np.array(graph_len)

    # This should be unnecessary, but just to be safe.
    return graphs[:num_time_steps], node_attr, edge_attr, u_list, input_r, gt_rhs, precond, diag, graph_len



# FEMDataset(name='circle_low_res', diffusivity=0.5, time_step=1e-3, num_time_steps=50, data_features=data_features)
# Dataset Modality: same shape, same diffusivity 
#                   same shape, different diffusivity
#                   different shape, same diffusivity
#                   different shape, different diffusivity

class HeatDataset(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None, use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        diffusivities=config['diffusivities'] if config is not None else [0.5]
        time_step=config['time_step'] if config is not None else 1e-3
        num_time_steps=config['num_time_steps'] if config is not None else 50
        num_inits=config['num_inits'] if config is not None else 1
        epsilon=0.5 # percentage of high frequency data for augmenting with high frequency data
        

        self.transform = TwoHop() if augment_edge else None

        if domain_files_path is not None and domain_files_path.endswith('.npy'):
            self.graphs = []
            self.high_freq_graphs = []
            data = np.load(domain_files_path, allow_pickle=True)
            ind = 0
            data_shape = data[0]['x'].shape[0]
            if augment_edge: 
                undirected_edge_index = to_undirected(data[0]['edge_index'])
                twohop_edges_index = twohop(undirected_edge_index)
                twohop_edges_attr = torch.ones(twohop_edges_index.shape[-1], data[0]['edge_attr'].shape[-1])

                # augmented_edge = torch.tensor([(x, y) for x in np.arange(data_shape) for y in np.arange(data_shape)]).T
                # augmented_edge = torch_geometric.utils.to_undirected(augmented_edge)

            for d in data:
                if use_high_freq:
                    if use_pred_x: 
                        x = torch.cat([d['x'][:,:2], d['x_high_freq'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                    else:
                        x = torch.cat([d['x'][:,:2], d['rhs_high_freq'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                elif high_freq_aug:
                    # TODO: add percentage of high freqency data to use
                    if use_pred_x:
                        if np.random.rand() > 0.5:
                            x = torch.cat([d['x'][:,:2], d['x_high_freq'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                        else:
                            x = torch.cat([d['x'][:,:2], d['u'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                    else:
                        if np.random.rand() > 0.5:
                            x = torch.cat([d['x'][:,:2], d['rhs_high_freq'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                        else:
                            x = torch.cat([d['x'][:,:2], d['u'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                else:
                    if use_pred_x:
                        x = torch.cat([d['x'][:,:2], d['u'].reshape(-1, 1).float(),  d['x'][:, -1].reshape(-1, 1)], dim=-1)
                    else:
                        x = d['x']
                
                if augment_edge:
                    edge_attr = torch.cat([ d['edge_attr'], twohop_edges_attr], dim=0)
                    edge_index = torch.cat([ d['edge_index'], twohop_edges_index], dim=-1)
                    assert edge_index.shape[0] == 2
                    assert edge_attr.shape[-1] == d['edge_attr'].shape[-1]
                else:
                    edge_attr = d['edge_attr']
                    edge_index = d['edge_index']

                graph_data = Data(x=x, # d['x'], \
                                edge_attr=edge_attr, 
                                edge_index=edge_index,
                                y=d['rhs'] if not use_high_freq else d['rhs_high_freq'],
                                u=d['u'] if not use_high_freq else d['x_high_freq'],
                                p=d['p'],
                                diag=torch.from_numpy(d['A_diag']).unsqueeze(-1).float() ,
                                # diag = d['A_diag'],
                                r=d['r'],
                                rhs=d['rhs'] if not use_high_freq else d['rhs_high_freq'],
                                u_next=d['u_next'] if not use_high_freq else d['x_high_freq'])
                # if augment_edge: graph_data = self.transform(graph_data)
                self.graphs.append(graph_data)
                
                ind += 1

        else:
            super().__init__(domain_files_path=domain_files_path, name=name)

            all_data = [(f, d) for d in diffusivities for f in self.domain_files]
            self.num_initializations = num_inits
            
            self.graphs = []
            self.meta_data = {'num_time_steps':[], 'diffusivities':[], 'time_steps':[]}
            counter = 0
            for _ in range(self.num_initializations):
                for f, d in all_data:
                    graphs, node_attr_list, edge_attr_list, u, r, rhs, precond, diag, graph_len = get_finite_element_graphs(f, d, time_step=time_step, num_time_steps=num_time_steps)
                    self.meta_data['num_time_steps'].extend(graph_len + counter*num_time_steps)  # the start position of each trajectory
                    self.meta_data['diffusivities'].extend([d]*len(graph_len))
                    self.meta_data['time_steps'].extend([time_step]*len(graph_len))
                    counter += 1
                    for i in range(len(graphs)):
                        self.graphs.append(Data(x=torch.from_numpy(node_attr_list[i]).float(), \
                                                edge_attr=torch.from_numpy(edge_attr_list[i]).float(), \
                                                edge_index=torch.from_numpy(graphs[i].edge_index.T).long(), \
                                                # p=torch.from_numpy(precond[i]).float(), \
                                                y=torch.from_numpy(rhs[i]).float(),
                                                u=torch.from_numpy(u[i]).float(),
                                                u_next=torch.from_numpy(node_attr_list[i][:, 2]).reshape(-1, 1).float(), 
                                                diag=diag[i].unsqueeze(-1).float(), 
                                                r=torch.from_numpy(r[i]).float(),
                                                rhs=torch.from_numpy(rhs[i]).float()))

            
        # prepare data related model parameters
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1
        self.dirichlet_idx = 3
        self.b_dim = self.graphs[0].x.shape[0]

        self.graphs = self.graphs[:use_data_num]

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
    train_dataset = HeatDataset(name="circle_low_res", config=heat_train_config)
    test_dataset = HeatDataset(name="circle_low_res", config=heat_test_config)
