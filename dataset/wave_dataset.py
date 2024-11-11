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
sys.path.append('../../')
# sys.path.append('../../python/example/phys_gnn/')
# sys.path.append('../../python/py_phys_sim/')
# sys.path.append('../../python/')
sys.path.append('../python/example/phys_gnn/')
sys.path.append('../python/py_phys_sim/')
sys.path.append('../python/')

from py_phys_sim.common.common import print_error, load_sparse_matrix, ndarray
from py_phys_sim.common.project_path import root_path
from py_phys_sim.common.tri_mesh import get_boundary_vertices
from py_phys_sim.core.py_phys_sim_core import WaveEquation2d, StdIntMatrixX2d
from finite_element_graph import FiniteElementGraph



def get_finite_element_graphs(domain_file, C, H=2e-3, num_time_steps=50, visualize=False, output_dir_name="wave"):

    visualize = True
    mesh = trimesh.load(domain_file, process=False)
    nodes = mesh.vertices
    faces = mesh.faces
    nodes = to_np_float(nodes)
    faces = to_np_int(faces)
    all_boundary_nodes = get_boundary_vertices(nodes, faces)
    nodes = nodes[:, :2]
    boundary_flattened = []
    boundary_flattened = [item for sublist in all_boundary_nodes for item in sublist]
    print(len(boundary_flattened))

    np.save('circle_low_res_nodes.npy', nodes)
    np.save('circle_low_res_elements.npy', faces)

    # Simulation options.
    sim_opt = {
        'solver': 'pcg',    # Alternative: 'pardiso'.
        'preconditioner': 'identity',    # Can delete this key if using Pardiso.
        'solver_abs_tol': '1e-10',
        'solver_rel_tol': '1e-10',
        'verbose': '0',
    }

    print('building dataset ..... ')
    graphs = []
    edge_attr_list = []
    node_attr_list = []
    gt_rhs = []
    gt_lhs = []
    r_list = []
    prev_list = []
    gt_u_next = []
    gt_u_dot_next = []
    bound_idx = 0
    graph_len = []
    precond = []
    diag_list = []
    gt_u = []


    # Construct the initial value.
    min_x = np.min(nodes[:, 0])
    max_x = np.max(nodes[:, 0])
    min_y = np.min(nodes[:, 1])
    max_y = np.max(nodes[:, 1])


    dirichlet = []
    for bn in all_boundary_nodes:
        # Select a random ratio of free and neumann boundaries.
        bn_num = len(bn)
        dirichlet_node_num = int(np.random.uniform(low=0.5, high=0.8) * bn_num)
        # Randomly select a starting point.
        dirichlet_node_begin = np.random.randint(bn_num)
        bn2 = bn + bn
        dirichlet += bn2[dirichlet_node_begin:dirichlet_node_begin + dirichlet_node_num]

    dirichlet = [int(i) for i in dirichlet]

    # dirichlet = boundary_flattened
    #### initialization method one
    center = ndarray([min_x + max_x, min_y + max_y]) / 2
    radii = np.max([max_x - min_x, max_y - min_y]) / 2
    nodes_normalized = (nodes - ndarray([center[0] + radii * 0.4, center[1]])) / radii
    initial_values = ndarray(np.zeros(nodes.shape[0]))
    # initial_rates = np.cos(nodes_normalized[:, 0] * np.pi / 2) * np.cos(nodes_normalized[:, 1] * np.pi / 2) - 0.5
    initial_rates = np.sin(np.pi * nodes_normalized[:, 0] * np.random.rand(1) / 2) * np.sin(np.pi * nodes_normalized[:, 1] * np.random.rand(1) / 2) - (np.random.rand(1) / 2)
    initial_rates[boundary_flattened] = 0
    initial_rates = ndarray(initial_rates)



    #### initialization with gaussion smoothing 
    # num_points = 1024
    # x = to_np_float(np.linspace(-1, 1, num_points))
    # u = to_np_float(np.random.normal(-1, 1, size=(num_points, num_points)))
    # # Smooth out u by Gaussian kernels.
    # u = to_np_float(gaussian_filter(u, sigma=num_points * 0.08))
    # # Rescale it to [-1, 1]
    # u_max = np.max(u[:])
    # u_min = np.min(u[:])
    # u = (u - u_min) / (u_max - u_min) * 2 - 1
    # # Now u is the initial value at the lattice grid [-1, 1] x [-1, 1] where along each dimension
    # # we have num_points points. We now want to interpolate initial values based on node positions.
    # f = interpolate.interp2d(x, x, u, kind='cubic')
    # initial_values = ndarray(np.zeros(nodes.shape[0]))
    # initial_rates = ndarray([f(x, y) for x, y in nodes]).ravel()
    # initial_rates[dirichlet] = 0
    # initial_rates = ndarray(initial_rates)

    # Simulation.
    sim = WaveEquation2d()
    # dirichlet_node = [int(i) for i in boundary_nodes]
    print('propagation speed', C, 'timestep',  H)
    sim.Initialize(C, H, domain_file, dirichlet) #, initial_values, dirichlet_node)
    
    # dirichlet = boundary_flattened
    interior_mask = to_np_float(sim.interior_nodes())
    boundary_mask = to_np_float(sim.boundary_nodes())
    dirichlet_mask = to_np_float(sim.dirichlet_nodes())
    non_dirichlet_mask = 1.0 - to_np_float(sim.dirichlet_nodes())
    init_boundary = initial_values * dirichlet_mask

    print(dirichlet_mask.sum(), interior_mask.sum(), nodes.shape[0])

    # In the case of heat equations, we use all boundary nodes as the dirichlet nodes, but
    # other PDEs may have different dirichlet and neumann nodes.
    graph = FiniteElementGraph(nodes, faces,
        dirichlet_node=dirichlet, neumann_node=[],
        node_attr_name=['u', 'u_dot', 'u_next', 'u_dot_next', 'rhs', 'lhs'],
        edge_attr_name=['M', 'K'],  # For debugging purposes only.
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
    M_idx = graph.edge_attr_name_index['M']
    graph.edge_attr[:, M_idx] = to_np_float(sim.GetMassElements(e))
    K_idx = graph.edge_attr_name_index['K']
    graph.edge_attr[:, K_idx] = to_np_float(sim.GetStiffnessElements(e))
    u_idx = graph.node_attr_name_index['u']
    u_dot_idx = graph.node_attr_name_index['u_dot']
    u_next_idx = graph.node_attr_name_index['u_next']
    u_dot_next_idx = graph.node_attr_name_index['u_dot_next']
    rhs_idx =  graph.node_attr_name_index['rhs']
    lhs_idx = graph.node_attr_name_index['lhs']

    # 
    r = (graph.edge_attr[:, M_idx])**2
    # boundary_elements = [True if (e0 in dirichlet or e1 in dirichlet) else False for e0, e1 in graph.edge_index]
    boundary_elements = np.logical_or(dirichlet_mask[graph.edge_index[:, 0]], dirichlet_mask[graph.edge_index[:, 1]])
    r[boundary_elements] = 0 # set edges with boundary nodes to 0
    r = 1.0 / np.max([1.0, np.sqrt(r.sum())])
    edge_identity_mask = (graph.edge_index[:, 0] == graph.edge_index[:, 1])
    r = edge_identity_mask * r
    diag_idx = graph.edge_index[:, 0] == graph.edge_index[:, 1]
    r = to_np_float(diag_idx) * r
    r = r[..., np.newaxis]


    # Simulation until the results no longer change or until we have collected enough data.
    node_values_history = [initial_values,]
    node_rates_history = [initial_rates,]
    # while len(graphs) < num_time_steps:


    cur_step_idx = 0
    # while cur_step_idx < num_time_steps:
    while len(graphs) < num_time_steps:
        print('[WaveEquationDataset::__init__]: {}/{} ready.'.format(cur_step_idx, num_time_steps))

        delta_node_rates = sim.Forward(node_values_history[-1], node_rates_history[-1], sim_opt)
        # print(delta_node_rates)

        delta_node_rates = ndarray(delta_node_rates)
        # print(delta_node_rates.mean(), delta_node_rates.max())

        node_rates = node_rates_history[-1] - delta_node_rates
        node_values = node_values_history[-1] + H * node_rates # first order approx
        # remove these two lines
        # Create the data item that characterize the simuation from values[-1] to new_values.
        # graph_t = deepcopy(graph)
        # Specify u and u_next.
        graph.node_attr[:, u_idx] = node_values_history[-1].copy()
        graph.node_attr[:, u_next_idx] = node_values.copy()
        graph.node_attr[:, u_dot_idx] = node_rates_history[-1].copy()
        graph.node_attr[:, u_dot_next_idx] = node_rates.copy()
        graph.node_attr[:, lhs_idx] = delta_node_rates.copy()
        node_values_history.append(ndarray(node_values))
        node_rates_history.append(node_rates)
        

        # Specify rhs. rhs = Ku
        rhs = to_np_float(sim.ComputeStiffnessVectorProduct(graph.node_attr[:, u_idx])) * interior_mask
        graph.node_attr[:, rhs_idx] = rhs
        rhs = np.expand_dims(rhs, -1)
        
        u = np.expand_dims(graph.node_attr[:, u_idx], -1)
        lhs = np.expand_dims(delta_node_rates, -1)
        diag = np.expand_dims(graph.edge_attr[:, M_idx][diag_idx], axis=-1)

        node_attr = np.hstack([nodes, \
                                    np.expand_dims(graph.node_attr[:, u_idx], axis=-1), \
                                    np.expand_dims(graph.node_attr[:, u_dot_idx], axis=-1), \
                                    np.expand_dims(graph.node_attr[:, rhs_idx], axis=-1),
                                    np.expand_dims(dirichlet_mask, axis=-1)])
        edge_attr = np.hstack([edge_len, \
                        np.expand_dims(graph.edge_attr[:, M_idx], axis=-1), \
                        np.expand_dims(graph.edge_attr[:, K_idx], axis=-1)])

        u_dot_next = np.expand_dims(graph.node_attr[:, u_dot_next_idx], -1)
        u_next = np.expand_dims(graph.node_attr[:, u_next_idx], -1)
        prev = np.hstack([np.expand_dims(graph.node_attr[:, u_idx], -1),
                                np.expand_dims(graph.node_attr[:, u_dot_idx], -1)])

        # post process to remove dirichlet nodes from everywhere and resize the graph
        



        # append all results
        if lhs.mean() != 0:
            diag_list.append(diag.copy())
            node_attr_list.append(node_attr.copy())
            edge_attr_list.append(edge_attr.copy())
            r_list.append(r.copy())
            gt_rhs.append(rhs.copy())
            gt_lhs.append(lhs.copy())
            gt_u.append(u.copy())
            prev_list.append(prev.copy())
            gt_u_dot_next.append(u_dot_next.copy())
            gt_u_next.append(u_next.copy())
        
            import copy
            graphs.append(copy.deepcopy(graph))
        # graphs.append(graph)
        # remove below line
        # graphs.append(graph)
        # node_values_history.append(ndarray(node_values))
        # node_rates_history.append(node_rates)
        # Update states.
        bound_idx += 1
    
    if visualize:
        # 3D plot.
        vis_path = Path(Path(root_path) / "torch" / "dataset" /  output_dir_name )
        os.makedirs(vis_path, exist_ok=True)
        for i in range(len(graphs)):
            dot_size = 5
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            print(nodes)
            print(faces)
            ax.plot_trisurf(nodes[:, 0], nodes[:, 1], node_values_history[i], triangles=faces,
                cmap='coolwarm')
            ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], 0, s=dot_size, color='black', marker='o', label='Dirichlet (few)')
            
            # make the panes transparent
            # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # # make the grid lines transparent
            # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.set_zlim([np.min(node_values_history), np.max(node_values_history)])
            fig.savefig( vis_path / '{:04d}.png'.format(i))
        plt.close()


    # This should be unnecessary, but just to be safe.
    # node_attr = np.array(node_attr)
    # prev = np.array(prev)
    # gt_rhs=np.array(gt_rhs)
    # gt = np.array(gt)
    
    # print(node_attr.shape)
    # print(prev.shape)
    # print(gt.shape)
    # print(edge_attr.shape)
    # print(r.shape)
    # print(diag.shape)

    if node_values_history[-1].mean() > 10:
        print('[SOLVER DOES NOT CONVERGE]')
        exit(1)

    print(gt_lhs[1])
    return_dict = {
        'graphs':graphs,
        'node_attr':node_attr_list, 
        'edge_attr':edge_attr_list, 
        'prev':prev_list, 
        'u_dot_next':gt_u_dot_next, 
        'u_next':gt_u_next, 
        'r': r_list, 
        'rhs': gt_rhs, 
        'lhs':gt_lhs, 
        'diag':diag_list, 
        'graph_len':[len(graphs)]
    }
    return return_dict



class WaveDataset(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None, use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        torch.set_num_threads(16)
        num_speed=config['num_speed'] if config is not None else None
        time_step=config['time_step'] if config is not None else 2e-3
        num_time_steps=config['num_time_steps'] if config is not None else 100
        data_features=config['data_features'] if config is not None else None 
        num_inits=config['num_inits'] if config is not None else 1
        generation=config['generation'] if config is not None else True

        # domain: generalizability of data
        c = config['speed']

        # self.transform = TwoHop() if augment_edge else None

        if domain_files_path is not None and domain_files_path.endswith('.npy'):
            self.graphs = []
            self.high_freq_graphs = []
            data = np.load(domain_files_path, allow_pickle=True)
            ind = 0
            data_shape = data[0]['x'].shape[0]

            for d in data:
                if use_pred_x: # replace rhs with x 
                    rhs_dirichlet = d['rhs'] * d['x'][:,3].unsqueeze(-1)
                    x = torch.cat([d['x'][:,:7], rhs_dirichlet], dim=-1)
                else:
                    x = d['x']
            
                edge_attr = d['edge_attr']
                edge_index = d['edge_index']

                graph_data = Data(x=x, # d['x'], \
                                edge_attr=edge_attr, 
                                edge_index=edge_index,
                                y=d['rhs'],
                                diag = d['diag'],
                                r=d['r'],
                                rhs=d['rhs'],
                                u_dot_next=d['u_dot_next'],
                                x_next=d['u_dot_next'],
                                u_next=d['u_next']) # the u_next here is actually u_dot next which is the x in Ax=b
                # if augment_edge: graph_data = self.transform(graph_data)
                self.graphs.append(graph_data)
                
                ind += 1

        else:
            super().__init__(domain_files_path=domain_files_path, name=name)
            
            if c is None:
                # 1e-1 to 1e1 seems to be a good range for 2D.
                if c is None: c = 0.1
                else: c = np.random.uniform(low=1e-1, high=1e1, size=num_speed)

            self.all_data = [(f, d) for d in c for f in self.domain_files]
            self.num_initializations = num_inits
            self.meta_data = {'num_time_steps':[], 'num_speed':[], 'time_steps':[]}
            self.graphs = []
            counter = 0
            for _ in range(self.num_initializations):
                for f, d in self.all_data:
                    # graphs, node_attr_list, edge_attr_list, prev, gt_p, r, rhs, precond, diag, graph_len = get_finite_element_graphs(f, d, time_step=time_step, num_time_steps=num_time_steps)
                    # edge_index, node_attr_list, edge_attr_list, prev, gt_u_dot, r_list, rhs, diag, graph_len = get_finite_element_graphs(f, d, H=time_step, num_time_steps=num_time_steps)
                    return_dict =  get_finite_element_graphs(f, d, H=time_step, num_time_steps=num_time_steps)
                    graphs= return_dict['graphs']
                    node_attr_list = return_dict['node_attr']
                    edge_attr_list= return_dict['edge_attr'] 
                    prev= return_dict['prev'] 
                    gt_u_dot= return_dict['u_dot_next']
                    gt_u= return_dict['u_next']
                    r_list= return_dict['r']
                    gt_rhs= return_dict['rhs'] 
                    gt_lhs= return_dict['lhs']
                    diag_list = return_dict['diag'] 
                    graph_len = return_dict['graph_len']

                    print('gt_lhs', gt_lhs[1])

                    self.meta_data['num_time_steps'].extend(graph_len + [counter]*num_time_steps)
                    self.meta_data['num_speed'].extend([d] * len(graph_len))
                    self.meta_data['time_steps'].extend([time_step]*len(graph_len))
                    counter += 1
                    for i in range(num_time_steps):
                        self.graphs.append(Data(x=torch.from_numpy(node_attr_list[i]).float(), \
                                                edge_attr=torch.from_numpy(edge_attr_list[i]).float(), \
                                                edge_index=torch.from_numpy(graphs[i].edge_index.T).long(), \
                                                y=torch.from_numpy(gt_rhs[i]).float(),
                                                rhs=torch.from_numpy(gt_rhs[i]).float(),
                                                lhs = torch.from_numpy(gt_lhs[i]).float(),
                                                u_dot_next=torch.from_numpy(gt_u_dot[i]).reshape(-1, 1).float(), 
                                                x_next=torch.from_numpy(gt_lhs[i]).reshape(-1, 1).float(), 
                                                u_next=torch.from_numpy(gt_u[i]).reshape(-1, 1).float(), 
                                                diag=torch.from_numpy(diag_list[i]).float(), 
                                                r=torch.from_numpy(r_list[i]).float(),
                                                prev=torch.from_numpy(prev[i]).float()
                                                ))



            if data_features is not None:
                self.data_features = data_features
            
        # prepare data related model parameters
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1 # u_dot
        self.dirichlet_idx = 5
        self.b_dim = self.graphs[0].x.shape[0]
        self.graphs = self.graphs[:use_data_num]

        # try on only one data
        # self.graphs = [self.graphs[0]]
        

    def to(self, int_dtype, float_dtype, device):
        for g in self.graphs:
            g.to(int_dtype, float_dtype, device)

    def get_data(self):
        return self.graphs

    def save(self, file_name='./data2d.npy'):
        data = np.array([x.to_dict() for x in self.graphs])
        np.save(file_name, data)        
        with open(f'{file_name}.pickle', 'wb') as handle:
            pickle.dump(self.meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


if __name__ == "__main__":
    # train_dataset = WaveDataset(name="eight_low_res", config=wave_train_config)
    test_dataset = WaveDataset(name="eight_low_res", config=wave_test_config)
