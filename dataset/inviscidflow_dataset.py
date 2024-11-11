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
from utils.data_config import inviscidflow_train_config, inviscidflow_test_config
from base.base_dataset import FEMDataset


def get_finite_element_graphs(domain_file, density, time_step=1e-3, num_time_steps=50, max_subdataset_size=100, output_dir_name="inviscidflow", visualize=False, generation=False):
    visualize = True
    mesh = trimesh.load(domain_file, process=False)
    nodes = mesh.vertices
    faces = mesh.faces
    nodes = to_np_float(nodes)
    faces = to_np_int(faces)

    np.save('cyn_low_res_node.npy', nodes)
    np.save('cyn_low_res_elements.npy', faces)
    boundary_nodes = get_boundary_vertices(nodes, faces)
    nodes = nodes[:, :2]
    outer_boundary_index = np.argmax([len(x) for x in boundary_nodes])
    print(outer_boundary_index)
    outer_boundary = boundary_nodes[outer_boundary_index]
    print(len(outer_boundary))
    other_obstacle = []
    for i in range(len(boundary_nodes)): 
        if i not in  [outer_boundary_index] : 
            other_obstacle += boundary_nodes[i]
    all_boundary_nodes = []
    for l in boundary_nodes:
        all_boundary_nodes += l 

    print(len(all_boundary_nodes))
    print(len(other_obstacle))
    top = []
    bottom = []
    rest_vertical = []
    leftband = []
    rightband = []
    rest_horizonal = []
    width = nodes[:, 0].max() - nodes[:, 0].min()
    height = nodes[:, 1].max() - nodes[:, 1].min()
    for i, b_ind in enumerate(all_boundary_nodes):
        p = nodes[b_ind]
        x, y  = p
        if x < width * 1e-3:
            leftband.append(b_ind)
        elif x > width - 1e-3:
            rightband.append(b_ind)
        else:
            rest_horizonal.append(b_ind)

        if y < height * 1e-3:
            bottom.append(b_ind)
        elif y > height - 1e-3:
            top.append(b_ind)
        else:
            rest_vertical.append(b_ind)
    # Simulate num_time_steps of the fluid scene using randomly generated initial velocities
    # and boundary conditions.

    # Simulation options.
    node_num = nodes.shape[0]
    sim_opt = {
        'solver': 'pcg',
        'preconditioner': 'incomplete_cholesky',
        'acceleration_structure': 'bvh',
        # Don't change the solver_abs_tol unless you understand what you are doing.
        'solver_abs_tol': str(0.01 * time_step / node_num),
        'solver_rel_tol': '0',
        'verbose': "1"
    }

    print('building dataset ..... ')
    graphs = []
    edge_attr = []
    node_attr = []
    prev = []
    gt = []
    r_list = [] 
    rhs = []
    precond = []
    diag_list = []
    graph_len = []

    bound_idx = 0


    while len(graphs) < num_time_steps:
        print('[InviscidFlowDataset]: {}/{} ready.'.format(len(graphs), num_time_steps))
        # Randomly generate initial velocities, Dirichlet boundary conditions, and Neumann boundary
        # inlet_velocity = 10.0
        # h = 0.01
        # rho = 0.15


        dirichlet = []
        neumann = []
        influx = []
        obstacle = []
        for bn in [outer_boundary]:
            # Select a random ratio of free and neumann boundaries.
            bn_num = len(bn)
            dirichlet_node_num = int(np.random.uniform(low=0.1, high=0.25) * bn_num)
            # Randomly select a starting point.
            dirichlet_node_begin = np.random.randint(bn_num)

            influx_node_num = int(np.random.uniform(low=0.1, high=0.25) * bn_num)
            obstacle_node_num = int(np.random.uniform(low=0.2, high=0.35) * bn_num)

            bn2 = bn + bn
            dirichlet += bn2[dirichlet_node_begin:dirichlet_node_begin + dirichlet_node_num]
            influx += bn2[dirichlet_node_begin + dirichlet_node_num + obstacle_node_num:dirichlet_node_begin + dirichlet_node_num + obstacle_node_num + influx_node_num]

            if dirichlet_node_begin + dirichlet_node_num <= bn_num:
                neumann += bn[:dirichlet_node_begin] + bn[dirichlet_node_begin + dirichlet_node_num:]
            else:
                neumann += bn[(dirichlet_node_begin + dirichlet_node_num - bn_num):dirichlet_node_begin]

            assert len(dirichlet) + len(neumann) == len(bn), f'{(len(dirichlet), len(neumann), len(outer_boundary))}'
            
        dirichlet = [int(i) for i in dirichlet]
        neumann += other_obstacle

        obstacle = [x for x in neumann if x not in influx]
        assert np.sum(np.isin(np.array(neumann), np.array(influx))) == len(influx), f'{neumann} {influx}'
        assert len(dirichlet) + len(neumann) == len(all_boundary_nodes), f'{(len(dirichlet), len(neumann), len(all_boundary_nodes))}'


        points_on_left = []
        points_on_right = []
        points_on_lower = []
        points_on_upper = []

        for x in influx:
            if x in leftband:
                points_on_left.append(x)
            if x in rightband:
                points_on_right.append(x)
            if x in top:
                points_on_upper.append(x)
            if x in bottom:
                points_on_lower.append(x)

        num_points = 1024
        x = to_np_float(np.linspace(nodes.min(), nodes.max(), num_points))
        u = to_np_float(np.random.uniform(1.0, 10.0, size=(num_points)))
        f = interpolate.interp1d( x, u, kind='cubic' )

        initial_velocities = np.zeros((nodes.shape[0], 2))

        initial_velocities[points_on_left, 0] = to_np_float([f(x) for x in nodes[points_on_left][:,0]])
        initial_velocities[points_on_right, 0] = - to_np_float([f(x) for x in nodes[points_on_right][:,0]])
        initial_velocities[points_on_lower, 1] =  to_np_float([f(x) for x in nodes[points_on_lower][:,1]])
        initial_velocities[points_on_upper, 1] = - to_np_float([f(x) for x in nodes[points_on_upper][:,1]])

        # initial_velocities[points_on_left, 0] = np.abs(np.random.randn(len(points_on_left))) 
        # initial_velocities[points_on_right, 0] = - np.abs(np.random.randn(len(points_on_right))) 
        # initial_velocities[points_on_lower, 1] =  np.abs(np.random.randn(len(points_on_lower))) 
        # initial_velocities[points_on_upper, 1] = - np.abs(np.random.randn(len(points_on_upper))) 

        initial_velocities = ndarray(initial_velocities)
        # neumann is a list of nodes that must be on the boundary and represents part of the boundary
        # whose fluid velocities are fixed. The rest of the boundary nodes will be free nodes: p = 0
        # and velocity unconstrained. The reason why they are called "Neumann" and "Dirichlet" is because
        # pressure (p) is the variable we attempt to solve:
        # - For nodes whose velocities are given -> grad p is constrained by the velocity's projection along
        #   the boundary normal direction (hence the name "Neumann");
        # - For nodes whose p = 0, this is Dirichlet by definition.
        # TODO

        



        # Gravitational acceleration. Don't plan to change it, and don't think it is necessary to change.
        g = np.zeros(nodes.shape)
        g[:, -1] = -9.81

        pde = InviscidEulerEquation2d()
        # if need to load
        # with open(Path(Path(root_path) / "torch" / "dataset" / output_dir_name ) / "pde.pkl", 'rb') as f:
        #     loaded_pkl = pickle.load(f)
        # pde.Initialize(loaded_pkl["density"], loaded_pkl["time_step"], loaded_pkl["domain_file"], loaded_pkl["initial_velocities"], loaded_pkl["dirichlet"])
        # import pdb; pdb.set_trace()

        # pde.Initialize will detect the boundary nodes and use those not included in neumann as the free
        # boundary.
        pde.Initialize(density, time_step, domain_file, initial_velocities, dirichlet)

        path = Path(Path(root_path) / "torch" / "dataset" / output_dir_name )
        path.mkdir(parents=True, exist_ok=True)
        save_pde = {}
        save_pde['density'] = density
        save_pde['time_step'] = time_step
        save_pde['domain_file'] = domain_file
        save_pde['initial_velocities'] = initial_velocities
        save_pde['dirichlet'] = dirichlet
        save_pde['sim_opt'] = sim_opt
        with open(str(path / "pde.pkl"), 'wb') as fw:
            pickle.dump(save_pde, fw)
        # pde.SaveToFile(str(path / "pde"))
                
        neumann_mask = to_np_float(pde.neumann_boundary_nodes())
        assert int(sum(neumann_mask)) == len(neumann), f'{int(sum(neumann_mask))}, {len(neumann)}' # TODO
        interior_mask = to_np_float(pde.interior_nodes())
        # Project the initial velocities to make sure it is incompressible.
        initial_pressure = pde.SolvePressure(initial_velocities, sim_opt)
        initial_velocities = to_np_float(pde.ProjectVelocity(initial_velocities, initial_pressure, sim_opt))
        graph = FiniteElementGraph(nodes, faces,
            dirichlet_node=dirichlet, neumann_node=neumann,
            # Goal: predict pressure (p) from fluid velocity (u, v).
            node_attr_name=['u', 'v', 'p', 'rhs',
                'u_next', 'v_next',
                'damping_surface', 'damping_volume'], # TODO stiff * p = rhs
            edge_attr_name=['stiffness'],
            element_attr_name=[])

        #####
        # # Display the triangulation.
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.triplot(nodes[:, 0], nodes[:, 1], faces)
        # ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], color='tab:green', marker='o', label='Dirichlet (few)')
        # ax.scatter(nodes[neumann, 0], nodes[neumann, 1], color='tab:red', marker='o', label='Neumann (more)')
        # ax.set_aspect('equal')
        # ax.legend()
        # plt.show()
        # #####

        u_idx = graph.node_attr_name_index['u']
        v_idx = graph.node_attr_name_index['v']
        p_idx = graph.node_attr_name_index['p']
        rhs_idx = graph.node_attr_name_index['rhs']
        u_next_idx = graph.node_attr_name_index['u_next']
        v_next_idx = graph.node_attr_name_index['v_next']
        damping_surface_idx = graph.node_attr_name_index['damping_surface']
        damping_volume_idx = graph.node_attr_name_index['damping_volume']

        # Specify mass and stiffness.
        edge_idx = to_np_int(graph.edge_index)
        e0 = nodes[graph.edge_index[:, 0] ]
        e1 = nodes[graph.edge_index[:, 1] ]
        edge_len = np.sqrt(np.sum((e0 - e1) ** 2, axis=1))[..., np.newaxis]
            
        e = StdIntMatrixX2d(edge_idx.shape[0])
        for i, (e0, e1) in enumerate(edge_idx):
            e[i] = [int(e0), int(e1)]
        K_idx = graph.edge_attr_name_index['stiffness']
        graph.edge_attr[:, K_idx] = to_np_float(pde.GetStiffnessElements(e))

        # Simulation until the results no longer change or until we have collected enough data.
        values = initial_velocities.copy()
        frame_idx = 0
        subdataset_idx = 0
        while len(graphs) < num_time_steps:
            # Simulate one step.
            old_values = values.copy()
            # Advection.
            values = pde.AdvectArrayField(values, values, sim_opt)
            # Body forces.
            values = pde.ApplyAcceleration(values, to_np_float(g), sim_opt)
            # Poisson equation -- this is the part where we prefer to have neural networks.
            pressures = pde.SolvePressure(values, sim_opt)
            # Use pressures to update the velocity.
            new_values = pde.ProjectVelocity(values, pressures, sim_opt)

            if np.max(np.abs(old_values - new_values)[:]) < np.max(initial_velocities[:]) * 1e-2 or subdataset_idx > max_subdataset_size:
                # If the velocity change is < 1% of the maximal initial velocities, we terminate.
                break

            graph_t = deepcopy(graph)
            # Specify u and u_next.
            graph_t.node_attr[:, u_idx] = to_np_float(values)[:, 0]
            graph_t.node_attr[:, v_idx] = to_np_float(values)[:, 1]
            graph_t.node_attr[:, p_idx] = to_np_float(pressures)
            graph_t.node_attr[:, u_next_idx] = to_np_float(new_values)[:, 0]
            graph_t.node_attr[:, v_next_idx] = to_np_float(new_values)[:, 1]
            ds = to_np_float(pde.damping_surface())
            dv = to_np_float(pde.damping_volume())
            graph_t.node_attr[:, damping_surface_idx] = ds
            graph_t.node_attr[:, damping_volume_idx] = dv
            # Compute the right-hand side.
            graph_t.node_attr[:, rhs_idx] = -(ds + dv) * (neumann_mask + interior_mask)

            # 1 / sqrt(sum( stiffness entries^2)) where sum includes edges whose two nodes are not dirichlet
            r = ( graph_t.edge_attr[:, K_idx])**2
            edge_index = np.array(graph_t.edge_index).astype(np.int64)
            dirichlet_mask = np.array(graph_t.node_dirichlet_mask)
            dirichlet_elements = np.logical_or(dirichlet_mask[edge_index[:, 0]], dirichlet_mask[edge_index[:, 1]])
            r[dirichlet_elements] = 0 # set edges with boundary nodes to 0
            # r = 1.0 / np.max([1.0, np.sqrt(r.sum())])
            r = 1.0 / np.sqrt(r.sum())
            # r = np.array([[r]]*graph.edge_attr.shape[0])
            diag_idx = graph_t.edge_index[:, 0] == graph_t.edge_index[:, 1]
            r = to_np_float(diag_idx) * r

            diag = graph_t.edge_attr[:, K_idx][diag_idx]

            
            node_attr.append(np.hstack([nodes,  \
                                        np.expand_dims(graph_t.node_interior_mask, -1), \
                                        np.expand_dims(graph_t.node_dirichlet_mask, -1), \
                                        np.expand_dims(graph_t.node_neumann_mask, -1), \
                                        np.expand_dims(graph_t.node_attr[:, u_idx], -1), \
                                        np.expand_dims(graph_t.node_attr[:, v_idx], -1), \
                                        np.expand_dims(graph_t.node_attr[:, rhs_idx], -1)]))
            edge_attr.append(np.hstack([edge_len, \
                                        np.expand_dims(graph_t.edge_attr[:, K_idx], -1)] ))
            gt.append(np.expand_dims(graph_t.node_attr[:, p_idx], -1))
            graphs.append(graph_t)
            r_list.append(r)
            prev.append(np.hstack([np.expand_dims(graph_t.node_attr[:, u_idx], -1),
                                    np.expand_dims(graph_t.node_attr[:, v_idx], -1)]))
            diag_list.append(np.expand_dims(diag, -1))
            rhs.append(np.expand_dims(graph_t.node_attr[:, rhs_idx], -1))


        # neumann is a list of nodes that must be on 
            values = to_np_float(new_values)

            '''
            # K = graph_t.message_passing_matrix(graph_t.edge_attr[:, graph_t.edge_attr_name_index['stiffness']])
            # b = graph_t.node_attr[:, rhs_idx].ravel()
            # p = graph_t.node_attr[:, p_idx].ravel()
            # print("K max: ", np.max(K))
            # print("K min: ", np.min(K))
            # print("b max: ", np.max(b))
            # print("b min: ", np.min(b))
            # print("p min: ", np.min(p))
            # print("p max: ", np.max(p))
            # print("ds min: ", np.min(ds))
            # print("ds max: ", np.max(ds))
            # print("dv min: ", np.min(dv))
            # print("dv max: ", np.max(dv))
            # exit(0)
            ###


            ####
            # K = graph_t.message_passing_matrix(graph_t.edge_attr[:, graph_t.edge_attr_name_index['stiffness']])
            # K_2 = to_np_float(pde.K())
            # print("K, K_2: ", np.max(np.abs(K - K_2)))
            # print(K.shape)
            # b = graph_t.node_attr[:, rhs_idx].ravel()
            # # b_1 =  -(ds + dv) * (neumann_mask + interior_mask)
            # # # b_1 = (neumann_mask + interior_mask)
            # # b_2 = to_np_float(pde.rhs())
            # # print("b, b_2: ", np.max(np.abs(b - b_2)))
            # # print("b_1, b_2: ", np.max(np.abs(b_1 - b_2)))
            # print(b.shape)
            # p = graph_t.node_attr[:, p_idx].ravel()
            # print(p.shape)
            # print("np.max(np.abs(K @ p - b)):", np.max(np.abs(K @ p - b)))
            # print("np.max(p):", np.max(p))
            # print("np.min(p):", np.min(p))
            # left_index = (neumann_mask + interior_mask).astype(bool)
            # right_index = 1 - left_index

            # K_left = K[right_index, :]
            # K_ll = K_left[:, left_index]
            # print("K_ll:", np.max(K_ll))
            # print("K_ll:", np.min(K_ll))
            # print("K:", np.max(K))
            # print("K:", np.min(K))

            # p_tmp = K_left @ (p * left_index)
            # print("p_tmp:", np.max(p_tmp))
            # print("p_tmp:", np.min(p_tmp))

            # print("left_index.shape: ", left_index.shape)
            # print("right_index.shape: ", right_index.shape)
            # # import pdb; pdb.set_trace()
            # p_tmp_2 = (K @ (p * left_index.astype(np.float64))) * left_index.astype(np.float64) + p * right_index.astype(np.float64)

            # print("p_tmp_2:", np.max(np.abs((p_tmp_2 - b))))
            # print("np.max(np.abs(K @ p - b)):", np.max(np.abs(K @ p - b)))
            # print("0.01 * time_step / node_num):", 0.01 * time_step / node_num)
            # sys.exit(0)

            #####
            '''
            

            if generation:
                path = Path(Path(root_path) / "torch" / "dataset" / output_dir_name / str(bound_idx) / str(frame_idx))
                path.mkdir(parents=True, exist_ok=True)
                
                with open(path / 'node.npy', 'wb') as fw:
                    np.save(fw, node_attr[-1])
                with open(path / 'edge.npy', 'wb') as fw:
                    np.save(fw, edge_attr[-1])
                with open(path / 'edge_index.npy', 'wb') as fw:
                    np.save(fw, graph_t.edge_index.T)
                with open(path / 'gt.npy', 'wb') as fw:
                    np.save(fw, gt[-1])
                with open(path / 'r.npy', 'wb') as fw:
                    np.save(fw, r)
                with open(path / 'diag.npy', 'wb') as fw:
                    np.save(fw, diag)
                with open(path / '', 'wb') as fw:
                    np.save()

            if visualize:
                dot_size = 30.0
                fig = plt.figure(figsize=(30, 20))
                ax = fig.add_subplot(211)
                ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, values[:, 0],
                    vmin=-10, vmax=10, cmap='coolwarm')
                ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], color='tab:green', marker='o', s=dot_size, label='Dirichlet (few)')
                ax.scatter(nodes[obstacle, 0], nodes[obstacle, 1], color='tab:red', marker='o', s=dot_size, label='obstacle (more)')
                ax.scatter(nodes[influx, 0], nodes[influx, 1], color='tab:blue', marker='o', s=dot_size, label='influx (more)')

                ax.set_aspect('equal')
                # ax.legend()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('u', fontsize=30, loc='left')

                ax = fig.add_subplot(212)
                ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, values[:, 1],
                    vmin=-10, vmax=10, cmap='coolwarm')
                ax.scatter(nodes[dirichlet, 0], nodes[dirichlet, 1], color='tab:green', marker='o', s=dot_size, label='Dirichlet (few)')
                ax.scatter(nodes[obstacle, 0], nodes[obstacle, 1], color='tab:red', marker='o', s=dot_size, label='obstacle (more)')
                ax.scatter(nodes[influx, 0], nodes[influx, 1], color='tab:blue', marker='o', s=dot_size, label='influx (more)')
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('v', fontsize=30, loc='left')
                # ax.legend()
                vis_path = Path(Path(root_path) / "torch" / "dataset" / output_dir_name / "render3" )
                vis_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(vis_path / '{:04d}_{:04d}.png'.format(bound_idx, frame_idx))
                plt.close()

            frame_idx += 1
            subdataset_idx += 1

        bound_idx += 1
    
    graph_len = np.array(graph_len)

    # graphs, node_attr_list, edge_attr_list, prev, gt_p, r, rhs, precond, diag, graph_len
    return graphs[:num_time_steps], node_attr[:num_time_steps], edge_attr[:num_time_steps], prev, gt[:num_time_steps], r_list, rhs, precond, diag_list, graph_len


class InviscidFlowDataset(FEMDataset):
    def __init__(self, domain_files_path=None, name=None, config=None, use_data_num=None, use_high_freq=False, augment_edge=False, use_pred_x=False, high_freq_aug=False):
        torch.set_num_threads(16)
        num_densities=config['num_densities'] if config is not None else None
        time_step=config['time_step'] if config is not None else 1e-2
        num_time_steps=config['num_time_steps'] if config is not None else 100
        data_features=config['data_features'] if config is not None else None 
        num_inits=config['num_inits'] if config is not None else 1
        generation=config['generation'] if config is not None else True

        # domain: generalizability of data
        densities = config['density']

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
                                # diag=torch.from_numpy(d['diag']).unsqueeze(-1).float() ,
                                diag = d['diag'],
                                r=d['r'],
                                rhs=d['rhs'],
                                p_next=d['p_next'],
                                u_next=d['p_next'])
                # if augment_edge: graph_data = self.transform(graph_data)
                self.graphs.append(graph_data)
                
                ind += 1

        else:
            super().__init__(domain_files_path=domain_files_path, name=name)
            
            if densities is None:
                # 1e-1 to 1e1 seems to be a good range for 2D.
                if num_densities is None: densities = [1e-3]
                else: densities = np.random.uniform(low=1e-1, high=1e1, size=num_densities)

            self.all_data = [(f, d) for d in densities for f in self.domain_files]
            self.num_initializations = num_inits
            self.meta_data = {'num_time_steps':[], 'num_densities':[], 'time_steps':[]}
            self.graphs = []
            counter = 0
            for _ in range(self.num_initializations):
                for f, d in self.all_data:
                    graphs, node_attr_list, edge_attr_list, prev, gt_p, r, rhs, precond, diag, graph_len = get_finite_element_graphs(f, d, time_step=time_step, num_time_steps=num_time_steps)
                    self.meta_data['num_time_steps'].extend(graph_len + counter*num_time_steps)
                    self.meta_data['num_densities'].extend([d] * len(graph_len))
                    self.meta_data['time_steps'].extend([time_step]*len(graph_len))
                    counter += 1
                    for i in range(len(graphs)):
                        self.graphs.append(Data(x=torch.from_numpy(node_attr_list[i]).float(), \
                                                edge_attr=torch.from_numpy(edge_attr_list[i]).float(), \
                                                edge_index=torch.from_numpy(graphs[i].edge_index.T).long(), \
                                                # p=torch.from_numpy(precond[i]).float(), \
                                                y=torch.from_numpy(rhs[i]).float(),
                                                # u=torch.from_numpy(u[i]).float(),
                                                p_next=torch.from_numpy(gt_p[i]).reshape(-1, 1).float(), 
                                                u_next=torch.from_numpy(gt_p[i]).reshape(-1, 1).float(), 
                                                diag=torch.from_numpy(diag[i]).float(), 
                                                r=torch.from_numpy(r[i]).reshape(-1,1).float(),
                                                rhs=torch.from_numpy(rhs[i]).float()))



            if data_features is not None:
                self.data_features = data_features
            
        # prepare data related model parameters
        self.node_attr_dim = self.graphs[0].x.shape[-1]
        self.edge_attr_dim = self.graphs[0].edge_attr.shape[-1] # minus the dual_edge_index
        self.num_edges = self.graphs[0].edge_attr.shape[0]
        self.output_dim = 1 # pressure
        self.dirichlet_idx = 3
        self.neumann_idx = 4
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
    train_dataset = InviscidFlowDataset(name="circle_low_res", config=inviscidflow_train_config)
    test_dataset = InviscidFlowDataset(name="circle_low_res", config=inviscidflow_test_config)
