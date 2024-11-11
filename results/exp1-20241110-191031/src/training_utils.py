from audioop import avg
from multiprocessing.context import get_spawning_popen
import sys
import struct
from grpc import xds_channel_credentials
from numpy import diagonal, identity
import torch
import torch_geometric
from tqdm import tqdm
from torch import _dirichlet_grad, no_grad, cat, save, load
from os import popen, path
from os.path import join, exists
import pickle
from pathlib import Path
import time
from torch.optim.lr_scheduler import _LRScheduler
from math import pow

sys.path.append('../')
from pcg import jacobi_torch, ic_torch_st, ic_torch_optimize_st
from utils.utils import load_vectorxr, write_vectorxr
from utils.data_utils import *
from utils.data_config import heat_train_config, heat_test_config
from base.base_dataset import FEMDataset
sys.path.append('../../')



from scipy.sparse import csr_matrix
# from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve_triangular

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def train_epoch(args, train_loader, model, 
        loss_dict, optimizer, lr_scheduler, device, 
        tb_writer=None, tb_rate=1, total_steps=0, epoch=0, exp_name="inviscid_flow", logger=None):

    '''
    Runs one training epoch

    train_loader: train dataset loader
    model: pytorch model
    loss_dict: dictionary of loss functions
    optimizer: optimizer
    device: pytorch device
    parallel: if True, train on parallel gpus
    tb_writer: tensorboard writer
    tb_rate: tensorboard logging rate
    total_steps: current step
    '''

    total_loss = 0
    total_num = 0
    running_loss = 0
    total_x_loss = 0
    total_rhs_loss = 0
    total_precond_loss = 0
    total_diag_loss1 = 0
    total_diag_loss2 = 0
    total_kappa_loss = 0
    total_num_edges = 0
    total_num_nodes = 0
    decay = (1 - 0.95**(epoch//100)) if args.decay else 1.0
    avg_loss = 0

    loss_function_x, loss_sign_x = loss_dict['x']
    loss_function_b, loss_sign_b = loss_dict['b']
    loss_function_diag, loss_sign_diag = loss_dict['diag']
    loss_function_diag2, loss_sign_diag2 = loss_dict['diag2']

    model.train()

    tqdm_itr = tqdm(train_loader, position=1, desc='Training', leave=True)
    for i, data in enumerate(tqdm_itr):
        data.to(device)
        data = data
        optimizer.zero_grad()

        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        # augmented_edge_index = data.augmented_edge_index
        r = data.r
        gt = data.y
        rhs = data.rhs
        diag = data.diag
        u_next = data.u_next
        
        batch_idx = data.batch

        batch_size = len(torch.unique(batch_idx))
        pred_rhs, ((L, D, D_logits), mean_edge_index), pred_x_next = model(node_attr, edge_attr, edge_index, diag=diag, input_r=r, input_x=u_next, batch_idx=batch_idx, \
                    include_r=args.use_r, use_global=args.use_global, diagonalize=args.diagonalize, use_pred_x=args.use_pred_x)

        total_num += node_attr.shape[0]
        total_num_nodes += pred_x_next.shape[0]
        total_num_edges += mean_edge_index.shape[1]

        # ||pred_u_next - u_next_gt|| 
        loss = 0 
        if args.x_loss_weight > 1e-3: 
            x_loss = loss_sign_x * loss_function_x(pred_x_next, u_next)
            loss += x_loss * args.x_loss_weight 
            total_x_loss += x_loss.item() * args.x_loss_weight
            avg_x_loss = total_x_loss / total_num
            avg_loss += avg_x_loss 
        
        # ||pred_rhs - rhs_gt|| 
        if args.rhs_loss_weight > 1e-10: 
            rhs_loss =  loss_sign_b * loss_function_b(pred_rhs, rhs)
            loss += rhs_loss * args.rhs_loss_weight 
            total_rhs_loss += rhs_loss.item() * args.rhs_loss_weight
            avg_rhs_loss = total_rhs_loss / total_num
            avg_loss += avg_rhs_loss 

        # if args.diag_loss_weight > 0.01 and D is not None:
        gt_diag = data.diag 
        if args.diag_loss_weight > 1e-3:
            diag_loss = loss_sign_diag * loss_function_diag(D, gt_diag).mean()
            loss += diag_loss * args.diag_loss_weight

            if False:
                pred_L = torch_geometric.utils.to_dense_adj(mean_edge_index, batch=batch_idx, edge_attr=L).squeeze(-1)
                pred_A = pred_L @ pred_L.transpose(2, 1)
                pred_diag = torch.diagonal(pred_A, dim1=1, dim2=2)
                gt_diag = gt_diag.reshape(len(torch.unique(batch_idx)), -1)
                diag_loss = loss_function_diag(pred_diag, gt_diag)
                diag_loss2 = loss_function_diag2(pred_diag, gt_diag)
            total_diag_loss1 += diag_loss.item() * args.diag_loss_weight
            avg_diag_loss1 = total_diag_loss1 / total_num_nodes
        
        
        if args.diag_loss2_weight > 1e-10:
            diag_loss2 = loss_sign_diag2 * loss_function_diag2(D, gt_diag).mean()
            loss += diag_loss2 * args.diag_loss_weight
            total_diag_loss2 += diag_loss2.item() * args.diag_loss_weight
            avg_diag_loss2 = total_diag_loss2 / total_num_nodes
            

        if args.precond_loss_weight > 0.01:
            gt_precond_val = data.p[:, -1]
            gt_precond_edge_index = data.p[:, :2].long()
            gt_precond = torch_geometric.utils.to_dense_adj(gt_precond_edge_index.T, batch=batch_idx, edge_attr=gt_precond_val)
            pred_precond = torch_geometric.utils.to_dense_adj(mean_edge_index, batch=batch_idx, edge_attr=L).squeeze(-1) 
            ilu_loss = loss_function_diag(pred_precond, gt_precond)
            loss += ilu_loss * args.precond_loss_weight
            total_precond_loss += ilu_loss.item() * args.precond_loss_weight
            avg_precond_loss = total_precond_loss / total_num_edges

        loss.backward()
        optimizer.step()
        if not 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
            lr_scheduler.step()

        total_loss += loss.item()

        if args.precond_loss_weight > 0.01: avg_loss = avg_loss + avg_precond_loss 
        if args.diag_loss_weight > 1e-3: avg_loss = avg_loss + avg_diag_loss1 
        total_steps += 1
        log_str = 'Train loss: {:.4e} '.format(avg_loss)
        if args.x_loss_weight > 1e-3: log_str += ' x loss {:.4e} '.format(avg_x_loss)
        if args.rhs_loss_weight > 1e-10: log_str += 'rhs loss {:.4e} '.format(avg_rhs_loss)
        if args.precond_loss_weight > 0.01: log_str += 'precond loss {:4e}'.format(avg_precond_loss)
        if args.diag_loss_weight > 1e-3: log_str += 'diag loss {:4e} '.format(avg_diag_loss1)
        if args.diag_loss2_weight > 1e-10: log_str += ' diag loss2 {:4e}'.format(avg_diag_loss2)
        tqdm_itr.set_postfix_str(log_str)

        if logger is not None:
            logger.info(f'Iter {i}' + log_str)

    if (tb_writer is not None) and (total_steps % tb_rate == 0):
        tb_writer.add_scalar("Loss_Train/running_loss", scalar_value=running_loss / tb_rate, global_step=total_steps)
        running_loss = 0.0

    if (tb_writer is not None):
        tb_writer.add_scalar("Loss_Train/loss", scalar_value=avg_loss, global_step=total_steps)
        if args.x_loss_weight > 1e-3: tb_writer.add_scalar("Loss_Train/x_loss_epoch", scalar_value=avg_x_loss, global_step=total_steps)
        if args.rhs_loss_weight > 1e-10: tb_writer.add_scalar("Loss_Train/rhs_loss_epoch", scalar_value=avg_rhs_loss, global_step=total_steps)
        if args.precond_loss_weight > 0.01: tb_writer.add_scalar("Loss_Train/precond_epoch", scalar_value=avg_precond_loss, global_step=total_steps)
        if args.diag_loss_weight > 0.01 and D is not None: tb_writer.add_scalar("Loss_Train/diag1_epoch", scalar_value=avg_diag_loss1, global_step=total_steps)
        if args.diag_loss2_weight > 0.01 and D is not None: tb_writer.add_scalar("Loss_Train/diag2_epoch", scalar_value=avg_diag_loss2, global_step=total_steps)
    
    return total_loss / total_num, total_steps


def val_epoch(args, val_loader, model, loss_dict, device, 
    tb_writer=None, total_steps=0, epoch=0, visualization_freq=50, visualize=None, 
    save_dir='results/test-exp/', logger=None):
    
    '''
    Runs one validation epoch

    val_loader: validation dataset loader
    model: pytorch model
    loss_function: loss function
    device: pytorch device
    parallel: if True, run on parallel gpus
    tb_writer: tensorboard writer
    total_steps: current training step/batch number
    epoch: current training epoch
    '''
    total_loss = 0
    total_num = 0
    total_x_loss = 0
    total_rhs_loss = 0
    total_precond_loss = 0
    total_kappa_loss = 0
    total_diag_loss1 = 0
    total_diag_loss2 = 0
    total_num_edges = 0
    total_pcg_iteration = 0
    total_num_shapes = 0
    avg_loss = 0
    
    loss_function_x, loss_sign_x = loss_dict['x']
    loss_function_b, loss_sign_b = loss_dict['b']
    loss_function_diag, loss_sign_diag = loss_dict['diag']
    loss_function_diag2, loss_sign_diag2 = loss_dict['diag2']

    model.eval()
    simulate = args.simulate
    tqdm_itr = tqdm(val_loader, position=1, desc='Validation', leave=True)
    simulation_iteration = 0

    for i, data in enumerate(tqdm_itr):
        data.to(device)

        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        # augmented_edge_index = data.augmented_edge_index
        r = data.r
        gt = data.y
        rhs = data.rhs
        diag = data.diag
        u_next = data.u_next
        batch_idx = data.batch
        batch_size = len(torch.unique(batch_idx))
        with no_grad():
            # using the networ to predict x, and L such that pred_rhs = L L.T pred_x_next   -->  ||pred_rhs - rhs_gt|| 
            pred_rhs, ((L, D, D_logits), mean_edge_index), pred_x_next = model(node_attr, edge_attr, edge_index, diag=diag, input_r = r, input_x=torch.zeros_like(u_next, device=u_next.device), batch_idx=batch_idx, \
                                                    include_r=args.use_r, use_global=args.use_global, diagonalize=args.diagonalize, use_pred_x=args.use_pred_x)
        
            total_num += node_attr.shape[0]
            total_num_edges += mean_edge_index.shape[1]
                
            # ||pred_u_next - u_next_gt|| 
            if args.x_loss_weight > 1e-3: 
                x_loss = loss_sign_x * loss_function_x(pred_x_next, u_next)
                total_loss += total_x_loss  
                total_x_loss += x_loss.item() * args.x_loss_weight
                avg_x_loss = total_x_loss / total_num
                avg_loss += avg_x_loss  
                
            # ||pred_rhs - rhs_gt|| 
            if args.rhs_loss_weight > 1e-10: 
                rhs_loss = loss_sign_b * loss_function_b(pred_rhs, rhs)
                total_loss += total_rhs_loss 
                total_rhs_loss += rhs_loss.item() * args.rhs_loss_weight 
                avg_rhs_loss = total_rhs_loss / total_num
                avg_loss += avg_rhs_loss * args.rhs_loss_weight 
            
            gt_diag = data.diag 
            if args.diag_loss_weight > 1e-3 and D is not None:
                diag_loss = loss_sign_diag * loss_function_diag(D, gt_diag).mean()
                if False:
                    pred_L = torch_geometric.utils.to_dense_adj(mean_edge_index, batch=batch_idx, edge_attr=L).squeeze(0).squeeze(-1)
                    pred_A = pred_L @ pred_L.T
                    pred_diag = torch.diagonal(pred_A).reshape(-1, 1)
                    print('diff', torch.abs(pred_diag - D).mean())
                    diag_loss = loss_function_diag(pred_diag, gt_diag)
                    diag_loss2 = loss_function_diag2(pred_diag, gt_diag)
                total_loss += diag_loss * args.diag_loss_weight
                total_diag_loss1 += diag_loss.item() * args.diag_loss_weight
                avg_diag_loss1 = total_diag_loss1 / total_num
                avg_loss += avg_diag_loss1 

            if args.diag_loss2_weight > 1e-10:
                diag_loss2 = loss_sign_diag2 * loss_function_diag2(D, gt_diag).mean()
                total_loss += diag_loss2 * args.diag_loss2_weight 
                total_diag_loss2 += diag_loss2.item() * args.diag_loss2_weight
                avg_diag_loss2 = total_diag_loss2 / total_num
                avg_loss += avg_diag_loss2
                
            if args.precond_loss_weight > 0.01: 
                gt_precond_val = data.p[:, -1]
                gt_precond_edge_index = data.p[:, :2].long()
                gt_precond = torch_geometric.utils.to_dense_adj(gt_precond_edge_index.T, batch=batch_idx, edge_attr=gt_precond_val)
                pred_precond = torch_geometric.utils.to_dense_adj(mean_edge_index, batch=batch_idx, edge_attr=L).squeeze(-1)
                if D is not None: pred_precond[:,range(pred_precond.shape[-1]), range(pred_precond.shape[-1])] = D.view(batch_size, -1)
                ilu_loss = loss_function_diag(pred_precond, gt_precond)
                total_precond_loss += ilu_loss.item()
                total_loss +=  total_precond_loss * args.precond_loss_weight

            if simulate:
                h = 0.01
                options = {
                    'solver': 'pcg',
                    'preconditioner': 'network',
                    'abs_tol' : 1e-9,
                    'rel_tol' : 0.0,
                    'max_iter': 2000, 
                    'verbose': '0',
                }
                simulation_iteration, convergenet_iterations_model = pcg_simulate_torch(args, model, data, options, use_r=args.use_r, epoch=epoch, visualize=visualize, save_dir=save_dir)
                options['preconditioner'] = 'ilu'
                simulation_iteration_ilu, convergent_iterations_ilu = pcg_simulate_torch(args, model, data, options, use_r=args.use_r, epoch=epoch, save_dir=save_dir)
                options['preconditioner'] = 'jacobi'
                simulation_iteration_jacobi, convergent_iterations_jacobi = pcg_simulate_torch(args, model, data, options, use_r=args.use_r, epoch=epoch, save_dir=save_dir)
                options['preconditioner'] = 'identity'
                simulation_iteration_identity, convergenet_iterations_identity = pcg_simulate_torch(args, model, data, options, use_r=args.use_r, epoch=epoch, save_dir=save_dir)


                total_pcg_iteration += simulation_iteration
                total_num_shapes += len(torch.unique(batch_idx))
                print('iteration number network:' , convergenet_iterations_model, simulation_iteration)
                print('iteration number ilu:' , convergent_iterations_ilu, simulation_iteration_ilu)
                print('iteration number jacobi:', convergent_iterations_jacobi , simulation_iteration_jacobi)
                print('iteration number identity:', convergenet_iterations_identity , simulation_iteration_identity)

                simulate = False

            if args.precond_loss_weight > 0.01: avg_precond_loss = total_precond_loss / total_num_edges
            log_str = 'Val loss: {:.4e} , simulation_iteration '.format(avg_loss, simulation_iteration)
            if args.x_loss_weight > 1e-3: log_str += 'x loss {:.4e} '.format(avg_loss)
            if args.rhs_loss_weight > 1e-10: log_str += ' rhs loss {:.4e} '.format(avg_rhs_loss)
            if args.precond_loss_weight > 0.1: log_str += 'precond loss {:4e}'.format(avg_precond_loss)
            if args.diag_loss_weight > 1e-3: log_str += 'diag loss1 {:4e} '.format(avg_diag_loss1)
            if args.diag_loss2_weight > 1e-3: log_str += 'diag loss2 {:4e}'.format(avg_diag_loss2)
            tqdm_itr.set_description('Validation')
            tqdm_itr.set_postfix_str(log_str)    
            if logger is not None:
                logger.info(f'Iter {i}' + log_str)

    val_loss = total_loss / total_num
    pcg_iteration = total_pcg_iteration / max(1, total_num_shapes)
    if (tb_writer is not None):
        tb_writer.add_scalar("Loss_Val/loss", scalar_value=avg_loss, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/pcg iteration", scalar_value=pcg_iteration, global_step=total_steps)
        if args.x_loss_weight > 1e-3: tb_writer.add_scalar("Loss_Val/x_loss_epoch", scalar_value=avg_x_loss, global_step=total_steps)
        if args.rhs_loss_weight > 1e-10: tb_writer.add_scalar("Loss_Val/rhs_loss_epoch", scalar_value=avg_rhs_loss, global_step=total_steps)
        if args.precond_loss_weight > 0.01: tb_writer.add_scalar("Loss_Val/precond_epoch", scalar_value=avg_precond_loss, global_step=total_steps)
        if args.diag_loss_weight > 1e-3: tb_writer.add_scalar("Loss_Val/diag1_epoch", scalar_value=avg_diag_loss1, global_step=total_steps)
        if args.diag_loss2_weight > 1e-3: tb_writer.add_scalar("Loss_Val/diag2_epoch", scalar_value=avg_diag_loss2, global_step=total_steps)

    return pcg_iteration, val_loss

def pcg_simulate_torch(args, model, data, options, use_r=False, \
                diffusivity=0.5, timestep=0.01, \
                epoch=0, visualize=None, save_dir='./results/test-exp/'):

    from subprocess import call
    from torch_geometric.utils import to_dense_adj
    

    h = 0.01
    solve_triag = args.solve_triag
    tols_dict = ['1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '1e-12']#, '1e-14', '1e-16']
    rel_tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10 , 1e-12]#, 1e-14, 1e-16]
    abs_tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10 , 1e-12]#, 1e-14, 1e-16]
    if args.dataset in ['navier-stokes', 'inviscid_flow', 'flow', 'flowmultisource', 'inviscidflowmultisource', 'navierstokesmultisource', 'flowgeneralize', 'flowoutdomain']:
        rel_tols = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        abs_tols = [ x * h / data.x.shape[0] for x in abs_tols]
    options['rel_tols'] = rel_tols
    options['abs_tols'] = abs_tols
    options['tol_dict'] = tols_dict
    

    dirichlet_node = np.where(torch_to_np_float(data.x[:, model.dirichlet_idx] > 0 ))[0]
    dirichlet_node = [int(idx) for idx in dirichlet_node]


    # prepare A and b
    if args.dataset in ['navier-stokes', 'inviscid_flow', 'flow', 'flowmultisource', 'inviscidflowmultisource', 'navierstokesmultisource', 'flowgeneralize', 'flowoutdomain']:
        edge_attr_a = data.edge_attr[:, -1]
    elif args.dataset in ['heat', 'heatmultisource', 'heatoutdomain', 'heatgeneralize']:
        edge_attr_a = data.edge_attr[:, -1] + data.edge_attr[:, -2]
    elif args.dataset in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
        edge_attr_a = data.edge_attr[:, 1] # edge_len 0, M: 1, K: 2
    else:
        edge_attr_a = data.edge_attr[:, -1]
    
    A = to_dense_adj(data.edge_index, edge_attr=edge_attr_a).squeeze(0)
    if args.dataset not in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
        A[dirichlet_node] = 0
        A[:, dirichlet_node] = 0
        dirichlet_pair = [(x,x) for x in dirichlet_node]
        for x in dirichlet_pair: A[x] = 1
        
    b = data.rhs#.reshape(-1)
    if args.dataset not in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
        b[dirichlet_node] = 0
    num_nodes = A.shape[0]

    # print('A', A)
    

    with torch.no_grad():
        pred_rhs, ((L, D, D_logits), mean_edge_index), pred_x = model(data.x, data.edge_attr, data.edge_index, diag=data.diag, input_r=data.r, input_x=data.u_next, batch_idx=data.batch, \
                    include_r=args.use_r, use_global=args.use_global, diagonalize=args.diagonalize, use_pred_x=args.use_pred_x) #  input_x=data.u_next
    x =  pred_x #.reshape(-1)
    if options['preconditioner'] in ['network', 'predefined']:
        L_mat = to_dense_adj(mean_edge_index, edge_attr=L).squeeze(0).squeeze(-1)
        L_mat = L_mat * D_logits
        if args.dataset not in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
            L_mat[dirichlet_node] = 0
            L_mat[:, dirichlet_node] = 0
            for x in dirichlet_pair: L_mat[x] = 1
        
        assert len(L_mat.shape) == 2 and L_mat.shape[0] == L_mat.shape[1]
        
        if solve_triag:
            preconditioner = csr_matrix(L_mat.detach().cpu().numpy())
        else:
            preconditioner = torch.cholesky_inverse(L_mat)
        x = pred_x
    else:
        x = torch.zeros((A.shape[0], 1), device=A.device)

    
    if options['preconditioner'] in ['ILU', 'ilu', 'incomplete_cholesky']:
        M = ic_torch_optimize_st(A)
        if solve_triag:
            preconditioner = csr_matrix(M.detach().cpu().numpy())
        else:
            preconditioner = torch.cholesky_inverse(M)
    elif options['preconditioner'] in ['network', 'predefined']:
        pass
    elif options['preconditioner'] in ['jacobi']:
        preconditioner = jacobi_torch(A)
        solve_triag = False
    else:
        preconditioner = torch.eye(A.shape[0], device=A.device).float()
        solve_triag = False


    if solve_triag:
        max_iter, convergent_iterations = cg_np(preconditioner, A, x, b, options, solve_triag)
    else:
        max_iter, convergent_iterations = cg_torch(preconditioner, A, x, b, options)

    if visualize is not None:
        M = ic_torch_optimize_st(A)
        L_vis = {'A': A.cpu().numpy(), 'ic': M.cpu().numpy(), 'pred': L_mat.cpu().numpy()}
        visualize(data, batch_pred_u_next=pred_x, batch_pred_rhs=pred_rhs, L_vis=L_vis, epoch=epoch, save_dir=save_dir)

    return max_iter, convergent_iterations
    

  


def cg_torch(preconditioner, A, x, b, options):

    
    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    max_iter = options['max_iter']
    tol_dict = options['tol_dict']
    flags = [True for x in rel_tols]


    x = x.float()
    A = A.float()
    b = b.float()
    preconditioner = preconditioner.float()

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y
    convergent_iterations = {}
    for i in range(max_iter):
        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap
        
        for j in range(len(rel_tols)-1):
            if flags[j] and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[j] + abs_tols[j]:
                convergent_iterations[tol_dict[j]] = i
                flags[j] = False
                
        if  torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            convergent_iterations[tol_dict[-1]] = i
            return i, convergent_iterations
        y_next   = torch.mm( preconditioner, r_next )
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
    
    return max_iter, convergent_iterations





def cg_np(preconditioner, A, x, b, options, solve_triag=True):
    
    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    max_iter = options['max_iter']
    tols_dict = options['tol_dict']
    flags = [True for x in rel_tols]


    x = x.detach().cpu().float().numpy()
    A = A.detach().cpu().float().numpy()
    b = b.detach().cpu().float().numpy()

    # r = A @ x - b
    r =  A.dot(x) - b
    if solve_triag:
        r_new = r.reshape(-1, 1)
        r_new = np.concatenate([r_new, np.zeros_like(r_new)], axis=-1)
        y0 = spsolve_triangular(preconditioner.transpose(), r_new, lower=False)
        y = spsolve_triangular(preconditioner, y0, lower=True)
        y = y[:, 0].reshape(-1,1)
    else:
        y = np.dot( preconditioner, r )
    p = -y
    convergent_iterations = {}
    for i in range(max_iter):
        
        Ap       = np.dot( A , p )
        alpha    = np.dot(r.T, y)/np.dot( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap


        for j in range(len(rel_tols)-1):
            if flags[j] and np.abs(r_next).max() <= np.abs(b).max() * rel_tols[j] + abs_tols[j]:
                end_cg_time = time.time()
                print(f'{tols_dict[j]} Pcg Converged in {i} steps time ')
                convergent_iterations[rel_tols[j]] = i
                flags[j] = False
                
        if  np.abs(r_next).max() <= np.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            end_cg_time = time.time()
            print(f'{tols_dict[-1]} Pcg Converged in {i} ')
            convergent_iterations[rel_tols[-1]] = i
            return i, convergent_iterations
        
        if solve_triag:
            r_next_new = r_next.reshape(-1, 1)
            r_next_new = np.concatenate([r_next, np.zeros_like(r_next_new)], axis=-1)
            y0 = spsolve_triangular(preconditioner.transpose(), r_next_new, lower=False)
            y_next = spsolve_triangular(preconditioner, y0, lower=True)
            y_next = y_next[:, 0].reshape(-1, 1)
        else:
            y_next = np.dot( preconditioner, r_next )
    
        beta     = np.dot(y_next.T, (r_next - r))/ np.dot(r.T, y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')


    
    return max_iter, convergent_iterations


