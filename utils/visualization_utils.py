import os
import sys
import matplotlib.pyplot as plt
import logging
from utils.data_utils import *

def visualize_heat(data, batch_pred_u_next=None, batch_pred_rhs=None, L_vis=None, epoch=0, save_dir='results/exp-test', dirichlet_idx=3):
    save_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)        

    batch_idx = data.batch
    batch_size = torch.unique(batch_idx).size(0)

    data = data#.clone().detach().cpu().numpy()
    node_dim = data.x.shape[-1]
    num_node = data.x.shape[0]
    node_attr = data.x.reshape((batch_size, -1, node_dim)).clone().cpu().numpy()
    # gt = data.y.clone().cpu().numpy()
    # pred = batch_pred_u_next.clone().cpu().numpy()
    
    gt = data.y.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_u = data.u.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_u_next = data.u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_rhs = data.rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_u_next = batch_pred_u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_rhs = batch_pred_rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    node_pos = node_attr[:, :,:2]
    dirichlet_mask = node_attr[:, :, dirichlet_idx]
    dirichlet_nodes = np.where(dirichlet_mask > 0)[1].reshape(batch_size, -1, 1)

    # u_next = node_attr[:, :, 2]

    if L_vis is not None:
        mid_range = num_node//2 - 5
        L_real = L_vis['A'].reshape(num_node, num_node)
        L_real_vis = np.concatenate([np.concatenate([L_real[:10, :10], L_real[:10, mid_range:mid_range+10], L_real[:10, -10:]]), 
                                    np.concatenate([L_real[mid_range:mid_range+10, :10], L_real[mid_range:mid_range+10, mid_range:mid_range+10], L_real[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_real[-10:, :10], L_real[-10:, mid_range:mid_range+10], L_real[-10:, -10:]])], axis=-1)
        L_ic = L_vis['ic'].reshape(num_node, num_node)
        L_ic_vis = np.concatenate([np.concatenate([L_ic[:10, :10], L_ic[:10, mid_range:mid_range+10], L_ic[:10, -10:]]), 
                                    np.concatenate([L_ic[mid_range:mid_range+10, :10], L_ic[mid_range:mid_range+10, mid_range:mid_range+10], L_ic[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_ic[-10:, :10], L_ic[-10:, mid_range:mid_range+10], L_ic[-10:, -10:]])], axis=-1)
        L_pred = L_vis['pred'].reshape(num_node, num_node)
        L_pred_vis = np.concatenate([np.concatenate([L_pred[:10, :10], L_pred[:10, mid_range:mid_range+10], L_pred[:10, -10:]]), 
                                    np.concatenate([L_pred[mid_range:mid_range+10, :10], L_pred[mid_range:mid_range+10, mid_range:mid_range+10], L_pred[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_pred[-10:, :10], L_pred[-10:, mid_range:mid_range+10], L_pred[-10:, -10:]])], axis=-1)


    
    for data_idx in range(batch_size):
        cur_node_pos = node_pos[data_idx]
        cur_dirichlet_mask = dirichlet_nodes[data_idx]
        cur_gt_u = gt_u[data_idx]
        cur_gt_u_next = gt_u_next[data_idx]
        cur_gt_rhs = gt_rhs[data_idx]
        cur_pred_u_next = pred_u_next[data_idx]
        cur_pred_rhs = pred_rhs[data_idx]

        # Extract the current and next u.
        pred_diff_u_next = np.max(np.abs(cur_pred_u_next - cur_gt_u_next))
        # Visualize the field before and after simulation.
        fig = plt.figure()
        for i, (value, name) in enumerate([(cur_gt_u_next, ' gt u'), (cur_pred_u_next, 'pred u'), (cur_pred_u_next - cur_gt_u_next, 'd/f: {:.2f}'.format(pred_diff_u_next))]):
            ax = fig.add_subplot(141 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.3, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_u_next.png')
        pred_diff_rhs = np.max(np.abs(cur_pred_rhs - cur_gt_rhs))
        fig2 = plt.figure()
        for i, (value, name) in enumerate([(cur_gt_rhs, ' gt rhs'), (cur_pred_rhs, 'pred rhs'), (cur_pred_rhs - cur_gt_rhs, 'd/f: {:.2f}'.format(pred_diff_rhs))]):
            ax = fig2.add_subplot(141 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.3, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_rhs.png')
        # visualize the predicted L and gt L
        if L_vis is not None:
            fig3 = plt.figure(figsize=(40, 10))
            for i, (value, name) in enumerate([(L_real_vis, 'A'), (L_ic_vis, 'ic'), (L_pred_vis, 'pred')]):
                ax = fig3.add_subplot(131+i)
                im = ax.imshow(value, cmap='coolwarm')
                fig3.colorbar(im, orientation='vertical')
                ax.set_title(f'{name}')
            plt.savefig(f'{save_dir}/epoch_{epoch}_L_{data_idx}.png')

        
    plt.close('all')


def visualize_flow(data, batch_pred_u_next=None, batch_pred_rhs=None, L_vis=None, epoch=0, save_dir='results/exp-test', dirichlet_idx=3):
    save_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)        

    batch_idx = data.batch
    batch_size = torch.unique(batch_idx).size(0)

    data = data#.clone().detach().cpu().numpy()
    node_dim = data.x.shape[-1]
    num_node = data.x.shape[0]
    node_attr = data.x.reshape((batch_size, -1, node_dim)).clone().cpu().numpy()
    # gt = data.y.clone().cpu().numpy()
    # pred = batch_pred_u_next.clone().cpu().numpy()
    
    gt = data.y.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_p_next = data.u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_rhs = data.rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_u_next = batch_pred_u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_rhs = batch_pred_rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    node_pos = node_attr[:, :,:2]
    dirichlet_mask = node_attr[:, :, dirichlet_idx]
    dirichlet_nodes = np.where(dirichlet_mask > 0)[1].reshape(batch_size, -1, 1)

    # u_next = node_attr[:, :, 2]

    if L_vis is not None:
        mid_range = num_node//2 - 5
        L_real = L_vis['A'].reshape(num_node, num_node)
        L_real_vis = np.concatenate([np.concatenate([L_real[:10, :10], L_real[:10, mid_range:mid_range+10], L_real[:10, -10:]]), 
                                    np.concatenate([L_real[mid_range:mid_range+10, :10], L_real[mid_range:mid_range+10, mid_range:mid_range+10], L_real[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_real[-10:, :10], L_real[-10:, mid_range:mid_range+10], L_real[-10:, -10:]])], axis=-1)
        L_ic = L_vis['ic'].reshape(num_node, num_node)
        L_ic_vis = np.concatenate([np.concatenate([L_ic[:10, :10], L_ic[:10, mid_range:mid_range+10], L_ic[:10, -10:]]), 
                                    np.concatenate([L_ic[mid_range:mid_range+10, :10], L_ic[mid_range:mid_range+10, mid_range:mid_range+10], L_ic[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_ic[-10:, :10], L_ic[-10:, mid_range:mid_range+10], L_ic[-10:, -10:]])], axis=-1)
        L_pred = L_vis['pred'].reshape(num_node, num_node)
        L_pred_vis = np.concatenate([np.concatenate([L_pred[:10, :10], L_pred[:10, mid_range:mid_range+10], L_pred[:10, -10:]]), 
                                    np.concatenate([L_pred[mid_range:mid_range+10, :10], L_pred[mid_range:mid_range+10, mid_range:mid_range+10], L_pred[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_pred[-10:, :10], L_pred[-10:, mid_range:mid_range+10], L_pred[-10:, -10:]])], axis=-1)

        min_val = np.min([np.min(L_ic_vis), np.min(L_pred_vis)])
        max_val = np.max([np.max(L_ic_vis), np.max(L_pred_vis)])



    for data_idx in range(batch_size):
        cur_node_pos = node_pos[data_idx]
        cur_dirichlet_mask = dirichlet_nodes[data_idx]
        cur_gt_u_next = gt_p_next[data_idx]
        cur_gt_rhs = gt_rhs[data_idx]
        cur_pred_u_next = pred_u_next[data_idx]
        cur_pred_rhs = pred_rhs[data_idx]

        # Extract the current and next u.
        pred_diff_u_next = np.max(np.abs(cur_pred_u_next - cur_gt_u_next))
        # Visualize the field before and after simulation.
        fig = plt.figure(figsize=(40, 8))
        for i, (value, name) in enumerate([(cur_gt_u_next, ' gt p'), (cur_pred_u_next, 'pred p'), (cur_pred_u_next - cur_gt_u_next, 'd/f: {:.8f}'.format(pred_diff_u_next))]):
            ax = fig.add_subplot(131 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.6, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_u_next.png')
        pred_diff_rhs = np.max(np.abs(cur_pred_rhs - cur_gt_rhs))
        fig2 = plt.figure(figsize=(40, 8))
        print(cur_gt_rhs.shape)
        for i, (value, name) in enumerate([(cur_gt_rhs, ' gt rhs'), (cur_pred_rhs, 'pred rhs'), (cur_pred_rhs - cur_gt_rhs, 'd/f: {:.8f}'.format(pred_diff_rhs))]):
            ax = fig2.add_subplot(131 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.6, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_rhs.png')
        # visualize the predicted L and gt L
        if L_vis is not None:
            fig3 = plt.figure(figsize=(40, 10))
            for i, (value, name) in enumerate([(L_real_vis, 'real'), (L_ic_vis, 'ic'), (L_pred_vis, 'pred')]):
                ax = fig3.add_subplot(131+i)
                im = ax.imshow(value, cmap='coolwarm', vmin=min_val, vmax=max_val)
                ax.set_title(f'{name}')
            fig3.colorbar(im, orientation='vertical')
            plt.savefig(f'{save_dir}/epoch_{epoch}_L_{data_idx}.png')

        
    plt.close('all')
    

def visualize_wave(data, batch_pred_u_next=None, batch_pred_rhs=None, L_vis=None, epoch=0, save_dir='results/exp-test', dirichlet_idx=5):
    save_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)        

    batch_idx = data.batch
    batch_size = torch.unique(batch_idx).size(0)

    data = data#.clone().detach().cpu().numpy()
    node_dim = data.x.shape[-1]
    num_node = data.x.shape[0]
    node_attr = data.x.reshape((batch_size, -1, node_dim)).clone().cpu().numpy()
    # gt = data.y.clone().cpu().numpy()
    # pred = batch_pred_u_next.clone().cpu().numpy()
    
    gt = data.y.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_p_next = data.u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_rhs = data.rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_u_next = batch_pred_u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_rhs = batch_pred_rhs.reshape((batch_size, -1)).clone().cpu().numpy()
    prev = data.prev[:, 0].reshape((batch_size, -1)).clone().cpu().numpy()
    node_pos = node_attr[:, :,:2]
    dirichlet_mask = node_attr[:, :, dirichlet_idx]
    dirichlet_nodes = np.where(dirichlet_mask > 0)[1].reshape(batch_size, -1, 1)

    # u_next = node_attr[:, :, 2]

    if L_vis is not None:
        mid_range = num_node//2 - 5
        L_real = L_vis['A'].reshape(num_node, num_node)
        L_real_vis = np.concatenate([np.concatenate([L_real[:10, :10], L_real[:10, mid_range:mid_range+10], L_real[:10, -10:]]), 
                                    np.concatenate([L_real[mid_range:mid_range+10, :10], L_real[mid_range:mid_range+10, mid_range:mid_range+10], L_real[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_real[-10:, :10], L_real[-10:, mid_range:mid_range+10], L_real[-10:, -10:]])], axis=-1)
        L_ic = L_vis['ic'].reshape(num_node, num_node)
        L_ic_vis = np.concatenate([np.concatenate([L_ic[:10, :10], L_ic[:10, mid_range:mid_range+10], L_ic[:10, -10:]]), 
                                    np.concatenate([L_ic[mid_range:mid_range+10, :10], L_ic[mid_range:mid_range+10, mid_range:mid_range+10], L_ic[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_ic[-10:, :10], L_ic[-10:, mid_range:mid_range+10], L_ic[-10:, -10:]])], axis=-1)
        L_pred = L_vis['pred'].reshape(num_node, num_node)
        L_pred_vis = np.concatenate([np.concatenate([L_pred[:10, :10], L_pred[:10, mid_range:mid_range+10], L_pred[:10, -10:]]), 
                                    np.concatenate([L_pred[mid_range:mid_range+10, :10], L_pred[mid_range:mid_range+10, mid_range:mid_range+10], L_pred[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_pred[-10:, :10], L_pred[-10:, mid_range:mid_range+10], L_pred[-10:, -10:]])], axis=-1)

        min_val = np.min([np.min(L_ic_vis), np.min(L_pred_vis)])
        max_val = np.max([np.max(L_ic_vis), np.max(L_pred_vis)])
    
    for data_idx in range(batch_size):
        cur_node_pos = node_pos[data_idx]
        cur_dirichlet_mask = dirichlet_nodes[data_idx]
        cur_gt_u_next = gt_p_next[data_idx]
        cur_gt_rhs = gt_rhs[data_idx]
        cur_pred_u_next = pred_u_next[data_idx]
        cur_pred_rhs = pred_rhs[data_idx]
        cur_prev = prev[data_idx]
        min_x = cur_node_pos[:, 0].min()
        max_x = cur_node_pos[:, 0].max()
        min_y = cur_node_pos[:, 1].min()
        max_y = cur_node_pos[:, 1].max()

        H = 2e-3
        # Extract the current and next u.
        pred_diff_u_next = np.max(np.abs(cur_pred_u_next - cur_gt_u_next))
        # Visualize the field before and after simulation.
        fig = plt.figure(figsize=(40, 8))
        for i, (value, name) in enumerate([(cur_gt_u_next, ' gt p'), (cur_pred_u_next, 'pred p'), (cur_pred_u_next - cur_gt_u_next, 'd/f: {:.8f}'.format(pred_diff_u_next))]):
            ax = fig.add_subplot(131 + i, projection='3d')
            value = np.array(value).reshape(-1)
            # recompute u_next:
            value_u = cur_prev + H * value
            # ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.plot_trisurf(cur_node_pos[:, 0], cur_node_pos[:, 1], value_u, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], 0.0, color='tab:green', marker='o', s=0.6, label='boundary')
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_u_next.png')
        pred_diff_rhs = np.max(np.abs(cur_pred_rhs - cur_gt_rhs))
        fig2 = plt.figure(figsize=(40, 8))
        print(cur_gt_rhs.shape)
        for i, (value, name) in enumerate([(cur_gt_rhs, ' gt rhs'), (cur_pred_rhs, 'pred rhs'), (cur_pred_rhs - cur_gt_rhs, 'd/f: {:.8f}'.format(pred_diff_rhs))]):
            ax = fig2.add_subplot(131 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.6, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_rhs.png')
        # visualize the predicted L and gt L
        if L_vis is not None:
            fig3 = plt.figure(figsize=(40, 10))
            for i, (value, name) in enumerate([(L_real_vis, 'real'), (L_ic_vis, 'ic'), (L_pred_vis, 'pred')]):
                ax = fig3.add_subplot(131+i)
                im = ax.imshow(value, cmap='coolwarm', vmin=min_val, vmax=max_val)
                ax.set_title(f'{name}')
            fig3.colorbar(im, orientation='vertical')
            plt.savefig(f'{save_dir}/epoch_{epoch}_L_{data_idx}.png')

        
    plt.close('all')


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    return l


def visualize_ablation(data, batch_pred_u_next=None, L_vis=None, epoch=0, save_dir='results/exp-test', dirichlet_idx=3):
    save_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)        

    batch_idx = data.batch
    batch_size = torch.unique(batch_idx).size(0)

    data = data#.clone().detach().cpu().numpy()
    node_dim = data.x.shape[-1]
    num_node = data.x.shape[0]
    node_attr = data.x.reshape((batch_size, -1, node_dim)).clone().cpu().numpy()
    # gt = data.y.clone().cpu().numpy()
    # pred = batch_pred_u_next.clone().cpu().numpy()
    
    gt = data.y.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_u = data.u.reshape((batch_size, -1)).clone().cpu().numpy()
    gt_u_next = data.u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    pred_u_next = batch_pred_u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    node_pos = node_attr[:, :,:2]
    dirichlet_mask = node_attr[:, :, dirichlet_idx]
    dirichlet_nodes = np.where(dirichlet_mask > 0)[1].reshape(batch_size, -1, 1)

    # u_next = node_attr[:, :, 2]

    if L_vis is not None:
        mid_range = num_node//2 - 5
        L_real = L_vis['A'].reshape(num_node, num_node)
        L_real_vis = np.concatenate([np.concatenate([L_real[:10, :10], L_real[:10, mid_range:mid_range+10], L_real[:10, -10:]]), 
                                    np.concatenate([L_real[mid_range:mid_range+10, :10], L_real[mid_range:mid_range+10, mid_range:mid_range+10], L_real[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_real[-10:, :10], L_real[-10:, mid_range:mid_range+10], L_real[-10:, -10:]])], axis=-1)
        L_ic = L_vis['ic'].reshape(num_node, num_node)
        L_ic_vis = np.concatenate([np.concatenate([L_ic[:10, :10], L_ic[:10, mid_range:mid_range+10], L_ic[:10, -10:]]), 
                                    np.concatenate([L_ic[mid_range:mid_range+10, :10], L_ic[mid_range:mid_range+10, mid_range:mid_range+10], L_ic[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_ic[-10:, :10], L_ic[-10:, mid_range:mid_range+10], L_ic[-10:, -10:]])], axis=-1)
        L_pred = L_vis['pred'].reshape(num_node, num_node)
        L_pred_vis = np.concatenate([np.concatenate([L_pred[:10, :10], L_pred[:10, mid_range:mid_range+10], L_pred[:10, -10:]]), 
                                    np.concatenate([L_pred[mid_range:mid_range+10, :10], L_pred[mid_range:mid_range+10, mid_range:mid_range+10], L_pred[mid_range:mid_range+10, -10:]]),
                                    np.concatenate([L_pred[-10:, :10], L_pred[-10:, mid_range:mid_range+10], L_pred[-10:, -10:]])], axis=-1)


    
    for data_idx in range(batch_size):
        cur_node_pos = node_pos[data_idx]
        cur_dirichlet_mask = dirichlet_nodes[data_idx]
        cur_gt_u = gt_u[data_idx]
        cur_gt_u_next = gt_u_next[data_idx]
        cur_pred_u_next = pred_u_next[data_idx]

        # Extract the current and next u.
        pred_diff_u_next = np.max(np.abs(cur_pred_u_next - cur_gt_u_next))
        # Visualize the field before and after simulation.
        fig = plt.figure()
        for i, (value, name) in enumerate([(cur_gt_u_next, ' gt u'), (cur_pred_u_next, 'pred u'), (cur_pred_u_next - cur_gt_u_next, 'd/f: {:.2f}'.format(pred_diff_u_next))]):
            ax = fig.add_subplot(141 + i)
            value = np.array(value).reshape(-1)
            ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=-1, vmax=1, cmap='coolwarm')
            ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', marker='o', s=0.3, label='boundary')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        plt.savefig(f'{save_dir}/epoch_{epoch}_data_{data_idx}_u_next.png')
        # visualize the predicted L and gt L
        if L_vis is not None:
            fig3 = plt.figure(figsize=(40, 10))
            for i, (value, name) in enumerate([(L_real_vis, 'A'), (L_ic_vis, 'ic'), (L_pred_vis, 'pred')]):
                ax = fig3.add_subplot(131+i)
                im = ax.imshow(value, cmap='coolwarm')
                fig3.colorbar(im, orientation='vertical')
                ax.set_title(f'{name}')
            plt.savefig(f'{save_dir}/epoch_{epoch}_L_{data_idx}.png')

        
    plt.close('all')


'''
def visualize_flow_old(data, batch_pred_u_next, epoch=0, save_dir='results/exp-test', dirichlet_idx=3):
    
    save_dir = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)        

    batch_idx = data.batch
    batch_size = torch.unique(batch_idx).size(0)

    data = data#.clone().detach().cpu().numpy()
    node_dim = data.x.shape[-1]
    node_attr = data.x.reshape((batch_size, -1, node_dim)).clone().cpu().numpy()
    # gt = data.y.clone().cpu().numpy()
    # pred = batch_pred_u_next.clone().cpu().numpy()
    
    gt = data.y.reshape((batch_size, -1)).clone().cpu().numpy()
    pred = batch_pred_u_next.reshape((batch_size, -1)).clone().cpu().numpy()
    node_pos = node_attr[:, :,:2]
    dirichlet_mask = 1 - node_attr[:, :, dirichlet_idx]
    # dirichlet_mask = 1 - node_attr[:, :, dirichlet_idx]
    dirichlet_nodes = np.where(dirichlet_mask > 0)[1].reshape(batch_size, -1, 1)

    # u = node_attr[:, :, 2]

    for data_idx in range(batch_size):
        cur_node_pos = node_pos[data_idx]
        cur_dirichlet_mask = dirichlet_nodes[data_idx]
        # cur_u = u[data_idx]
        cur_gt_u_next = gt[data_idx]
        cur_pred_u_next = pred[data_idx]

        # Extract the current and next u.
        pred_diff = (np.abs(cur_pred_u_next - cur_gt_u_next)) / (np.abs(cur_gt_u_next))
        # Visualize the field before and after simulation.
        fig = plt.figure()
        for i, (value, name) in enumerate([(cur_gt_u_next, 'gt after'), (cur_pred_u_next, 'pred after'), (pred_diff, 'relative d/f')]): #(cur_pred_u_next - cur_gt_u_next, 'relative d/f: {:.2f}'.format(pred_diff))]):
            ax = fig.add_subplot(141 + i)
            value = np.array(value).reshape(-1)
            if name != 'relative d/f':
                ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=np.min(cur_gt_u_next), vmax=np.max(cur_gt_u_next), cmap='coolwarm')
                ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', s=0.01, marker='o', label='boundary')
            else:
                ax.tripcolor(cur_node_pos[:, 0], cur_node_pos[:, 1], value, vmin=0, vmax=1, cmap='coolwarm')
                ax.scatter(cur_node_pos[cur_dirichlet_mask, 0], cur_node_pos[cur_dirichlet_mask, 1], color='tab:green', s=0.01, marker='o', label='boundary')
            ax.set_aspect('equal')def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    return l
.png')
    plt.close('all')
'''

