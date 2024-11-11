import dis
from importlib import import_module
import os
import pickle
import time
import sys
from sys import path
import time
from pathlib import Path
from subprocess import call

import argparse
import numpy as np
import tqdm
import logging

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import TwoHop

from torch.utils.tensorboard import SummaryWriter

# for parallel: 
from torch_geometric.data import DataListLoader

import importlib
from dataset.synthetic_dataset import SyntheticDataset
from dataset.heat_dataset import HeatDataset
from dataset.heat_dataset_multisource import HeatDatasetMultiSource
from dataset.inviscidflow_dataset import InviscidFlowDataset
from dataset.inviscidflow_dataset_multisource import InviscidFlowDatasetMultiSource
from config import build_parser
from utils.training_utils import train_epoch, val_epoch
from utils.utils import get_free_gpus, save_model, load_model, get_delta_minute, ExpLR
from utils.data_config import heat_train_config, heat_test_config, inviscidflow_train_config, inviscidflow_test_config, wave_train_config, wave_test_config, poisson3d_train_config, poisson3d_test_config
from utils.visualization_utils import create_logger, visualize_flow, visualize_heat, visualize_wave
from utils.distance_metric_utils import distance_metric


num_threads = 8

# #########
# # local application imports
# # get path to root of the project
# mgn_code_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
# path.append(mgn_code_dir)

#########
# load config file
parser = build_parser()
args = parser.parse_args()
args.exp_name = args.exp_name + "-" + time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(save_dir, exist_ok=True)
logger = create_logger('global_logger', save_dir + '/log.txt')
logger.info(args)
num_threads = args.cpu
torch.set_num_threads(num_threads)


#########
# prepare train / test sets
if args.dataset in ['flow', 'inviscidflowmultisource', 'flowmultisource', 'flowoutdomain', 'flowgeneralize']:
    inviscidflow_train_config['name'] = args.mesh
    inviscidflow_train_config['density'] = [float(x) for x in args.param.split('-')[:-1]]
    inviscidflow_test_config['name'] = args.mesh
    inviscidflow_test_config['density'] = [float(args.param.split('-')[-1])]
    print(inviscidflow_train_config, inviscidflow_test_config)
    train_dataset = InviscidFlowDatasetMultiSource(domain_files_path=args.train_path, config=inviscidflow_train_config, use_data_num=args.use_data_num,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = InviscidFlowDatasetMultiSource(domain_files_path=args.test_path, config=inviscidflow_test_config, use_data_num=args.use_data_num,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    print('training dataset length', len(train_dataset), 'test data length', len(test_dataset))
    visualization_func = visualize_flow
elif args.dataset in ['heat', 'heatmultisource', 'heatoutdomain', 'heatgeneralize']:
    heat_train_config['name'] = args.mesh
    heat_train_config['diffusivities'] = [float(x) for x in args.param.split('-')[:-1]]
    heat_test_config['name'] = args.mesh
    heat_test_config['diffusivities'] = [float(args.param.split('-')[-1])]
    print(heat_train_config, heat_test_config)
    train_dataset = HeatDatasetMultiSource(config=heat_train_config,  use_data_num=4800,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = HeatDatasetMultiSource(config=heat_test_config, use_data_num=1,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    visualization_func = visualize_heat
    print('training dataset length', len(train_dataset), 'test data length', len(test_dataset))
elif args.dataset in ['heat3d', 'heat3dmultisource', 'heat3doutdomain', 'heat3dgeneralize']:
    heat_train_config['name'] = args.mesh
    heat_train_config['diffusivities'] = [float(x) for x in args.param.split('-')[:-1]]
    heat_test_config['name'] = args.mesh
    heat_test_config['diffusivities'] = [float(args.param.split('-')[-1])]
    train_dataset = HeatDataset3DMultiSource(config=heat_train_config,  use_data_num=args.use_data_num,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = HeatDataset3DMultiSource(config=heat_test_config, use_data_num=1,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    visualization_func = visualize_heat
    print('training dataset length', len(train_dataset), 'test data length', len(test_dataset))
elif args.dataset in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
    wave_train_config['name'] = args.mesh
    wave_train_config['speed'] = [float(x) for x in args.param.split('-')[:-1]]
    wave_test_config['name'] = args.mesh
    wave_test_config['speed'] = [float(args.param.split('-')[-1])]
    print(wave_train_config)
    print(wave_test_config)
    train_dataset = WaveDatasetMultiSource(domain_files_path=args.train_path, config=wave_train_config, use_data_num=wave_train_config['total_data_num'],
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = WaveDatasetMultiSource(domain_files_path=args.test_path, config=wave_test_config, use_data_num=1,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    print('training dataset length', len(train_dataset), 'test data length', len(test_dataset))
    visualization_func = visualize_wave
elif args.dataset in ['synthetic', 'syn']:
    poisson3d_train_config['name'] = args.mesh
    poisson3d_test_config['name'] = args.mesh
    train_dataset = SyntheticDataset(domain_files_path=args.train_path, config=poisson3d_train_config,  use_data_num=args.use_data_num,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = SyntheticDataset(domain_files_path=args.test_path, config=poisson3d_test_config, use_data_num=1,
                                use_high_freq=args.high_freq, augment_edge=args.augment_edge, use_pred_x = args.use_pred_x, high_freq_aug=args.high_freq_aug)
    visualization_func = None
    print('training dataset length', len(train_dataset), 'test data length', len(test_dataset))
else:
    raise Exception(f' dataset {args.dataset} is not defined')

node_attr_dim = train_dataset.node_attr_dim
edge_attr_dim = train_dataset.edge_attr_dim
num_edges = train_dataset.num_edges
out_node_dim = train_dataset.output_dim
out_b_dim = train_dataset.b_dim
dirichlet_idx = train_dataset.dirichlet_idx


#########
# build model
importlib.invalidate_caches()
model_def = importlib.import_module('models.' + args.model)

model = model_def.Net( args, 
        node_attr_dim, edge_attr_dim, 
        out_node_dim, out_b_dim, 
        num_edges=num_edges,
        out_dim_node=args.hidden_dim, 
        out_dim_edge=args.hidden_dim, 
        hidden_dim_node=args.hidden_dim, 
        hidden_dim_edge=args.hidden_dim, 
        hidden_layers_node=args.hidden_layers_encoder,
        hidden_layers_edge=args.hidden_layers_encoder, 
        num_iterations=args.num_iterations, # num_heads=args.num_atten_heads,  
        hidden_dim_processor_node=args.hidden_dim, 
        hidden_dim_processor_edge=args.hidden_dim, 
        hidden_layers_processor_node=args.hidden_layers_processor,
        hidden_layers_processor_edge=args.hidden_layers_processor,
        hidden_dim_decoder=args.hidden_dim,
        hidden_layers_decoder=args.hidden_layers_decoder,
        dirichlet_idx=dirichlet_idx,
        norm_type=args.norm)
#########
# device settings
free_gpus=[0, 1, 2]
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    free_gpus = list(set(free_gpus).intersection(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))))
device = torch.device("cuda:0") if torch.cuda.is_available() and len(free_gpus) > 0 else "cpu"

# TODO: use training set: it seems like this introduces domain gap and does not converge.
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# train_loader = DataLoader(test_dataset[:args.use_data_num], batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#########
# optimizer settings
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler == 'ExpLR':
    lr_scheduler = ExpLR(optimizer, decay_steps=4e4, min_lr=1e-8) 
elif args.scheduler == 'OneCycleLR':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.scheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, min_lr=1e-8)
elif args.scheduler == 'MultiStepLR':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 800], gamma=0.1)
else:
    raise Exception('LR scheduler not recognized')

loss_dict = distance_metric(args.loss)


#########
# reload model, if available
model, lr_scheduler, last_epoch = load_model(args.ckpt, 
    'best_val.pt', model, 
    'best_val_scheduler.pt', lr_scheduler,  
    'last_epoch.pt')

model = model.to(device)
# model = model.double()

if args.ckpt=='': 
    args.ckpt = os.path.join(save_dir, 'model')
    os.makedirs(args.ckpt, exist_ok=True)


#########
# tensorboard settings
tb_dir = os.path.join(args.save_dir, args.exp_name, 'tb')
# create a summary writer.
tb_writer = None if not args.tensorboard else SummaryWriter(tb_dir)
total_steps = (last_epoch + 1) * len(train_loader) #assumes batch size does not change on continuation

########
# save relevant files
src_dir = os.path.join(args.save_dir, args.exp_name, 'src')
os.makedirs(src_dir)
call('cp ./utils/training_utils.py '+src_dir, shell=True)
call('cp ./utils/data_config.py '+src_dir, shell=True)
call(f'cp ./models/{args.model}.py '+src_dir, shell=True)
call('cp ./scripts/train*.sh '+src_dir, shell=True)

########
# training loop
best_pcg_iteration, best_loss = val_epoch(args, test_loader, model, loss_dict, device, visualization_freq=args.log_freq, \
                        visualize=visualization_func ,save_dir=save_dir, logger=logger)

best_save_metric = best_loss if best_pcg_iteration == 0 else best_pcg_iteration
start_time = time.time()
tqdm_itr = tqdm.trange(last_epoch + 1, args.epochs, position=0, leave=True)
for i in tqdm_itr:
    tqdm_itr.set_description('Epoch')
    logger.info('Epoch: {}/{}, lr: {}'.format(i, args.epochs, optimizer.param_groups[0]['lr']))
    train_loss, total_steps = train_epoch(args, train_loader, model, loss_dict,  
        optimizer, lr_scheduler, device, tb_writer, args.log_freq, total_steps, epoch=i, logger=logger)
    tqdm_itr.refresh()
    if i % args.val_freq == 0:
        pcg_iteration, val_loss = val_epoch(args, test_loader, model, loss_dict, device, tb_writer, total_steps, i, visualization_freq=args.log_freq, \
                             visualize=visualization_func, save_dir=save_dir, logger=logger)
        tqdm_itr.refresh()
        logger.info('Finish Training of Epoch: {}, Train loss: {}, Val Loss: {}'.format(i, train_loss, val_loss))
    
        save_metric = val_loss if pcg_iteration == 0 else pcg_iteration
        if save_metric < best_save_metric:
            best_save_metric = save_metric
            save_model(args.ckpt, 
                'best_val.pt', model, 
                'best_val_scheduler.pt', lr_scheduler,  
                'last_epoch.pt', i)

        print('saveing model ... ')
        save_model(args.ckpt, 
        'latest_model.pt', model, 
        'latest_val_scheduler.pt', lr_scheduler,  
        'latest_last_epoch.pt', i)

    if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
        lr_scheduler.step(val_loss)
    elif 'MultiStepLR' in str(lr_scheduler.__class__):
        lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.add_scalar("Opt/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=total_steps)
        tb_writer.add_scalar("Profile/epoch_time", scalar_value=get_delta_minute(start_time), global_step=i)



