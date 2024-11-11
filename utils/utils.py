
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
from pcg import jacobi_torch, ic_torch_optimize_st
from utils.data_utils import *
from utils.data_config import heat_train_config, heat_test_config
from base.base_dataset import FEMDataset


def load_vectorxr(file_name):
    f = open(file_name, 'rb')
    row_num = struct.unpack('@i', f.read(4))[0]
    
    value = []
    for _ in range(row_num):
        v = struct.unpack('@d', f.read(8))[0]
        value.append(v)
    return np.array(value)

def write_vectorxr(file_name, data):
    f=open(file_name,"wb")
    # print(data)
    data = data.tolist()
    myfmt='d'*len(data)
    bin=struct.pack('i', len(data))
    #  You can use 'd' for double and < or > to force endinness
    bin+=struct.pack(myfmt,*data)
    f.write(bin)
    f.close()


# convert (pred, gt) pair of preconditioners to dense representations
def convert_precond(pred, gt, pred_edge_index, batch, max_num_nodes):
    pred_precond = torch_geometric.utils.to_dense_adj(pred_edge_index, batch=batch, edge_attr=pred, max_num_nodes=max_num_nodes)

    gt_precond = torch_geometric.utils.to_dense_adj(gt[:, :2].long(), batch=batch, edge_attr=gt[:, -1], max_num_nodes=max_num_nodes)

    return pred_precond, gt_precond

def get_free_gpus(mem_threshold=20000):

    '''
    Gets current free gpus

    mem_threshold: maximum allowed memory currently in use to be considered a free gpu
    '''

    with popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used') as f:
        gpu_info = f.readlines()
    
    memory_available = [int(x.split()[2]) for x in gpu_info]
    free_gpus = [i for i, mem in enumerate(memory_available) if mem <= mem_threshold]
    free_gpus = [0]
    return free_gpus

def save_model(checkpoint_dir, 
        model_filename, model, 
        scheduler_filename=None, scheduler=None,
        epoch_filename=None, epoch=None):

    '''
    Saves model checkpoint
    '''

    save(model.state_dict(), join(checkpoint_dir, model_filename))

    if (scheduler_filename is not None) and (scheduler is not None):
        save(scheduler.state_dict(), join(checkpoint_dir, scheduler_filename))

    if epoch is not None:
        save(epoch, join(checkpoint_dir, epoch_filename))

def load_model(checkpoint_dir, 
        model_filename, model,  
        scheduler_filename=None, scheduler=None,  
        epoch_filename=None):

    '''
    Loads model checkpoint
    '''

    if exists(join(checkpoint_dir, model_filename)) and model is not None:
        model.load_state_dict(load(join(checkpoint_dir, model_filename)))

    if exists(join(checkpoint_dir, scheduler_filename)) and scheduler is not None:
        scheduler.load_state_dict(load(join(checkpoint_dir, scheduler_filename)))

    if exists(join(checkpoint_dir, epoch_filename)):
        last_epoch = load(join(checkpoint_dir, epoch_filename))
    else:
        last_epoch = -1

    return model, scheduler, last_epoch

# def load_decoded_A(checkpoint_dir, model_filename):
#     model, _, _ =  load_model(checkpoint_dir, 
#             model_filename, model,  
#             scheduler_filename=None, scheduler=None,  
#             epoch_filename=None)
    

def get_delta_minute(start_time):
    # start_time in seconds
    return (time.time() - start_time) / 60.

class ExpLR(_LRScheduler):
    '''
    Exponential learning rate scheduler
    Based on procedure described in LTS
    If min_lr==0 and decay_steps==1, same as torch.optim.lr_scheduler.ExpLR 
    '''

    def __init__(self, optimizer, decay_steps=10000, gamma=0.1, min_lr=1e-6, last_epoch=-1):

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.gamma = gamma
        self.decay_steps = decay_steps

        super(ExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [min_lr + max(base_lr - min_lr, 0) * pow(self.gamma, self.last_epoch / self.decay_steps) 
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
