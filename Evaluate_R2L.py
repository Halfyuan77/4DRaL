import sys
import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, SubsetRandomSampler

from config.config import get_config
from dataset.ntu4d import EvaluateDataset
from model.resnet18_place import resnet_place_stu

from utils.train_tools import *
import time

cfg = get_config()
cfg.batch_size = 40
cfg.debug = True


def generate_database(model, test_data_loader, state='db', is_gpu=True):
    
    model.eval()
    model.to('cuda')
    num_iterations = 100
    count = 0
    total_time = 0.0
    with torch.no_grad():
        
        all_global_descs = []
        all_local_feats = []
        all_global_descs_gpu = []
        for _, (imgs, _) in enumerate(tqdm(test_data_loader)):    
            imgs = imgs.to(device)
            
            start_time = time.time()
            global_desc= model(imgs)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            count += 1
            if count % num_iterations == 0:
                average_time = total_time / count
                print(f"Average inference time: {average_time:.6f} seconds")
                count = 0 
                total_time = 0.0
            if is_gpu:
                all_global_descs_gpu.append(global_desc)
            else:
                all_global_descs.append(global_desc.detach().cpu().numpy())
            
    if is_gpu:
        return torch.concatenate(all_global_descs_gpu, dim=0)
    else:
        return np.concatenate(all_global_descs, axis=0)


def generate_database_v1(model, test_data_loader, state='db', is_gpu=True):
    
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        
        all_global_descs = []
        all_local_feats = []
        all_global_descs_gpu = []
        for _, (imgs, _) in enumerate(tqdm(test_data_loader)):    
            imgs = imgs.to(device)
            _, global_desc= model(imgs)
            if is_gpu:
                all_global_descs_gpu.append(global_desc)
            else:
                all_global_descs.append(global_desc.detach().cpu().numpy())
            
    if is_gpu:
        return torch.concatenate(all_global_descs_gpu, dim=0)
    else:
        return np.concatenate(all_global_descs, axis=0)

def generate_database_v2(model, test_data_loader, state='db', is_gpu=True):
    
    model.eval()
    model.to('cuda')
    num_iterations = 100
    count = 0
    total_time = 0.0
    with torch.no_grad():
        
        all_global_descs = []
        all_local_feats = []
        all_global_descs_gpu = []
        for _, (imgs, _) in enumerate(tqdm(test_data_loader)):    
            imgs = imgs.to(device)
            
            start_time = time.time()
            _,_, global_desc= model(imgs)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            count += 1
            if count % num_iterations == 0:
                average_time = total_time / count
                print(f"Average inference time: {average_time:.6f} seconds")
                count = 0 
                total_time = 0.0
            if is_gpu:
                all_global_descs_gpu.append(global_desc)
            else:
                all_global_descs.append(global_desc.detach().cpu().numpy())

    if is_gpu:
        return torch.concatenate(all_global_descs_gpu, dim=0)
    else:
        return np.concatenate(all_global_descs, axis=0)


if __name__ == "__main__":

    device = torch.device("cuda")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    
    sensor = 'radar'
    is_gpu = True
    writer = None
    db_feature = None
    
    model = resnet_place_stu()
    model = model.cuda()
    model.load_state_dict(torch.load(cfg.checkpoint + 'R2L_table' +'.pth'))
    n_params = sum([param.nelement() for param in model.parameters()])
    
    print(f'Number of model parameters: {n_params}')
    
    test_name = ['ntu4d', 'test_a', 'test_b']
    db_loader_list = []
    test_loader_list = []
    
    for name in test_name:
        db_dataset = EvaluateDataset(dataset_name=name, dataset_path=cfg.dataset_path,  is_query=False, sensor='lidar', is_gray=True)
        test_dataset = EvaluateDataset(dataset_name=name, dataset_path=cfg.dataset_path,  is_query=True, sensor='radar', is_gray=True)
        db_loader_list.append(DataLoader(dataset=db_dataset, num_workers=0, batch_size=cfg.batch_size, shuffle=False))
        test_loader_list.append(DataLoader(dataset=test_dataset, num_workers=0, batch_size=cfg.batch_size, shuffle=False))    
    
    recalls = None
    
    for loader_index in range(len(db_loader_list)):

        print('\n', test_name[loader_index], 'generate database feature')
        db_feature = generate_database_v1(model, db_loader_list[loader_index], state='db', is_gpu=is_gpu)
        print('\n', test_name[loader_index], 'generate query feature')
        query_feature = generate_database_v1(model, test_loader_list[loader_index], state='test', is_gpu=is_gpu)

        if is_gpu:
            recalls = evaluate_gpu(db_feature, query_feature, cfg.dataset_path, 0, writer=writer, evaluate_name=test_name[loader_index])
        else:
            recalls = evaluate(db_feature, query_feature, 0)
    
    print('finish')
