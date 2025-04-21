import cv2
import os
import torch
import pickle
import faiss
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
from dataset.ntu4d import TrainingTuple, EvaluationTuple
import matplotlib.pyplot as plt


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.1
        # print(self.margin)
        # self.margin = 0.01
    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()


def evaluate_gpu(global_features, query_features, dataset_path, epoch, writer, evaluate_name='ntu4d'):

    dot_product = torch.matmul(query_features, global_features.t())  # (4306, 10839)

    query_norm = torch.norm(query_features, dim=1, keepdim=True) ** 2  # (4306, 1)

    global_norm = torch.norm(global_features, dim=1) ** 2  # (10839,)

    distances = query_norm + global_norm - 2 * dot_product  # (4306, 10839)

    k = 25
    top_k_50_distances, top_k_50_indices = torch.topk(distances, k, dim=1, largest=False, sorted=True)

    top_k_10_indices = torch.zeros(top_k_50_distances.shape[0], 10).to(int)

    for b_idx in range(top_k_50_distances.shape[0]):
        unique_distance, inverse_indices = torch.unique(top_k_50_distances[b_idx], return_inverse=True)
        seen = set()
        count = 0
        for i, idx in enumerate(inverse_indices):
            if idx.item() not in seen:
                seen.add(idx.item())
                top_k_10_indices[b_idx][count] = i
                count += 1
            if count == 10:
                break

    top_k_10_indices = top_k_10_indices.to(top_k_50_indices.device)
    top_10_indices = torch.gather(top_k_50_indices, dim=1, index=top_k_10_indices)
    n_values = [1, 5, 10]
    print('====> Search...')

    path = dataset_path
    if evaluate_name == 'ntu4d':
        path += '/ntu4d'
        query_filepath = os.path.join(path, 'ntu4d_val_evaluation_query_25.pickle')
    elif evaluate_name == 'test_a':
        path += '/sjtu-rsvi/test_a'
        query_filepath = os.path.join(path, 'sjtu-rsvi_test_a_evaluation_query_25.pickle')
    elif evaluate_name == 'test_b':
        path += '/sjtu-rsvi/test_b'
        query_filepath = os.path.join(path, 'sjtu-rsvi_test_b_evaluation_query_25.pickle')
    
    queries: Dict[int, TrainingTuple] = pickle.load(open(query_filepath, 'rb'))
    correct_at_n = np.zeros(len(n_values))
    whole_test_size = 0
    
    print('====> Calculate...')
    
    faid_inx = []
    faid_pos = []

    top_10_indices = top_10_indices.detach().cpu().numpy()
    for qIx, pred in enumerate(top_10_indices):

        if len(queries[qIx]['positives']) == 0:
            continue
        whole_test_size+=1

        is_find = False
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], queries[qIx]['positives'])):
                correct_at_n[i:] += 1
                is_find = True
                break
        
        if not is_find:
            faid_inx.append(queries[qIx]['index'])
            faid_pos.append(queries[qIx]['positives'])

    recall_at_n = correct_at_n / whole_test_size

    recalls = {}
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print(evaluate_name,"====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
    
    return recalls

 
def evaluate(global_features, query_features, epoch, writer, debug=False, test_flag=True):

    # 全局特征
    global_features = np.vstack(global_features)
    query_features = np.vstack(query_features)

    query_feat = query_features.astype('float32')
    db_feat = global_features.astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(query_feat.shape[1])
    faiss_index.add(db_feat)

    print('====> Search...')
    n_values = [1, 5, 10]
    _, predictions = faiss_index.search(query_feat, max(n_values))

    query_filepath = os.path.join('/data/hny/dataset/4Dradar', 'ntu4d_val_evaluation_query_25.pickle')
    queries: Dict[int, TrainingTuple] = pickle.load(open(query_filepath, 'rb'))


    correct_at_n = np.zeros(len(n_values))
    whole_test_size = 0

    
    print('====> Calculate...')
    for qIx, pred in enumerate(predictions):

        # query_frame = list(dict)

        if len(queries[qIx]['positives']) == 0:
            continue
        whole_test_size+=1

        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], queries[qIx]['positives'])):
                correct_at_n[i:] += 1
                break
        
    recall_at_n = correct_at_n / whole_test_size

    recalls = {}
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
    
    if not debug:
        writer.add_scalar('recall_1', recall_at_n[0], ((epoch)))
        writer.add_scalar('recall_5', recall_at_n[1], ((epoch)))
        writer.add_scalar('recall_10', recall_at_n[2], ((epoch)))
    
    return recalls






