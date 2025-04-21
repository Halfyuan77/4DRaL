import os
import cv2
import pickle
from typing import List
from typing import Dict
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert (position.shape == (2,) or position.shape == (3,))

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, index: int, file: str, northing:float, easting:float, indices: np.array):
        # position: x, y position in meters
        # "index": index,
        # "file": row["file"],
        # "northing": row["northing"],
        # "easting": row["easting"],
        # "positives": indices,  # Indices of the positive matches

        self.index = index
        self.file = file
        self.northing = northing
        self.easting = easting
        self.indices = indices

    def to_tuple(self):
        return self.id, self.file, self.northing, self.easting, self.indices


def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices

class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, sensor='radar', is_rotation=True, is_gray=True):
        
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)

        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.file_template = "{:05d}.jpg"
        self.query_len = len(self.queries)
        self.db_len = 9786
        self.bev_path = None
        self.is_rotation = is_rotation
        
        self.all_numbers = set(range(self.db_len))
        self.positives_per_query = 1
        self.negatives_per_query = 4
        
        if sensor == 'radar':
            self.bev_path = 'radar_bev/'
        elif sensor == 'lidar':
            self.bev_path = 'lidar_bev/'
        else:
            print('error!!!')
        print(f"Number of queries: {len(self)}")

        self.is_gray = is_gray
        # pc_loader must be set in the inheriting class
    def __len__(self):
        # return len(self.queries)
        return self.query_len

    def get_bev_file(self, ndx):
        
        path = self.dataset_path + '/ntu4d/train/query/' + self.bev_path
        bev_file = path + self.file_template.format(self.queries[ndx].id)
 
        return bev_file
    
    def __getitem__(self, ndx):
        
        # Load point cloud and apply transform
        bev_query_name = self.get_bev_file(ndx)

        if self.is_gray:
            query = cv2.imread(bev_query_name, cv2.IMREAD_GRAYSCALE)
        else:
            query = cv2.imread(bev_query_name)
        if self.is_rotation:
            mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(-30,30), 1)
            query = cv2.warpAffine(query, mat, query.shape[:2])
        
        positive_list = self.get_positives(ndx)
        if self.is_gray:
            query = np.expand_dims(query, axis=0)
            positive = cv2.imread(positive_list[0], cv2.IMREAD_GRAYSCALE)
        else:
            query = query.transpose(2,0,1)
            positive = cv2.imread(positive_list[0])
        query = (query.astype(np.float32))/256
        
        if self.is_rotation:           
            mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(-30,30), 1)
            positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        
        if self.is_gray:    
            positive = np.expand_dims(positive, axis=0)
        else:
            positive = positive.transpose(2,0,1)
        
        positive = (positive.astype(np.float32))/256
        
        negative_list = self.get_non_negatives(ndx)
        negatives = []

        for neg_i in negative_list:
            
            if self.is_gray:  
                negative = cv2.imread(neg_i, cv2.IMREAD_GRAYSCALE)
            else:
                negative = cv2.imread(neg_i)
            if self.is_rotation:
                mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(-30,30), 1)
                negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
                
            if self.is_gray: 
                negative = np.expand_dims(negative, axis=0)
            else:
                negative = negative.transpose(2,0,1)
                
            negative = (negative.astype(np.float32))/256
            negatives.append(torch.from_numpy(negative))
        
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, ndx

    def get_positives(self, ndx):
        
        if ndx >= self.query_len:    
            path = self.dataset_path + '/ntu4d/train/query/' + self.bev_path
        else:
            path = self.dataset_path + '/ntu4d/train/database/' + self.bev_path
        
        pos_inx = list(self.queries[ndx].positives)
        positives = []
        
        if(len(pos_inx) < 1):
            print('error: ', ndx)
        
        pos_inx = random.sample(pos_inx, self.positives_per_query)
        for idx in range(len(pos_inx)):
            # if pos_inx[idx] > self.query_len:
            #     positives.append(path + self.file_template.format(pos_inx[idx] - self.query_len))
            # else:
            positives.append(path + self.file_template.format(pos_inx[idx]))
        return positives

    def get_non_negatives(self, ndx):
        
        if ndx >= self.query_len:    
            path = self.dataset_path + '/ntu4d/train/query/' + self.bev_path
        else:
            path = self.dataset_path + '/ntu4d/train/database/' + self.bev_path
        
        neg_inx = self.queries[ndx].non_negatives
        lst_set = set(neg_inx)
        neg_inx = list(self.all_numbers.difference(lst_set))
        neg_inx = random.sample(neg_inx, self.negatives_per_query)
        positives = []

        for idx in range(len(neg_inx)):
            # if pos_inx[idx] > self.query_len:
            #     positives.append(path + self.file_template.format(pos_inx[idx] - self.query_len))
            # else:
            positives.append(path + self.file_template.format(neg_inx[idx]))

        return positives

class EvaluateDataset(Dataset):
    
    def __init__(self, dataset_name, dataset_path, is_query=True, sensor='radar', is_gray=False):
        
        # remove_zero_points: remove points with all zero coords
        # assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)

        self.dataset_path = dataset_path
        self.query_filepath = None
        if dataset_name == 'ntu4d':
            self.dataset_path += '/ntu4d'
            self.query_filepath = os.path.join(self.dataset_path, 'ntu4d_val_evaluation_query_25.pickle')
        elif dataset_name == 'test_a':
            self.dataset_path += '/sjtu-rsvi/test_a'
            self.query_filepath = os.path.join(self.dataset_path, 'sjtu-rsvi_test_a_evaluation_query_25.pickle')
        elif dataset_name == 'test_b':
            self.dataset_path += '/sjtu-rsvi/test_b'
            self.query_filepath = os.path.join(self.dataset_path, 'sjtu-rsvi_test_b_evaluation_query_25.pickle')

        self.queries: Dict[int, EvaluationTuple] = pickle.load(open(self.query_filepath, 'rb'))

        # self.dataset_path = dataset_path

        self.file_template = "{:05d}.jpg"
        self.query_len = 0
        self.bev_path = None
        self.is_query = is_query
        if is_query:
            self.query_len = len(self.queries)
            if sensor == 'radar':
                self.bev_path = '/test/query/radar_bev/'
            elif sensor == 'lidar':
                self.bev_path = '/test/query/lidar_bev/'
            else:
                print('error!!!')
        else:
            self.query_len = len(os.listdir(self.dataset_path+'/test/database/radar_bev/'))
            if sensor == 'radar':
                self.bev_path = '/test/database/radar_bev/'
            elif sensor == 'lidar':
                self.bev_path = '/test/database/lidar_bev/'     
            else:
                print('error!!!')   

        if is_query:
            print(dataset_name, f"===>Number of queries: {len(self)}")
        else:
            print(dataset_name, f"===>Number of database: {len(self)}") 
        self.is_gray = is_gray

        # pc_loader must be set in the inheriting class
    def __len__(self):
        # return len(self.queries)
        return self.query_len
    
    def __getitem__(self, ndx):
        
        # Load point cloud and apply transform
        if self.is_query:
            bev_query_name = self.dataset_path + self.bev_path + self.file_template.format(self.queries[ndx]['index'])
        else:
            bev_query_name = self.dataset_path + self.bev_path + self.file_template.format(ndx)

        if self.is_gray:
            query = cv2.imread(bev_query_name, cv2.IMREAD_GRAYSCALE)
            query = np.expand_dims(query, axis=0)
        else:
            query = cv2.imread(bev_query_name)
            query = query.transpose(2,0,1)
        
        query = (query.astype(np.float32))/256
        
        return query, ndx


def collate_fn_kd(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None, None

    query, lidar_query, positive, lidar_positive, negatives, lidar_negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    lidar_query=np.array(lidar_query)
    lidar_positive=np.array(lidar_positive)    
    
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    lidar_query = data.dataloader.default_collate(lidar_query)
    lidar_positive = data.dataloader.default_collate(lidar_positive)
        
    negatives = torch.cat(negatives, 0)
    lidar_negatives = torch.cat(lidar_negatives, 0)
    indices = list(indices)
    
    return query, lidar_query, positive, lidar_positive, negatives, lidar_negatives, indices


class TrainingKD_Dataset(Dataset):
    
    def __init__(self, dataset_path, query_filename, is_rotation=False, is_gray=True, negatives_per_query=4):
        
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)

        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.file_template = "{:05d}.jpg"
        self.query_len = len(self.queries)
        self.db_len = 9786
        self.bev_path = None
        self.is_rotation = is_rotation
        
        self.all_numbers = set(range(self.db_len))
        self.positives_per_query = 1
        self.negatives_per_query = negatives_per_query
        
        self.radar_path = 'radar_bev/'
        self.lidar_path = 'lidar_bev/'
            
        print(f"trainning set len:: {len(self)}")

        self.is_gray = is_gray
        # pc_loader must be set in the inheriting class
    def __len__(self):
        # return len(self.queries)
        return self.query_len

    def get_bev_file(self, ndx):
    
        radar_name = self.dataset_path + '/ntu4d/train/query/' + self.radar_path
        radar_file = radar_name + self.file_template.format(self.queries[ndx].id)
        lidar_name = self.dataset_path + '/ntu4d/train/query/' + self.lidar_path
        lidar_file = lidar_name + self.file_template.format(self.queries[ndx].id)
 
        return radar_file, lidar_file

    def __getitem__(self, ndx):
        
        # Load point cloud and apply transform
        radar_file, lidar_file = self.get_bev_file(ndx)

        if self.is_gray:
            radar_query = cv2.imread(radar_file, cv2.IMREAD_GRAYSCALE)
            lidar_query = cv2.imread(lidar_file, cv2.IMREAD_GRAYSCALE)
        else:
            radar_query = cv2.imread(radar_file)
            lidar_query = cv2.imread(lidar_file)

        if self.is_rotation:
            mat = cv2.getRotationMatrix2D((radar_query.shape[1]//2, radar_query.shape[0]//2 ), np.random.randint(-30,30), 1)
            radar_query = cv2.warpAffine(radar_query, mat, radar_query.shape[:2])
            lidar_query = cv2.warpAffine(lidar_query, mat, lidar_query.shape[:2])
        
        radar_list, lidar_list = self.get_positives(ndx)
        
        if self.is_gray:
            radar_query = np.expand_dims(radar_query, axis=0)
            lidar_query = np.expand_dims(lidar_query, axis=0)
            radar_positive = cv2.imread(radar_list[0], cv2.IMREAD_GRAYSCALE)
            lidar_positive = cv2.imread(lidar_list[0], cv2.IMREAD_GRAYSCALE)
        else:
            radar_query = radar_query.transpose(2,0,1)
            lidar_query = lidar_query.transpose(2,0,1)
            radar_positive = cv2.imread(radar_list[0])
            lidar_positive = cv2.imread(lidar_list[0])
        
        radar_query = (radar_query.astype(np.float32))/256
        lidar_query = (lidar_query.astype(np.float32))/256
        
        
        if self.is_rotation:           
            mat = cv2.getRotationMatrix2D((radar_positive.shape[1]//2, radar_positive.shape[0]//2 ), np.random.randint(-30,30), 1)
            radar_positive = cv2.warpAffine(radar_positive, mat, radar_positive.shape[:2])
            lidar_positive = cv2.warpAffine(lidar_positive, mat, lidar_positive.shape[:2])
            
        if self.is_gray:    
            radar_positive = np.expand_dims(radar_positive, axis=0)
            lidar_positive = np.expand_dims(lidar_positive, axis=0)
        else:
            radar_positive = radar_positive.transpose(2,0,1)
            lidar_positive = lidar_positive.transpose(2,0,1)
        
        radar_positive = (radar_positive.astype(np.float32))/256
        lidar_positive = (lidar_positive.astype(np.float32))/256
        
        radar_list, lidar_list = self.get_non_negatives(ndx)
        radar_negatives = []
        lidar_negatives = []
        for index in range(len(radar_list)):
            
            if self.is_gray:  
                radar_negative = cv2.imread(radar_list[index], cv2.IMREAD_GRAYSCALE)
                lidar_negative = cv2.imread(lidar_list[index], cv2.IMREAD_GRAYSCALE)
            else:
                radar_negative = cv2.imread(radar_list[index])
                lidar_negative = cv2.imread(lidar_list[index])
                
            if self.is_rotation:
                mat = cv2.getRotationMatrix2D((radar_negative.shape[1]//2, radar_negative.shape[0]//2 ), np.random.randint(-30,30), 1)
                radar_negative = cv2.warpAffine(radar_negative, mat, radar_negative.shape[:2]) 
                lidar_negative = cv2.warpAffine(lidar_negative, mat, lidar_negative.shape[:2]) 
                
            if self.is_gray: 
                radar_negative = np.expand_dims(radar_negative, axis=0)
                lidar_negative = np.expand_dims(lidar_negative, axis=0)
            else:
                radar_negative = radar_negative.transpose(2,0,1)
                lidar_negative = lidar_negative.transpose(2,0,1)
                
            radar_negative = (radar_negative.astype(np.float32))/256
            radar_negatives.append(torch.from_numpy(radar_negative))

            lidar_negative = (lidar_negative.astype(np.float32))/256
            lidar_negatives.append(torch.from_numpy(lidar_negative))
            
        radar_negatives = torch.stack(radar_negatives, 0)
        lidar_negatives = torch.stack(lidar_negatives, 0)
        
        return radar_query, lidar_query, radar_positive, lidar_positive, radar_negatives, lidar_negatives, int(self.queries[ndx].id)

    def get_positives(self, ndx):
        
        if ndx >= self.query_len:    
            radar_path = self.dataset_path + '/ntu4d/train/query/' + self.radar_path
            lidar_path = self.dataset_path + '/ntu4d/train/query/' + self.lidar_path
        else:
            radar_path = self.dataset_path + '/ntu4d/train/database/' + self.radar_path
            lidar_path = self.dataset_path + '/ntu4d/train/database/' + self.lidar_path
        
        pos_inx = list(self.queries[ndx].positives)
        radar_positives = []
        lidar_positives = []
        if(len(pos_inx) < 1):
            print('error: ', ndx)
        
        pos_inx = random.sample(pos_inx, self.positives_per_query)
        for idx in range(len(pos_inx)):
            # if pos_inx[idx] > self.query_len:
            #     positives.append(path + self.file_template.format(pos_inx[idx] - self.query_len))
            # else:
            radar_positives.append(radar_path + self.file_template.format(pos_inx[idx]))
            lidar_positives.append(lidar_path + self.file_template.format(pos_inx[idx]))
        return radar_positives, lidar_positives

    def get_non_negatives(self, ndx):
        
        if ndx >= self.query_len:    
            radar_path = self.dataset_path + '/ntu4d/train/query/' + self.radar_path
            lidar_path = self.dataset_path + '/ntu4d/train/query/' + self.lidar_path
        else:
            radar_path = self.dataset_path + '/ntu4d/train/database/' + self.radar_path
            lidar_path = self.dataset_path + '/ntu4d/train/database/' + self.lidar_path

        neg_inx = self.queries[ndx].non_negatives
        lst_set = set(neg_inx)
        neg_inx = list(self.all_numbers.difference(lst_set))
        neg_inx = random.sample(neg_inx, self.negatives_per_query)
        radar_positives = []
        lidar_positives = []
        
        for idx in range(len(neg_inx)):
            # if pos_inx[idx] > self.query_len:
            #     positives.append(path + self.file_template.format(pos_inx[idx] - self.query_len))
            # else:
            radar_positives.append(radar_path + self.file_template.format(neg_inx[idx]))
            lidar_positives.append(lidar_path + self.file_template.format(neg_inx[idx]))
        return radar_positives, lidar_positives


def main():

    train_dataset = TrainingKD_Dataset(dataset_path='/data/hny/dataset/4Dradar', query_filename='train_queries_train.pickle')
    _,_,_,_,_,_,_ = train_dataset.__getitem__(20)
    idx = train_dataset.get_non_negatives(0)
    
    test_dataset = EvaluateDataset(dataset_name='ntu4d', is_query=False)
    _,_ = test_dataset.__getitem__(20)
    
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=12, 
                                        batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    for query, indices in training_data_loader:
        print('1')
        
    print('finish...')

if __name__ == "__main__":
    main()
