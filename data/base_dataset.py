import os
import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args, is_inference=False):
        self.path = args.path
        if is_inference:
            squence = ['longdress']
            file_list_name = os.path.join(self.path, 'test.txt')
        else:
            squence = ['longdress', 'loot']
            # squence = ['longdress', 'loot', 'longdress_9bit', 'loot_9bit']
            # squence = ['longdress_9bit', 'loot_9bit']
            file_list_name = os.path.join(self.path, 'train.txt')

        if os.path.isfile(file_list_name):
            print('reading dataset from {}'.format(file_list_name))
            with open(file_list_name, 'r') as f:
                lines = f.readlines()
                self.pair_list = [item.strip() for item in lines]
        else:
            print('Do not find the file list, building the training/testing set')
            self.pair_list = self.get_data_list(file_list_name, squence)
        self.cube_size = args.cube_size
        

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        comparssed_name = self.pair_list[index]
        gt_name = self.to_gt_name(comparssed_name)
        decompressed_points = np.loadtxt(comparssed_name).astype(np.int32)
        gt_points = np.loadtxt(gt_name).astype(np.int32)
        decompressed = self.convert_to_onehot(decompressed_points)
        gt = self.convert_to_onehot(gt_points)
        return {
            'gt_point_cloud': gt,
            'input_point_cloud': decompressed,
            'name': comparssed_name,
        }

    def to_gt_name(self, name):
        name_list = name.split('/')
        name_list[-4] = 'gt'
        return '/'.join(name_list)

    def convert_to_onehot(self, points):
        inputs = np.zeros((self.cube_size, self.cube_size, self.cube_size)) 
        inputs[points[:,0], points[:,1], points[:,2]]  = 1 
        inputs = torch.tensor(inputs).type(torch.float32)[None, ]
        return inputs

    def get_data_list(self, txt_name, squence, write_txt=True):
        gt_files, input_files = [], []
        for item in squence:
            item_gt_files = glob.glob(os.path.join(self.path, 'gt', item, '*/*.txt'))
            item_gt_files = ['/'.join(item.split('/')[-3:]) for item in item_gt_files]
            gt_files.extend(item_gt_files)
            for rate in ['r00','r01','r02','r03','r04','r05','r06','r07']:
            # for rate in ['r01','r02','r03','r04','r05','r06']:
            # for rate in ['r00']:
                item_input_files = sorted(glob.glob(os.path.join(self.path, rate, item, '*/*.txt')))
                input_files.extend(item_input_files)
        gt_files = set(gt_files)
        final_files = []
        for item in tqdm(input_files): # filter out
            if '/'.join(item.split('/')[-3:]) in gt_files:
                final_files.append(item)
        
        if write_txt:
            print('writing the file list to {}'.format(txt_name))
            with open(txt_name, 'w') as f:
                for item in final_files:
                    f.write(item + '\n')

        return final_files


