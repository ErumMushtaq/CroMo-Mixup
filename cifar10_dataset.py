import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import sys
import torchvision.transforms as T

from collections import OrderedDict

class SimSiam_Dataloader(Dataset):
    def __init__(self, xtrain , ytrain , X_dtype=torch.float32, y_dtype=torch.float32,  is_knn = 0):

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype

        self.train_data = xtrain
        self.label_data = ytrain
        
        self.is_knn = is_knn

        data_normalize_mean = (0.4914, 0.4822, 0.4465)
        data_normalize_std = (0.247, 0.243, 0.261)
        self.basic_transform = T.Compose(
            [   
                T.Normalize(data_normalize_mean, data_normalize_std),
            ])

        self.transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

        self.transform_prime = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y = self.label_data[idx]
        # print(self.train_data[idx].shape)
        if self.is_knn == 0:
            x1 = self.transform(self.train_data[idx])
            x2 = self.transform_prime(self.train_data[idx])           
            return x1, x2, y
        else:
            x = self.train_data[idx]
            return x, y

