import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

class GenericDataset(Dataset):
    def __init__(self, xtrain, ytrain, transforms=None):

        self.train_data = xtrain
        self.label_data = ytrain

        self.transforms = transforms

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y = self.label_data[idx]
        result_x = []
        if self.transforms != None:
            for transform in self.transforms:
                result_x.append(transform(self.train_data[idx]))
        else:
            result_x.append(self.train_data[idx])
        return result_x, y


class TensorDataset(Dataset):
    def __init__(self, xtrain, ytrain, transform=None):

        self.train_data = xtrain
        self.label_data = ytrain

        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __get_dataset__(self):
        return self.train_data, self.label_data

    def __getitem__(self, idx):
        y = self.label_data[idx]
        if self.transform != None:
            x1 = self.transform(self.train_data[idx])
        else:
            x1 = self.train_data[idx]
        return x1, y
    

class SimSiam_Dataset(Dataset):
    def __init__(self, xtrain, ytrain, transform, transform_prime):

        self.train_data = xtrain
        self.label_data = ytrain

        self.transform = transform
        self.transform_prime = transform_prime

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y = self.label_data[idx]
        x1 = self.transform(self.train_data[idx]) if self.transform is not None else self.train_data[idx]
        x2 = self.transform_prime(self.train_data[idx]) if self.transform_prime is not None else self.train_data[idx]

        # x1 = normalize_to_neg_one_to_one(x1)
        return x1, x2, y

class Diffusion_Dataset(Dataset):
    def __init__(self, xtrain, ytrain, transform):

        self.train_data = xtrain
        self.label_data = ytrain

        self.transform = transform
        # sself.transform_prime = transform_prime

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # print("diffusion")
        # print(idx)
        y = self.label_data[idx]
        x1 = self.transform(self.train_data[idx]) if self.transform is not None else self.train_data[idx]
        # x2 = self.transform_prime(self.train_data[idx]) if self.transform_prime is not None else self.train_data[idx]
        return x1, y, idx

class Guided_Diffusion_Dataset(Dataset):
    def __init__(self, xtrain, ytrain, cluster_ids, transform, transform_prime):

        self.train_data = xtrain
        self.label_data = ytrain
        self.cluster_id = cluster_ids

        self.transform = transform
        self.transform_prime = transform_prime

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y = self.label_data[idx]
        ci = se;f/cluster_id[idx]
        x1 = self.transform(self.train_data[idx]) if self.transform is not None else self.train_data[idx]
        x2 = self.transform_prime(self.train_data[idx]) if self.transform_prime is not None else self.train_data[idx]
        return x1, y, ci

# class Sup_Dataset(Dataset):
#     def __init__(self, xtrain, ytrain):

#         self.train_data = xtrain
#         self.label_data = ytrain
#         self.transform =  T.Compose([
#         T.RandomCrop(32, padding=4),
#         T.RandomHorizontalFlip(),
#         T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
#     ])

#     def __len__(self):
#         return len(self.train_data)

#     def __getitem__(self, idx):
#         y = self.label_data[idx]
#         x1 = self.transform(self.train_data[idx])
#         # x2 = self.transform_prime(self.train_data[idx])
#         return x1, y

