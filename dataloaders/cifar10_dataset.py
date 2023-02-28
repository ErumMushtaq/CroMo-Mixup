
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


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
        x1 = self.transform(self.train_data[idx])
        x2 = self.transform_prime(self.train_data[idx])
        return x1, x2, y

