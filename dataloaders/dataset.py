import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
import gdown
import numpy as np
from PIL import Image


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root, train=True, transform=None,
                target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                base_url =  "https://drive.google.com/uc?export=download&id=1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj"
                file_name = "tiny-imagenet-200.zip"

                file_path = os.path.join(root, file_name)

                if not os.path.exists(root):
                    os.makedirs(root)

                if not os.path.exists(file_path):
                    print(f"Downloading Tiny ImageNet to {file_path}")
                    gdown.download(base_url, file_path, quiet=False)

                print(f"Extracting Tiny ImageNet to {root}")

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(root)

                # Remove the downloaded zip file
                os.remove(file_path)


        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        # original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # if hasattr(self, 'logits'):
        #     return img, target, original_img, self.logits[index]

        return img, target


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

    # def get_xtrain(self):
    #     return self.train_data

class Org_SimSiam_Dataset(Dataset):
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
        return self.train_data[idx], x1, x2, y

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

class Unlabeled_Dataset(Dataset):
    def __init__(self, xtrain,  transform):

        self.train_data = xtrain
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        x1 = self.transform(self.train_data[idx]) if self.transform is not None else self.train_data[idx]
        return x1

# class Guided_Diffusion_Dataset(Dataset):
#     def __init__(self, xtrain, ytrain, cluster_ids, transform, transform_prime):

#         self.train_data = xtrain
#         self.label_data = ytrain
#         self.cluster_id = cluster_ids

#         self.transform = transform
#         self.transform_prime = transform_prime

#     def __len__(self):
#         return len(self.train_data)

#     def __getitem__(self, idx):
#         y = self.label_data[idx]
#         ci = se;f/cluster_id[idx]
#         x1 = self.transform(self.train_data[idx]) if self.transform is not None else self.train_data[idx]
#         x2 = self.transform_prime(self.train_data[idx]) if self.transform_prime is not None else self.train_data[idx]
#         return x1, y, ci

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

