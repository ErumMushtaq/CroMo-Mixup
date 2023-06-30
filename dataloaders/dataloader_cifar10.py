import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from dataloaders.dataset import SimSiam_Dataset, TensorDataset


def get_cifar10(transform=None, transform_prime=None, classes=[5,5], valid_rate = 0.05, seed = 0, batch_size = 128, num_worker = 8):

    ind = np.cumsum(classes)[:-1]
    tasks = np.split(np.arange(sum(classes)), ind, axis=0)

    trainset=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed
    testset=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed

    Ytrain = trainset.targets
    Xtrain = trainset.data
    Ytest = testset.targets
    Xtest = testset.data

    train_data_loaders = []
    test_data_loaders = []
    validation_data_loaders = []
    train_data_loaders_knn = []
    train_data_loaders_pure = []
    train_data_loaders_linear = []

    for task in tasks:
        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for t in task:
            ind = np.where(Ytrain == t)[0]
            xtrain.append(Xtrain[ind])
            ytrain.append([t]*len(ind))
            ind = np.where(Ytest == t)[0]
            xtest.append(Xtest[ind])
            ytest.append([t]*len(ind))
        xtrain = torch.Tensor(np.concatenate(xtrain, axis=0)/255).permute(0,3,1,2)
        ytrain = torch.tensor(np.array(ytrain).reshape(-1),dtype=int) 
        xtest = torch.Tensor(np.concatenate(xtest, axis=0)/255).permute(0,3,1,2)
        ytest = torch.tensor(np.array(ytest).reshape(-1),dtype=int) 

        r=np.arange(len(xtrain))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(valid_rate*len(r))
        ivalid=r[:nvalid]
        itrain=r[nvalid:]
        xvalid = xtrain[ivalid].clone()
        yvalid = ytrain[ivalid].clone()
        xtrain = xtrain[itrain].clone()
        ytrain = ytrain[itrain].clone()

        data_normalize_mean = (0.4914, 0.4822, 0.4465)
        data_normalize_std = (0.247, 0.243, 0.261)
        transform_test = transforms.Compose([   
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ])
        transform_linear = transforms.Compose([
                    transforms.RandomResizedCrop(32), # scale=(0.2, 1.0) is possible
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(data_normalize_mean, data_normalize_std),
                ])

        linear_batch_size = 256
        train_dataset = SimSiam_Dataset(xtrain, ytrain, transform, transform_prime)
        train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_worker , pin_memory=True))
        train_data_loaders_knn.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
        train_data_loaders_pure.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
        test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest,transform=transform_test), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True))
        validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid,transform=transform), batch_size=batch_size, shuffle=False, num_workers = 8))
        train_data_loaders_linear.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_linear), batch_size=linear_batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))

    return train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders, train_data_loaders_linear, train_data_loaders_pure
