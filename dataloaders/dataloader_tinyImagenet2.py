import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from dataloaders.dataset import SimSiam_Dataset, TensorDataset, GenericDataset, TinyImagenet
import torchvision


def get_tinyImagenet(transform, transform_prime, classes=[50,50,50,50], valid_rate = 0.05, seed = 0, batch_size = 128, num_worker = 8, valid_transform = None, dl_type = 'class_incremental', org_data = False):

    ind = np.cumsum(classes)[:-1]
    tasks = np.split(np.arange(sum(classes)), ind, axis=0)

    trainset=torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train",transform=transforms.Compose([transforms.ToTensor()]))   
    testset=torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val",transform=transforms.Compose([transforms.ToTensor()]))   

    Ytrain = np.array(trainset.targets)
    Ytest = np.array(testset.targets)

    train_data_loaders = []
    test_data_loaders = []
    validation_data_loaders = []
    train_data_loaders_knn = []
    train_data_loaders_pure = []
    train_data_loaders_linear = []
    train_data_loaders_generic = []

    print(dl_type)
    if dl_type == 'class_incremental':

        for task in tasks:
            xtrain = []
            ytrain = []
            xtest = []
            ytest = []
            for t in task:
                ind = np.where(Ytrain == t)[0]
                for i in ind:
                    xtrain.append(trainset[i][0])
                ytrain.append([t]*len(ind))
                ind = np.where(Ytest == t)[0]
                for i in ind:
                    xtest.append(testset[i][0])
                ytest.append([t]*len(ind))
            xtrain = torch.stack(xtrain)
            ytrain = torch.tensor(np.array(ytrain).reshape(-1),dtype=int) 
            xtest = torch.stack(xtest)
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

            # data_normalize_mean = (0.4802, 0.4480, 0.3975)
            # data_normalize_std = (0.2770, 0.2691, 0.2821)

            data_normalize_mean = (0.485, 0.456, 0.406)  #from infomax
            data_normalize_std = (0.229, 0.224, 0.225)
            transform_knn = transforms.Compose( [   
                    transforms.Normalize(data_normalize_mean, data_normalize_std),
                ])

            random_crop_size = 64
            if valid_transform is None:
                transform_test = transforms.Compose([
                        transforms.Resize(int(random_crop_size*(8/7)), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                        transforms.CenterCrop(random_crop_size),
                        transforms.Normalize(data_normalize_mean, data_normalize_std),
                    ] )

                transform_linear = transforms.Compose( [
                        transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize(data_normalize_mean, data_normalize_std),
                    ] )
            else:
                transform_test = valid_transform
                transform_linear = valid_transform
                linear_batch_size = 128

            linear_batch_size = 256    
            train_dataset = SimSiam_Dataset(xtrain, ytrain, transform, transform_prime)
            train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_worker , pin_memory=False)) #, timeout=500
            train_data_loaders_knn.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_knn), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            train_data_loaders_pure.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest,transform=transform_test), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True))
            validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid,transform=transform), batch_size=batch_size, shuffle=False, num_workers = 8))
            train_data_loaders_linear.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_linear), batch_size=linear_batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            train_data_loaders_generic.append(DataLoader(GenericDataset(xtrain, ytrain,transforms=None), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
    else: #Data Increment
        xtrain_by_class = []
        ytrain_by_class = []
        xtest_by_class = []
        ytest_by_class = []
        for t in np.arange(sum(classes)): #Arrange class wise
            ind = np.where(Ytrain == t)[0]
            for i in ind:
                xtrain_by_class.append(trainset[i][0])
            ytrain_by_class.append([t]*len(ind))
            ind_ = np.where(Ytest == t)[0]
            for i in ind_:
                xtest_by_class.append(testset[i][0])
            ytest_by_class.append([t]*len(ind_))
        
        xtrain_by_class = torch.stack(xtrain_by_class) #0th index for the class ID, rest is on the format 0,3,1,2
        ytrain_by_class = torch.Tensor(np.array(ytrain_by_class))
        xtest_by_class = torch.stack(xtest_by_class)
        ytest_by_class = torch.Tensor(np.array(ytest_by_class))

        train_num = int(xtrain_by_class.shape[1]/len(tasks)) #total images per class/# of tasks
        test_num = int(xtest_by_class.shape[1]/len(tasks))

        for t in range(len(tasks)):
            nvalid = int(valid_rate*train_num)

            xvalid = xtrain_by_class[:, t*(train_num):t*(train_num)+nvalid].clone().reshape(-1, 3, 64, 64)
            yvalid = ytrain_by_class[:, t*(train_num):t*(train_num)+nvalid].clone().reshape(-1).type(torch.int64)
            xtrain = xtrain_by_class[:, (t*train_num)+nvalid:(t+1)*train_num].clone().reshape(-1, 3, 64, 64)
            ytrain = ytrain_by_class[:, (t*train_num)+nvalid:(t+1)*train_num].clone().reshape(-1).type(torch.int64)
            xtest = xtest_by_class[:, t*test_num:(t+1)*test_num].clone().reshape(-1, 3, 64, 64)
            ytest = ytest_by_class[:, t*test_num:(t+1)*test_num].clone().reshape(-1)
            print(xtrain.shape)
            # data_normalize_mean = (0.4802, 0.4480, 0.3975)
            # data_normalize_std = (0.2770, 0.2691, 0.2821)

            data_normalize_mean = (0.485, 0.456, 0.406)  #from infomax
            data_normalize_std = (0.229, 0.224, 0.225)

            transform_knn = transforms.Compose( [transforms.Normalize(data_normalize_mean, data_normalize_std),])
            random_crop_size = 64

            if valid_transform is None:
                transform_test = transforms.Compose([
                        transforms.Resize(int(random_crop_size*(8/7)), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                        transforms.CenterCrop(random_crop_size),
                        transforms.Normalize(data_normalize_mean, data_normalize_std),
                    ] )

                transform_linear = transforms.Compose( [
                        transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                        transforms.RandomHorizontalFlip(),
                        transforms.Normalize(data_normalize_mean, data_normalize_std),
                    ] )
            else:
                transform_test = valid_transform
                transform_linear = valid_transform
                linear_batch_size = 128

            linear_batch_size = 256    
            train_dataset = SimSiam_Dataset(xtrain, ytrain, transform, transform_prime)
            train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_worker , pin_memory=False)) #, timeout=500
            train_data_loaders_knn.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_knn), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            train_data_loaders_pure.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest,transform=transform_test), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True))
            validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid,transform=transform), batch_size=batch_size, shuffle=False, num_workers = 8))
            train_data_loaders_linear.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_linear), batch_size=linear_batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
            train_data_loaders_generic.append(DataLoader(GenericDataset(xtrain, ytrain,transforms=None), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))

    return train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders, train_data_loaders_linear, train_data_loaders_pure, train_data_loaders_generic
