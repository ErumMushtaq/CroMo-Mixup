import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from dataloaders.cifar10_dataset import SimSiam_Dataset, Sup_Dataset

def find_task(classes,label):
    cur_label = 0
    for i in range(len(classes)):
        cur_label += classes[i]
        if label < cur_label:
            break
    return i
def get_cifar10(transform=None, transform_prime=None, classes=[5,5], valid_rate = 0.05, seed = 0, batch_size = 128, num_worker = 8):
    pc_valid= valid_rate
    dat = {}
    if transform == None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dat['train']=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=transform_train)#normalization added
        # dat['test']=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transforms.Compose(transform_test))#normalization removed
        dat['test']=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transform_test)#normalization removed
    else:
        dat['train']=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed
        dat['test']=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed


    data = {}
    lower_bound = 0
    upper_bound = 0
    size=[3,32,32]
    for n, num_class in enumerate(classes):
        upper_bound += num_class
        data[n]={}
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image, target in loader:
            n = find_task(classes,target.numpy()[0])
            data[n][s]['x'].append(image)
            data[n][s]['y'].append(target.numpy()[0])

    for n in data.keys():
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        

    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    train_data_loaders = []
    test_data_loaders = []
    validation_data_loaders = []
    train_data_loaders_knn = []
    for k in range(len(classes)):
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']

        if transform == None:
            train_dataset = Sup_Dataset(xtrain, ytrain)
        else:
            train_dataset = SimSiam_Dataset(xtrain, ytrain, transform, transform_prime)
        # train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, prefetch_factor = 8, pin_memory=True, persistent_workers=True))
        train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_worker , pin_memory=True))

        train_data_loaders_knn.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
        test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True))
        validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid), batch_size=batch_size, shuffle=False, num_workers = 8))

    return train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders
