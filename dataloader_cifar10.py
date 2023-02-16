import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle


def get_cifar10(classes=[5,5],valid_rate = 0.05, seed = 0,batch_size = 128):
    pc_valid= valid_rate
    dat = {}
    dat['train']=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed
    dat['test']=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed
    data = {}
    lower_bound = 0
    upper_bound = 0
    size=[3,32,32]
    for n, num_class in enumerate(classes):
        upper_bound += num_class
        data[n]={}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            data[n][s]={'x': [],'y': []}
            for image, target in loader:
                if target.numpy()[0] < upper_bound and target.numpy()[0] >= lower_bound:
                    data[n][s]['x'].append(image)
                    data[n][s]['y'].append(target.numpy()[0])
    
        for s in ['train','test']:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        
        lower_bound = upper_bound

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
    for k in range(len(classes)):
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']

        train_data_loaders.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True))
        test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest), batch_size=batch_size, shuffle=True))
        validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid), batch_size=batch_size, shuffle=True))

    return train_data_loaders, test_data_loaders, validation_data_loaders
