import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle


#Erum: Adding SimSiam augmentation class from ssfl work: mean std value taken from the new paper
class SimSiamTransform:
    def __init__(self, image_size):
        image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur 

        self.transform = T.Compose([
            # T.ToPILImage(), #already PIL
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]),std=torch.tensor([0.247, 0.243, 0.261]))
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        # return x1
        return x1, x2

def get_cifar10(classes=[5,5],valid_rate = 0.05, seed = 0,batch_size = 128):
    pc_valid= valid_rate
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    dat = {}
    dat['train']=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=SimSiamTransform(32))
    # dat['train']=datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data = {}
    lower_bound = 0
    upper_bound = 0
    size=[3,32,32]
    for n, num_class in enumerate(classes):
        upper_bound += num_class
        data[n]={}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            # data[n][s]={'x': [],'y': []}
            # for image, target in loader:
            #     if target.numpy()[0] < upper_bound and target.numpy()[0] >= lower_bound:
            #         data[n][s]['x'].append(image)
            #         data[n][s]['y'].append(target.numpy()[0])
            
            if s == 'train': # SimCLR augmentations
                data[n][s]={'x1': [],'x2':[], 'y': []}
                for image, target in loader: #if train then x1 and x2
                    if target.numpy()[0] < upper_bound and target.numpy()[0] >= lower_bound:
                        data[n][s]['x1'].append(image[0])
                        data[n][s]['x2'].append(image[1])
                        data[n][s]['y'].append(target.numpy()[0])
            else:
                data[n][s]={'x': [],'y': []}
                for image, target in loader:
                    # print(image[0].shape) if train then x1 and x2
                    if target.numpy()[0] < upper_bound and target.numpy()[0] >= lower_bound:
                        data[n][s]['x'].append(image)
                        data[n][s]['y'].append(target.numpy()[0])


        for s in ['train','test']:
            if s in 'train':
                data[n][s]['x1']=torch.stack(data[n][s]['x1']).view(-1,size[0],size[1],size[2])
                data[n][s]['x2']=torch.stack(data[n][s]['x2']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
            else:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        
        lower_bound = upper_bound

    # TODO: Validation: it shouldn't have SimCLR transforms. Can be done when we need it, skippinng for now and taking val set from test
    for t in data.keys():
        r=np.arange(data[t]['test']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['test']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['test']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['test']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['test']['y'][itrain].clone()
    # for t in data.keys():
    #     r=np.arange(data[t]['train']['x'].size(0))
    #     r=np.array(shuffle(r,random_state=seed),dtype=int)
    #     nvalid=int(pc_valid*len(r))
    #     ivalid=torch.LongTensor(r[:nvalid])
    #     itrain=torch.LongTensor(r[nvalid:])
    #     data[t]['valid']={}
    #     data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #     data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #     data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #     data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    train_data_loaders = []
    test_data_loaders = []
    validation_data_loaders = []
    for k in range(len(classes)):
        x1train=data[k]['train']['x1']
        x2train=data[k]['train']['x2']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']

        print(x1train.shape)
        print(x2train.shape)

        train_data_loaders.append(DataLoader(TensorDataset(x1train, x2train, ytrain), batch_size=batch_size, shuffle=True))
        test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest), batch_size=batch_size, shuffle=True))
        validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid), batch_size=batch_size, shuffle=True))

    return train_data_loaders, test_data_loaders, validation_data_loaders
