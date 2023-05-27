import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.transforms as T
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from dataloaders.dataset import SimSiam_Dataset, TensorDataset


CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
CIFAR100_LABELS_LIST = np.array(CIFAR100_LABELS_LIST)

sclass = []
sclass.append(['beaver','dolphin', 'otter', 'seal', 'whale'])                       #aquatic mammals
sclass.append(['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'])               #fish
sclass.append(['orchid', 'poppy', 'rose', 'sunflower', 'tulip'])                    #flowers
sclass.append(['bottle', 'bowl', 'can', 'cup', 'plate'])                            #food
sclass.append(['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'])              #fruit and vegetables
sclass.append(['clock', 'keyboard', 'lamp', 'telephone', 'television'])             #household electrical devices
sclass.append(['bed', 'chair', 'couch', 'table', 'wardrobe'])                       #household furniture
sclass.append(['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'])           #insects
sclass.append(['bear', 'leopard', 'lion', 'tiger', 'wolf'])                         #large carnivores
sclass.append(['bridge', 'castle', 'house', 'road', 'skyscraper'])                  #large man-made outdoor things
sclass.append(['cloud', 'forest', 'mountain', 'plain', 'sea'])                      #large natural outdoor scenes
sclass.append(['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'])            #large omnivores and herbivores
sclass.append(['fox', 'porcupine', 'possum', 'raccoon', 'skunk'])                   #medium-sized mammals
sclass.append(['crab', 'lobster', 'snail', 'spider', 'worm'])                       #non-insect invertebrates
sclass.append(['baby', 'boy', 'girl', 'man', 'woman'])                              #people
sclass.append(['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'])               #reptiles
sclass.append(['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'])                  #small mammals
sclass.append(['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'])  #trees
sclass.append(['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'])            #vehicles 1
sclass.append(['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'])             #vehicles 2
sclass = np.array(sclass)

def get_cifar100_superclass(transform, transform_prime, classes=[50,50], valid_rate = 0.05, seed = 0, batch_size = 128, num_worker = 8):

    tasks = []
    for i in range(5):
        temp = []
        for s in sclass:
            temp.append(np.where(CIFAR100_LABELS_LIST==s[i])[0][0])
        tasks.append(temp)

    trainset=datasets.CIFAR100('./data/cifar100/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed
    testset =datasets.CIFAR100('./data/cifar100/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))#normalization removed

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
        xtrain = torch.Tensor(np.concatenate(xtrain, axis=0))
        ytrain = torch.tensor(np.array(ytrain).reshape(-1),dtype=int) 
        xtest = torch.Tensor(np.concatenate(xtest, axis=0))
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

        data_normalize_mean = (0.5071, 0.4865, 0.4409)
        data_normalize_std = (0.2673, 0.2564, 0.2762)

        random_crop_size = 32
        transform_test = transforms.Compose([
                transforms.Resize(int(random_crop_size*(8/7)), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), 
                transforms.CenterCrop(random_crop_size),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ] )
        transform_knn = transforms.Compose( [   
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ])
        transform_linear = transforms.Compose( [
                transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ] )

        linear_batch_size = 256    
        train_dataset = SimSiam_Dataset(xtrain, ytrain, transform, transform_prime)
        train_data_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_worker , pin_memory=True))
        train_data_loaders_knn.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_knn), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
        train_data_loaders_pure.append(DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))
        test_data_loaders.append(DataLoader(TensorDataset(xtest,ytest,transform=transform_test), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory=True))
        validation_data_loaders.append(DataLoader(TensorDataset(xvalid,yvalid,transform=transform), batch_size=batch_size, shuffle=False, num_workers = 8))
        train_data_loaders_linear.append(DataLoader(TensorDataset(xtrain, ytrain,transform=transform_linear), batch_size=linear_batch_size, shuffle=True, num_workers = num_worker, pin_memory=True))

    return train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders, train_data_loaders_linear, train_data_loaders_pure
