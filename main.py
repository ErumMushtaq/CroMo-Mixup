import numpy as np
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.autograd import Variable
from byol_pytorch import BYOL
from torchvision import models

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import copy
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(7)

from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from torch import autograd
from torchvision import datasets,transforms
from sklearn.utils import shuffle

from dataloader_cifar10 import get_cifar10


# Hyper-Parameters
batch_size = 512 #512 in SimSiam paper
lr = 0.001  #paper 0.05
epoch = 200 # 200 epoch
cuda_device = 0

device = torch.device("cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu")

train_data_loaders, test_data_loaders, validation_data_loaders = get_cifar10(classes=[5,5],valid_rate = 0.05, batch_size=batch_size, seed = 0)

model = models.resnet18(pretrained=True)
model.to(device)
learner = BYOL(model, image_size = 32, hidden_layer = 'avgpool', use_momentum = False)       # turn off momentum in the target encoder, image size 32, imagenet 256. 
learner.to(device) #automatically detects from model

# SimSiam uses SGD, with lr = lr*BS/256 from paper + https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py), why 256? shall it be 32 for cifar10
init_lr = lr*batch_size/256
optimizer = torch.optim.SGD(learner.parameters(), init_lr, momentum=0.9, weight_decay=0.0001)


# Training Loop 
#TODO: add knn accuracy as well.
loss_ = []
for _ in range(epoch):
    epoch_loss = []
    for x, y in train_data_loaders[0]:
        x = x.to(device)
        loss = learner(x)
        epoch_loss.append(loss.item())
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
    print(np.mean(epoch_loss))
    loss_.append(np.mean(epoch_loss))

print(loss)



# for x,y in train_data_loaders[0]:
#     print(y)




