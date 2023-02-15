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
import kornia

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import copy
import os
from torch import nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(7)

from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from torch import autograd
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from dataloader_cifar10 import get_cifar10, SimSiamTransform

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Hyper-Parameters taken from new paper
batch_size = 512 #512 in SimSiam paper
lr = 0.05 #paper 0.05
epoch = 4 # 1000 epoch
cuda_device = 0


#Dataloader
device = torch.device("cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu")
train_data_loaders, test_data_loaders, validation_data_loaders = get_cifar10(classes=[10], valid_rate = 0.05, batch_size=batch_size, seed = 0)

#Model and Learner 
model = models.resnet18(pretrained=False)
model.to(device)
augment_fn = nn.Sequential(kornia.augmentation.RandomHorizontalFlip())
augment_fn2 = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(),kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5)))
learner = BYOL(model, image_size = 32, hidden_layer = 'avgpool', projection_size = 256, projection_hidden_size = 2048, use_momentum = False, augment_fn = augment_fn,  augment_fn2 = augment_fn2)
learner.to(device) #automatically detects from model

# Optimizer and Scheduler
# SimSiam uses SGD, with lr = lr*BS/256 from paper + https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py)
init_lr = lr*batch_size/256
optimizer = torch.optim.SGD(learner.parameters(), init_lr, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loaders), eta_min=0, last_epoch=-1)

#Training Loop 
#TODO: add knn accuracy as well.
loss_ = []
for epoch_counter in range(epoch):
    epoch_loss = []
    for x, y in train_data_loaders[0]:
        x = x.to(device)
        loss = learner(x)
        epoch_loss.append(loss.item())
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if epoch_counter >= 10: #warmup of 10 epochs #from SimCLR
                scheduler.step()       
    print(np.mean(epoch_loss))
    loss_.append(np.mean(epoch_loss))

plt.plot(loss_)
plt.savefig('loss_iteration')
# save your encoder network
save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'lr':init_lr,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss_':loss_,
            }, is_best=False, filename='checkpoint_{:04f}.pth.tar'.format(lr))
print(loss)



# for x,y in train_data_loaders[0]:
#     print(y)




