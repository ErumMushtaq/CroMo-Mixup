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
import wandb

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import copy
import os
from torch import nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(4)

from collections import OrderedDict
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from torch import autograd
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from dataloader_cifar10 import get_cifar10
import random
import math
from eval_metrics import linear_evaluation, Knn_Validation
from linear_classifer import LinClassifier

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Hyper-Parameters taken from new paper
batch_size = 512 #512 in SimSiam paper
lr = 0.2 #paper 0.05
epoch = 1000 # 1000 epoch
cuda_device = 0

min_scale = 0.08 
data_normalize_mean = (0.4914, 0.4822, 0.4465)
data_normalize_std = (0.247, 0.243, 0.261)
random_crop_size = 32

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization:
    def __call__(self, img):
        return torchvision.transforms.functional.solarize(img,threshold = 0.5)#th value is compatible with PIL documentation

class Clamp:
    def __call__(self, img):
        img = torch.clamp(img,min=0.0,max=1.0)
        return img

transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    random_crop_size, 
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC, # Only in VicReg
                ),
                Clamp(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0), 
                transforms.RandomApply([Solarization()], p=0.0), # Only in VicReg
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ]
        )

transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    random_crop_size, 
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC, # Only in VicReg
                ),
                Clamp(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1), 
                transforms.RandomApply([Solarization()], p=0.2), # Only in VicReg
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ]
        )

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
#wandb logging


#Dataloader
device = torch.device("cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu")
train_data_loaders, test_data_loaders, validation_data_loaders = get_cifar10(classes=[10], valid_rate = 0.05, batch_size=batch_size, seed = 0)

#Model and Learner 
model = models.resnet18(pretrained=False)
model.to(device)
learner = BYOL(model, image_size = 32, hidden_layer = 'avgpool', projection_size = 256, projection_hidden_size = 2048, use_momentum = False, augment_fn = transform,  augment_fn2 = transform_prime)
learner.to(device) #automatically detects from model

# Optimizer and Scheduler
# SimSiam uses SGD, with lr = lr*BS/256 from paper + https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py)
init_lr = lr*batch_size/256
optimizer = torch.optim.SGD(learner.parameters(), init_lr, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=2e-3) #scheduler + values ref: infomax paper
wandb.init(project="SSL Project", name="SimSiam"
            + "-e" + str(epoch) + "-b" + str(batch_size) + "-lr" + str(init_lr))

#Training Loop 
#TODO: add knn accuracy as well.
loss_ = []
for epoch_counter in range(epoch):
    learner.online_encoder.train()
    learner.online_predictor.train()
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
    # adjust_learning_rate(optimizer, lr, epoch_counter, epoch)       
    print(np.mean(epoch_loss))
    loss_.append(np.mean(epoch_loss))
    knn_acc = Knn_Validation(learner.online_encoder,train_data_loaders[0],validation_data_loaders[0],device=device, K=20) #TODO: cheeck 200 a good HP for cifar10?
    wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})
    wandb.log({" Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})  
    wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
    
    print(f'knn acc: {knn_acc}')

classifier =    LinClassifier().to(device)
lin_optimizer = torch.optim.SGD(classifier.parameters(),  0.001, momentum=0.9, weight_decay=0.0001)
test_loss, test_acc1, test_acc5 = linear_evaluation(learner.online_encoder, train_data_loaders[0],test_data_loaders[0],lin_optimizer,classifier, epochs= 500)

# plt.plot(loss_)
# plt.savefig('loss_iteration')

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




