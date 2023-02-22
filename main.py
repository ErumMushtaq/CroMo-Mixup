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
import argparse

import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import copy
import os
from torch import nn
from LRScheduler import LinearWarmupCosineAnnealingLR
from SimSiamScheduler import SimSiamScheduler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(5)

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
from Simsiam_pytorch import Encoder, Predictor, SimSiam


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Hyper-Parameters taken from new paper
batch_size = 512 #512 in SimSiam paper
lr = 0.06 #paper 0.05
epoch = 800 # 1000 epoch
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
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        return x

class Solarization:
    def __call__(self, img):
        return torchvision.transforms.functional.solarize(img,threshold = 0.5)#th value is compatible with PIL documentation



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings


    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_warmup_epochs', type=int, default=0)
    parser.add_argument('--pretrain_warmup_lr', type=float, default=0)
    parser.add_argument('--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=5e-4)
    parser.add_argument('--init_pretrain_epoch', type=int, default=1)
    parser.add_argument('--final_pretrain_epoch', type=int, default=800)

    parser.add_argument('--cuda_device', type=int, default=0, metavar='N',
                        help='device id')

    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='Learning rate')

    args = parser.parse_args()
    return args
# transform = T.Compose([
#             T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
#             T.RandomHorizontalFlip(),
#             T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
#             T.RandomGrayscale(p=0.2),
#             T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

# transform_prime = T.Compose([
#             T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
#             T.RandomHorizontalFlip(),
#             T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
#             T.RandomGrayscale(p=0.2),
#             T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

# def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
#     """Decay the learning rate based on schedule"""
#     cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
#     for param_group in optimizer.param_groups:
#         if 'fix_lr' in param_group and param_group['fix_lr']:
#             param_group['lr'] = init_lr
#         else:
#             param_group['lr'] = cur_lr
#wandb logging

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    #Dataloader
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders = get_cifar10(classes=[10], valid_rate = 0.00, batch_size=batch_size, seed = 0)

    #Model and Learner 
    # model = models.resnet50(pretrained=False)
    # model.to(device)
    # learner = BYOL(model, image_size = 32, hidden_layer = 'avgpool', projection_size = 64, projection_hidden_size = 2048, use_momentum = False, augment_fn = transform,  augment_fn2 = transform_prime)
    # learner.to(device) #automatically detects from model

    proj_hidden = 2048
    proj_out = 2048
    pred_hidden = 512
    pred_out = 2048
    encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out)
    predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
    model = SimSiam(encoder, predictor)
    model.to(device) #automatically detects from model

    # Optimizer and Scheduler
    # SimSiam uses SGD, with lr = lr*BS/256 from paper + https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py)
    init_lr = args.lr #*batch_size/256
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrain_base_lr*args.pretrain_batch_size/256., momentum=0.9, weight_decay= 5e-4)
    # #TODO:double check this Scheduler values
    # scheduler = LinearWarmupCosineAnnealingLR(
    #                         optimizer,
    #                         warmup_epochs=10,
    #                         max_epochs=epochs,
    #                         warmup_start_lr=args.warmup_start_lr,
    #                         eta_min=2e-4,
    #                     )
    # scheduler = SimSiamScheduler(optimizer, 
    #                              warmup_epochs=args.pretrain_warmup_epochs, warmup_lr=args.pretrain_warmup_lr*args.pretrain_batch_size/256., 
    #                              num_epochs=args.final_pretrain_epoch, base_lr=args.pretrain_base_lr*args.pretrain_batch_size/256., final_lr=0, iter_per_epoch=len(train_data_loaders), 
    #                              constant_predictor_lr=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
    wandb.init(project="SSL Project", name="SimSiam"
                + "-e" + str(epoch) + "-b" + str(batch_size) + "-lr" + str(args.pretrain_base_lr))

    #Training Loop 
    loss_ = []
    for epoch_counter in range(epoch):
        model.train()
        epoch_loss = []
        for x1, x2, y in train_data_loaders[0]:
            #x = x.to(device)
            loss = model(x1, x2)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        #TODO: do HP with linear warmup scheduler as well
        scheduler.step()

        # adjust_learning_rate(optimizer, lr, epoch_counter, epoch)       
        print(np.mean(epoch_loss))
        loss_.append(np.mean(epoch_loss))
        #TODO: knn predict
        if epoch_counter % 10 == 0:
            knn_acc = Knn_Validation(model, train_data_loaders_knn[0],test_data_loaders[0],device=device, K=200,sigma=0.5) 
            wandb.log({" Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})


    classifier =    LinClassifier().to(device)
    lin_epoch = 100
    lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.1, momentum=0.9) # Infomax: no weight decay, epoch 100, cosine scheduler
    lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=2e-4) #scheduler + values ref: infomax paper
    test_loss, test_acc1, test_acc5 = linear_evaluation(model, train_data_loaders_knn[0],test_data_loaders[0],lin_optimizer,classifier, lin_scheduler, epochs= lin_epoch)


    # save your encoder network
    save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'resnet18',
                    'lr':init_lr,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss_':loss_,
                }, is_best=False, filename='checkpoint_{:04f}.pth.tar'.format(lr))




