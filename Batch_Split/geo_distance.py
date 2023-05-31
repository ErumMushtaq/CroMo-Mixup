import os
import sys
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import torch
import torchvision.transforms as T
import torchvision
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot
from models.linear_classifer import LinearClassifier
from models.ssl import  SimSiam, Siamese, Encoder, Predictor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"



def extract_subspace(model, loader, device = None):
    model.eval()
    outs = []
    for x,y in loader:
        x = x.to(device)
        out = model(x).cpu().detach().numpy()
        outs.append(out)


    outs = np.concatenate(outs)
    outs = outs.transpose()
    outs = torch.tensor(outs)
    # print(outs.shape) #dxN
    U, S, V = torch.svd(outs)
    # print(U.shape) #dxd
    # print(V.shape) #Nxd
    return U



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        return x
def add_args(parser):
    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('-cs', '--class_split', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)
    parser.add_argument('--appr', type=str, default='basic', help='Approach name, basic, PFR') #approach
    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)
    parser.add_argument('--normalization', type=str, default='batch', help='normalization method: batch, group or none')
    parser.add_argument('--weight_standard', action='store_true', default=False, help='weight standard for conv layers')
    parser.add_argument('--cuda_device', type=int, default=0, metavar='N',
                        help='device id')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='num of workers')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if args.dataset == "cifar10":
        get_dataloaders = get_cifar10
        num_classes=10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print(device)
    num_worker = int(8/len(args.class_split))
    if len(args.class_split) == 10:
        num_worker = 2
    # Step 1: Initialize Model and Load PreTrained Model
    if 'infomax' in args.appr or 'barlow' in args.appr:
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr)
        model = Siamese(encoder)
        model.to(device) # Automatically detects from model

        base_encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr)
        base_model = Siamese(base_encoder)
        base_model.to(device)

    # filename ='./checkpoints/final_checkpoint_{:04f}_cs_{}_bs_{}.pth.tar'.format(args.pretrain_base_lr, args.class_split, args.pretrain_batch_size)
    # print(filename)
    # dict_ = torch.load(filename, map_location='cpu')
    # model.load_state_dict(dict_['state_dict'])

    # filename = './checkpoints/basecase.tar'
    # filename = './checkpoints/Task0.tar'
    # basedict_ = torch.load(filename, map_location='cpu')
    # base_model.load_state_dict(basedict_['state_dict'])

    # Step 2: Load DataLoaders
    if 'infomax' in args.appr:
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=1.0), 
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.1),
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])
    if 'barlow' in args.appr: #ref: they do not have Gaussian Blur https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/augmentations/asymmetric.yaml
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=1.0), 
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.1),
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

    batch_size = []
    for k in range(len(args.class_split)):
        batch_size.append(512)
    train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, train_data_loaders_pure = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker)



    orthobases = []
    for i in range(len(args.class_split)):
        U = extract_subspace(base_model, test_data_loaders[i], device)
        orthobases.append(U)  #Extract U


    sing_val = []
    for i in range(len(args.class_split)):
        for j in range(len(args.class_split)):
            cross_U = (orthobases[i].T)@orthobases[j]  #Uj^T*U_i
            U, S, V = torch.svd(cross_U) 
            sing_val.append(S.mean())
    print(sing_val)           
                # print(math.degrees(math.acos(S)))







