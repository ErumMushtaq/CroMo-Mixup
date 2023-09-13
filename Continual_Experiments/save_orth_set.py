import os
import sys
import wandb
import argparse
import numpy as np

from tqdm import tqdm
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
import torch
import torchvision.transforms as T
import torchvision

from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot
from models.linear_classifer import LinearClassifier
from models.ssl import  SimSiam, Siamese, Encoder, Predictor

from trainers.train_simsiam import train_simsiam
from trainers.train_infomax import train_infomax
from trainers.train_barlow import train_barlow

from trainers.train_PFR import train_PFR_simsiam
from trainers.train_PFR_contrastive import train_PFR_contrastive_simsiam
from trainers.train_contrastive import train_contrastive_simsiam
from trainers.train_ering import train_ering_simsiam

from torchsummary import summary
import random
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from utils.lars import LARS
from copy import deepcopy
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
import torch.nn as nn
import time
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset

from itertools import cycle
from torchvision import transforms
from copy import deepcopy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        return x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Args():
    normalization = 'batch'
    weight_standard = False
    same_lr = False
    pretrain_batch_size = 512
    pretrain_warmup_epochs = 10
    pretrain_warmup_lr = 3e-3
    pretrain_base_lr = 0.03
    pretrain_momentum = 0.9
    pretrain_weight_decay = 5e-4
    min_lr = 0.00
    lambdap = 1.0
    appr = 'barlow_PFR'
    knn_report_freq = 10
    cuda_device = 3
    num_workers = 8
    contrastive_ratio = 0.001
    dataset = 'cifar100'
    class_split = [20,20,20,20,20]
    epochs = [500,500,500,500,500]
    cov_loss_weight = 1.0
    sim_loss_weight = 250.0
    info_loss = 'invariance'
    lambda_norm = 1.0
    subspace_rate = 0.99
    lambda_param = 5e-3
    bsize = 32
    msize = 150
    proj_hidden = 2048
    proj_out = 2048 #infomax 64
    pred_hidden = 512
    pred_out = 2048
    epsilon=0.9

args = Args()

if args.dataset == "cifar10":
    get_dataloaders = get_cifar10
    num_classes=10
elif args.dataset == "cifar100":
    get_dataloaders = get_cifar100
    num_classes=100
assert sum(args.class_split) == num_classes
assert len(args.class_split) == len(args.epochs)

num_worker = args.num_workers
#device
device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
print(device)

#wandb init
wandb.init(project="CSSL",  entity="yavuz-team",
            mode="disabled",
            config=args,
            name= str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" 
            + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split))

if 'infomax' in args.appr or 'barlow' in args.appr:
    transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur()], p=0.5), 
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

    transform_prime = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur()], p=0.5), 
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])
    
#Dataloaders
print("Creating Dataloaders..")
#Class Based
_, _, _, _, train_data_loaders_linear, _, _  = get_dataloaders(transform, transform_prime, \
                                    classes=args.class_split, valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)


device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
print(device)
if 'infomax' in args.appr or 'barlow' in args.appr:
    proj_hidden = args.proj_hidden
    proj_out = args.proj_out
    encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr)
    model = Siamese(encoder)
    model.to(device) #automatically detects from model
    #load model here
model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)
model_path = "./checkpoints/checkpoint_cifar100-algocassle_barlow-e[500, 500, 500, 500, 500]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False_first_task_chkpt.pth.tar"
model.load_state_dict(torch.load(model_path)['state_dict'])

from functools import reduce
def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')[:-1]
     return reduce(getattr, names, module)

orth_set = {}
for name, param in model.encoder.backbone.named_parameters():
    if "weight" in name:
        module = get_module_by_name(model.encoder.backbone, name)
        if isinstance(module, torch.nn.Conv2d):
            orth_set[name] = None

def activation_collection(model, loader, device, orth_set):
    start = time.time()
    activation = {}
    def getActivation(id):
        # the hook signature
        def hook(model, input, output):
            activation[id].append(input[0].detach())
        return hook

    hooks = []
    for name, _ in model.encoder.backbone.named_parameters():
        if "weight" in name:
            module = get_module_by_name(model.encoder.backbone, name)
            if isinstance(module, torch.nn.Conv2d):
                activation[name] = []
                hooks.append(module.register_forward_hook(getActivation(name)))

    model.eval()
    for batch_index, (x, _) in enumerate(loader):
        x=x.to(device)
        _ = model.encoder.backbone(x)
        if batch_index > len(loader)/10-1:
            break
            
    for hook in hooks:
        hook.remove()

    for name in activation.keys():
        activation[name] = torch.cat(activation[name],dim=0)
        if "shortcut" not in name:
            activation[name] = F.pad(activation[name], (1, 1, 1, 1), "constant", 0)

    for name in activation.keys():
        module = get_module_by_name(model.encoder.backbone, name)
        unfolder = torch.nn.Unfold(module.kernel_size[0], dilation=1, padding=0, stride= module.stride[0])
        act = activation[name]
        mat = unfolder(act.to(device))
        mat = mat.permute(0,2,1)
        mat = mat.reshape(-1, mat.shape[2])
        mat = mat.T
    
        if orth_set[name] is not None:
            U = orth_set[name].to(device)
            projected = U @ U.T @ mat
            remaining = mat - projected
            activation[name] = remaining.cpu()
        else:
            activation[name] = mat.cpu()
    end = time.time()
    print(f'Activations collection time {end-start}')
    return activation 

def expand_orth_set(activations, orth_set, args, device):
    for key in activations.keys():
        if orth_set[key] == None:
            projected = torch.zeros(1)
        else:
            projected = orth_set[key]  @ orth_set[key].T @ activations[key] 

        remaining = (activations[key] - projected).to(device)
        remaining = remaining@remaining.T
        #find svds of remaining
        U, S, _ = torch.svd(remaining.cpu())
        #find how many singular vectors will be used
        total = torch.norm(activations[key])**2
        proj_norm = torch.norm(projected)**2
        for i in range(len(S)):
            hand = proj_norm + torch.sum(S[0:i+1])
            if i == 0 and hand / total > args.epsilon:
                break
            elif hand / total > args.epsilon:
                break
            
        print(U[:,0:i+1].shape)
        if orth_set[key] == None:
            orth_set[key] = U[:,0:i+1].cpu()
        else:
            orth_set[key] = torch.cat((orth_set[key], U[:,0:i+1]),dim=1).cpu()
            orth_set[key], _ = torch.qr(orth_set[key])

activations = activation_collection(model, train_data_loaders_linear[0], device, orth_set)
expand_orth_set(activations, orth_set, args, device)

import pickle 
with open('orrth_set.pkl', 'wb') as f:
    pickle.dump(orth_set, f)