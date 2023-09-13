import os
import sys
import time
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from loss import BarlowTwinsLoss
from models.ssl import  Siamese, Encoder

from utils.lars import LARS
from utils.eval_metrics import Knn_Validation_cont

from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

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
    cuda_device = 1
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
    scale_loss=0.1
    lambdacs=5

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

data_normalize_mean = (0.5071, 0.4865, 0.4409)
data_normalize_std = (0.2673, 0.2564, 0.2762)
random_crop_size = 32

transform_linear = transforms.Compose( [
            transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(data_normalize_mean, data_normalize_std),
        ] )

#Dataloaders
print("Creating Dataloaders..")
#Class Based
train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, train_data_loaders_pure, train_data_loaders_generic  = get_dataloaders(transform, transform_prime, \
                                    classes=args.class_split, valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)
_, train_data_loaders_knn_all, test_data_loaders_all, _, train_data_loaders_linear_all, train_data_loaders_pure_all, _ = get_dataloaders(transform, transform_prime, \
                                        classes=[num_classes], valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)


def get_cone(loader, model, device, quantile=0.05):
    features = torch.Tensor([])
    model.eval()
    for x, _ in loader:
        out = model.encoder.backbone(x.to(device)).detach().cpu().squeeze()
        features = torch.cat((features, out), dim=0)
    mean = torch.mean(features, dim=0)
    scores = torch.cosine_similarity(mean, features)
    cs = torch.quantile(scores, q=quantile)
    return mean, cs

def store_samples(loader, num):
    x_data = loader.dataset.train_data
    select = np.random.randint(0,x_data.shape[0],num)
    return torch.Tensor(x_data[select])

def train(model, loader, knn_train_data_loaders, memory, epochs, cone_mean, cone_cs, transform, transform_prime, transform_linear, device, args):
    
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    # Optimizer and Scheduler
    init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
    optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

    loader.dataset.transforms = [transform, transform_prime, transform_linear]
    for epoch in range(epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        coss_loss = []
        coss_loss_old = []
        for x, _ in loader:
            x1, x2, x3 = x[0], x[1], x[2]
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

            #ering
            x1_old = torch.Tensor([]).to(device)
            x2_old = torch.Tensor([]).to(device)
            x3_old = torch.Tensor([]).to(device)
            indices = np.random.randint(0,memory.shape[0],128)
            for ind in indices:
                k = memory[ind:ind+1].to(device)
                x1_old = torch.cat((x1_old, transform(k)), dim=0)
                x2_old = torch.cat((x2_old, transform_prime(k)), dim=0)
                x3_old = torch.cat((x3_old, transform_linear(k)), dim=0)

            x1 = torch.cat((x1,x1_old))
            x2 = torch.cat((x2,x2_old))

            #barlow
            z1,z2 = model(x1, x2)
            loss =  cross_loss(z1, z2)

            #cosine
            m3 = model.encoder.backbone(x3).squeeze()
            scores = torch.cosine_similarity(cone_mean.to(device), m3)
            cossine_loss = torch.max(torch.tensor(0), scores-cone_cs).mean()
            m3_old = model.encoder.backbone(x3_old).squeeze()
            scores = torch.cosine_similarity(cone_mean.to(device), m3_old)
            cossine_loss_old = torch.max(torch.tensor(0), cone_cs-scores).mean()
            loss += args.lambdacs * (cossine_loss+15*cossine_loss_old)
            # cossine_loss=0
            # cossine_loss_old=0


            epoch_loss.append(loss.item())
            coss_loss.append(cossine_loss.item())
            coss_loss_old.append(cossine_loss_old.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        scheduler.step()
        end = time.time()
        if (epoch) % args.knn_report_freq == 0:
            knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:2], test_data_loaders[:2], device=device, K=200, sigma=0.5) 
            print(task_acc_arr)
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} | Cos Loss: {np.mean(coss_loss):.4f}  | Cos Loss Old: {np.mean(coss_loss_old):.4f} | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} | Cos Loss: {np.mean(coss_loss):.4f}  | Cos Loss Old: {np.mean(coss_loss_old):.4f} ')
    
    return model, optimizer

# device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
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

mean, cs = get_cone(train_data_loaders_linear[0], model, device, quantile=0.05)
memory = store_samples(train_data_loaders_generic[0], 500)

model, opt = train(model, train_data_loaders_generic[1], train_data_loaders_knn, memory, 500, mean, cs, \
    transform, transform_prime, transform_linear, device, args)