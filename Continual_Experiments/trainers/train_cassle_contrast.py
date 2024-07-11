import time
import wandb
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

import torch.nn as nn
from itertools import cycle

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from copy import deepcopy
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from utils.lars import LARS
#https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/predictive_mse.py

from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset, SimSiam_Dataset
import torchvision.transforms as transforms
from torchvision import transforms as  T,utils
from tqdm import tqdm

def correct_top_k(outputs, targets, top_k=(1,5)):
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result

def linear_test(net, data_loader, classifier, epoch, device, task_num):
    # evaluate model:
    net.eval() # for not update batchnorm
    linear_loss = 0.0
    num = 0
    total_loss, total_correct_1, total_num, test_bar = 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data_tuple in test_bar:
            data, target = [t.to(device) for t in data_tuple]
            output = net(data)
            if classifier is not None:  #else net is already a classifier
                output = classifier(output) 
            linear_loss = F.cross_entropy(output, target)
            
            # Batchsize for loss and accuracy
            num = data.size(0)
            total_num += num 
            total_loss += linear_loss.item() * num 
            # Accumulating number of correct predictions 
            correct_top_1 = correct_top_k(output, target, top_k=[1])    
            total_correct_1 += correct_top_1[0]
            test_bar.set_description('Lin.Test Epoch: [{}] Loss: {:.4f} ACC: {:.2f}% '
                                     .format(epoch,  total_loss / total_num,
                                             total_correct_1 / total_num * 100
                                             ))
        acc_1 = total_correct_1/total_num*100
        wandb.log({f" {task_num} Linear Layer Test Loss ": linear_loss / total_num, "Linear Epoch ": epoch})
        wandb.log({f" {task_num} Linear Layer Test - Acc": acc_1, "Linear Epoch ": epoch})
    return total_loss / total_num, acc_1  

def linear_train(net, data_loader, train_optimizer, classifier, scheduler, epoch, device, task_num):

    net.eval() # for not update batchnorm 
    total_num, train_bar = 0, tqdm(data_loader)
    linear_loss = 0.0
    total_correct_1 = 0.0
    for data_tuple in train_bar:
        # Forward prop of the model with single augmented batch
        pos_1, target = data_tuple
        pos_1 = pos_1.to(device)
        feature_1 = net(pos_1)
        # Batchsize
        batchsize_bc = feature_1.shape[0]
        features = feature_1
        targets = target.to(device)
        logits = classifier(features.detach()) 
        # Cross Entropy Loss 
        linear_loss_1 = F.cross_entropy(logits, targets)

        # Number of correct predictions
        linear_correct_1 = correct_top_k(logits, targets, top_k=[1])
    
        # Backpropagation part
        train_optimizer.zero_grad()
        linear_loss_1.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc
        linear_loss += linear_loss_1.item() * batchsize_bc
        total_correct_1 += linear_correct_1[0] 

        acc_1 = total_correct_1/total_num*100
        # # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f} ACC: {:.2f}'.format(\
                epoch, linear_loss / total_num, acc_1))
    scheduler.step()
    acc_1 = total_correct_1/total_num*100   
    wandb.log({f" {task_num} Linear Layer Train Loss ": linear_loss / total_num, "Linear Epoch ": epoch})
    wandb.log({f" {task_num} Linear Layer Train - Acc": acc_1, "Linear Epoch ": epoch})
        
    return linear_loss/total_num, acc_1


def linear_evaluation(net, data_loaders,test_data_loaders,train_optimizer,classifier, scheduler, epochs, device, task_num):
    train_X = torch.Tensor([])
    train_Y = torch.tensor([],dtype=int)
    for loader in data_loaders:
        train_X = torch.cat((train_X, loader.dataset.train_data), dim=0)
        train_Y = torch.cat((train_Y, loader.dataset.label_data), dim=0)
    data_loader = DataLoader(TensorDataset(train_X, train_Y,transform=data_loaders[0].dataset.transform), batch_size=256, shuffle=True, num_workers = 5, pin_memory=True)

    test_X = torch.Tensor([])
    test_Y = torch.tensor([],dtype=int)
    for loader in test_data_loaders:
        test_X = torch.cat((test_X, loader.dataset.train_data), dim=0)
        test_Y = torch.cat((test_Y, loader.dataset.label_data), dim=0)
    test_data_loader = DataLoader(TensorDataset(test_X, test_Y,transform=test_data_loaders[0].dataset.transform), batch_size=256, shuffle=True, num_workers = 5, pin_memory=True)

    for epoch in range(1, epochs+1):
        linear_train(net,data_loader,train_optimizer,classifier,scheduler, epoch, device, task_num)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1 = linear_test(net, test_data_loader, classifier, epoch, device, task_num)

    return test_loss, test_acc1, classifier


def store_samples(loader, task_id, num):
    x_data = loader.dataset.train_data
    select = np.random.randint(0,x_data.shape[0],num)
    labels = torch.ones(num,dtype=torch.long) * task_id 
    return torch.Tensor(x_data[select]), labels


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)

class Predictor(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return out
    
def process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args):
    z1,z2 = model(x1, x2)
    loss =  cross_loss(z1, z2)
    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

def process_batch_cassle_ering(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args):
    x1 = torch.cat((x1,x1_old))
    x2 = torch.cat((x2,x2_old))

    z1, z2 = model(x1, x2)

    loss_new = cross_loss(z1, z2) 
    loss = loss_new
    
    lossKD = 0.0
    
    f1Old = oldModel.encoder(x1).squeeze().detach()
    f2Old = oldModel.encoder(x2).squeeze().detach()
    p2_1 = model.temporal_projector(z1)
    p2_2 = model.temporal_projector(z2)

    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
                            + cross_loss(p2_2, f2Old).mean() * 0.5) )
    loss += lossKD 

    
    
    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

# def process_batch_ering_contrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old):



#     # x1 = torch.cat((x1, x1_old))
#     # x2 = torch.cat((x2, x2_old))
#     z1, z2 = model(x1, x2)
#     z1old, z2old = model(x1_old, x2_old)
#     # f2o_old = oldModel.encoder(x2_old).detach()

#     z1new = torch.cat([z1, z1old])
#     z2new = torch.cat([z2, features_old]) #consistency + contrast

#     # print(z1new.shape)
#     # print(z2new.shape)
#     # print(features_old.shape)

#     loss_new = cross_loss(z1new, z2new) 
#     loss = loss_new
    
#     lossKD = 0.0
    
#     if args.lambdap != 0.0:
#         f1Old = oldModel.encoder(x1).squeeze().detach()
#         f2Old = oldModel.encoder(x2).squeeze().detach()
#         p2_1 = model.temporal_projector(z1)
#         p2_2 = model.temporal_projector(z2)

#         lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
#                                 + cross_loss(p2_2, f2Old).mean() * 0.5) )
#         loss += lossKD 
    
#     epoch_loss.append(loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step() 

#     return model, optimizer

def process_batch_ering_contrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old):
    curr_task_size = x2.shape[0]
    x1 = torch.cat((x1, x1_old))
    x2 = torch.cat((x2, x2_old))
    z1, z2 = model(x1, x2)
    f1Old = oldModel.encoder(x1).squeeze().detach()
    f2Old = oldModel.encoder(x2).squeeze().detach()

    # Consistency
    z2new = torch.cat([z2[:curr_task_size], f1Old[curr_task_size:]]) #same aug + consistency + contrast

    # print(z1new.shape)
    # print(z2new.shape)
    # print(features_old.shape)

    loss_new = cross_loss(z1, z2new) 
    loss = loss_new
    
    lossKD = 0.0
    
    p2_1 = model.temporal_projector(z1)
    p2_2 = model.temporal_projector(z2)

    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
                            + cross_loss(p2_2, f2Old).mean() * 0.5) )
    loss += lossKD 
    
    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

# Augmentation
def process_batch_ering_augcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old, distil_loss, x, x_old_bs, transform2, transform2_prime):
    curr_task_size = x2.shape[0]
    lam = np.random.beta(args.alpha, args.alpha)
    # for ind in range(curr_task_size):
    #     x_old_bs[ind:ind+1] = transform2(x_old_bs[ind:ind+1])
    #     x[ind:ind+1] = transform2(x[ind:ind+1])

    mix_x1 = lam * x1 + (1 - lam) * x1_old
    mix_x2 = lam * x2 + (1 - lam) * x2_old

    # mix_x1 = lam * x1 + (1 - lam) * x1_old
    # mix_x2 = lam * x2 + (1 - lam) * x2_old

    x1 = torch.cat((x1, x1_old))
    x2 = torch.cat((x2, x2_old))

    z1, z2 = model(x1, x2)
    mixz1, mixz2 = model(mix_x1, mix_x2)

    org_loss = cross_loss(z1, z2).mean()
    ood_loss = 0.5 * (lam * cross_loss(mixz1, z1[:curr_task_size]).mean() + (1-lam) * cross_loss(mixz1, z1[curr_task_size:]).mean() )+ \
            0.5 * (lam * cross_loss(mixz2, z2[:curr_task_size]).mean() + (1-lam) * cross_loss(mixz2, z2[curr_task_size:]).mean())
    # mixz1, mixz2 = oldModel(mix_x1, mix_x2)
    # z_org, z_old = model(x, x_old_bs)

    # mixed_z1 = lam * z1[:curr_task_size] * (1 - lam) * z1[curr_task_size:]
    # mixed_z2 = lam * z2[:curr_task_size] * (1 - lam) * z2[curr_task_size:]

    # org_loss = cross_loss(z1, z2)
    # ood_loss = 0.5 * (lam * cross_loss(mixz1, z1[:curr_task_size]).mean() + (1-lam) * cross_loss(mixz1, z1[curr_task_size:])) +  0.5 * (lam * cross_loss(mixz2, z2[:curr_task_size]).mean() + (1-lam) * cross_loss(mixz2, z2[curr_task_size:]))

    #new 
    # ood_loss = 0.5 * (lam * cross_loss(mixz1, z_org).mean() + (1-lam) * cross_loss(mixz1, z_old)) +  0.5 * (lam * cross_loss(mixz2, z_old).mean() + (1-lam) * cross_loss(mixz2, z_org).mean())
    loss = org_loss + ood_loss

    # f1Old = oldModel.encoder(x1).squeeze().detach()
    # f2Old = oldModel.encoder(x2).squeeze().detach()

    # # Consistency
    # z2new = torch.cat([z2[:curr_task_size], f1Old[curr_task_size:]]) #same aug + consistency + contrast

    # # print(z1new.shape)
    # # print(z2new.shape)
    # # print(features_old.shape)

    # loss_new = cross_loss(z1, z2new) 
    # loss = loss_new
    
    # lossKD = 0.0
    
    # p2_1 = model.temporal_projector(z1)
    # p2_2 = model.temporal_projector(z2)

    # lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
    #                         + cross_loss(p2_2, f2Old).mean() * 0.5) )
    # loss += lossKD 
    
    epoch_loss.append(org_loss.item())
    distil_loss.append(ood_loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

# Augmentation
def process_batch_ering_inputmixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old, distil_loss, x, x_old_bs, transform2, transform2_prime):
    curr_task_size = x2.shape[0]
    if curr_task_size < args.replay_bs:
        old_task_size = curr_task_size
    else:
        old_task_size = args.replay_bs
    lam = np.random.beta(args.alpha, args.alpha)
    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

    # x1 = torch.cat((x1, mix_x1))
    # x2 = torch.cat((x2, mix_x2))

    #     x1 = torch.cat((x1, mix_x1))
    # x2 = torch.cat((x2, mix_x2))

    z1, z2 = model(mix_x1, mix_x2)
    org_loss = cross_loss(z1, z2).mean()
    loss = org_loss
    epoch_loss.append(org_loss.item())
    distil_loss.append(0.0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

# Augmentation
def process_batch_ering_iomixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old, distil_loss, x, x_old_bs, transform2, transform2_prime):
    curr_task_size = x2.shape[0]
    if curr_task_size < args.replay_bs:
        old_task_size = curr_task_size
    else:
        old_task_size = args.replay_bs
    lam = np.random.beta(args.alpha, args.alpha)
    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

    x1 = torch.cat((x1, mix_x1))
    x2 = torch.cat((x2, mix_x2))

    z1, z2 = model(x1, x2)
    z1old, z2old = model(x1_old, x2_old)
    org_loss = cross_loss(z1[:curr_task_size], z2[:curr_task_size]).mean() 

    ood_loss = 0.5 * (lam * cross_loss(z1[curr_task_size:], z1[:old_task_size]).mean() + (1-lam) * cross_loss(z1[curr_task_size:], z1old[:old_task_size]).mean())+ \
             0.5 * (lam * cross_loss(z2[curr_task_size:], z2[:old_task_size]).mean() + (1-lam) * cross_loss(z2[curr_task_size:], z2old[:old_task_size]).mean())
    loss = org_loss + ood_loss    
    epoch_loss.append(org_loss.item())
    distil_loss.append(ood_loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer    

# Augmentation
def process_batch_ering_principled_iomixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old, distil_loss, x, x_old_bs, transform2, transform2_prime):
    curr_task_size = x2.shape[0]
    if curr_task_size < args.replay_bs:
        old_task_size = curr_task_size
    else:
        old_task_size = args.replay_bs
    lam = 0.7 #np.random.beta(args.alpha, args.alpha)
    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

    x1 = torch.cat((x1, mix_x1))
    x2 = torch.cat((x2, mix_x2))

    z1, z2 = model(x1, x2)
    org_loss = cross_loss(z1[:curr_task_size], z2[:curr_task_size]).mean() 
    ood_loss = 0.5 * (lam * cross_loss(z1[curr_task_size:], z1[:old_task_size]).mean() )+ \
             0.5 * (lam * cross_loss(z2[curr_task_size:], z2[:old_task_size]).mean())
    loss = org_loss + ood_loss    
    epoch_loss.append(org_loss.item())
    distil_loss.append(ood_loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer  

def process_batch_mixed_distillation_contrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old):
    curr_task_size = x2.shape[0]
    if curr_task_size < args.replay_bs:
        old_task_size = curr_task_size
    else:
        old_task_size = args.replay_bs
    lam = np.random.beta(args.alpha, args.alpha) #0.7
    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

    x1_hat = torch.cat((x1, mix_x1))
    x2_hat = torch.cat((x2, mix_x2))
    z1, z2 = model(x1_hat, x2_hat)

    x1_ = torch.cat((x1, x1_old[:old_task_size]))
    x2_ = torch.cat((x2, x2_old[:old_task_size]))
    f1Old = oldModel.encoder(x1_).squeeze().detach()
    f2Old = oldModel.encoder(x2_).squeeze().detach()
    org_loss = cross_loss(z1[:curr_task_size], z2[:curr_task_size])
    mixed_loss = 0.5*(lam* cross_loss(z1[curr_task_size:], z1[:old_task_size]) + (1-lam)* cross_loss(z1[curr_task_size:], f1Old[curr_task_size:]))+\
            0.5*(lam* cross_loss(z2[curr_task_size:], z2[:old_task_size]) + (1-lam)* cross_loss(z2[curr_task_size:], f2Old[curr_task_size:]))

    loss = org_loss + mixed_loss
    
    lossKD = 0.0
    
    p2_1 = model.temporal_projector(z1[:curr_task_size])
    p2_2 = model.temporal_projector(z2[:curr_task_size])

    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old[:curr_task_size]).mean() * 0.5
                            + cross_loss(p2_2, f2Old[:curr_task_size]).mean() * 0.5) )
    loss += lossKD 
    
    epoch_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return model, optimizer

def train_cassle_barlow_ering_contrast(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    if args.temp_proj == 'nonlinear':
        model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
    else:
        model.temporal_projector = nn.Identity().to(device)
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        if task_id == 0 and args.start_chkpt == 1:
            model_path = "./checkpoints/checkpoint_cifar100-algobarlow_ering_negcontrast-e[750, 750, 750, 750, 750]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = args.epochs[task_id]
        else:
            # Optimizer and Scheduler
            model.task_id = task_id
            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            if task_id != 0 and args.same_lr != True:
                init_lr = init_lr / 10

            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper                
            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                epoch_loss = []
                distil_loss = []
                kd_loss_cur = []
                kd_loss_old = []
                loss1= []
                loss2=[]
                itr = 0
                if task_id == 0:
                    for x1, x2, _ in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        model, optimizer = process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args)
                else:
                    for cur_data in loader:
                        x1, x2, _ = cur_data
                        x1, x2 = x1.to(device), x2.to(device)
                        x1_old = torch.Tensor([]).to(device)
                        x2_old = torch.Tensor([]).to(device)
                        f2_old = torch.Tensor([]).to(device)
                        if args.replay_bs == args.msize:
                            replay_batchsize = x_old.shape[0]
                        elif args.replay_bs == args.pretrain_batch_size:
                            replay_batchsize = x1.shape[0]
                        else:
                            replay_batchsize = args.replay_bs
                        indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                        x_old_bs = x_old[indices]
                        for ind in indices:
                            x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                            x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                            # f2_old = torch.cat((f2_old, features_old[ind:ind+1]), dim=0)
                        x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                        # if args.appr == 'barlow_ering_contrast':
                        model, optimizer = process_batch_cassle_ering(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args)
                        # elif args.appr == 'barlow_ering_negcontrast':
                        #     model, optimizer = process_batch_ering_contrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, f2_old)
                        # elif args.appr == 'barlow_ering_augcontrast':
                        #     model, optimizer = process_batch_ering_augcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, f2_old, distil_loss, x, x_old_bs, transform2, transform2_prime)
                        # elif args.appr == 'barlow_ering_inputmixcontrast':
                        #     model, optimizer = process_batch_ering_inputmixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, f2_old, distil_loss, x, x_old_bs, transform2, transform2_prime)


                epoch_counter += 1
                scheduler.step()
                loss_.append(np.mean(epoch_loss))
                end = time.time()
                print('epoch end')
                if (epoch+1) % args.knn_report_freq == 0:
                    knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                    wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                    for i, acc in enumerate(task_acc_arr):
                        wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} | Knn:  {knn_acc*100:.2f}')
                    print(task_acc_arr)
                else:
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} ')
            
                wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
                wandb.log({" Average Aug Train Loss ": np.mean(distil_loss), " Epoch ": epoch_counter})
                wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
            # # save your encoder network
            if task_id == 0:
                file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
                torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)
    

        # oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id < len(train_data_loaders)-1:
           lin_epoch = 1
           num_class = np.sum(args.class_split[:task_id+1])
           classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
           lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
           lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
           linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
        # features = oldModel.encoder(x_samp.to(device)).detach()
        # features_old = torch.cat([features_old, features])

    return model, loss_, optimizer

def train_cassle_barlow_inputmixup(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    if args.temp_proj == 'nonlinear':
        model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
    else:
        model.temporal_projector = nn.Identity().to(device)
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        if task_id == 0 and args.start_chkpt == 1:
            model_path = "./checkpoints/checkpoint_cifar100-algobarlow_ering_negcontrast-e[750, 750, 750, 750, 750]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = args.epochs[task_id]
        else:
            # Optimizer and Scheduler
            model.task_id = task_id
            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            if task_id != 0 and args.same_lr != True:
                init_lr = init_lr / 10

            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper                
            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                epoch_loss = []
                distil_loss = []
                kd_loss_cur = []
                kd_loss_old = []
                loss1= []
                loss2=[]
                itr = 0
                if task_id == 0:
                    for x, x1, x2, _ in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        model, optimizer = process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args)
                else:
                    for cur_data in loader:
                        x, x1, x2, _ = cur_data
                        x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                        x1_old = torch.Tensor([]).to(device)
                        x2_old = torch.Tensor([]).to(device)
                        f2_old = torch.Tensor([]).to(device)
                        replay_batchsize = args.replay_bs
                        indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                        x_old_bs = x_old[indices]
                        for ind in indices:
                            x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                            x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                            # f2_old = torch.cat((f2_old, features_old[ind:ind+1]), dim=0)
                        x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                        model, optimizer = process_batch_ering_inputmixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, f2_old, distil_loss, x, x_old_bs, transform2, transform2_prime)


                epoch_counter += 1
                scheduler.step()
                loss_.append(np.mean(epoch_loss))
                end = time.time()
                print('epoch end')
                if (epoch+1) % args.knn_report_freq == 0:
                    knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                    wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                    for i, acc in enumerate(task_acc_arr):
                        wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} | Knn:  {knn_acc*100:.2f}')
                    print(task_acc_arr)
                else:
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} ')
            
                wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
                wandb.log({" Average Aug Train Loss ": np.mean(distil_loss), " Epoch ": epoch_counter})
                wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
            # # save your encoder network
            if task_id == 0:
                file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
                torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)
    

        # oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id < len(train_data_loaders)-1:
           lin_epoch = 1
           num_class = np.sum(args.class_split[:task_id+1])
           classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
           lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
           lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
           linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
        # features = oldModel.encoder(x_samp.to(device)).detach()
        # features_old = torch.cat([features_old, features])

    return model, loss_, optimizer

def train_cassle_barlow_iomixup(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    if args.temp_proj == 'nonlinear':
        model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
    else:
        model.temporal_projector = nn.Identity().to(device)
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        if task_id == 0 and args.start_chkpt == 1:
            model_path = "./checkpoints/checkpoint_cifar100-algobarlow_ering_negcontrast-e[750, 750, 750, 750, 750]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = args.epochs[task_id]
        else:
            # Optimizer and Scheduler
            model.task_id = task_id
            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            if task_id != 0 and args.same_lr != True:
                init_lr = init_lr / 10

            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper                
            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                epoch_loss = []
                distil_loss = []
                kd_loss_cur = []
                kd_loss_old = []
                loss1= []
                loss2=[]
                itr = 0
                if task_id == 0:
                    for x, x1, x2, _ in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        model, optimizer = process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args)
                else:
                    for cur_data in loader:
                        x, x1, x2, _ = cur_data
                        x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                        x1_old = torch.Tensor([]).to(device)
                        x2_old = torch.Tensor([]).to(device)
                        f2_old = torch.Tensor([]).to(device)
                        replay_batchsize = args.replay_bs
                        indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                        x_old_bs = x_old[indices]
                        for ind in indices:
                            x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                            x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                            # f2_old = torch.cat((f2_old, features_old[ind:ind+1]), dim=0)
                        x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                        model, optimizer = process_batch_ering_iomixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, f2_old, distil_loss, x, x_old_bs, transform2, transform2_prime)

                epoch_counter += 1
                scheduler.step()
                loss_.append(np.mean(epoch_loss))
                end = time.time()
                print('epoch end')
                if (epoch+1) % args.knn_report_freq == 0:
                    knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                    wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                    for i, acc in enumerate(task_acc_arr):
                        wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} | Knn:  {knn_acc*100:.2f}')
                    print(task_acc_arr)
                else:
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} ')
            
                wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
                wandb.log({" Average Aug Train Loss ": np.mean(distil_loss), " Epoch ": epoch_counter})
                wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
            # # save your encoder network
            if task_id == 0:
                file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
                torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)
    

        # oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id < len(train_data_loaders)-1:
           lin_epoch = 1
           num_class = np.sum(args.class_split[:task_id+1])
           classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
           lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
           lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
           linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
        # features = oldModel.encoder(x_samp.to(device)).detach()
        # features_old = torch.cat([features_old, features])

    return model, loss_, optimizer

def train_cassle_barlow_principled_iomix(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    if args.temp_proj == 'nonlinear':
        model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
    else:
        model.temporal_projector = nn.Identity().to(device)
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        if task_id == 0 and args.start_chkpt == 1:
            model_path = "./checkpoints/checkpoint_cifar100-algobarlow_ering_negcontrast-e[750, 750, 750, 750, 750]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = args.epochs[task_id]
        else:
            # Optimizer and Scheduler
            model.task_id = task_id
            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            if task_id != 0 and args.same_lr != True:
                init_lr = init_lr / 10

            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper                
            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                epoch_loss = []
                distil_loss = []
                kd_loss_cur = []
                kd_loss_old = []
                loss1= []
                loss2=[]
                itr = 0
                if task_id == 0:
                    for x, x1, x2, _ in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        model, optimizer = process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args)
                else:
                    for cur_data in loader:
                        x, x1, x2, _ = cur_data
                        x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                        x1_old = torch.Tensor([]).to(device)
                        x2_old = torch.Tensor([]).to(device)
                        f2_old = torch.Tensor([]).to(device)
                        replay_batchsize = args.replay_bs
                        indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                        x_old_bs = x_old[indices]
                        for ind in indices:
                            x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                            x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                            # f2_old = torch.cat((f2_old, features_old[ind:ind+1]), dim=0)
                        x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                        model, optimizer = process_batch_ering_principled_iomixcontrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old, distil_loss, x, x_old_bs, transform2, transform2_prime)

                epoch_counter += 1
                scheduler.step()
                loss_.append(np.mean(epoch_loss))
                end = time.time()
                print('epoch end')
                if (epoch+1) % args.knn_report_freq == 0:
                    knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                    wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                    for i, acc in enumerate(task_acc_arr):
                        wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} | Knn:  {knn_acc*100:.2f}')
                    print(task_acc_arr)
                else:
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} ')
            
                wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
                wandb.log({" Average Aug Train Loss ": np.mean(distil_loss), " Epoch ": epoch_counter})
                wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
            # # save your encoder network
            if task_id == 0:
                file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
                torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)
    

        # oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id < len(train_data_loaders)-1:
           lin_epoch = 1
           num_class = np.sum(args.class_split[:task_id+1])
           classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
           lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
           lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
           linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
        # features = oldModel.encoder(x_samp.to(device)).detach()
        # features_old = torch.cat([features_old, features])

    return model, loss_, optimizer

def train_cassle_barlow_mixed_distillation(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    if args.temp_proj == 'nonlinear':
        model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
    else:
        model.temporal_projector = nn.Identity().to(device)
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        if task_id == 0 and args.start_chkpt == 1:
            model_path = "./checkpoints/checkpoint_cifar100-algobarlow_ering_negcontrast-e[750, 750, 750, 750, 750]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = args.epochs[task_id]
        else:
            # Optimizer and Scheduler
            model.task_id = task_id
            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            if task_id != 0 and args.same_lr != True:
                init_lr = init_lr / 10

            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper                
            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                epoch_loss = []
                distil_loss = []
                kd_loss_cur = []
                kd_loss_old = []
                loss1= []
                loss2=[]
                itr = 0
                if task_id == 0:
                    for x, x1, x2, _ in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        model, optimizer = process_batch(x1, x2, model, cross_loss, optimizer, epoch_loss, args)
                else:
                    for cur_data in loader:
                        x, x1, x2, _ = cur_data
                        x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                        x1_old = torch.Tensor([]).to(device)
                        x2_old = torch.Tensor([]).to(device)
                        f2_old = torch.Tensor([]).to(device)
                        replay_batchsize = args.replay_bs
                        indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                        x_old_bs = x_old[indices]
                        for ind in indices:
                            x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                            x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                            # f2_old = torch.cat((f2_old, features_old[ind:ind+1]), dim=0)
                        x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                        model, optimizer = process_batch_mixed_distillation_contrast(x1, x2, x1_old, x2_old, model, cross_loss, oldModel, optimizer, epoch_loss, args, features_old)

                epoch_counter += 1
                scheduler.step()
                loss_.append(np.mean(epoch_loss))
                end = time.time()
                print('epoch end')
                if (epoch+1) % args.knn_report_freq == 0:
                    knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                    wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                    for i, acc in enumerate(task_acc_arr):
                        wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} | Knn:  {knn_acc*100:.2f}')
                    print(task_acc_arr)
                else:
                    print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time: {end-start:.1f}s | Loss: {np.mean(epoch_loss):.4f} | Aug Loss: {np.mean(distil_loss):.4f} ')
            
                wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
                wandb.log({" Average Aug Train Loss ": np.mean(distil_loss), " Epoch ": epoch_counter})
                wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
            # # save your encoder network
            if task_id == 0:
                file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
                torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)
    

        # oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id < len(train_data_loaders)-1:
           lin_epoch = 1
           num_class = np.sum(args.class_split[:task_id+1])
           classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
           lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
           lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
           linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
        # features = oldModel.encoder(x_samp.to(device)).detach()
        # features_old = torch.cat([features_old, features])

    return model, loss_, optimizer


def train_infomax_iomix(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)
    old_model = None
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)
    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr
        # init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10
            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
        covarince_loss = CovarianceLoss(args.proj_out, device=device)
        old_covarince_loss = CovarianceLoss(args.proj_out, device=device)
        if args.info_loss == 'error_cov':
            err_covarince_loss = ErrorCovarianceLoss(args.proj_out ,device=device)

        loss_ = []
        old_covarince_loss = CovarianceLoss(args.proj_out, device=device)
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            distil_loss = []
            if task_id == 0:
                for x, x1, x2, y in loader:
                    x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                    z1_cur, z2_cur = model(x1, x2)
                    z1 = F.normalize(z1_cur, p=2)
                    z2 = F.normalize(z2_cur, p=2)
                    cov_loss =  covarince_loss(z1, z2)
                    sim_loss =  invariance_loss(z1, z2)
                    loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)
                    epoch_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

            else:
                for cur_data in loader:
                    x, x1, x2, _ = cur_data
                    x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    f2_old = torch.Tensor([]).to(device)
                    replay_batchsize = args.replay_bs
                    indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                    x_old_bs = x_old[indices]
                    for ind in indices:
                        x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                    x1_old, x2_old = x1_old.to(device), x2_old.to(device)

                    curr_task_size = x2.shape[0]
                    if curr_task_size < args.replay_bs:
                        old_task_size = curr_task_size
                    else:
                        old_task_size = args.replay_bs
                    lam = np.random.beta(args.alpha, args.alpha)
                    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
                    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

                    x1 = torch.cat((x1, mix_x1))
                    x2 = torch.cat((x2, mix_x2))

                    z1, z2 = model(x1, x2)
                    z1old, z2old = model(x1_old, x2_old)
                    z1 = F.normalize(z1, p=2)
                    z2 = F.normalize(z2, p=2)
                    z1old = F.normalize(z1old, p=2)
                    z2old = F.normalize(z2old, p=2)

                    cov_loss =  covarince_loss(z1[:curr_task_size], z2[:curr_task_size])
                    sim_loss =  invariance_loss(z1[:curr_task_size], z2[:curr_task_size])
                    org_loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)

                    cov_ood_loss = 0.25 * (lam * old_covarince_loss(z1[curr_task_size:], z1[:old_task_size]) + (1-lam) * old_covarince_loss(z1[curr_task_size:], z1old[:old_task_size]))+ \
                        0.25 * (lam * old_covarince_loss(z2[curr_task_size:], z2[:old_task_size]) + (1-lam) * old_covarince_loss(z2[curr_task_size:], z2old[:old_task_size]))
                    sim_ood_loss = 0.25 * (lam * invariance_loss(z1[curr_task_size:], z1[:old_task_size]) + (1-lam) * invariance_loss(z1[curr_task_size:], z1old[:old_task_size]))+ \
                        0.25 * (lam * invariance_loss(z2[curr_task_size:], z2[:old_task_size]) + (1-lam) * invariance_loss(z2[curr_task_size:], z2old[:old_task_size]))   
                    ood_loss =   args.sim_loss_weight* sim_ood_loss +cov_ood_loss *  args.cov_loss_weight 

                    loss = org_loss + ood_loss
                    epoch_loss.append(org_loss.item())
                    distil_loss.append(ood_loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model)  # save t-1 model
        oldModel.to(device)
        oldModel.train()

        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        #torch.save({
        #                'state_dict': model.state_dict(),
        #                'optimizer' : optimizer.state_dict(),
        #                'encoder': model.encoder.backbone.state_dict(),
        #            }, file_name)
        
        if task_id < len(train_data_loaders)-1:
            lin_epoch = 1
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  

        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
    return model, loss_, optimizer

def train_cassle_infomax_mixed_distillation(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime):    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)
    old_model = None
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)
    x_old = torch.Tensor([]).to(device)
    features_old = torch.Tensor([]).to(device)
    y_old = torch.tensor([],dtype=torch.long).to(device)

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr
        # init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10
            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
        covarince_loss = CovarianceLoss(args.proj_out, device=device)
        old_covarince_loss = CovarianceLoss(args.proj_out, device=device)
        if args.info_loss == 'error_cov':
            err_covarince_loss = ErrorCovarianceLoss(args.proj_out ,device=device)

        loss_ = []
        old_covarince_loss = CovarianceLoss(args.proj_out, device=device)
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            distil_loss = []
            if task_id == 0:
                for x, x1, x2, y in loader:
                    x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                    z1_cur, z2_cur = model(x1, x2)
                    z1 = F.normalize(z1_cur, p=2)
                    z2 = F.normalize(z2_cur, p=2)
                    cov_loss =  covarince_loss(z1, z2)
                    sim_loss =  invariance_loss(z1, z2)
                    loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)
                    epoch_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
            else:
                for cur_data in loader:
                    x, x1, x2, _ = cur_data
                    x, x1, x2 = x.to(device), x1.to(device), x2.to(device)
                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    f2_old = torch.Tensor([]).to(device)
                    replay_batchsize = args.replay_bs
                    indices = np.random.randint(0,x_old.shape[0], replay_batchsize)
                    x_old_bs = x_old[indices]
                    for ind in indices:
                        x1_old = torch.cat((x1_old, transform(x_old[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(x_old[ind:ind+1])), dim=0)
                    x1_old, x2_old = x1_old.to(device), x2_old.to(device)

                    curr_task_size = x2.shape[0]
                    if curr_task_size < args.replay_bs:
                        old_task_size = curr_task_size
                    else:
                        old_task_size = args.replay_bs
                    lam = np.random.beta(args.alpha, args.alpha)
                    # lam = 0.7 #np.random.beta(args.alpha, args.alpha)
                    mix_x1 = lam * x1[:old_task_size] + (1 - lam) * x1_old[:old_task_size]
                    mix_x2 = lam * x2[:old_task_size] + (1 - lam) * x2_old[:old_task_size]

                    x1_hat = torch.cat((x1, mix_x1))
                    x2_hat = torch.cat((x2, mix_x2))
                    z1, z2 = model(x1_hat, x2_hat)
                    z1 = F.normalize(z1, p=2)
                    z2 = F.normalize(z2, p=2)

                    x1_ = torch.cat((x1, x1_old[:old_task_size]))
                    x2_ = torch.cat((x2, x2_old[:old_task_size]))
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()

                    f1Old = F.normalize(f1Old, p=2)
                    f2Old = F.normalize(f2Old, p=2)

                    cov_loss =  covarince_loss(z1[:curr_task_size], z2[:curr_task_size])
                    sim_loss =  invariance_loss(z1[:curr_task_size], z2[:curr_task_size])
                    org_loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)

                    # org_loss = cross_loss(z1[:curr_task_size], z2[:curr_task_size])
                    cov_ood_loss = 0.25 * (lam * old_covarince_loss(z1[curr_task_size:], z1[:old_task_size]) + (1-lam) * old_covarince_loss(z1[curr_task_size:], f1Old[:old_task_size]))+ \
                        0.25 * (lam * old_covarince_loss(z2[curr_task_size:], z2[:old_task_size]) + (1-lam) * old_covarince_loss(z2[curr_task_size:], f2Old[:old_task_size]))
                    sim_ood_loss = 0.25 * (lam * invariance_loss(z1[curr_task_size:], z1[:old_task_size]) + (1-lam) * invariance_loss(z1[curr_task_size:], f1Old[:old_task_size]))+ \
                        0.25 * (lam * invariance_loss(z2[curr_task_size:], z2[:old_task_size]) + (1-lam) * invariance_loss(z2[curr_task_size:], f2Old[:old_task_size]))   
                    ood_loss =   args.sim_loss_weight* sim_ood_loss +cov_ood_loss *  args.cov_loss_weight 

                    loss = org_loss + ood_loss
                    lossKD = 0.0
                    p2_1 = model.temporal_projector(z1[:curr_task_size])
                    p2_2 = model.temporal_projector(z2[:curr_task_size])
                    p2_1 = F.normalize(p2_1, p=2)
                    p2_2 = F.normalize(p2_2, p=2)

                    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old[:curr_task_size]).mean() * 0.5
                                            + cross_loss(p2_2, f2Old[:curr_task_size]).mean() * 0.5) )
                    loss += lossKD 
                    epoch_loss.append(org_loss.item())
                    distil_loss.append(ood_loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel.to(device)
        oldModel.train()

        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        #torch.save({
        #                'state_dict': model.state_dict(),
        #                'optimizer' : optimizer.state_dict(),
        #                'encoder': model.encoder.backbone.state_dict(),
        #            }, file_name)
        
        if task_id < len(train_data_loaders)-1:
            lin_epoch = 1
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  

        x_samp, y_samp = store_samples(loader, task_id, args.msize)
        x_samp, y_samp = x_samp.to(device), y_samp.to(device)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)
    return model, loss_, optimizer



