import time
import wandb
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
import random
import torchvision

import torch.nn as nn
from tqdm import tqdm


from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset


from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from copy import deepcopy
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from utils.lars import LARS
import torchvision.transforms as transforms
import torchvision.transforms as T
#https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/predictive_mse.py

class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        return x
    
def correct_top_k(outputs, targets, top_k=(1,5)):
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result
    
def finetune_contrastive_first_task(net, contrastive_projector, data_loader, optimizer, epochs,  device):
    for epoch in range(1, epochs+1):
        net.eval() # for not update batchnorm 
        total_num, train_bar = 0, tqdm(data_loader)
        linear_loss = 0.0
        for data_tuple in train_bar:
            # Forward prop of the model with single augmented batch
            pos_1, targets = data_tuple
            pos_1 = pos_1[0].to(device)
            features = net(pos_1)

            # Batchsize
            batchsize_bc = features.shape[0]
            targets = torch.ones(targets.shape[0],dtype=torch.long).to(device) * 0 
            targets = targets.to(device)
            
            c_weights = torch.nn.functional.normalize(contrastive_projector.weight,dim=1)
            logits = features.detach() @ c_weights.T

            # Cross Entropy Loss 
            linear_loss = F.cross_entropy(logits, targets)

            # Backpropagation part
            optimizer.zero_grad()
            linear_loss.backward()
            optimizer.step()

            # Accumulating number of examples, losses and correct predictions
            total_num += batchsize_bc
            linear_loss += linear_loss.item() * batchsize_bc
            train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f}'.format(epoch, linear_loss / total_num))

def finetune_contrastive_v1(net, data_loader, task_id, optimizer, epochs, device):
    for epoch in range(1, epochs+1):
        net.eval() # for not update batchnorm 
        total_num, train_bar = 0, tqdm(data_loader)
        linear_loss = 0.0
        for data_tuple in train_bar:
            # Forward prop of the model with single augmented batch
            pos_1, targets = data_tuple
            pos_1 = pos_1[0].to(device)
            features = net(pos_1)
            
            #logits = net.contrastive_projector(features) 
            
            c_weights = torch.nn.functional.normalize(net.contrastive_projector.weight,dim=1)
            logits = features @ c_weights.T

            # Batchsize
            batchsize_bc = features.shape[0]
            targets = torch.ones(targets.shape[0],dtype=torch.long).to(device) * task_id 
            targets = targets.to(device)
            
            # Cross Entropy Loss 
            linear_loss = F.cross_entropy(logits, targets)

            # Backpropagation part
            optimizer.zero_grad()
            linear_loss.backward()
            net.contrastive_projector.weight.grad[0:task_id] = torch.zeros(net.contrastive_projector.weight.grad[0:task_id].shape).to(device)
            optimizer.step()

            # Accumulating number of examples, losses and correct predictions
            total_num += batchsize_bc
            linear_loss += linear_loss.item() * batchsize_bc

            train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f}'.format(epoch, linear_loss / total_num))
        wandb.log({f" Contrastive loss {task_id}": linear_loss / total_num, " Epoch ": epoch})
    return linear_loss/total_num

def train_cassle_contrastive_v1_barlow(model, train_data_loaders_generic, knn_train_data_loaders, test_data_loaders, transform, transform_prime, device, args):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)

    contrastive_projector = nn.Linear(512, len(train_data_loaders_generic), bias=False).to(device)

    #will be changed later!!
    data_normalize_mean = (0.5071, 0.4865, 0.4409)
    data_normalize_std = (0.2673, 0.2564, 0.2762)
    random_crop_size = 32

    transform_linear = transforms.Compose( [
                transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ] )

    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    for task_id, loader in enumerate(train_data_loaders_generic):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        loader.dataset.transforms = [transform, transform_prime]
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_task = []
            epoch_loss_kd = []
            for x, y in loader:
                x1, x2 = x  
                x1, x2 = x1.to(device), x2.to(device)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()
                    p2_1 = model.temporal_projector(z1)
                    p2_2 = model.temporal_projector(z2)
                    
                    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
                                           + cross_loss(p2_2, f2Old).mean() * 0.5) )
                    loss += lossKD 
                else:
                    lossKD = torch.tensor(0)
                
                epoch_loss_task.append(loss.item())
                epoch_loss_kd.append(lossKD.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss_task))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task),  " Average KD Loss ": np.mean(epoch_loss_kd)  , " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id == 0:
            loader.dataset.transforms = [transform_linear]
            lin_optimizer = torch.optim.SGD(contrastive_projector.parameters(), 1e-3, momentum=0.9, weight_decay=0) 
            finetune_contrastive_first_task(model, contrastive_projector, loader, lin_optimizer, 5,  device)
        else:
            loader.dataset.transforms = [transform_linear]
            model.contrastive_projector = contrastive_projector 
            lin_optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=0)
            finetune_contrastive_v1(model, loader, task_id, lin_optimizer, 10, device)

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer


def collect_activations(model, loader, num, task_id, device):
    model.eval()
    outs = []
    for x,_ in loader:
        x = x[0].to(device)
        out = model(x).cpu().detach().numpy()
        outs.append(out)

    outs = np.concatenate(outs)
    
    select = np.random.randint(0,outs.shape[0],num)
    labels = torch.ones(num,dtype=torch.long) * task_id 
    return torch.tensor(outs[select]), labels

def finetune_contrastive_v2(net, data_loader, old_representations, old_labels, task_id, optimizer, epochs, device):
    for epoch in range(1, epochs+1):
        net.eval() # for not update batchnorm 
        total_num, train_bar = 0, tqdm(data_loader)
        linear_loss = 0.0
        for x,_ in train_bar:
            # Forward prop of the model with single augmented batch
            x = x[0].to(device)
            features = net(x)

            select = np.random.randint(0,old_representations.shape[0],100)
            all_features = torch.cat((old_representations[select].to(device), features),dim=0)
            logits = net.contrastive_projector(all_features) 
            
            # Batchsize
            batchsize_bc = all_features.shape[0]
            targets = torch.ones(features.shape[0],dtype=torch.long).to(device) * task_id 
            targets = targets.to(device)

            all_targets = torch.cat((old_labels[select].to(device),targets))
            
            # Cross Entropy Loss 
            linear_loss = F.cross_entropy(logits, all_targets)

            # Backpropagation part
            optimizer.zero_grad()
            linear_loss.backward()
            #net.contrastive_projector.weight.grad[0:task_id] = torch.zeros(net.contrastive_projector.weight.grad[0:task_id].shape).to(device)
            optimizer.step()

            # Accumulating number of examples, losses and correct predictions
            total_num += batchsize_bc
            linear_loss += linear_loss.item() * batchsize_bc

            train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f}'.format(epoch, linear_loss / total_num))
        wandb.log({f" Contrastive loss {task_id}": linear_loss / total_num, " Epoch ": epoch})
    return linear_loss/total_num


def train_cassle_contrastive_v2_barlow(model, train_data_loaders_generic, knn_train_data_loaders, test_data_loaders, transform, transform_prime, device, args):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)

    contrastive_projector = nn.Linear(512, len(train_data_loaders_generic), bias=False).to(device)

    #will be changed later!!
    data_normalize_mean = (0.5071, 0.4865, 0.4409)
    data_normalize_std = (0.2673, 0.2564, 0.2762)
    random_crop_size = 32

    transform_linear = transforms.Compose( [
                transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ] )

    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([])
    y_old = torch.tensor([],dtype=torch.long)

    for task_id, loader in enumerate(train_data_loaders_generic):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        loader.dataset.transforms = [transform, transform_prime]
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_task = []
            epoch_loss_kd = []
            for x, y in loader:
                x1, x2 = x  
                x1, x2 = x1.to(device), x2.to(device)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()
                    p2_1 = model.temporal_projector(z1)
                    p2_2 = model.temporal_projector(z2)
                    
                    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
                                           + cross_loss(p2_2, f2Old).mean() * 0.5) )
                    loss += lossKD 
                else:
                    lossKD = torch.tensor(0)
                
                epoch_loss_task.append(loss.item())
                epoch_loss_kd.append(lossKD.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss_task))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task),  " Average KD Loss ": np.mean(epoch_loss_kd)  , " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id != 0:
            loader.dataset.transforms = [transform_linear]
            model.contrastive_projector = contrastive_projector 
            lin_optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=0) 
            finetune_contrastive_v2(model, loader, x_old, y_old, task_id, lin_optimizer, 10, device)
 
        loader.dataset.transforms = [transform_linear]
        x_rep, y_rep = collect_activations(model, loader, 50, task_id, device)
        x_old = torch.cat((x_old, x_rep), dim=0)
        y_old = torch.cat((y_old, y_rep), dim=0)

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer

def store_samples(loader, task_id, num):
    x_data = loader.dataset.train_data
    select = np.random.randint(0,x_data.shape[0],num)
    labels = torch.ones(num,dtype=torch.long) * task_id 
    return torch.Tensor(x_data[select]), labels

def finetune_contrastive_v3(net, data_loader, old_samples, old_labels, new_batch_size, task_id, optimizer, epochs, device):
    
    data_normalize_mean = (0.5071, 0.4865, 0.4409)
    data_normalize_std = (0.2673, 0.2564, 0.2762)
    random_crop_size = 32
    transform_linear = transforms.Compose( [
            transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(data_normalize_mean, data_normalize_std),
        ] )

    old_data_loader = DataLoader(TensorDataset(old_samples,old_labels,transform=transform_linear), batch_size=new_batch_size, shuffle=True, 
                            num_workers = 5, pin_memory=True)
    current_data_loader = DataLoader(data_loader.dataset, batch_size=new_batch_size, shuffle=True, num_workers = 5, pin_memory=True)
    
    for epoch in range(1, epochs+1):
        net.eval() # for not update batchnorm 
        total_num, train_bar = 0, tqdm(current_data_loader)
        linear_loss = 0.0

        dataloader_iterator = iter(old_data_loader)
        for x1, _ in train_bar:
            try:
                x2, y2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(old_data_loader)
                x2, y2 = next(dataloader_iterator)

            x1 = x1[0]
            y1 = torch.ones(x1.shape[0],dtype=torch.long) * task_id
            x_all = torch.cat((x1, x2), dim=0)
            y_all = torch.cat((y1, y2), dim=0)
            x_all = x_all.to(device)
            y_all = y_all.to(device)

            # Forward prop of the model with single augmented batch
            features = net(x_all)
            logits = net.contrastive_projector(features) 
            
            #c_weights = torch.nn.functional.normalize(net.contrastive_projector.weight,dim=1)
            #logits = features @ c_weights.T
            
            # Cross Entropy Loss 
            linear_loss = F.cross_entropy(logits, y_all)

            # Backpropagation part
            optimizer.zero_grad()
            linear_loss.backward()
            # net.contrastive_projector.weight.grad[0:task_id] = torch.zeros(net.contrastive_projector.weight.grad[0:task_id].shape).to(device)
            optimizer.step()

            # Accumulating number of examples, losses and correct predictions
            batchsize_bc = features.shape[0]
            total_num += batchsize_bc
            linear_loss += linear_loss.item() * batchsize_bc

            train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f}'.format(epoch, linear_loss / total_num))
        wandb.log({f" Contrastive loss {task_id}": linear_loss / total_num, " Epoch ": epoch})
    return linear_loss/total_num

def train_cassle_contrastive_v3_barlow(model, train_data_loaders_generic, knn_train_data_loaders, test_data_loaders, transform, transform_prime, device, args):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)

    contrastive_projector = nn.Linear(512, len(train_data_loaders_generic), bias=False).to(device)
     

    #will be changed later!!
    data_normalize_mean = (0.5071, 0.4865, 0.4409)
    data_normalize_std = (0.2673, 0.2564, 0.2762)
    random_crop_size = 32

    transform_linear = transforms.Compose( [
                transforms.RandomResizedCrop(random_crop_size,  interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ] )

    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    x_old = torch.Tensor([])
    y_old = torch.tensor([],dtype=torch.long)

    for task_id, loader in enumerate(train_data_loaders_generic):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        loader.dataset.transforms = [transform, transform_prime]
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_task = []
            epoch_loss_kd = []
            for x, y in loader:
                x1, x2 = x  
                x1, x2 = x1.to(device), x2.to(device)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()
                    p2_1 = model.temporal_projector(z1)
                    p2_2 = model.temporal_projector(z2)
                    
                    lossKD = args.lambdap * ((cross_loss(p2_1, f1Old).mean() * 0.5
                                           + cross_loss(p2_2, f2Old).mean() * 0.5) )
                    loss += lossKD 
                else:
                    lossKD = torch.tensor(0)
                
                epoch_loss_task.append(loss.item())
                epoch_loss_kd.append(lossKD.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss_task))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f}  | KDLoss: {np.mean(epoch_loss_kd):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task),  " Average KD Loss ": np.mean(epoch_loss_kd)  , " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        if task_id != 0:
            loader.dataset.transforms = [transform_linear]
            model.contrastive_projector = contrastive_projector 
            lin_optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=0) 
            finetune_contrastive_v3(model, loader, x_old, y_old, 64, task_id, lin_optimizer, 10, device)
 
        loader.dataset.transforms = None
        x_samp, y_samp = store_samples(loader, task_id, 50)
        x_old = torch.cat((x_old, x_samp), dim=0)
        y_old = torch.cat((y_old, y_samp), dim=0)

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer





