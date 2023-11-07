import time
import wandb
import torch
import os
import numpy as np
from copy import deepcopy
import math
import torch.nn.functional as F
from models.gaussian_diffusion.basic_unet import EMA
import torchvision.transforms as T
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataloaders.dataset import SimSiam_Dataset, TensorDataset, GenericDataset, Diffusion_Dataset, Unlabeled_Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms as T, utils
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from copy import deepcopy
from torch.optim import Adam, AdamW
from models.gaussian_diffusion.basic_unet import UNet_conditional
from models.gaussian_diffusion.openai_unet import UNetModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet2DModel
from models.gaussian_diffusion.eval_utils.fid_evaluation import FIDEvaluation
from sklearn.cluster import KMeans



from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from utils.lars import LARS
#https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/predictive_mse.py

from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset
from tqdm import tqdm
# import tqdm
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

def train_simsiam(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)
    old_model = None

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10
            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:
                x1, x2 = x1.to(device), x2.to(device)

                f1 = model.encoder.backbone(x1).squeeze() # NxC
                f2 = model.encoder.backbone(x2).squeeze() # NxC
                z1 = model.encoder.projector(f1) # NxC
                z2 = model.encoder.projector(f2) # NxC

                p1 = model.predictor(z1) # NxC
                p2 = model.predictor(z2) # NxC   

                loss_one = loss_fn(p1, z2.detach())
                loss_two = loss_fn(p2, z1.detach())
                loss = 0.5*loss_one + 0.5*loss_two
                loss = loss.mean()

                epoch_loss.append(loss.item())
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
            
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        #torch.save({
        #                'state_dict': model.state_dict(),
        #                'optimizer' : optimizer.state_dict(),
        #                'encoder': model.encoder.backbone.state_dict(),
        #            }, file_name)
        
        if task_id < len(train_data_loaders)-1:
            lin_epoch = 100
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  



    return model, loss_, optimizer




def train_infomax(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args):
    
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
        if args.info_loss == 'error_cov':
            err_covarince_loss = ErrorCovarianceLoss(args.proj_out ,device=device)

        loss_ = []
        old_covarince_loss = CovarianceLoss(args.proj_out, device=device)
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1_cur,z2_cur = model(x1, x2)

                z1 = F.normalize(z1_cur, p=2)
                z2 = F.normalize(z2_cur, p=2)

                cov_loss =  covarince_loss(z1, z2)
                sim_loss =  invariance_loss(z1, z2)
                loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)

                epoch_loss.append(loss.item())
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
            

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        #torch.save({
        #                'state_dict': model.state_dict(),
        #                'optimizer' : optimizer.state_dict(),
        #                'encoder': model.encoder.backbone.state_dict(),
        #            }, file_name)
        
        if task_id < len(train_data_loaders)-1:
            lin_epoch = 100
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  

    return model, loss_, optimizer


def train_barlow_diffusion(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args, diffusion_model, noise_scheduler, train_data_loaders_diffusion, test_data_loaders_diffusion, transform, transform_prime, diffusion_tr, transform_knn):
    
    epoch_counter = 0
    model.temporal_projector = nn.Sequential(
            nn.Linear(args.proj_out, args.proj_hidden, bias=False),
            nn.BatchNorm1d(args.proj_hidden),
            nn.ReLU(),
            nn.Linear(args.proj_hidden, args.proj_out),
        ).to(device)
    old_model = None
    old_diffusion_model = None
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)
    old_difftrain_dataloader = None
    memory_x, memory_y, memory_cid = None, None, None

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:
                x1, x2 = x1.to(device), x2.to(device)
                if task_id > 0:
                    x1_old = torch.Tensor([])
                    x2_old = torch.Tensor([])
                    x1_old, x2_old = x1_old.to(device), x2_old.to(device)
                    indices = np.random.choice(len(memory_x), size=min(args.diff_train_bs, len(memory_x)), replace=False)
                    for ind in indices:
                        x1_old = torch.cat((x1_old, transform(memory_x[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory_x[ind:ind+1])), dim=0)
                    x1 = torch.cat([x1, x1_old], dim=0)
                    x2 = torch.cat([x2, x2_old], dim=0)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)
                epoch_loss.append(loss.item())
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

            if task_id > 0 and (epoch+1) % args.image_report_freq == 0:
                # diff_train_dl, ema_model, task_id, epoch
                print_images_to_wandb(args, memory_x, old_diffusion_model, task_id, epoch, text= "SSL Training")
            if args.is_debug:
                break

            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)
    
        if task_id < len(train_data_loaders)-1:
            if args.is_debug is not True:
                lin_epoch = 100
            else:
                lin_epoch = 1
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


        #send model initialization, update dataloader, send the previous model to sample new images for training, train the diffusion model 
        kmeans_clustering = None
        if task_id < len(train_data_loaders)-1: # do not calculate for last task
            train_dl_diff, test_dl_diff = train_data_loaders_diffusion[task_id], test_data_loaders_diffusion[task_id] #use original dataloaders for class label conditioning
            if args.clustering_label:
                n_cluster = 20*(task_id+1)
                kmeans_clustering = KNNmeans_cluster_training(n_cluster=n_cluster, encoder=model.encoder, current_knn_dataloader=knn_train_data_loaders[task_id], memory=memory_x, task_id=task_id, transform_knn=transform_knn, device=device)
                xtrain = torch.Tensor([])
                for xx1, xx2, y in train_data_loaders_diffusion[task_id]:
                    xtrain = torch.cat([xx2, xtrain], dim=0)

                xvector, yvector_cid = get_cluster_ids_of_unlabeled_exemplar_set(xtrain, model.encoder, kmeans_clustering, n_cluster, transform_knn, args, device)
                train_dl_diff = get_dataloader(xvector=xvector, yvector=yvector_cid, batchsize=args.diff_train_bs, transform=diffusion_tr, transform_prime=None, num_workers=4, )
 
            diffusion_model_weights = train_diffusion_model(diffusion_model, noise_scheduler, args, train_dl_diff, test_dl_diff, task_id, device, diffusion_tr, memory_x, memory_y, memory_cid ,200*(task_id+1), kmeans_clustering, model.encoder, transform_knn)
            diffusion_model.load_state_dict(diffusion_model_weights)
            old_diffusion_model = deepcopy(diffusion_model).requires_grad_(False)
            print('Sample Samples for next task')
            memory_x, memory_y = get_exemplar_set_by_sampling(diffusion_model, args, device, noise_scheduler, task_id, state='train', iteration_=4, n_cluster=10)
            if args.clustering_label:
                memory_x, memory_cid = get_cluster_ids_of_unlabeled_exemplar_set(xvector, model.encoder, kmeans_clustering, n_cluster, transform_knn, args, device)

    return model, loss_, optimizer


def train_diffusion_model(model, noise_scheduler, args, train_dataloader, test_dataloader, task_id, device, diffusion_tr, memory_x, memory_y, memory_cid, n_clusters, kmeans_clustering, encoder, knn_transform):
    ema_model = deepcopy(model).eval().requires_grad_(False)
    ema = EMA(0.995)
    model.to(device)
    ema_model.to(device)
    diffoptimizer = AdamW(model.parameters(), lr=args.diff_train_lr,  weight_decay=args.diff_weight_decay)
    diff_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=diffoptimizer,
                        num_warmup_steps=50,
                        num_training_steps=(len(train_dataloader) * args.diff_epochs[task_id]),)
    scaler = torch.cuda.amp.GradScaler()
    mse = nn.MSELoss()

    for epoch in range(args.diff_epochs[task_id]):
        loss_ = []
        model.train()
        start = time.time()
        for images, images2, labels in train_dataloader:
            diffoptimizer.zero_grad()
            images = images.to(device)
            if args.class_condition or args.clustering_label:
                labels = labels.long().to(device)
            else:
                labels = None
            if task_id > 0: # sample images from the old model
                x_old = torch.Tensor([])
                y_old = torch.Tensor([])
                indices = np.random.choice(len(memory_x), size=min(args.diff_train_bs, len(memory)), replace=False)
                for ind in indices:
                    x_old = torch.cat((x_old, diff_transform(memory_x[ind:ind+1])), dim=0)
                    if args.class_condition:
                        y_old = torch.cat((y_old, memory_y[ind:ind+1]), dim=0)
                    if args.clustering_label:
                        y_old = torch.cat((y_old, memory_cid[ind:ind+1]), dim=0)
                    x_old = x_old.to(device)
                if labels is not None:
                    y_old = y_old.to(device)
                    labels = torch.cat((labels, y_old), dim = 0)
                images = torch.cat([images, x1_], dim = 0)

            timesteps = torch.randint(low=1, high=args.num_train_timesteps, size=(images.shape[0],)).to(device)
            noise = torch.randn_like(images)
            x_t = noise_scheduler.add_noise(images, noise, timesteps)

            if args.unet_model == 'diffusers':
                predicted_noise = model(x_t, timesteps, labels, return_dict=False)[0]
            else:
                predicted_noise = model(x_t, timesteps, labels)
            loss = mse(noise, predicted_noise)
            scaler.scale(loss).backward()
            scaler.step(diffoptimizer)
            scaler.update()
            ema.step_ema(ema_model, model)
            diff_scheduler.step()
            loss_.append(loss.item())
            
            if args.is_debug:
                break
        end = time.time()
        print('Diffusion epoch end')
        
        # break
        if (epoch+1) % args.image_report_freq == 0:
            print('Sample Samples for Diffusion Model FID Evaluation')
            if args.clustering_label:
                total_labels = n_clusters
            elif args.class_condition:
                total_labels = np.sum(args.class_split[:task_id+1])
            else:
                total_labels = None
            print(total_labels)
            fid_x, fid_y = get_exemplar_set_by_sampling(model, args, device, noise_scheduler, task_id, state='fid', iteration_=len(test_dataloader), n_cluster=total_labels)
            fid_dl = get_dataloader(xvector=fid_x, yvector=None, batchsize=args.diff_train_bs, transform=diffusion_tr, transform_prime=None, num_workers=4, )
            results_folder = './results/'+str(args.class_split)+'task_id'+str(task_id)+'e'+str(args.epochs)+'de'+str(args.diff_epochs)
            if not os.path.isdir(results_folder):
                os.makedirs(results_folder)
            print('Step 2: FID Score Eval')
            fid_score = FID_evaluate(args, ema_model, fid_dl, test_dataloader, results_folder, device, epoch)
            print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(loss_):.4f}  | FID:  {fid_score*100:.2f}')
            print_images_to_wandb(args, fid_dl, ema_model, task_id, epoch, text= "Diffusion Training")
        else:
            print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(loss_):.4f} ')
        wandb.log({"train_mse": np.mean(loss_), "learning_rate": diff_scheduler.get_last_lr()[0]})        
        wandb.log({"train_mse": np.mean(loss_),"epoch": epoch})
    return ema_model.state_dict()


def sample_images(model, n, args, device, noise_scheduler, task_id, cluster_centers):
    model.eval()
    if args.clustering_label or args.class_condition:
        if n > cluster_centers: #add logic of classes seen so far
            num = int(n/cluster_centers)
            labels_ = torch.tensor([[i]*num for i in range(cluster_centers)]).flatten()
        else:
            labels_ = torch.tensor([[i]*1 for i in range(cluster_centers)]).flatten()
        labels_ = labels_.to(device)
        labels = labels_.flatten().to(device) #sample num Images per class
        sampled_bs = len(labels_)
    else:
        labels_ = None
        sampled_bs = args.sample_bs
        labels = None

    with torch.inference_mode():
        x = torch.randn(sampled_bs, 3, args.image_size, args.image_size).to(device)
        for i, t in enumerate(tqdm(noise_scheduler.timesteps)): #timesteps in reverse order
            with torch.no_grad():
                if args.unet_model == 'diffusers':
                    predicted_noise = model(x, t, labels).sample
                else:
                    ts = (torch.ones(sampled_bs) * t).long().to(device)
                    predicted_noise = model(x, ts, labels)
            x = noise_scheduler.step(predicted_noise, t, x).prev_sample
            if args.is_debug:
                break
        x = (x.clamp(-1, 1) + 1) / 2 #unnormalize
    return x, labels_


def get_exemplar_set_by_sampling(model, args, device, noise_scheduler, task_id, state='train', iteration_=4, n_cluster=10,):
    if state == 'fid':
        iteration =  iteration_ - 1
    else:
        iteration = args.msize/args.diff_train_bs
    if args.is_debug:
        iteration = 4
    xvector = torch.Tensor([]).to(device)
    yvector = torch.Tensor([]).to(device)
    for i in range(int(iteration)):
        print(' Sampling Iteration Number: '+str(i+1)+', Total Iterations: '+str(int(iteration)))
        sampled_images, labels = sample_images(model, args.diff_train_bs, args, device, noise_scheduler, task_id, n_cluster)
        xvector = torch.cat((sampled_images, xvector), dim=0)
        if labels is not None:
            yvector = torch.cat((labels, yvector), dim=0)
        if i ==2 and args.is_debug:
            break

    return xvector, yvector



def get_cluster_ids_of_unlabeled_exemplar_set(xvector, encoder, kmeans_clustering, n_clusters, transform_knn, args, device):
    termination_condition = math.ceil(xvector.shape[0]/args.pretrain_batch_size)-1
    xset = torch.Tensor([]).to(device)
    yset_ci = torch.Tensor([]).to(device)
    for i in range(termination_condition):
        if i == termination_condition-1:
            x = xvector[i*args.pretrain_batch_size:]
        else:
            x = xvector[i*args.pretrain_batch_size:(i+1)*args.pretrain_batch_size]
        x = x.to(device)
        x_ = torch.Tensor([]).to(device)
        for ind in range(x.shape[0]):
            x_ = torch.cat((x_, transform_knn(x[ind:ind+1])), dim=0)
        # images = knn_transform(x_)
        features = encoder.backbone(x_).squeeze().detach().cpu().numpy()
        cluster_ids = torch.Tensor(kmeans_clustering.predict(features)).to(device)
        xset = torch.cat((xset, x), dim = 0)
        yset_ci = torch.cat((yset_ci, cluster_ids), dim = 0)
    return xset, yset_ci



def get_dataloader(xvector, yvector=None, batchsize=64, transform=None, transform_prime=None, num_workers=4, ):
    if yvector is None:
        dataset = Unlabeled_Dataset(xvector.to('cpu'), transform)
    else:
        dataset = SimSiam_Dataset(xvector.to('cpu'), yvector.to('cpu'), transform, transform_prime)
    dl = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers = num_workers, pin_memory=True)
    return dl
    

def print_images_to_wandb(args, diff_train_dl, ema_model, task_id, epoch, text= "Diffusion Training"):
    if text == "Diffusion Training":
        for images2 in diff_train_dl:
            # utils.save_image(images2,  str('./results/' +str(text)+'task_id'+str(task_id)+'epoch'+str(epoch)+'emasample-.png'), nrow = int(math.sqrt(images2.shape[0])))
            # images2 = images2.type(torch.uint8) # To plot
            wandb.log({" sampled_images (ema model) "+str(text):     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in images2]})
            break
    else:
        indices = np.random.choice(len(diff_train_dl), size=min(args.diff_train_bs, len(diff_train_dl)), replace=False)
        images2 = diff_train_dl[indices]
            # utils.save_image(images2,  str('./results/' +str(text)+'task_id'+str(task_id)+'epoch'+str(epoch)+'emasample-.png'), nrow = int(math.sqrt(images2.shape[0])))
            # images2 = images2.type(torch.uint8) # To plot
        wandb.log({" sampled_images (ema model) "+str(text):     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in images2]})
        # break

def FID_evaluate(args, ema_model, train_dl, test_dl, results_folder, device, epoch):
    if args.is_debug:
        num_fid_samples=args.diff_train_bs
    else:
        num_fid_samples=(len(test_dl)-1)*args.diff_train_bs
    fid_scorer = FIDEvaluation(
    batch_size=args.sample_bs,
    dl=test_dl, #TODO: Train or eval dataloader?
    sampler=ema_model,
    channels=3,
    accelerator=None,
    stats_dir=results_folder,
    device=device,
    num_fid_samples=num_fid_samples, #for drop last false case
    inception_block_idx=2048,
    args=args)

    fid_score = fid_scorer.fid_score(train_dl, num_fid_samples)
    wandb.log({"FID Score ": fid_score, "epoch ": epoch})
    return fid_score


def KNNmeans_cluster_training(n_cluster=20, encoder=None, current_knn_dataloader=None, memory=None, task_id=0, transform_knn=None, device='cpu' ):
    ind = 0
    encoder.to(device)
    encoder.eval()
    train_features = torch.Tensor([]).to(device)

    # Step 1: Concatenate current task + memory data to construct cluster IDs
    for x, y in current_knn_dataloader:
        x = x.to(device)
        features = encoder.backbone(x)
        train_features = torch.cat((train_features, features), dim=0)
        if ind == 10:
            break
        ind += 1

    if  memory is not None: 
        for i in range(10):  
            x = torch.Tensor([])   
            indices = np.random.choice(len(memory), size=min(args.diff_train_bs, len(memory)), replace=False) 
            for ind in indices:
                x = torch.cat((x, transform_knn(memory[ind:ind+1])), dim=0)
            x = x.to(device)
            features = encoder.backbone(x)
            features = nn.functional.normalize(features.squeeze()).detach().cpu().numpy()
            train_features = torch.cat((train_features, features), dim=0)
    # print(train_features.shape)
    # print(int(n_cluster))
    kmeans = KMeans(n_clusters=int(n_cluster), random_state=0, n_init=10).fit(train_features.squeeze().detach().cpu().numpy())
    return kmeans

def get_cluster_ids_of_labeled_exemplar_set(xvector, yvector, encoder, kmeans_clustering, n_clusters, knn_transform, ): #if needed by any chance
    termination_condition = math.ceil(xvector.shape[0]/args.pretrain_batch_size)-1
    for i in range(termination_condition):
        if i == termination_condition-1:
            x = xvector[i*args.pretrain_batch_size:]
            labels = yvector[i*args.pretrain_batch_size:]
        else:
            x = xvector[i*args.pretrain_batch_size:(i+1)*args.pretrain_batch_size]
            labels = yvector[i*args.pretrain_batch_size:(i+1)*args.pretrain_batch_size]
        x = x.to(device)
        labels = labels.to(device)
        images = knn_transform(x)
        features = encoder.backbone(images).squeeze().detach().cpu().numpy()
        cluster_ids = kmeans_clustering.predict(features)
        if i == 0:
            xset = images
            yset_ci = cluster_ids
            yset = labels            
        else:
            xset = torch.cat((xset, x2), dim = 0)
            yset_ci = torch.cat((yset_ci, cluster_ids), dim = 0)
            yset = torch.cat((yset, labels), dim = 0)
    return xset, yset, yset_ci

