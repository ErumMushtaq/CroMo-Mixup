import time
import wandb
import torch
import numpy as np
from copy import deepcopy
import math
import torch.nn.functional as F
from models.gaussian_diffusion.basic_unet import EMA
import torchvision.transforms as T
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataloaders.dataset import SimSiam_Dataset, TensorDataset, GenericDataset, Diffusion_Dataset
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
                    #use diffusiondata loader to get sampled images
                    dataloader_iterator = iter(old_train_dataloader)
                    # for cur_data in dataloader_iterator:
                    try:
                        x1_, x2_, _ = next(dataloader_iterator)
                        x1_, x2_ = x1_.to(device), x2_.to(device)
                    except StopIteration:
                        dataloader_iterator = iter(old_data_loader)
                        x1_, x2_, _  = next(dataloader_iterator)
                        x1_, x2_ = x1_.to(device), x2_.to(device)

                    x1 = torch.cat((x1, x1_), dim = 0)
                    x2 = torch.cat((x2, x2_), dim = 0)
                    # print(x1.shape)
                # x1, x2 = x1.to(device), x2.to(device)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)
                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                if args.is_debug:
                    break
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
                print_images(old_difftrain_dataloader, old_diffusion_model, task_id, epoch, text= "SSL Training")

            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
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

        #train a diffusion model
        #send model initialization, update dataloader, send the previous model to sample new images for training, train the diffusion model 
        kmeans_clustering = None
        if task_id < len(train_data_loaders)-1: # do not calculate for last task
            train_dl_diff, test_dl_diff = train_data_loaders_diffusion[task_id], test_data_loaders_diffusion[task_id] #use original dataloaders for class label conditioning
            if args.clustering_label:  #if cluster id labeling, then obtain a predictor, predict labels for the diffusion dataloader
                n_cluster = 200*(task_id+1)
                kmeans_clustering = KNNmeans_clustering(n_cluster=n_cluster, encoder=model.encoder, current_knn_dataloader=knn_train_data_loaders[task_id], old_diffusion_dataloader=old_difftrain_dataloader, task_id=task_id, transform_knn=transform_knn, device=device)
                train_dl_diff = get_clustering_dataloader(args, kmeans_clustering, train_data_loaders_diffusion[task_id], model.encoder,  transform_knn, diffusion_tr, device)

            diffusion_model_weights = train_diffusion_model(diffusion_model, old_diffusion_model, noise_scheduler, args, train_dl_diff, test_dl_diff, task_id, device, diffusion_tr, old_difftrain_dataloader,transform, transform_prime,200*(task_id+1), kmeans_clustering, model.encoder,transform_knn)
            diffusion_model.load_state_dict(diffusion_model_weights)
            old_diffusion_model = deepcopy(diffusion_model).requires_grad_(False)
            print('Sample Samples for next task')
            old_train_dataloader, old_difftrain_dataloader = get_dataloaders(diffusion_model, args.sample_bs, args, device, noise_scheduler, task_id, transform, transform_prime,  diffusion_tr, 'train', len(test_dl_diff), 200*(task_id+1)) #state='train', iteration_=4,n_cluster=0
            if args.clustering_label: 
                old_difftrain_dataloader = get_clustering_dataloader(args, kmeans_clustering, old_difftrain_dataloader, model.encoder,  transform_knn, diffusion_tr, device)



    return model, loss_, optimizer

def train_diffusion_model(model, old_model, noise_scheduler, args, train_dataloader, test_dataloader, task_id, device, diffusion_tr, old_difftrain_dataloader,transform, transform_prime,n_clusters,kmeans_clustering, encoder, transform_knn):
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
            if args.class_condition:
                labels = labels.to(device)
            else:
                labels = None
            if task_id > 0: # sample images from the old model
                dataloader_iterator = iter(old_difftrain_dataloader)
                try:
                    x1_, x2_, labels_ = next(dataloader_iterator)
                    x1_, x2_,labels_= x1_.to(device), x2_.to(device),labels_.to(device)
                except StopIteration:
                    dataloader_iterator = iter(old_difftrain_dataloader)
                    x1_, x2_, labels_ = next(dataloader_iterator)
                    x1_, x2_,labels_= x1_.to(device), x2_.to(device),labels_.to(device)
                # sampled_images, labels_ = sample_images(old_model, args.sample_bs, args, device, noise_scheduler, task_id)
                # x1_ = diffusion_tr(sampled_images)
                if labels is not None:
                    labels = torch.cat((labels, labels_), dim = 0)
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
            # calculate FID
            # print("Step1: Generate Fake samples")
            # if args.class_condition:
            _, old_difftrain_dataloader = get_dataloaders(ema_model, args.diff_train_bs, args, device, noise_scheduler, task_id, transform, transform_prime,  diffusion_tr, state='fid', iteration_ = len(test_dataloader), n_cluster=n_clusters)
            if args.clustering_label:
                old_difftrain_dataloader = get_clustering_dataloader(args, kmeans_clustering, old_difftrain_dataloader, encoder,  transform_knn, diffusion_tr, device)

                # _, old_difftrain_dataloader = get_dataloaders(ema_model, args.diff_train_bs, args, device, noise_scheduler, task_id, transform, transform_prime,  diffusion_tr, state='fid', iteration_ = len(test_dataloader), n_cluster=n_clusters)
            results_folder = './results/'+str(args.class_split)+'task_id'+str(task_id)+'e'+str(args.epochs)+'de'+str(args.diff_epochs)
            results_folder = Path(results_folder)
            results_folder.mkdir(exist_ok = True)
            # print('Step 2: FID Score Eval')
            fid_score = FID_evaluate(args, ema_model, old_difftrain_dataloader, test_dataloader, results_folder, device, epoch)
            print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(loss_):.4f}  | FID:  {fid_score*100:.2f}')
            print_images(old_difftrain_dataloader, ema_model, task_id, epoch, text= "Diffusion Training")
        else:
            print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(loss_):.4f} ')
        
        wandb.log({"train_mse": np.mean(loss_),
                        "learning_rate": diff_scheduler.get_last_lr()[0]})
            
        wandb.log({"train_mse": np.mean(loss_),
                        "epoch": epoch})
        # print(f'Epoch {epoch:2d}  | Loss: {np.mean(loss_):.4f}')
    return ema_model.state_dict()
        # if (epoch+1) % self.args.knn_report_freq == 0:


def sample_images(model, n, args, device, noise_scheduler, task_id, cluster_centers):
    model.eval()
    if args.clustering_label:
        if n > cluster_centers: #add logic of classes seen so far
            num = int(n/cluster_centers)
            labels_ = torch.tensor([[i]*num for i in range(cluster_centers)]).flatten()
        else:
            labels_ = torch.tensor([[i]*1 for i in range(cluster_centers)]).flatten()
        labels = labels_.flatten().to(device) #sample num Images per class
    elif args.class_condition:
        if n > np.sum(args.class_split[:task_id+1]): #add logic of classes seen so far
            num = int(n/np.sum(args.class_split[:task_id+1]))
            # print(np.sum(args.class_split[:task_id+1]))
            labels_ = torch.tensor([[i]*num for i in range(np.sum(args.class_split[:task_id+1]))]).flatten()
        else:
            labels_ = torch.tensor([[i]*1 for i in range(np.sum(args.class_split[:task_id+1]))]).flatten()
        labels = labels_.flatten().to(device) #sample num Images per class
    else:
        if n > np.sum(args.class_split[:task_id+1]): #add logic of classes seen so far
            num = int(n/np.sum(args.class_split[:task_id+1]))
            labels_ = torch.tensor([[i]*num for i in range(np.sum(args.class_split[:task_id+1]))]).flatten()
        else:
            labels_ = torch.tensor([[i]*1 for i in range(np.sum(args.class_split[:task_id+1]))]).flatten()
        labels = None


    with torch.inference_mode():
        x = torch.randn((len(labels_), 3, args.image_size, args.image_size)).to(device)
        for i, t in enumerate(tqdm(noise_scheduler.timesteps)): #timesteps in reverse order
            # print(len(labels))
            with torch.no_grad():
                if args.unet_model == 'diffusers':
                    predicted_noise = model(x, t, labels).sample
                else:
                    ts = (torch.ones(len(labels_)) * t).long().to(device)
                    predicted_noise = model(x, ts, labels)
            x = noise_scheduler.step(predicted_noise, t, x).prev_sample
            if args.is_debug:
                break
            # break
        # x = (x / 2 + 0.5).clamp(0, 1)
        x = (x.clamp(-1, 1) + 1) / 2 #unnormalize
    return x, labels_


def get_clustering_dataloader(args, kmeans_clustering, train_data_loaders_diffusion, encoder, knn_transform, diff_transform, device):
    xset = []
    yset =  []
    index = 0
    encoder.eval()
    for x1, x2, y in train_data_loaders_diffusion:
        x2 = x2.to(device)
        images = knn_transform(x2)
        features = encoder.backbone(images).squeeze().detach().cpu().numpy()
        # print(features.shape) #.astype('double')
        cluster_ids = kmeans_clustering.predict(features)
        # print(cluster_ids)
        if index == 0:
            xset = images
            yset = cluster_ids
        else:
            xset = torch.cat((xset, x2), dim = 0)
            yset = torch.cat((yset, cluster_ids), dim = 0)
    
    diff_train_dataset = SimSiam_Dataset(xset.to('cpu'), yset, diff_transform, None)
    diff_train_dl = DataLoader(diff_train_dataset , batch_size=args.diff_train_bs, shuffle=True, num_workers = 4, pin_memory=True)
    return diff_train_dl

    

def get_dataloaders(model, n, args, device, noise_scheduler, task_id, transform, transform_prime, diff_transform, state='train', iteration_=4,n_cluster=0):
    if state == 'fid':
        iteration =  iteration_ - 1
        diff_transform = T.Compose([
            T.Resize(args.image_size),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        iteration = args.msize/args.diff_train_bs
    if args.is_debug:
        iteration = 4
    for i in range(int(iteration)):
        print(' Sampling Iteration Number: '+str(i+1)+', Total Iterations: '+str(int(iteration)))
        sampled_images, labels = sample_images(model, args.diff_train_bs, args, device, noise_scheduler, task_id, n_cluster)
        if i == 0:
            xvector = sampled_images
            yvector = labels
        else:
            xvector = torch.cat((sampled_images, xvector), dim=0)
            if labels is not None:
                yvector = torch.cat((labels, yvector), dim=0)
        if i ==2 and args.is_debug:
            break

    train_dataset = SimSiam_Dataset(xvector.to('cpu'), yvector.to('cpu'), transform, transform_prime)
    diff_train_dataset = SimSiam_Dataset(xvector.to('cpu'), yvector.to('cpu'), diff_transform, None)

    train_dl = DataLoader(train_dataset, batch_size=args.replay_bs, shuffle=True, num_workers = 4, pin_memory=True)
    diff_train_dl = DataLoader(diff_train_dataset , batch_size=args.diff_train_bs, shuffle=True, num_workers = 4, pin_memory=True)

    return train_dl, diff_train_dl

def print_images(diff_train_dl, ema_model, task_id, epoch, text= "Diffusion Training"):
    for images, images2, labels in diff_train_dl:
        utils.save_image(images2,  str('./results/' +str(text)+'task_id'+str(task_id)+'epoch'+str(epoch)+'emasample-.png'), nrow = int(math.sqrt(images2.shape[0])))
        # images2 = images2.type(torch.uint8) # To plot
        wandb.log({" sampled_images (ema model) "+str(text):     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in images2]})

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


def KNNmeans_clustering(n_cluster=200, encoder=None, current_knn_dataloader=None, old_diffusion_dataloader=None, task_id=0, transform_knn=None, device='cpu' ):
    ind = 0
    encoder.to(device)
    # for data in zip(current_knn_dataloader): #cuurent task's knn data loader
    for x, y in current_knn_dataloader:
        x = x.to(device)
        features = encoder.backbone(x)
        features = nn.functional.normalize(features.squeeze()).detach().cpu().numpy()
        if ind == 0:
            train_features = features
        elif ind > 0 and ind <=10:
            train_features = np.concatenate((train_features, features), axis=0)
            if ind == 10:
                break
        ind += 1

    if  old_diffusion_dataloader is not None:        
        for x1, x2, y in old_diffusion_dataloader:
            x2 = x2.to(device)
            x2 = transform_knn(x2)
            features = encoder.backbone(x2)
            features = nn.functional.normalize(features.squeeze()).detach().cpu().numpy()
            train_features = np.concatenate((train_features, features), axis=0)
            if ind == 20:
                break
        ind += 1
    print("train features sahpe")
    print(train_features.shape)       
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(train_features)
    print(kmeans.labels_.shape)
    return kmeans



 




# def evaluate_diffusion_model():



