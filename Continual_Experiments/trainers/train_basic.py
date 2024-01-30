import time
import wandb
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

import torch.nn as nn

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from copy import deepcopy
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from utils.lars import LARS
#https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/predictive_mse.py

from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset
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
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
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
        covarince_loss = CovarianceLoss(args.proj_out, device=device, la_mu=args.la_mu, la_R=args.la_R)
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
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  

    return model, loss_, optimizer


def train_barlow(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args):
    
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
            lin_epoch = 100
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


    return model, loss_, optimizer



def info_nce_loss(features,args,device):

    labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)#there are only two views
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels

def train_simclr(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args):
    
    epoch_counter = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr
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
                z1,z2 = model(x1, x2)
                features = torch.cat((z1,z2),dim=0)

                logits, labels = info_nce_loss(features,args,device)
                loss =  criterion(logits, labels)
                
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
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)
        
        if task_id < len(train_data_loaders)-1:
            lin_epoch = 100
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(features_dim=model.encoder.backbone.output_dim, num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  


    return model, loss_, optimizer


