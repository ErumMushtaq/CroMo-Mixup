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
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss,BarlowTwinsLoss
from utils.lars import LARS

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def extract_subspace(model, loader, rate=0.99,device = None, Q_prev = None, task=None):
    model.eval()
    outs = []
    for x,y in loader:
        x = x.to(device)
        out = model(x).cpu().detach().numpy()
        outs.append(out)



    outs = np.concatenate(outs)
    outs = outs.transpose()
    outs = torch.tensor(outs)

    if Q_prev == None:
        projected = torch.zeros(1)
    else:
        Q_prev = Q_prev.to('cpu')
        projected = Q_prev  @ Q_prev.T @ outs 

    remaining = outs - projected
    U, S, V = torch.svd(remaining)
    for i in range(len(S)):
        total = torch.norm(outs)**2 
        hand =  torch.norm(projected)**2 + torch.norm(S[0:i+1])**2
        
        if hand / total > rate:
            break

    print(U[:,0:i+1].shape)

    if Q_prev == None:
        Q_prev = U[:,0:i+1]
    else:
        Q_prev = torch.cat((Q_prev, U[:,0:i+1]),dim=1)
        Q_prev, _ = torch.linalg.qr(Q_prev, mode="reduced")
    wandb.log({"Task": task, "LRD Space Used ": Q_prev.shape[1]/Q_prev.shape[0] })  
    print(Q_prev.shape)
    return Q_prev


def train_LRD_infomax(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, device, args):#just for 2 tasks
    
    epoch_counter = 0
    old_model = None
    criterion = nn.CosineSimilarity(dim=1)
    Q = None

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        if args.resume_checkpoint:
            file_name = 'checkpoints/infomax.tar'
            dict = torch.load(file_name)
            model.load_state_dict(dict['state_dict'])
            args.epochs[0] = 0

        project_dim = args.proj_out
        covarince_loss = CovarianceLoss(project_dim,device=device)

            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.pretrain_warmup_epochs , max_epochs=args.epochs[task_id],warmup_start_lr=args.pretrain_warmup_lr,eta_min=args.min_lr) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_task = []
            epoch_loss_kd = []
            epoch_loss_norm = []
            for x1, x2, y in loader:
                x1,x2 = x1.to(device), x2.to(device)
                f1 = model.encoder.backbone(x1).squeeze() # NxC
                f2 = model.encoder.backbone(x2).squeeze() # NxC

                if Q != None:#let's do projection
                    f1_projected = f1 @ Q @ Q.T  
                    f2_projected = f2 @ Q @ Q.T

                    f1 = f1#- f1_projected
                    f2 = f2# - f2_projected

                    norm_loss_1 = torch.norm(f1_projected,dim =1) / (torch.norm(f1,dim =1) + 0.0000001) 
                    norm_loss_1 = torch.mean(norm_loss_1)

                    norm_loss_2 = torch.norm(f2_projected,dim =1) / (torch.norm(f2,dim =1) + 0.0000001) 
                    norm_loss_2 = torch.mean(norm_loss_2)

                    loss_norm = (norm_loss_1 + norm_loss_2) / 2
                else:
                    loss_norm = torch.tensor(0)


                z1 = model.encoder.projector(f1) # NxC
                z2 = model.encoder.projector(f2) # NxC

                z1 = F.normalize(z1, p=2)
                z2 = F.normalize(z2, p=2)
                cov_loss =  covarince_loss(z1, z2)
                sim_loss =  invariance_loss(z1, z2)
                

                loss_task = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss) 

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()

                    lossKD = (-(criterion(f1_projected, f1Old).mean() * 0.5
                                            + criterion(f2_projected, f2Old).mean() * 0.5) )
                else:
                    lossKD = torch.tensor(0)
                


                epoch_loss_task.append(loss_task.item())
                epoch_loss_kd.append(lossKD.item())
                epoch_loss_norm.append(loss_norm.item())
                
                optimizer.zero_grad()
                loss = loss_task +  args.lambdap * lossKD + args.lambda_norm * loss_norm
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
                    print(f" Knn Accuracy Task- {str(i)} : {acc},  Epoch : {epoch_counter}")
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Norm_Loss: {np.mean(epoch_loss_norm):.4f}   | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Norm_Loss: {np.mean(epoch_loss_norm):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task), " Epoch ": epoch_counter, " Average KD Loss ": np.mean(epoch_loss_kd) , " Average Norm Loss ": np.mean(epoch_loss_norm) })  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder.backbone)  # save t-1 model
        oldModel.to(device)
        oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        Q = None # each time make Q empty
        
        Q = extract_subspace(model, knn_train_data_loaders[task_id], rate= args.subspace_rate,device = device, Q_prev = Q, task=task_id)
        Q = Q.to(device)


        #file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)
        #+"-CS"+str(args.class_split) + 'task_' + str(task_id) + 'lambdap_' + str(args.lambdap) + 'lambda_norm_' + str(args.lambda_norm) + 'same_lr_' + str(args.same_lr) + 'norm_' + str(normalization) + 'ws_' + str(args.weight_standard) + '.pth.tar' 
        
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) +  '_subspace_' + str(args.subspace_rate) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)


    return model, loss_, optimizer



def train_LRD_barlow(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, device, args):#just for 2 tasks
    
    epoch_counter = 0
    old_model = None
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)
    Q = None

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        if args.resume_checkpoint:
            file_name = 'checkpoints/infomax.tar'
            dict = torch.load(file_name)
            model.load_state_dict(dict['state_dict'])
            args.epochs[0] = 0

        project_dim = args.proj_out
        covarince_loss = CovarianceLoss(project_dim,device=device)

            
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.pretrain_warmup_epochs , max_epochs=args.epochs[task_id],warmup_start_lr=args.pretrain_warmup_lr,eta_min=args.min_lr) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
       
        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_task = []
            epoch_loss_kd = []
            epoch_loss_norm = []
            for x1, x2, y in loader:
                x1,x2 = x1.to(device), x2.to(device)
                f1 = model.encoder.backbone(x1).squeeze() # NxC
                f2 = model.encoder.backbone(x2).squeeze() # NxC

                if Q != None:#let's do projection
                    f1_projected = f1 @ Q @ Q.T  
                    f2_projected = f2 @ Q @ Q.T

                    f1 = f1 #- f1_projected #remove that part
                    f2 = f2 #- f2_projected #remove that part

                    norm_loss_1 = torch.norm(f1_projected,dim =1) / (torch.norm(f1,dim =1) + 0.0000001) 
                    norm_loss_1 = torch.mean(norm_loss_1)

                    norm_loss_2 = torch.norm(f2_projected,dim =1) / (torch.norm(f2,dim =1) + 0.0000001) 
                    norm_loss_2 = torch.mean(norm_loss_2)

                    loss_norm = (norm_loss_1 + norm_loss_2) / 2
                else:
                    loss_norm = torch.tensor(0)


                z1 = model.encoder.projector(f1) # NxC
                z2 = model.encoder.projector(f2) # NxC

                #z1 = F.normalize(z1, p=2)#remove that part
                #z2 = F.normalize(z2, p=2)#remove that part

                loss_task = cross_loss(z1, z2) 

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()

                    f1Old = f1Old @ Q @ Q.T  
                    f2Old = f2Old @ Q @ Q.T

                    lossKD = (-(criterion(f1_projected, f1Old).mean() * 0.5
                                            + criterion(f2_projected, f2Old).mean() * 0.5) )
                else:
                    lossKD = torch.tensor(0)
                


                epoch_loss_task.append(loss_task.item())
                epoch_loss_kd.append(lossKD.item())
                epoch_loss_norm.append(loss_norm.item())
                
                optimizer.zero_grad()
                loss = loss_task +  args.lambdap * lossKD + args.lambda_norm * loss_norm
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
                    print(f" Knn Accuracy Task- {str(i)} : {acc},  Epoch : {epoch_counter}")
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Norm_Loss: {np.mean(epoch_loss_norm):.4f}   | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Norm_Loss: {np.mean(epoch_loss_norm):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task), " Epoch ": epoch_counter, " Average KD Loss ": np.mean(epoch_loss_kd) , " Average Norm Loss ": np.mean(epoch_loss_norm) })  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder.backbone)  # save t-1 model
        oldModel.to(device)
        oldModel = oldModel.train()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        Q = None # each time make Q empty
        
        Q = extract_subspace(model, knn_train_data_loaders[task_id], rate= args.subspace_rate,device = device, Q_prev = Q, task=task_id)
        Q = Q.to(device)


        #file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)
        #+"-CS"+str(args.class_split) + 'task_' + str(task_id) + 'lambdap_' + str(args.lambdap) + 'lambda_norm_' + str(args.lambda_norm) + 'same_lr_' + str(args.same_lr) + 'norm_' + str(normalization) + 'ws_' + str(args.weight_standard) + '.pth.tar' 
        
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + 'updated_loss' + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)


    return model, loss_, optimizer



