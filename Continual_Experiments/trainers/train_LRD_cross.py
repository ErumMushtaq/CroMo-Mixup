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

def update_memory(memory_x, memory_y, dataloader, size, task):
    indices = np.random.choice(len(dataloader.dataset), size=size, replace=False)
    x, _ =  dataloader.dataset[indices]
    memory_x = torch.cat((memory_x, x), dim=0)
    y = torch.ones(x.shape[0],dtype=torch.long) * task
    memory_y = torch.cat((memory_y, y), dim=0)
    return memory_x, memory_y.to(dtype=torch.long)

def train_LRD_cross_barlow(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args):#just for 2 tasks
    
    memory_x = torch.Tensor()
    memory_y = torch.Tensor()

    epoch_counter = 0
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)
    Q = None

    crossentry_loss = nn.CrossEntropyLoss()

    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

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
            epoch_loss_contrastive = []
            for x1, x2, y in loader:
                x1,x2 = x1.to(device), x2.to(device)
                f1 = model.encoder.backbone(x1).squeeze() # NxC
                f2 = model.encoder.backbone(x2).squeeze() # NxC


                if task_id > 0:
                    #samples from old tasks with labels (task ids)
                    indices = np.random.choice(len(memory_x), size=min(args.bsize, len(memory_x)), replace=False)
                    x_old = memory_x[indices].to(device)
                    y_old = memory_y[indices].to(device)
                    #samples from the new task
                    indices = np.random.choice(len(train_data_loaders_pure[task_id].dataset), size=args.bsize, replace=False)
                    x_new, _ =  train_data_loaders_pure[task_id].dataset[indices]
                    x_new = x_new.to(device)
                    y_new = torch.ones(x_new.shape[0],dtype=torch.long).to(device) * task_id          
                    #concatenate
                    x = torch.cat((x_old,x_new),dim=0)
                    y = torch.cat((y_old,y_new),dim=0)
                    

                    #pass from the model
                    encoded_vectors = model.encoder.backbone(x).squeeze() # NxC

                    #do classification loss
                    outputs = contrastive_classifier(encoded_vectors) 
                    contrastive_loss = crossentry_loss(outputs, y)

                    
                    f1_projected = f1 @ Q @ Q.T  
                    f2_projected = f2 @ Q @ Q.T

                else:
                    contrastive_loss = torch.tensor(0)



                z1 = model.encoder.projector(f1) # NxC
                z2 = model.encoder.projector(f2) # NxC

                z1 = F.normalize(z1, p=2)
                z2 = F.normalize(z2, p=2)

                loss_task = cross_loss(z1, z2) 

                if task_id != 0: #do Distillation
                    f1Old = oldModel(x1).squeeze().detach()
                    f2Old = oldModel(x2).squeeze().detach()

                    lossKD = (-(criterion(f1_projected, f1Old).mean() * 0.5
                                            + criterion(f2_projected, f2Old).mean() * 0.5) )
                else:
                    lossKD = torch.tensor(0)
                


                epoch_loss_task.append(loss_task.item())
                epoch_loss_kd.append(lossKD.item())
                epoch_loss_contrastive.append(contrastive_loss.item())
                
                if task_id > 0:
                    #tune the classifier (might be optional in the future)
                    contrastive_optimizer.zero_grad()
                optimizer.zero_grad()
                loss = loss_task +  args.lambdap * lossKD + args.contrastive_ratio * contrastive_loss
                loss.backward()

                if task_id > 0:
                    #tune the classifier (might be optional in the future)
                    contrastive_optimizer.step()
            
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
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Contrastive_Loss: {np.mean(epoch_loss_contrastive):.4f}   | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss_task):.4f} | KDLoss: {np.mean(epoch_loss_kd):.4f} | Contrastive_Loss: {np.mean(epoch_loss_contrastive):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_task), " Epoch ": epoch_counter, " Average KD Loss ": np.mean(epoch_loss_kd) , " Average Contrastive Loss ": np.mean(epoch_loss_contrastive) })  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            

        oldModel = deepcopy(model.encoder.backbone)  # save t-1 model
        oldModel.to(device)
        oldModel.eval()
        for param in oldModel.parameters(): #Freeze old model
            param.requires_grad = False

        Q = None # each time make Q empty
        
        Q = extract_subspace(model, knn_train_data_loaders[task_id], rate= args.subspace_rate,device = device, Q_prev = Q, task=task_id)
        Q = Q.to(device)

        contrastive_classifier = nn.Linear(f1.shape[1], task_id+2, bias=False).to(device)
        contrastive_optimizer = torch.optim.SGD(contrastive_classifier.parameters(), lr=0.001)

        memory_x, memory_y = update_memory(memory_x,memory_y, train_data_loaders_pure[task_id], args.msize, task_id)

        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + 'updated_loss' + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

        

    return model, loss_, optimizer


