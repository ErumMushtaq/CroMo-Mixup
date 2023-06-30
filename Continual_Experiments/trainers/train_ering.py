import time
import wandb
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss
def update_memory(memory, dataloader, size):
    indices = np.random.choice(len(dataloader.dataset), size=size, replace=False)
    x, _ =  dataloader.dataset[indices]
    memory = torch.cat((memory, x), dim=0)
    return memory

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def train_ering_simsiam(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    memory = torch.Tensor()

    epoch_counter = 0
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0:
            init_lr = init_lr / 10
            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            epoch_loss_mem = []
            total_loss = []
            for x1, x2, y in loader:   
                optimizer.zero_grad()
                z1,z2,p1,p2 = model(x1, x2)

                loss_one = loss_fn(p1, z2.detach())
                loss_two = loss_fn(p2, z1.detach())
                loss = 0.5*loss_one + 0.5*loss_two
                loss_org = loss.mean()

                len_org = len(x1)
                loss_mem = 0.0
                len_mem = 0.0
                if task_id > 0:
                    indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    x = memory[indices].to(device)
                    x1, x2 = transform(x), transform_prime(x)

                    z1,z2,p1,p2 = model(x1, x2)
                    loss_one = loss_fn(p1, z2.detach())
                    loss_two = loss_fn(p2, z1.detach())
                    loss = 0.5*loss_one + 0.5*loss_two
                    loss_mem = loss.mean()

                    len_mem = args.bsize
                    epoch_loss_mem.append(loss_mem.item())

                loss = loss_org*len_org + loss_mem*len_mem
                loss = loss / (len_org+len_mem)
                epoch_loss.append(loss_org.item())
                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss))
            end = time.time()
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Current Loss: {np.mean(epoch_loss):.4f} | Mem Loss: {np.mean(epoch_loss_mem):.4f}  | Total Loss: {np.mean(total_loss):.4f}    | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Curent Loss: {np.mean(epoch_loss):.4f}  | Mem Loss: {np.mean(epoch_loss_mem):.4f}  | Total Loss: {np.mean(total_loss):.4f}  ')

            wandb.log({" Average Current Task Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Total Loss ": np.mean(total_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Mem Loss ": np.mean(epoch_loss_mem), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)

    return model, loss_, optimizer


def train_ering_infomax(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    memory = torch.Tensor()
    epoch_counter = 0
    for task_id, loader in enumerate(train_data_loaders):
        past_data_loader = train_data_loaders[0]
        # Optimizer and Scheduler
        # init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        init_lr = args.pretrain_base_lr
        if task_id != 0:
            init_lr = init_lr / 10
        
        if args.resume_checkpoint:
            file_name = 'checkpoints/infomax.tar'
            dict = torch.load(file_name)
            model.load_state_dict(dict['state_dict'])
            args.epochs[0] = 0
            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
        covarince_loss = CovarianceLoss(args.proj_out, device=device)
        if args.info_loss == 'error_cov':
            err_covarince_loss = ErrorCovarianceLoss(args.proj_out ,device=device)

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            epoch_loss_mem = []
            total_loss = []
            for x1, x2, y in loader:   
                optimizer.zero_grad()
                x1, x2 = x1.to(device), x2.to(device)
                len_org = len(x1)
                loss_mem = torch.tensor(0)
                len_mem = 0.0
                if task_id > 0:                    
                    indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    x = memory[indices].to(device)
                    xx1, xx2 = transform(x), transform_prime(x)

                    x1 = torch.cat([x1, xx1], dim=0)
                    x2 = torch.cat([x2, xx2], dim=0)

                z1,z2 = model(x1, x2)
                z1 = F.normalize(z1, p=2)
                z2 = F.normalize(z2, p=2)
                cov_loss =  covarince_loss(z1, z2)

                if args.info_loss == 'invariance':
                    sim_loss =  invariance_loss(z1, z2)
                elif args.info_loss == 'error_cov': 
                    sim_loss = err_covarince_loss(z1, z2)
                loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss)
                loss_org = loss

                # loss = loss_org*len_org + loss_mem*len_mem
                # loss = loss / (len_org+len_mem)
                epoch_loss.append(loss_org.item())
                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                # print("done")
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss))
            end = time.time()
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Current Loss: {np.mean(epoch_loss):.4f} | Mem Loss: {np.mean(epoch_loss_mem):.4f}  | Total Loss: {np.mean(total_loss):.4f}    | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Curent Loss: {np.mean(epoch_loss):.4f}  | Mem Loss: {np.mean(epoch_loss_mem):.4f}  | Total Loss: {np.mean(total_loss):.4f}  ')

            wandb.log({" Average Current Task Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Total Loss ": np.mean(total_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Mem Loss ": np.mean(epoch_loss_mem), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer