import time
import wandb
import torch
import numpy as np
import torchvision.transforms as T

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont

def update_memory(memory, dataloader, size):
    indices = np.random.choice(len(dataloader.dataset), size=size, replace=False)
    x, _ =  dataloader.dataset[indices]
    memory = torch.cat((memory, x), dim=0)
    return memory


def train_ering(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, device, args, transform, transform_prime):
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
                loss_org = model(x1, x2)
                len_org = len(x1)
                loss_mem = 0.0
                len_mem = 0.0
                if task_id > 0:
                    indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    x = memory[indices].to(device)
                    x1, x2 = transform(x), transform_prime(x)
                    loss_mem = model(x1, x2)
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

        memory = update_memory(memory, knn_train_data_loaders[task_id], args.msize)

    return model, loss_, optimizer
