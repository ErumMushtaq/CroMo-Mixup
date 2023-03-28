import time
import wandb
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont

class Generation_Dataset(Dataset):
    def __init__(self, mean, cov, shape, dataset_len, transform, transform_prime):

        self.sampler = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        self.shape = shape
        self.dataset_len = dataset_len

        self.transform = transform
        self.transform_prime = transform_prime

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        x = self.sampler.sample().reshape(self.shape)
        x1 = self.transform(x)
        x2 = self.transform_prime(x)
        return x1, x2

def get_data_creator(dataloader, length, bs):
    data_normalize_mean = (0.4914, 0.4822, 0.4465)
    data_normalize_std = (0.247, 0.243, 0.261)
    transform_n = T.Compose(
            [   
                T.Normalize(data_normalize_mean, data_normalize_std),
            ])

    X, _ = dataloader.dataset[:]
    X = transform_n(X)
    size  = X.shape
    X = X.reshape(size[0], -1)

    cov = torch.cov(X.T)
    mean = torch.mean(X, dim=0)

    transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2)])

    transform_prime = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2)])

    custom_ds = Generation_Dataset(mean, cov, size[1:], length, transform, transform_prime)
    dl = torch.utils.data.DataLoader(custom_ds, batch_size=bs, shuffle=True, num_workers = 2 , pin_memory=True)
    return dl


def train_cov(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, device, args):
    epoch_counter = 0
    old_task_dl = None
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            epoch_loss_old = []
            total_loss = []
            if task_id == 0:
                for x1, x2, y in loader:   
                    loss = model(x1, x2)
                    epoch_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
            else:
                for data in zip(loader, old_task_dl): 
                    loss = 0.0
                    optimizer.zero_grad() 

                    length_org = len(data[0][0])
                    loss_org = model(data[0][0], data[0][1])
                    epoch_loss.append(loss_org.item())
                    loss += loss_org*length_org

                    length_old = len(data[1][0])
                    loss_old = model(data[1][0], data[1][1])
                    epoch_loss_old.append(loss_old.item())
                    loss += loss_old*length_old

                    loss /= (length_org + length_old)
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
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | New Loss: {np.mean(epoch_loss):.4f} | Old Loss: {np.mean(epoch_loss_old):.4f}  | Total Loss: {np.mean(total_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | New Loss: {np.mean(epoch_loss):.4f} | Old Loss: {np.mean(epoch_loss_old):.4f}  | Total Loss: {np.mean(total_loss):.4f}')
        
            wandb.log({" Average Current Task Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Total Loss ": np.mean(total_loss), " Epoch ": epoch_counter})  
            wandb.log({" Average Old Task Loss ": np.mean(epoch_loss_old), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        old_task_dl = get_data_creator(knn_train_data_loaders[task_id], int(len(loader)*loader.batch_size*args.ratio), int(loader.batch_size*args.ratio))
    return model, loss_, optimizer

