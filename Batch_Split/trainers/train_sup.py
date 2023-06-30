import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import linear_test


def train_sup(model, train_data_loaders, test_data_loaders, device, args):
    # Optimizer and Scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_base_lr, weight_decay=args.pretrain_weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrain_base_lr, weight_decay=args.pretrain_weight_decay, 
                                momentum=args.pretrain_momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.pretrain_base_lr, 
    #                                                 epochs=args.epochs, steps_per_epoch=len(train_data_loaders[0])*len(train_data_loaders))

    criterion = nn.CrossEntropyLoss()
    loss_ = []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        true_samples = 0
        total_samples = 0
        for data in zip(*train_data_loaders):
            for x1, _, y in data:  
                x1 = x1.to(device) 
                y = y.to(device)
                outputs = model(x1)
                loss = criterion(outputs, y)
                total_samples += len(x1)
                pred = torch.argsort(outputs, dim=-1, descending=True)[:, 0]
                true_samples += torch.sum((pred == y)).item()
                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
        scheduler.step()
        loss_.append(np.mean(epoch_loss))

        model.eval()
        true_samples_test = 0
        total_samples_test = 0
        for x1, y in test_data_loaders:
            x1 = x1.to(device) 
            y = y.to(device)
            outputs = model(x1)
            total_samples_test += len(x1)
            pred = torch.argsort(outputs, dim=-1, descending=True)[:, 0]
            true_samples_test += torch.sum((pred == y)).item()
        end = time.time()
        if (epoch+1) % args.knn_report_freq == 0:
            linear_test(model, test_data_loaders, None, epoch, device)

        wandb.log({" Global Linear Accuracy ": true_samples_test/total_samples_test*100, " Epoch ": epoch})
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

        print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} | Acc: {true_samples/total_samples*100:.2f}% \
| Acc Test: {true_samples_test/total_samples_test*100:.2f}')
    return model, loss_, optimizer
