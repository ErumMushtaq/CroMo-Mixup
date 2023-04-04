import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation, linear_test_sup

def train_sup(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, class_split_knntrain_data_loader, class_split_test_data_loader, device, args):

    # Optimizer and Scheduler
    # SimSiam uses SGD, with lr = lr*BS/256 from paper + https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py)
    init_lr = args.pretrain_base_lr#*args.pretrain_batch_size/256.
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
    
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=epochs,
    #                                             warmup_start_lr=args.warmup_start_lr, eta_min=2e-4)
    # scheduler = SimSiamScheduler(optimizer, 
    #                              warmup_epochs=args.pretrain_warmup_epochs, warmup_lr=args.pretrain_warmup_lr*args.pretrain_batch_size/256., 
    #                              num_epochs=args.final_pretrain_epoch, base_lr=args.pretrain_base_lr*args.pretrain_batch_size/256., final_lr=0, iter_per_epoch=len(train_data_loaders), 
    #                              constant_predictor_lr=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    #Training Loop for x1, x2, y in train_data_loaders[0]:
    loss_ = []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        iteration = 0
        for data in zip(*train_data_loaders):
            for x1,  y in data:  
                x1 = x1.to(device) 
                y = y.to(device)
                outputs = model(x1)
                loss = criterion(outputs, y)
                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                iteration += 1
        scheduler.step()
        loss_.append(np.mean(epoch_loss))
        end = time.time()
        if (epoch+1) % args.knn_report_freq == 0:
            loss, acc1, acc5 = linear_test_sup(model, test_data_loaders, epoch, device)
            # # class_split_test_data_loader
            # # knn_acc = Knn_Validation(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            # wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})
            # if len(class_split_test_data_loader) > 1: 
            #     for i in range(len(class_split_test_data_loader)):
            #         knn_acc_class = Knn_Validation(model, class_split_knntrain_data_loader[i], class_split_test_data_loader[i], device=device, K=200, sigma=0.5)
            #         wandb.log({" Knn Accuracy Class-"+str(i): knn_acc_class, " Epoch ": epoch})
                    

        #     print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        # else:
        print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    return model, loss_, optimizer

