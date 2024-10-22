import time
import wandb
import torch
import numpy as np

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation

def train_concate(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, class_split_knntrain_data_loader, class_split_test_data_loader, device, args):

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
    if args.algo == 'infomax':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.pretrain_warmup_epochs , max_epochs=args.epochs,warmup_start_lr=args.pretrain_warmup_lr,eta_min=args.min_lr) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
    if args.algo == 'simsiam':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    #Training Loop for x1, x2, y in train_data_loaders[0]:
    loss_ = []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        iteration = 0
        for data in zip(*train_data_loaders):
            class_type_count = 0
            for x1, x2, y in data:  
                if class_type_count == 0:
                    concate_x1 = x1
                    concate_x2 = x2
                    # concate_y = y
                else:
                    concate_x1 = torch.cat((concate_x1, x1), dim = 0)
                    concate_x2 = torch.cat((concate_x2, x2), dim = 0)
                    # concate_y = torch.cat(concate_y, y)

                class_type_count += 1
            # print(concate_x1.shape)
            loss = model(concate_x1, concate_x2)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            iteration += 1
        scheduler.step()
        loss_.append(np.mean(epoch_loss))
        end = time.time()
        if (epoch+1) % args.knn_report_freq == 0:
            class_split_test_data_loader
            knn_acc = Knn_Validation(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})
            # if len(class_split_test_data_loader) > 1: 
            #     for i in range(len(class_split_test_data_loader)):
            #         knn_acc_class = Knn_Validation(model, class_split_knntrain_data_loader[i], class_split_test_data_loader[i], device=device, K=200, sigma=0.5)
            #         wandb.log({" Knn Accuracy Class-"+str(i): knn_acc_class, " Epoch ": epoch})
                    

            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    return model, loss_, optimizer

