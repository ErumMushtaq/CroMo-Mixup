import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss
from copy import deepcopy


def update_moving_average(new_model, old_model):
    for current_params, ma_params in zip(new_model.parameters(), old_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        # old * self.beta + (1 - self.beta) * new
        ma_params.data = old_weight * 0.5 + 0.5 * up_weight
    # return new_model

def train(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, class_split_knntrain_data_loader, class_split_test_data_loader, device, args):

    init_lr = args.pretrain_base_lr        
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.pretrain_warmup_epochs , max_epochs=args.epochs,warmup_start_lr=args.pretrain_warmup_lr,eta_min=args.min_lr) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
    covarince_loss = CovarianceLoss(args.proj_out,device=device, R_eps_weight= args.R_eps_weight)
    if args.info_loss == 'error_cov':
        err_covarince_loss = ErrorCovarianceLoss(args.proj_out ,device=device)

    loss_ = []
    epoch_counter = 0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        cov_loss_ = []
        inv_loss_ = []
        for data in zip(*train_data_loaders):
            for x1, x2, y in data:   
                x1 = x1.to(device)
                x2 = x2.to(device)
                z1,z2 = model(x1, x2)

                z1 = F.normalize(z1, p=2)
                z2 = F.normalize(z2, p=2)
                cov_loss =  covarince_loss(z1, z2)

                if args.info_loss == 'invariance':
                    sim_loss =  invariance_loss(z1, z2)
                elif args.info_loss == 'error_cov': 
                    sim_loss = err_covarince_loss(z1, z2)

                loss = (args.sim_loss_weight * sim_loss) + (args.cov_loss_weight * cov_loss) 
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
<<<<<<< HEAD
                epoch_loss.append(loss.item())
                cov_loss_.append(cov_loss.item())
                inv_loss_.append(sim_loss.item())
=======


                old_model.load_state_dict(model.state_dict())
                update_moving_average(model, old_model)
                # print(epoch_loss)


        print('epoch finished') 
>>>>>>> 8644ebc7177e8e3e9a1b9b16622df85ec077e89d
        epoch_counter += 1
        scheduler.step()
        loss_.append(np.mean(epoch_loss))
        end = time.time()
        # covarince_loss.plot_eigs(epoch_counter)
        if (epoch+1) % args.knn_report_freq == 0:
            covarince_loss.plot_eigs(epoch_counter)
            class_split_test_data_loader
            knn_acc = Knn_Validation(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})
            if len(class_split_test_data_loader) > 1: 
                for i in range(len(class_split_test_data_loader)):
                    knn_acc_class = Knn_Validation(model, class_split_knntrain_data_loader[i], class_split_test_data_loader[i], device=device, K=200, sigma=0.5)
                    wandb.log({" Knn Accuracy Class-"+str(i): knn_acc_class, " Epoch ": epoch})
                    

            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    return model, loss_, optimizer

