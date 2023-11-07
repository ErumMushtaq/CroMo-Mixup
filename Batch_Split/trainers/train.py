import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from copy import deepcopy
from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset
from tqdm import tqdm
from utils.lars import LARS
import torch.nn as nn

def update_moving_average(new_model, old_model):
    for current_params, ma_params in zip(new_model.parameters(), old_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        # old * self.beta + (1 - self.beta) * new
        ma_params.data = old_weight * 0.5 + 0.5 * up_weight
    # return new_model

def correct_top_k(outputs, targets, top_k=(1,5)):
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result

def linear_test(net, data_loader, classifier, epoch, device, task_num, args):
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
            for i in range(args.val_class_split[task_num]):
                target[target== task_num*args.val_class_split[task_num]+i] = i
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

def linear_train(net, data_loader, train_optimizer, classifier, scheduler, epoch, device, task_num, args,):

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
        # for i in range(len(args.class_split)):
        # if task_num == 0:
        # if task_num == 4:
        #     print(target)
        for i in range(args.val_class_split[task_num]):
            target[target== task_num*args.val_class_split[task_num]+i] = i
        # if task_num == 4:
        #     print(target)
        # if task_num == 4:
        #     break

            # else:
            #     target[(target>=np.sum(args.class_split[i-1])) & (target<np.sum(args.class_split[i]))] = i
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

def linear_evaluation(net, data_loader,test_data_loader,train_optimizer,classifier, scheduler, epochs, device, task_num, args):
    for epoch in range(1, epochs+1):
        linear_train(net,data_loader,train_optimizer,classifier,scheduler, epoch, device, task_num, args)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1 = linear_test(net, test_data_loader, classifier, epoch, device, task_num, args)

    return test_loss, test_acc1, classifier

def linear_test_(net, data_loader, classifier, epoch, device, task_num, args):
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
            # Task ID conversion
            for i in range(len(args.val_class_split)):            
                if i ==0:
                    target[target<args.val_class_split[i]] = i
                else:
                    target[(target>=np.sum(args.val_class_split[:i])) & (target<np.sum(args.val_class_split[:i+1]))] = i
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

def linear_train_(net, data_loader, train_optimizer, classifier, scheduler, epoch, device, task_num, args):

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
        # Task ID conversion
        for i in range(len(args.val_class_split)):            
            if i ==0:
                target[target<args.val_class_split[i]] = i
            else:
                target[(target>=np.sum(args.val_class_split[:i])) & (target<np.sum(args.val_class_split[:i+1]))] = i
        # print(target)
        # exit()
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

def linear_evaluation_TP(net, data_loaders,test_data_loaders,train_optimizer,classifier, scheduler, epochs, device, task_num, args):
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
        linear_train_(net,data_loader,train_optimizer,classifier,scheduler, epoch, device, task_num, args)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1 = linear_test_(net, test_data_loader, classifier, epoch, device, task_num, args)

    return test_loss, test_acc1, classifier

# (model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)

def train_infomax(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args):

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
                z1, z2 = model(x1, x2)

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
                epoch_loss.append(loss.item())

        print('epoch finished') 
        epoch_counter += 1
        scheduler.step()
        loss_.append(np.mean(epoch_loss))
        end = time.time()
        # covarince_loss.plot_eigs(epoch_counter)
        if (epoch+1) % args.knn_report_freq == 0:
            # covarince_loss.plot_eigs(epoch_counter)
            # class_split_test_data_loader
            knn_acc, task_acc_arr = Knn_Validation_cont(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            # knn_acc = Knn_Validation(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})

            #WP
            if (epoch+1) % args.knn_report_freq*1 == 0:
                WP = []
                for i in range(len(args.val_class_split)):
                    lin_epoch = 100
                    num_class = args.val_class_split[i]
                    classifier = LinearClassifier(num_classes = num_class).to(device)
                    lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
                    lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
                    _, wp, _ =  linear_evaluation(model, train_data_loaders_linear[i], test_data_loaders[i], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, i, args)  
                    WP.append(wp)

                #TP
                # for i in range(len(args.class_split)):
                lin_epoch = 100
                num_class = len(args.val_class_split) #total number of tasks
                classifier = LinearClassifier(num_classes = num_class).to(device)
                lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
                lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
                _, tp, _ = linear_evaluation_TP(model, train_data_loaders_linear[:], test_data_loaders[:], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, 0, args) 

                wandb.log({" WP ": np.mean(WP), " Epoch ": epoch})
                wandb.log({" TP ": tp, " Epoch ": epoch})
                    

            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    return model, loss_, optimizer


def train_barlow(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args):
    
    epoch_counter = 0
    criterion = nn.CosineSimilarity(dim=1)
    cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

    # Optimizer and Scheduler
    init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
    optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

    loss_ = []
    for epoch in range(args.epochs):    
        start = time.time()
        model.train()
        epoch_loss = []
        for data in zip(*train_data_loaders):
            for x1, x2, y in data: 
                x1, x2 = x1.to(device), x2.to(device)
                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)
                # print(loss.item())
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
            knn_acc, task_acc_arr = Knn_Validation_cont(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})

            # #WP
            # if (epoch+1) % args.knn_report_freq*1 == 0:
            #     WP = []
            #     for i in range(len(args.val_class_split)):
            #         lin_epoch = 100
            #         num_class = args.val_class_split[i]
            #         classifier = LinearClassifier(num_classes = num_class).to(device)
            #         lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            #         lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            #         _, wp, _ =  linear_evaluation(model, train_data_loaders_linear[i], test_data_loaders[i], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, i, args)  
            #         WP.append(wp)

            #     #TP
            #     lin_epoch = 100
            #     num_class = len(args.val_class_split) #total number of tasks
            #     classifier = LinearClassifier(num_classes = num_class).to(device)
            #     lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            #     lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            #     _, tp, _ = linear_evaluation_TP(model, train_data_loaders_linear[:], test_data_loaders[:], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, 0, args) 

            #     wandb.log({" WP ": np.mean(WP), " Epoch ": epoch})
            #     wandb.log({" TP ": tp, " Epoch ": epoch})
                    
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    WP = []
    for i in range(len(args.val_class_split)):
        lin_epoch = 100
        num_class = args.val_class_split[i]
        classifier = LinearClassifier(num_classes = num_class).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
        _, wp, _ =  linear_evaluation(model, train_data_loaders_linear[i], test_data_loaders[i], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, i, args)  
        WP.append(wp)

    #TP
    lin_epoch = 100
    num_class = len(args.val_class_split) #total number of tasks
    classifier = LinearClassifier(num_classes = num_class).to(device)
    lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
    lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
    _, tp, _ = linear_evaluation_TP(model, train_data_loaders_linear[:], test_data_loaders[:], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, 0, args) 

    wandb.log({" WP ": np.mean(WP), " Epoch ": epoch})
    wandb.log({" TP ": tp, " Epoch ": epoch})
    return model, loss_, optimizer

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()
def train_simsiam(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args):
    

    init_lr = args.pretrain_base_lr

            
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

    loss_ = []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = []
        for data in zip(*train_data_loaders):
            for x1, x2, y in data: 
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

        scheduler.step()
        loss_.append(np.mean(epoch_loss))
        end = time.time()
        print('epoch end')
        # print('epoch end')
        if (epoch+1) % args.knn_report_freq == 0:
            knn_acc, task_acc_arr = Knn_Validation_cont(model, train_data_loaders_knn, test_data_loaders, device=device, K=200, sigma=0.5) 
            wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch})

            #WP
            # if (epoch+1) % args.knn_report_freq*1 == 0:

                    
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
        else:
            print(f'Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
    
        wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch})  
        wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch})

    WP = []
    for i in range(len(args.val_class_split)):
        lin_epoch = 100
        num_class = args.val_class_split[i]
        classifier = LinearClassifier(num_classes = num_class).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
        _, wp, _ =  linear_evaluation(model, train_data_loaders_linear[i], test_data_loaders[i], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, i, args)  
        WP.append(wp)

    #TP
    lin_epoch = 100
    num_class = len(args.val_class_split) #total number of tasks
    classifier = LinearClassifier(num_classes = num_class).to(device)
    lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
    lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
    _, tp, _ = linear_evaluation_TP(model, train_data_loaders_linear[:], test_data_loaders[:], lin_optimizer, classifier, lin_scheduler, lin_epoch, device, 0, args) 

    wandb.log({" WP ": np.mean(WP), " Epoch ": epoch})
    wandb.log({" TP ": tp, " Epoch ": epoch})

    return model, loss_, optimizer



