import time
import wandb
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from loss import invariance_loss,CovarianceLoss,ErrorCovarianceLoss, BarlowTwinsLoss
from utils.lars import LARS


def update_memory(memory, dataloader, size, device = 'cpu'):
    indices = np.random.choice(len(dataloader.dataset), size=size, replace=False)
    x, _ =  dataloader.dataset[indices]
    x = x.to(device)
    memory = torch.cat((memory, x), dim=0)
    return memory

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def train_ering_simsiam(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    memory = torch.Tensor().to(device)

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
                    # indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    # x = memory[indices].to(device)
                    # x1, x2 = transform(x), transform_prime(x)

                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    for ind in indices:
                        x1_old = torch.cat((x1_old, transform(memory[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory[ind:ind+1])), dim=0)
                    x1_old, x2_old = x1_old.to(device), x2_old.to(device)


                    z1,z2,p1,p2 = model(x1_old, x2_old)
                    loss_one = loss_fn(p1, z2.detach())
                    loss_two = loss_fn(p2, z1.detach())
                    loss = 0.5*loss_one + 0.5*loss_two
                    loss_mem = loss.mean()

                    len_mem = min(32*task_id, len(memory))
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

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)#,device = device)

    return model, loss_, optimizer


def train_ering_infomax(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    memory = torch.Tensor()#.to(device)
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
                    # indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    # x = memory[indices].to(device)
                    # xx1, xx2 = transform(x), transform_prime(x)

                    # x1_old = torch.Tensor([]).to(device)
                    # x2_old = torch.Tensor([]).to(device)
                    # indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    # for ind in indices:
                    #     x1_old = torch.cat((x1_old, transform(memory[ind:ind+1])), dim=0)
                    #     x2_old = torch.cat((x2_old, transform_prime(memory[ind:ind+1])), dim=0)
                    # x1_old, x2_old = x1_old.to(device), x2_old.to(device)


                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    memory_samples = memory[indices].to(device)
                    for ind in range(len(memory_samples)):
                        x1_old = torch.cat((x1_old, transform(memory_samples[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory_samples[ind:ind+1])), dim=0)
                    del memory_samples

                    x1 = torch.cat([x1, x1_old], dim=0)
                    x2 = torch.cat([x2, x2_old], dim=0)

                    del x1_old
                    del x2_old

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

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)#,device = device)
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer


def train_ering_barlow(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    epoch_counter = 0
    memory = torch.Tensor()#.to(device)
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0:
            init_lr = init_lr / 10
        
        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        # scheduler: adjut larnig rate: => https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py#L155
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
        cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:   
                optimizer.zero_grad()
                x1, x2 = x1.to(device), x2.to(device)

                if task_id > 0:                    
                    # indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    # x = memory[indices].to(device)
                    # xx1, xx2 = transform(x), transform_prime(x)

                    # x1_old = torch.Tensor([]).to(device)
                    # x2_old = torch.Tensor([]).to(device)
                    # indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    # for ind in indices:
                    #     x1_old = torch.cat((x1_old, transform(memory[ind:ind+1])), dim=0)
                    #     x2_old = torch.cat((x2_old, transform_prime(memory[ind:ind+1])), dim=0)
                    # x1_old, x2_old = x1_old.to(device), x2_old.to(device)

                    # x1 = torch.cat([x1, x1_old], dim=0)
                    # x2 = torch.cat([x2, x2_old], dim=0)

                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    memory_samples = memory[indices].to(device)
                    for ind in range(len(memory_samples)):
                        x1_old = torch.cat((x1_old, transform(memory_samples[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory_samples[ind:ind+1])), dim=0)
                    del memory_samples

                    x1 = torch.cat([x1, x1_old], dim=0)
                    x2 = torch.cat([x2, x2_old], dim=0)

                    del x1_old
                    del x2_old

                z1,z2 = model(x1, x2)
                loss =  cross_loss(z1, z2)
                epoch_loss.append(loss.item())
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
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)#,device = device)
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer



def info_nce_loss(features,args,device):

    labels = torch.cat([torch.arange(features.shape[0]/2) for i in range(2)], dim=0)#there are only two views
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train_ering_simclr(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    epoch_counter = 0
    memory = torch.Tensor()#.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0:
            init_lr = init_lr / 10
        
        optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)  
        # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:   
                optimizer.zero_grad()
                x1, x2 = x1.to(device), x2.to(device)

                if task_id > 0:                    
                    # indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    # x = memory[indices].to(device)
                    # xx1, xx2 = transform(x), transform_prime(x)

                    # x1_old = torch.Tensor([]).to(device)
                    # x2_old = torch.Tensor([]).to(device)
                    # indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    # for ind in indices:
                    #     x1_old = torch.cat((x1_old, transform(memory[ind:ind+1])), dim=0)
                    #     x2_old = torch.cat((x2_old, transform_prime(memory[ind:ind+1])), dim=0)
                    # x1_old, x2_old = x1_old.to(device), x2_old.to(device)

                    # x1 = torch.cat([x1, x1_old], dim=0)
                    # x2 = torch.cat([x2, x2_old], dim=0)

                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    indices = np.random.choice(len(memory), size=min(32*task_id, len(memory)), replace=False)
                    memory_samples = memory[indices].to(device)
                    for ind in range(len(memory_samples)):
                        x1_old = torch.cat((x1_old, transform(memory_samples[ind:ind+1])), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory_samples[ind:ind+1])), dim=0)
                    del memory_samples

                    x1 = torch.cat([x1, x1_old], dim=0)
                    x2 = torch.cat([x2, x2_old], dim=0)

                    del x1_old
                    del x2_old

                z1,z2 = model(x1, x2)
                features = torch.cat((z1,z2),dim=0)
                logits, labels = info_nce_loss(features,args,device)
                loss =  criterion(logits, labels)
                
                epoch_loss.append(loss.item())
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
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize)#, device)
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_lambdap_' + str(args.lambdap) + '_lambda_norm_' + str(args.lambda_norm) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)

    return model, loss_, optimizer

def collect_params(model, exclude_bias_and_bn=True):
    param_list = []
    for name, param in model.named_parameters():
        if exclude_bias_and_bn and any(
            s in name for s in ['bn', 'downsample.1', 'bias']):
            param_dict = {
                'params': param,
                'weight_decay': 0.,
                'lars_exclude': True}
            # NOTE: with the current pytorch lightning bolts
            # implementation it is not possible to exclude 
            # parameters from the LARS adaptation
        else:
            param_dict = {'params': param}
        param_list.append(param_dict)
    return param_list

# def loss_func(x, y):
#    # L2 normalization
#    x = F.normalize(x, dim=-1, p=2)
#    y = F.normalize(y, dim=-1, p=2)
#    return 2 - 2 * (x * y).sum(dim=-1)

def loss_func(p, z):
   # L2 normalization
   p = F.normalize(p, dim=-1, p=2)
   z = F.normalize(z, dim=-1, p=2)
   return 2 - 2 * (p * z.detach()).sum(dim=1).mean()

# def train_ering_byol(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
def train_ering_byol(model, train_data_loaders, knn_train_data_loaders, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime):
    epoch_counter = 0
    init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256
    model.initialize_EMA(0.99, 1.0, len(train_data_loaders[0])*sum(args.epochs))
    step_number = 0
    
    memory = torch.Tensor().to(device)
    # model.initialize_EMA(0.99, 1.0, len(train_data_loaders[0])*sum(args.epochs))
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        model.task_id = task_id
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256
        model_parameters = collect_params(model)
        # init_lr = args.pretrain_base_lr
        if task_id != 0 and args.same_lr != True:
            init_lr = init_lr / 10

        optimizer = LARS(model_parameters,lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id])    
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.pretrain_warmup_epochs , max_epochs=args.epochs[task_id],warmup_start_lr=args.min_lr,eta_min=args.min_lr) 
        # model.initialize_EMA(0.99, 1.0, len(loader)*args.epochs[task_id])
       
        loss_ = []
        
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss = []
            for x1, x2, y in loader:
                # print(y)
                x1, x2 = x1.to(device), x2.to(device)

                if task_id > 0:                    
                    x1_old = torch.Tensor([]).to(device)
                    x2_old = torch.Tensor([]).to(device)
                    indices = np.random.choice(len(memory), size=min(args.bsize, len(memory)), replace=False)
                    for ind in indices:
                        x1_old = torch.cat((x1_old, transform(memory[ind:ind+1].squeeze()).unsqueeze(0).to(device)), dim=0)
                        x2_old = torch.cat((x2_old, transform_prime(memory[ind:ind+1].squeeze()).unsqueeze(0).to(device)), dim=0)
                    x1_old, x2_old = x1_old.to(device), x2_old.to(device)

                    x1 = torch.cat([x1, x1_old], dim=0)
                    x2 = torch.cat([x2, x2_old], dim=0)

                f1 = model.encoder.backbone(x1).squeeze() # NxC
                f2 = model.encoder.backbone(x2).squeeze() # NxC
                z1 = model.encoder.projector(f1) # NxC
                z2 = model.encoder.projector(f2) # NxC

                p1 = model.predictor(z1) # NxC
                p2 = model.predictor(z2) # NxC   

                with torch.no_grad():
                    target_z1 = model.teacher_model(x1)
                    target_z2 = model.teacher_model(x2)

                loss_one = loss_func(p1, target_z2.detach())
                loss_two = loss_func(p2, target_z1.detach())
                loss = 0.5*loss_one + 0.5*loss_two
                loss = loss.mean()

                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_number += 1 
                model.update_moving_average(step_number)
                # ema_model = ema.update_model_average(ema_model, model)

            if args.is_debug:
                break

            scheduler.step()
            epoch_counter += 1
            loss_.append(np.mean(epoch_loss))
            end = time.time()
            print('epoch end')
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Loss: {np.mean(epoch_loss):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss), " Epoch ": epoch_counter})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})
            
        memory = update_memory(memory, train_data_loaders_pure[task_id], args.msize, device)
        file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'

        # save your encoder network
        torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'encoder': model.encoder.backbone.state_dict(),
                    }, file_name)
        
        # if task_id < len(train_data_loaders)-1:
        #     lin_epoch = 1
        #     num_class = np.sum(args.class_split[:task_id+1])
        #     classifier = LinearClassifier(num_classes = num_class).to(device)
        #     lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        #     lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
        #     linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  



    return model, loss_, optimizer