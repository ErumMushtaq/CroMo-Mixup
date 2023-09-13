import time
import wandb
import torch
import pickle 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from utils.eval_metrics import Knn_Validation_cont
from loss import BarlowTwinsLoss
from utils.lars import LARS

from models.linear_classifer import LinearClassifier
from torch.utils.data import DataLoader
from dataloaders.dataset import TensorDataset
from tqdm import tqdm

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def correct_top_k(outputs, targets, top_k=(1,5)):
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result

def linear_test(net, data_loader, classifier, epoch, device, task_num):
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

def linear_train(net, data_loader, train_optimizer, classifier, scheduler, epoch, device, task_num):

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


def linear_evaluation(net, data_loaders,test_data_loaders,train_optimizer,classifier, scheduler, epochs, device, task_num):
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
        linear_train(net,data_loader,train_optimizer,classifier,scheduler, epoch, device, task_num)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1 = linear_test(net, test_data_loader, classifier, epoch, device, task_num)

    return test_loss, test_acc1, classifier


def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')[:-1]
     return reduce(getattr, names, module)


def activation_collection(model, loader, device, orth_set):
    start = time.time()
    activation = {}
    def getActivation(id):
        # the hook signature
        def hook(model, input, output):
            activation[id].append(input[0].detach())
        return hook

    hooks = []
    for name, _ in model.encoder.backbone.named_parameters():
        if "weight" in name:
            module = get_module_by_name(model.encoder.backbone, name)
            if isinstance(module, torch.nn.Conv2d):
                activation[name] = []
                hooks.append(module.register_forward_hook(getActivation(name)))

    model.eval()
    for batch_index, (x, _) in enumerate(loader):
        x=x.to(device)
        _ = model.encoder.backbone(x)
        if batch_index > len(loader)/10-1:
            break
            
    for hook in hooks:
        hook.remove()

    for name in activation.keys():
        activation[name] = torch.cat(activation[name],dim=0)
        if "shortcut" not in name:
            activation[name] = F.pad(activation[name], (1, 1, 1, 1), "constant", 0)

    for name in activation.keys():
        module = get_module_by_name(model.encoder.backbone, name)
        unfolder = torch.nn.Unfold(module.kernel_size[0], dilation=1, padding=0, stride= module.stride[0])
        act = activation[name]
        mat = unfolder(act.to(device))
        mat = mat.permute(0,2,1)
        mat = mat.reshape(-1, mat.shape[2])
        mat = mat.T
    
        if orth_set[name] is not None:
            U = orth_set[name].to(device)
            projected = U @ U.T @ mat
            remaining = mat - projected
            activation[name] = remaining.cpu()
        else:
            activation[name] = mat.cpu()
    end = time.time()
    print(f'Activations collection time {end-start}')
    return activation 

def expand_orth_set(activations, orth_set, args, device):
    for key in activations.keys():
        if orth_set[key] == None:
            projected = torch.zeros(1)
        else:
            projected = orth_set[key]  @ orth_set[key].T @ activations[key] 

        remaining = (activations[key] - projected).to(device)
        remaining = remaining@remaining.T
        #find svds of remaining
        U, S, _ = torch.svd(remaining.cpu())
        #find how many singular vectors will be used
        total = torch.norm(activations[key])**2
        proj_norm = torch.norm(projected)**2
        for i in range(len(S)):
            hand = proj_norm + torch.sum(S[0:i+1])
            if i == 0 and hand / total > args.epsilon:
                break
            elif hand / total > args.epsilon:
                break
            
        print(U[:,0:i+1].shape)
        if orth_set[key] == None:
            orth_set[key] = U[:,0:i+1].cpu()
        else:
            orth_set[key] = torch.cat((orth_set[key], U[:,0:i+1]),dim=1).cpu()
            orth_set[key], _ = torch.qr(orth_set[key])

def train_gpm_barlow(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, train_data_loaders_linear, device, args):
    orth_set = {}
    layers = []
    for name, param in model.encoder.backbone.named_parameters():
        if "weight" in name:
            module = get_module_by_name(model.encoder.backbone, name)
            if isinstance(module, torch.nn.Conv2d):
                orth_set[name] = None
                layers.append(name)

    epoch_counter = 0
    for task_id, loader in enumerate(train_data_loaders):

        if task_id == 0 and args.start_chkpt == 1:
            model.temporal_projector = nn.Sequential(
                nn.Linear(args.proj_out, args.proj_hidden, bias=False),
                nn.BatchNorm1d(args.proj_hidden),
                nn.ReLU(),
                nn.Linear(args.proj_hidden, args.proj_out),
            ).to(device)
            
            model_path = "./checkpoints/checkpoint_cifar100-algocassle_barlow-e[500, 500, 500, 500, 500]-b256-lr0.3-CS[20, 20, 20, 20, 20]_task_0_same_lr_True_norm_batch_ws_False_first_task_chkpt.pth.tar"
            model.load_state_dict(torch.load(model_path)['state_dict'])
            model.task_id = task_id
            epoch_counter = 500
        else:        

            for _, module in model.encoder.backbone.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.weight.requires_grad=False
                    module.bias.requires_grad=False
                    module.track_running_stats = False

            init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
            
            optimizer = LARS(model.parameters(),lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay, eta=0.02, clip_lr=True, exclude_bias_n_norm=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id]) #eta_min=2e-4 is removed scheduler + values ref: infomax paper
            cross_loss = BarlowTwinsLoss(lambda_param= args.lambda_param, scale_loss =args.scale_loss)

            loss_ = []
            for epoch in range(args.epochs[task_id]):
                start = time.time()
                model.train()
                model.encoder.backbone.eval()
                model.encoder.projector.train()
                epoch_loss = []
                for x1, x2, y in loader:   
                    z1,z2 = model(x1, x2)
                    loss =  cross_loss(z1, z2)
                    epoch_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for key,p in model.encoder.backbone.named_parameters():
                            if key in layers:
                                if orth_set[key] == None:
                                    if p.grad != None:
                                        print(key)
                                    continue
                                grad = p.grad.data
                                projected = orth_set[key].to(device) @ orth_set[key].T.to(device) @ grad.view(grad.size(0), -1).T
                                p.grad.data = grad - projected.T.view(grad.size())
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

            file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_norm_' + str(args.normalization) + '_ws_' + str(args.weight_standard) + '.pth.tar'
            # save your encoder network
            torch.save({
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'encoder': model.encoder.backbone.state_dict(),
                        }, file_name)

        activations = activation_collection(model, train_data_loaders_linear[task_id], device, orth_set)
        expand_orth_set(activations, orth_set, args, device)

        filename = './checkpoints/orthset_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr) + "-CS" + str(args.class_split) + '_task_' + str(task_id) + '_same_lr_' + str(args.same_lr) + '_eps_' + str(args.epsilon) + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(orth_set, f)

        if task_id < len(train_data_loaders)-1 and not (task_id == 0 and args.start_chkpt == 1):
            lin_epoch = 100
            num_class = np.sum(args.class_split[:task_id+1])
            classifier = LinearClassifier(num_classes = num_class).to(device)
            lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
            lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
            linear_evaluation(model, train_data_loaders_linear[:task_id+1], test_data_loaders[:task_id+1], lin_optimizer,classifier, lin_scheduler, lin_epoch, device, task_id)  

    return model, loss_, optimizer

