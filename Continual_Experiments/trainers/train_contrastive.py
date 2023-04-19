import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from utils.lr_schedulers import LinearWarmupCosineAnnealingLR, SimSiamScheduler
from utils.eval_metrics import Knn_Validation_cont
from loss import invariance_loss,CovarianceLoss
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import time

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

def get_mean_vectors(model, data_loader, device, n_centers=20):
    data_normalize_mean = (0.4914, 0.4822, 0.4465)
    data_normalize_std = (0.247, 0.243, 0.261)
    random_crop_size = 32
    transform = transforms.Compose(
            [   
                transforms.Resize(int(random_crop_size*(8/7))), # In Imagenet: 224 -> 256 
                transforms.CenterCrop(random_crop_size),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ])
    # evaluate model:
    model.eval() 
    vectors = []
    with torch.no_grad():
        for x, y in data_loader: 
            x = x.to(device)
            out = model(x,projector = True)
            vectors.append(out.cpu().numpy())

    vectors = np.concatenate(vectors,axis = 0)
    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(vectors)
    centers = torch.tensor(kmeans.cluster_centers_)
    return centers


def contrastive_loss(z, centers):
    z = F.normalize(z, dim=-1, p=2)
    centers = F.normalize(centers, dim=-1, p=2)
    return (z.unsqueeze(1) * centers.unsqueeze(0)).sum(dim=-1).max(axis=1)[0].mean()#we want to decrease similarity
        

def train_contrastive_simsiam(model, train_data_loaders, knn_train_data_loaders, test_data_loaders, device, args):
    epoch_counter = 0
    centers = None
    for task_id, loader in enumerate(train_data_loaders):
        # Optimizer and Scheduler
        init_lr = args.pretrain_base_lr*args.pretrain_batch_size/256.
        if task_id != 0:
            init_lr = init_lr / 10

            
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=args.pretrain_momentum, weight_decay= args.pretrain_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs[task_id])

        loss_ = []
        for epoch in range(args.epochs[task_id]):
            start = time.time()
            model.train()
            epoch_loss_similarity = []
            epoch_loss_contrastive = []
            for x1, x2, y in loader:   
                z1,z2,p1,p2 = model(x1, x2)


                if centers != None:
                    loss_contrastive = contrastive_loss(torch.cat((z1,z2)), centers.to(device))
                else:
                    loss_contrastive = torch.tensor(0)

                loss_one = loss_fn(p1, z2.detach())
                loss_two = loss_fn(p2, z1.detach())
                loss = 0.5*loss_one + 0.5*loss_two
                loss_similarity = loss.mean() 


                epoch_loss_similarity.append(loss_similarity.item())
                epoch_loss_contrastive.append(loss_contrastive.item())
                optimizer.zero_grad()
                if loss_contrastive.item() == 0:
                    loss = loss_similarity
                else:
                    loss =  (1-args.contrastive_ratio) * loss_similarity + args.contrastive_ratio * loss_contrastive
                loss.backward()
                optimizer.step()
            print('epoch finished') 
            epoch_counter += 1
            scheduler.step()
            loss_.append(np.mean(epoch_loss_similarity))
            end = time.time()
            if (epoch+1) % args.knn_report_freq == 0:
                knn_acc, task_acc_arr = Knn_Validation_cont(model, knn_train_data_loaders[:task_id+1], test_data_loaders[:task_id+1], device=device, K=200, sigma=0.5) 
                wandb.log({" Global Knn Accuracy ": knn_acc, " Epoch ": epoch_counter})
                for i, acc in enumerate(task_acc_arr):
                    wandb.log({" Knn Accuracy Task-"+str(i): acc, " Epoch ": epoch_counter})
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s | Similarity Loss: {np.mean(epoch_loss_similarity):.4f}  | Contrastive Loss: {np.mean(epoch_loss_contrastive):.4f}  | Knn:  {knn_acc*100:.2f}')
                print(task_acc_arr)
            else:
                print(f'Task {task_id:2d} | Epoch {epoch:3d} | Time:  {end-start:.1f}s  | Similarity Loss: {np.mean(epoch_loss_similarity):.4f} | Contrastive Loss: {np.mean(epoch_loss_contrastive):.4f} ')
        
            wandb.log({" Average Training Loss ": np.mean(epoch_loss_similarity), " Epoch ": epoch_counter, " Average Contrastive Loss ": np.mean(epoch_loss_contrastive)})  
            wandb.log({" lr ": optimizer.param_groups[0]['lr'], " Epoch ": epoch_counter})

        if centers == None:
            centers =  get_mean_vectors(model, knn_train_data_loaders[task_id], device, n_centers=20)
        else:
            center =  get_mean_vectors(model, knn_train_data_loaders[task_id], device, n_centers=20)
            centers = torch.cat((centers,center),dim=0)

    return model, loss_, optimizer