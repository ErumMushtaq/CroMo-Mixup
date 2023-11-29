
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
import wandb
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def Knn_Validation_cont(encoder,train_data_loaders,test_data_loaders,device=None, K = 200,sigma = 0.1):#sigma is for
    """Extract features from validation split and search on train split features."""
    encoder.eval()
    encoder.to(device)
    # torch.cuda.empty_cache() #https://discuss.pytorch.org/t/what-is-torch-cuda-empty-cache-do-and-where-should-i-add-it/40975
    train_features_all = []
    train_labels_all = []
    with torch.no_grad():       
        for loader in train_data_loaders:
            train_features = []
            train_labels = []
            for batch_idx, (inputs, t_label) in enumerate(loader):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)

                # forward
                features = encoder(inputs).squeeze()
                features = nn.functional.normalize(features)
                train_features.append(features.data.t())
                train_labels.append(t_label.to(device))

            #train_labels = torch.LongTensor(train_data_loader.dataset.tensors[1]).cuda()
            train_features = torch.cat(train_features,dim = 1)
            train_labels = torch.cat(train_labels)
            train_features_all.append(train_features)
            train_labels_all.append(train_labels)

    total = 0
    correct = 0
    C = train_labels_all[-1].max() + 1
    # print(C)
    top1 = 0
    task_acc=[]
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).to(device)
        for task_id, loader in enumerate(test_data_loaders):
            tot = 0
            corr=0
            t1 = 0
            for _, (inputs, targets) in enumerate(loader):
                targets = targets.to(device)
                batch_size = inputs.size(0)
                features = encoder(inputs.to(device))

                #Task-wise result
                dist = torch.mm(features, train_features_all[task_id])
                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                
                candidates = train_labels_all[task_id].view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batch_size * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1 , C), yd_transform.view(batch_size, -1, 1)), 1)
                _, predictions = probs.sort(1, True)
            
                tot += targets.size(0)
                corr = predictions.eq(targets.data.view(-1,1))
                t1 = t1 + corr.narrow(1,0,1).sum().item()

                #All-Task result
                dist = torch.mm(features, torch.cat(train_features_all,dim = 1))
                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                
                candidates = torch.cat(train_labels_all).view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batch_size * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1 , C), yd_transform.view(batch_size, -1, 1)), 1)
                _, predictions = probs.sort(1, True)
            
                total += targets.size(0)
                correct = predictions.eq(targets.data.view(-1,1))
                top1 = top1 + correct.narrow(1,0,1).sum().item()
            task_acc.append(t1/tot)   
    top1 = top1 / total

    return top1, task_acc

def Knn_Validation(encoder,train_data_loader,validation_data_loader,device=None, K = 200,sigma = 0.1):#sigma is for
    """Extract features from validation split and search on train split features."""
    encoder.eval()
    encoder.to(device)
    # torch.cuda.empty_cache() #https://discuss.pytorch.org/t/what-is-torch-cuda-empty-cache-do-and-where-should-i-add-it/40975
    train_features = []
    train_labels = []
    total = 0
    with torch.no_grad():       
        for batch_idx, (inputs, t_label) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # forward
            features = encoder(inputs)
            features = nn.functional.normalize(features)
            train_features.append(features.data.t())
            train_labels.append(t_label.to(device))
            total += batch_size

        #train_labels = torch.LongTensor(train_data_loader.dataset.tensors[1]).cuda()
        train_features = torch.cat(train_features,dim = 1)
        train_labels = torch.cat(train_labels)

    total = 0
    correct = 0
    C = train_labels.max() + 1
    top1 = 0
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).to(device)
        for batch_idx, (inputs, targets) in enumerate(validation_data_loader):
            targets = targets.to(device)
            # targets = targets.to(device)(non_blocking=True)
            batch_size = inputs.size(0)
            features = encoder(inputs.to(device))

            dist = torch.mm(features, train_features)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batch_size * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1 , C), yd_transform.view(batch_size, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
           
            #retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            total += targets.size(0)
            correct = predictions.eq(targets.data.view(-1,1))
            top1 = top1 + correct.narrow(1,0,1).sum().item()
            #correct += retrieval.eq(targets.data).sum().item()
    top1 = top1 / total

    return top1


def correct_top_k(outputs, targets, top_k=(1,5)):
    """
    Find number of correct predictions for one batch.
    Args:
        outputs (torch.Tensor): Nx(class_number) Tensor containing logits.
        targets (torch.Tensor): N Tensor containing ground truths.
        top_k (Tuple): checking the ground truth is included in top-k prediction.
    Returns:
        List: List of number of top-1 and top-5 correct predictions.
    """
    with torch.no_grad():
        prediction = torch.argsort(outputs, dim=-1, descending=True)
        result= []
        for k in top_k:
            correct_k = torch.sum((prediction[:, 0:k] == targets.unsqueeze(dim=-1)).any(dim=-1).float()).item() 
            result.append(correct_k)
        return result

def linear_test(net, data_loader, classifier, epoch, device):
    # evaluate model:
    net.eval() # for not update batchnorm
    linear_loss = 0.0
    num = 0
    total_loss, total_correct_1, total_correct_5, total_num, test_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data_tuple in test_bar:
            data, target = [t.to(device) for t in data_tuple]

            # Forward prop of the model with single augmented batch
            output = net(data)

            # Logits by classifier
            if classifier is not None:  #else net is already a classifier
                output = classifier(output) 

            # Calculate Cross Entropy Loss for batch
            linear_loss = F.cross_entropy(output, target)
            
            # Batchsize for loss and accuracy
            num = data.size(0)
            total_num += num 
            
            # Accumulating loss 
            total_loss += linear_loss.item() * num 
            # Accumulating number of correct predictions 
            correct_top_1, correct_top_5 = correct_top_k(output, target, top_k=(1,5))    
            total_correct_1 += correct_top_1
            total_correct_5 += correct_top_5

            test_bar.set_description('Lin.Test Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% '
                                     .format(epoch,  total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100
                                             ))
        acc_1 = total_correct_1/total_num*100
        acc_5 = total_correct_5/total_num*100
        wandb.log({" Linear Layer Test Loss ": linear_loss / total_num, " Epoch ": epoch})
        wandb.log({" Linear Layer Test - Acc": acc_1, " Epoch ": epoch})
    return total_loss / total_num, acc_1 , acc_5 

def linear_train(net, data_loader, train_optimizer, classifier, scheduler, epoch, device):

    net.eval() # for not update batchnorm 
    total_num, train_bar = 0, tqdm(data_loader)
    linear_loss = 0.0
    total_correct_1, total_correct_5 = 0.0, 0.0
    for data_tuple in train_bar:
        # Forward prop of the model with single augmented batch
        pos_1, target = data_tuple
        pos_1 = pos_1.to(device)
        feature_1 = net(pos_1)
        # feature_1 = net.get_representation(pos_1) 

        # Batchsize
        batchsize_bc = feature_1.shape[0]
        features = feature_1
        targets = target.to(device)


        # Classifier with detach(for stop gradient to model)
        # Actually no need due to we use lin_optimizer seperate
        logits = classifier(features.detach()) 
        # Cross Entropy Loss 
        linear_loss_1 = F.cross_entropy(logits, targets)

        # Number of correct predictions
        linear_correct_1, linear_correct_5 = correct_top_k(logits, targets, top_k=(1, 5))
    

        # Backpropagation part
        train_optimizer.zero_grad()
        linear_loss_1.backward()
        train_optimizer.step()

        # Accumulating number of examples, losses and correct predictions
        total_num += batchsize_bc
        linear_loss += linear_loss_1.item() * batchsize_bc
        total_correct_1 += linear_correct_1 
        total_correct_5 += linear_correct_5


        acc_1 = total_correct_1/total_num*100
        # # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f} ACC: {:.2f}'.format(\
                epoch, linear_loss / total_num, acc_1))
    scheduler.step()
    acc_1 = total_correct_1/total_num*100
    acc_5 = total_correct_5/total_num*100       
    wandb.log({" Linear Layer Train Loss ": linear_loss / total_num, " Epoch ": epoch})
    wandb.log({" Linear Layer Train - Acc": acc_1, " Epoch ": epoch})
    # print(f'Linear Layer Train - Acc: {acc_1}')
        
    return linear_loss/total_num, acc_1, acc_5


def linear_evaluation(net, data_loader,test_data_loader,train_optimizer,classifier, scheduler, epochs, device):
    for epoch in range(1, epochs+1):
        linear_loss, linear_acc1, linear_acc5 = linear_train(net,data_loader,train_optimizer,classifier,scheduler, epoch, device)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1, test_acc5 = linear_test(net, test_data_loader, classifier, epoch, device)

    return test_loss, test_acc1, test_acc5, classifier


def gen_features(test_data_loader, net, classifier, device):
    net.eval()
    targets_list = []
    outputs_list = []

    #TODO: check if transform needed here?

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            feature = net(inputs)

            # Logits by classifier
            #outputs = classifier(feature) 
            #outputs_np = outputs.data.cpu().numpy()
            outputs_np = feature.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(test_data_loader)):
                print(idx+1, '/', len(test_data_loader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(targets, outputs, log_message, class_count):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    print(targets)
    plt.figure()
    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", class_count), #10
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    wandb.log({" t-SNE Plot "+log_message: [wandb.Image(plt)]})
    print('done!')

def get_t_SNE_plot(test_data_loader, encoder, classifier, device, log_message='global', class_count=10):
    #Code ref: https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py
    targets, outputs = gen_features(test_data_loader, encoder, classifier, device)
    tsne_plot(targets, outputs, log_message=log_message, class_count=class_count)


def linear_evaluation_task_confusion(model, classifier, test_data_loaders, args, device):
    total_correct_1, total_correct_5 = 0.0, 0.0
    total_num = 0.0
    total_correct_wp = 0.0
    # target_copy = target.copy()
    total_num_task_correct = 0.0
    for task_id in range(len(args.class_split)):
        for x, target in test_data_loaders[task_id]:
            x, target = x.to(device), target.to(device)
            features = model(x)
            logits = classifier(features.detach()) 
            task_probs = torch.nn.functional.softmax(logits, dim = 1)
            prediction_tp = torch.argmax(task_probs, dim = 1)
            prediction_wp = torch.argmax(task_probs, dim = 1)

            num = x.shape[0]
            total_num += num 

            # check TP:
            for i in range(len(args.class_split)):  #Map predictions to task ids          
                if i ==0:
                    prediction_tp[prediction_tp<args.class_split[i]] = i
                else:
                    prediction_tp[(prediction_tp>=np.sum(args.class_split[:i])) & (prediction_tp<np.sum(args.class_split[:i+1]))] = i

            correct_top_1 = torch.sum((prediction_tp == task_id).float()).item()
            total_num_task_correct += correct_top_1
            # Check WP for the correct TP instances

            correct_top_wp = torch.sum((prediction_wp[prediction_tp == task_id] == target[prediction_tp == task_id]).float()).item() #out of correct TP, how many are correct wp
            total_correct_wp +=  correct_top_wp 


            print(" Total Test samples "+str(num)+" of Task "+str(task_id)+": Correct TP: "+str(correct_top_1)+": Correct WP: "+str(correct_top_wp ))
            assert correct_top_wp<=correct_top_1
            assert total_num_task_correct<=total_num


    wp = total_correct_wp/total_num_task_correct
    tp = total_num_task_correct/total_num
    wandb.log({" Linear Layer Test - TP Acc": tp})
    wandb.log({" Linear Layer Test - WP Acc": wp})
    return wp, tp

# def linear_evaluation_WP(model, classifier, test_data_loaders, args, device):
#     total_correct_1, total_correct_5 = 0.0, 0.0
#     total_num = 0.0
#     for task_id in range(len(args.class_split)):
#         for x, target in test_data_loaders[task_id]:
#             x, target = x.to(device), target.to(device)
#             features = model(x)
#             logits = classifier(features.detach()) #[B, C]
#             if task_id == 4:
#                 print(target)
#             #Map to tasks
#             if task_id == 0:
#                 task_logits = logits[:,:args.class_split[task_id]]
#             else:
#                 task_logits = logits[:,np.sum(args.class_split[:task_id]):np.sum(args.class_split[:task_id+1])] #pick k logits only
#             for i in range(args.class_split[task_id]): #map target classes to 0 to k where k = cs of that task
#                 target[target== task_id*args.class_split[task_id]+i] = i
            
#             if task_id == 4:
#                 print(target)
#             task_probs = torch.nn.functional.softmax(task_logits, dim = 1) #[B, C]
#             prediction = torch.argmax(task_probs, dim = 1)
#             if task_id == 4:
#                 print(prediction)
#             correct_top_1 = torch.sum((prediction == target).float()).item()
#             total_correct_1 +=  correct_top_1 
#             num = x.shape[0]
#             total_num += num 
#     acc_1 = total_correct_1/total_num
#     wandb.log({" Linear Layer Test - WP Acc": acc_1, " Epoch ": 1})
#     print("Linear Layer Test - WP Acc "+str(acc_1))

#     return acc_1

# def linear_evaluation_TP(model, classifier, test_data_loaders, args, device):
#     total_correct_1, total_correct_5 = 0.0, 0.0
#     total_num = 0.0
#     for task_id in range(len(args.class_split)):
#         print(task_id)
#         for x, target in test_data_loaders[task_id]:
#             x, target = x.to(device), target.to(device)
            
#             features = model(x)
#             logits = classifier(features.detach()) 
#             task_probs = torch.nn.functional.softmax(logits, dim = 1)
#             prediction = torch.argmax(task_probs, dim = 1)

#             for i in range(len(args.class_split)):  #Map targets to task ids          
#                 if i ==0:
#                     target[target<args.class_split[i]] = i
#                     prediction[prediction<args.class_split[i]] = i
#                 else:
#                     target[(target>=np.sum(args.class_split[:i])) & (target<np.sum(args.class_split[:i+1]))] = i
#                     prediction[(prediction>=np.sum(args.class_split[:i])) & (prediction<np.sum(args.class_split[:i+1]))] = i


#             correct_top_1 = torch.sum((prediction == target).float()).item()
#             total_correct_1 += correct_top_1
#             num = x.shape[0]
#             total_num += num 

#             if task_id == 0:
#                 task_logits = logits[:,:args.class_split[task_id]]
#             else:
#                 task_logits = logits[:,np.sum(args.class_split[:task_id]):np.sum(args.class_split[:task_id+1])] #pick k logits only
#             for i in range(args.class_split[task_id]): #map target classes to 0 to k where k = cs of that task
#                 target_copy[target== task_id*args.class_split[task_id]+i] = i


#     acc_1 = total_correct_1/total_num
#     wandb.log({" Linear Layer Test - TP Acc": acc_1, " Epoch ": 1})
#     print("Linear Layer Test - TP Acc "+str(acc_1))

    # return acc_1



            

