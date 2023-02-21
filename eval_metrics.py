
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
import wandb


def Knn_Validation(encoder,train_data_loader,validation_data_loader,device=None, K = 200,sigma = 0.1,kn_predict=False):#sigma is for
    data_normalize_mean = (0.4914, 0.4822, 0.4465)
    data_normalize_std = (0.247, 0.243, 0.261)
    random_crop_size = 32
    transform = transforms.Compose(
            [   
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ])
    """Extract features from validation split and search on train split features."""
    encoder.eval()
    encoder.to(device)
    torch.cuda.empty_cache()
    train_features = []
    train_labels = []
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, t_label) in enumerate(train_data_loader):
            inputs = transform(inputs) # normalize
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # forward
            features = encoder(inputs)
            features = nn.functional.normalize(features)
            train_features.append(features.data.t())
            train_labels.append(t_label.cuda())
            total += batch_size

        #train_labels = torch.LongTensor(train_data_loader.dataset.tensors[1]).cuda()
        train_features = torch.cat(train_features,dim = 1)
        train_labels = torch.cat(train_labels)

    total = 0
    correct = 0
    C = train_labels.max() + 1
    top1 = 0
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(validation_data_loader):
            targets = targets.cuda(non_blocking=True)
            batch_size = inputs.size(0)
            inputs = transform(inputs)
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

def linear_test(net, data_loader, classifier, epoch):
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
    net.eval() # for not update batchnorm
    linear_loss = 0.0
    num = 0
    total_loss, total_correct_1, total_correct_5, total_num, test_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with torch.no_grad():
        for data_tuple in test_bar:
            data, target = [t.cuda() for t in data_tuple]
            data = transform(data)

            # Forward prop of the model with single augmented batch
            # feature = net.get_representation(data) 
            feature = net(data)

            # Logits by classifier
            output = classifier(feature) 

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



def linear_train(net, data_loader, train_optimizer, classifier, scheduler, epoch):
    data_normalize_mean = (0.4914, 0.4822, 0.4465)
    data_normalize_std = (0.247, 0.243, 0.261)
    random_crop_size = 32
    transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(random_crop_size), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(data_normalize_mean, data_normalize_std),
            ])

    net.eval() # for not update batchnorm 
    total_num, train_bar = 0, tqdm(data_loader)
    linear_loss = 0.0
    total_correct_1, total_correct_5 = 0.0, 0.0
    for data_tuple in train_bar:
        # Forward prop of the model with single augmented batch
        pos_1, target = data_tuple
        pos_1 = pos_1.cuda()
        pos_1 = transform(pos_1)
        feature_1 = net(pos_1)
        # feature_1 = net.get_representation(pos_1) 

        # Batchsize
        batchsize_bc = feature_1.shape[0]
        features = feature_1
        targets = target.cuda()


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



        # # This bar is used for live tracking on command line (batch_size -> batchsize_bc: to show current batchsize )
        train_bar.set_description('Lin.Train Epoch: [{}] Loss: {:.4f} '.format(\
                epoch, linear_loss / total_num))
    scheduler.step()
    acc_1 = total_correct_1/total_num*100
    acc_5 = total_correct_5/total_num*100       
    wandb.log({" Linear Layer Train Loss ": linear_loss / total_num, " Epoch ": epoch})
    wandb.log({" Linear Layer Train - Acc": acc_1, " Epoch ": epoch})
        
    return linear_loss/total_num, acc_1, acc_5


def linear_evaluation(net, data_loader,test_data_loader,train_optimizer,classifier, scheduler, epochs):
    for epoch in range(1, epochs+1):
        linear_loss, linear_acc1, linear_acc5 = linear_train(net,data_loader,train_optimizer,classifier,scheduler, epoch)
        with torch.no_grad():
            # Testing for linear evaluation
            test_loss, test_acc1, test_acc5 = linear_test(net, test_data_loader, classifier, epoch)

    return test_loss, test_acc1, test_acc5