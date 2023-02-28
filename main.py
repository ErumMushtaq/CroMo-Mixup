import os
import sys
import wandb
import argparse
import numpy as np

import torch
import torchvision.transforms as T

from dataloaders.dataloader_cifar10 import get_cifar10
from utils.eval_metrics import linear_evaluation
from models.linear_classifer import LinearClassifier
from models.simsiam import Encoder, Predictor, SimSiam
from trainers.train import train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(5)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_warmup_epochs', type=int, default=0)
    parser.add_argument('--pretrain_warmup_lr', type=float, default=0)
    parser.add_argument('--pretrain_base_lr', type=float, default=0.06)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--knn_report_freq', type=int, default=10)

    parser.add_argument('--cuda_device', type=int, default=0, metavar='N',
                        help='device id')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    #device
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")

    #wandb init
    wandb.init(project="SSL Project", 
                mode="disabled",
                config=args,
                name="SimSiam" + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr))

    #augmentations
    transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

    transform_prime = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

    #Dataloaders
    print("Creating Dataloaders..")
    train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders = get_cifar10(transform, transform_prime, \
                                        classes=[10], valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0)

    #Create Model
    print("Creating Model..")
    proj_hidden = 2048
    proj_out = 2048
    pred_hidden = 512
    pred_out = 2048
    encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out)
    predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
    model = SimSiam(encoder, predictor)
    model.to(device) #automatically detects from model

    #Training
    print("Starting Training..")
    model, loss, optimizer = train(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, device, args)

    #Test Linear classification acc
    print("Starting Classifier Training..")
    lin_epoch = 2
    classifier = LinearClassifier().to(device)
    lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.1, momentum=0.9) # Infomax: no weight decay, epoch 100, cosine scheduler
    lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=2e-4) #scheduler + values ref: infomax paper
    test_loss, test_acc1, test_acc5 = linear_evaluation(model, train_data_loaders_knn[0],test_data_loaders[0],lin_optimizer, classifier, lin_scheduler, epochs=lin_epoch)

    # save your encoder network
    save_checkpoint({
                    'epoch': args.epochs + 1,
                    'arch': 'resnet18',
                    'lr': args.pretrain_base_lr,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss': loss,
                }, is_best=False, filename='./checkpoints/checkpoint_{:04f}.pth.tar'.format(args.pretrain_base_lr))




