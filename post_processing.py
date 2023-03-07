import os
import sys
import wandb
import argparse
import numpy as np

import torch
import torchvision.transforms as T

from dataloaders.dataloader_cifar10 import get_cifar10
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot
from models.linear_classifer import LinearClassifier
from models.simsiam import Encoder, Predictor, SimSiam
from trainers.train import train
from torchsummary import summary


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


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
    parser.add_argument('--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cuda_device', type=int, default=5, metavar='N',
                        help='device id')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--knn_report_freq', type=int, default=10)

    parser.add_argument('--pretrained_dir', type=str, default=0, metavar='N',
                        help='directory where model is saved')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='num of workers')
    parser.add_argument('-cs', '--class_split', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    assert sum(args.class_split) == 10
    num_worker = int(8/len(args.class_split))
    if len(args.class_split) == 10:
        num_worker = 2
    wandb.init(project="SSL Project", 
                # mode="disabled",
                config=args,
                name="SimSiampostprocessing" + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split))

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

    #Class Based
    train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders = get_cifar10(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)
    train_data_loaders_all, train_data_loaders_knn_all, test_data_loaders_all, validation_data_loaders_all = get_cifar10(transform, transform_prime, \
                                        classes=[10], valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)

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

    # Load Previous Message
    if os.path.isfile(args.pretrained_dir):
        print("=> loading checkpoint '{}'".format(args.pretrained_dir))
        checkpoint = torch.load(args.pretrained_dir, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        lin_dict = checkpoint['classifier']

        model.load_state_dict(state_dict)
        classifier = LinearClassifier().to(device)
        classifier.load_state_dict(lin_dict)

        print("Starting T-SNE Plot..")
        for i in range(len(test_data_loaders)):
            get_t_SNE_plot(test_data_loaders[i], model, classifier, device, log_message=str(i), class_count=args.class_split[i])

    else:
        print(' The pretrained model does not exist ')