import os
import sys
import time
import wandb
import argparse

import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
from dataloaders.dataloader_cifar100_superclass import get_cifar100_superclass
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot
from models.linear_classifer import LinearClassifier
from models.simsiam import Encoder, Predictor, SimSiam, InfoMax, BarlowTwins
from trainers.train import train
from trainers.train_sup import train_sup
from trainers.train_concat import train_concate
from models.resnet import resnetc18
from models.resnet_org import resnetc18_bn
from transform import get_transform
import random
import torch.nn as nn

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

    parser.add_argument('-n','--normalization', type=str, default='batch', help='normalization method: batch, group or none')
    parser.add_argument('--weight_standard', action='store_true', default=True, help='weight standard for conv layers')

    # Training settings
    parser.add_argument('-bs','--pretrain_batch_size', type=int, default=512)
    parser.add_argument('-we','--pretrain_warmup_epochs', type=int, default=10)
    parser.add_argument('-wlr','--pretrain_warmup_lr', type=float, default=0.1)
    parser.add_argument('-lr','--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_param', type=float, default=5e-3)
    parser.add_argument('--la_mu', type=float, default=0.01)
    parser.add_argument('--la_R', type=float, default=0.01)

    parser.add_argument('-d','--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--knn_report_freq', type=int, default=5)
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=64)

    parser.add_argument('-gpu','--cuda_device', type=int, default=0, metavar='N', help='device id')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='num of workers')
    parser.add_argument('--algo', type=str, default='simsiam', help='ssl algorithm')
    parser.add_argument('--exp_type', type=str, default='basic',help='concat, basic')

    # Infomax Args
    parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    parser.add_argument('--sim_loss_weight', type=float, default=1000.0)
    parser.add_argument('--info_loss', type=str, default='invariance', help='infomax loss')
    parser.add_argument('--R_eps_weight', type=float, default=1e-8)

    parser.add_argument('-cs', '--class_split', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    num_worker = 2 if len(args.class_split) == 10 else int(8/len(args.class_split))

    if args.dataset == "cifar10":
        get_dataloaders = get_cifar10
        get_dataloaders_all = get_cifar10
        num_classes=10
    elif args.dataset == "cifar100sup":
        get_dataloaders = get_cifar100_superclass
        get_dataloaders_all = get_cifar100
        num_classes=100
    elif args.dataset == "cifar100":
        get_dataloaders = get_cifar100
        get_dataloaders_all = get_cifar100
        num_classes=100
    assert sum(args.class_split) == num_classes

    #device
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print(device)
    #wandb init
    wandb.init(project="CSSL", entity="yavuz-team",
                # mode="disabled",
                config=args,
                name=args.algo + "-" + args.dataset + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)
                +"-CS"+str(args.class_split) +'-'+args.normalization+'norm')

    #Dataloaders
    print("Creating Dataloaders..")
    batch_size = args.pretrain_batch_size
    transform, transform_prime = get_transform(args)
    train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, train_data_loaders_pure = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker)
    train_data_loaders_all, train_data_loaders_knn_all, test_data_loaders_all, _, train_data_loaders_linear_all, _ = get_dataloaders_all(transform, transform_prime, \
                                        classes=[num_classes], valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker)
    print(len(test_data_loaders_all))
    #Create Model
    if args.algo == 'simsiam':
        print("Creating Model for Simsiam..")
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        pred_hidden = 512
        pred_out = 2048
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard)
        predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
        model = SimSiam(encoder, predictor)
        model.to(device) #automatically detects from model
    if args.algo == 'infomax':
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard)
        model = InfoMax(encoder, project_dim=proj_out,device=device, la_mu=args.la_mu,la_R=args.la_R)
        model.to(device) #automatically detects from model
    if args.algo == 'barlowtwins':
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard)
        model = BarlowTwins(encoder, project_dim = proj_out, lambda_param = args.lambda_param, device=device)
        model.to(device) #automatically detects from model
    if args.algo == 'supervised':
        model = resnetc18(num_classes, normalization = args.normalization, weight_standard = args.weight_standard)
        model.to(device)

    #Training
    print("Starting Training..")
    if args.algo == 'supervised':
        model, loss, optimizer = train_sup(model, train_data_loaders, test_data_loaders_all[0], device, args)
        torch.save(model, "./checkpoints/"+str(time.time())+"model")
    else:
        if args.exp_type == 'basic':
            model, loss, optimizer = train(model, train_data_loaders, test_data_loaders_all[0], train_data_loaders_knn_all[0], train_data_loaders_knn, test_data_loaders, device, args)
        else:
            model, loss, optimizer = train_concate(model, train_data_loaders_all[0], test_data_loaders_all[0], train_data_loaders_knn_all[0], train_data_loaders_knn, test_data_loaders, device, args)

        torch.save(model, "./checkpoints/"+str(time.time())+"model")
        #Test Linear classification acc
        print("Starting Classifier Training..")
        lin_epoch = 100
        classifier = LinearClassifier(num_classes=num_classes).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
        test_loss, test_acc1, test_acc5, classifier = linear_evaluation(model, train_data_loaders_knn_all[0],test_data_loaders_all[0],lin_optimizer, classifier, lin_scheduler, epochs=lin_epoch, device=device) 

        #T-SNE Plot
        # print("Starting T-SNE Plot..")
        # get_t_SNE_plot(test_data_loaders_all[0], model, classifier, device)


        # save your encoder network
        save_checkpoint({
                        'epoch': args.epochs + 1,
                        'arch': 'resnet18',
                        'lr': args.pretrain_base_lr,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'loss': loss,
                        'encoder': model.encoder.backbone.state_dict(),
                        'classifier': classifier.state_dict(),
                    }, is_best=False, filename='./checkpoints/checkpoint_{:04f}_cs_{}_bs_{}.pth.tar'.format(args.pretrain_base_lr, args.class_split, args.pretrain_batch_size))




