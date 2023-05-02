import os
import sys
import wandb
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
import torch
import torchvision.transforms as T
import torchvision
# import data_utils
# from SSL.corinfomax_ssl.cifar10_tiny.data_utils import make_data

from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot
from models.linear_classifer import LinearClassifier
from models.ssl import  SimSiam, Siamese, Encoder, Predictor
from models.infomax_model import CovModel

from trainers.train_simsiam import train_simsiam
from trainers.train_infomax import train_infomax
from trainers.train_barlow import train_barlow

from trainers.train_PFR import train_PFR_simsiam
from trainers.train_LRD import train_LRD_infomax
from trainers.train_PFR_contrastive import train_PFR_contrastive_simsiam
from trainers.train_contrastive import train_contrastive_simsiam
from trainers.train_ering import train_ering_simsiam

from torchsummary import summary
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=[3,3],sigma=sigma)#kernel size and sigma are open problems but right now seems ok!
        return x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Model parameters
    parser.add_argument('--normalization', type=str, default='batch', help='normalization method: batch, group or none')
    parser.add_argument('--weight_standard', action='store_true', default=False, help='weight standard for conv layers')



    parser.add_argument('--same_lr', action='store_true', default=False, help='same lr for each task')
    parser.add_argument("--projector", default='4096-4096-128', type=str, help='projector MLP')

    parser.add_argument('--resume_checkpoint', action='store_true', default=False, help='start from second task LRD')
    


   
    

    # Training settings
    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_warmup_epochs', type=int, default=0)
    parser.add_argument('--pretrain_warmup_lr', type=float, default=0)
    parser.add_argument('--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=0.00)

    parser.add_argument('--lambdap', type=float, default=2.0)
    parser.add_argument('--appr', type=str, default='basic', help='Approach name, basic, PFR') #approach

    parser.add_argument('--knn_report_freq', type=int, default=10)
    parser.add_argument('--cuda_device', type=int, default=0, metavar='N', help='device id')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='num of workers')

    parser.add_argument('--contrastive_ratio', type=float, default=0.001)
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    parser.add_argument('-cs', '--class_split', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-e', '--epochs', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])

    # Infomax Args
    parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    parser.add_argument('--sim_loss_weight', type=float, default=250.0)
    parser.add_argument('--info_loss', type=str, default='invariance',
                        help='infomax loss')


    #LRD parameters
    parser.add_argument('--lambda_norm', type=float, default=1.0)
    parser.add_argument('--subspace_rate', type=float, default=0.99)

    # Barlow Twins Args
    parser.add_argument('--lambda_param', type=float, default=5e-3)
    parser.add_argument('--scale_loss', type=float, default=0.025)

    #Ering parameters
    parser.add_argument('--bsize', type=int, default=32, help='For Ering, number of samples that are sampled for each batch')
    parser.add_argument('--msize', type=str, default=150, help='For Ering, number of samples that are stored for each task (memory sample for each task)')

    #Architecture parameters
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)

    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)

    parser.add_argument("--normalize_on", action="store_true", help='l2 normalization after projection MLP')

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    if args.dataset == "cifar10":
        get_dataloaders = get_cifar10
        num_classes=10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    elif args.dataset == "cifar100":
        get_dataloaders = get_cifar100
        num_classes=100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    assert sum(args.class_split) == num_classes
    assert len(args.class_split) == len(args.epochs)
    
    num_worker = args.num_workers

    #device
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print(device)
    #wandb init
    wandb.init(project="CSSL",  entity="yavuz-team",
                #mode="disabled",
                config=args,
                name= str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split))

    if 'simsiam' in args.appr:
        #augmentations
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(mean=mean, std=std)])
    
    if 'infomax' in args.appr:
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=1.0), 
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.1),
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])
    if 'barlow' in args.appr: #ref: they do not have Gaussian Blur https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/augmentations/asymmetric.yaml
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=1.0), 
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.1),
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

    #Dataloaders
    print("Creating Dataloaders..")

    # #Class Based
    train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)
    _, train_data_loaders_knn_all, test_data_loaders_all, _, train_data_loaders_linear_all = get_dataloaders(transform, transform_prime, \
                                        classes=[num_classes], valid_rate = 0.00, batch_size=args.pretrain_batch_size, seed = 0, num_worker= num_worker)

    #Create Model
    ##!!! Make these model arguments
    if 'simsiam' in args.appr:
        print("Creating Model for Simsiam..")
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        pred_hidden = args.pred_hidden
        pred_out = args.pred_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard)
        predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
        model = SimSiam(encoder, predictor)
        model.to(device) #automatically detects from model
    if 'infomax' in args.appr or 'barlow' in args.appr:
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard)
        model = Siamese(encoder)
        # Infomax model
        # model = CovModel(args)
        model.to(device) #automatically detects from model

    #Training
    print("Starting Training..")
    if args.appr == 'basic_simsiam': #baseline setup
        model, loss, optimizer = train_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'basic_infomax': #baseline setup
        model, loss, optimizer = train_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'basic_barlow': #baseline setup
        model, loss, optimizer = train_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'PFR_simsiam': #CVPR paper
        model, loss, optimizer = train_PFR_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'contrastive_simsiam': #contrastive loss between new and old task samples
        model, loss, optimizer = train_contrastive_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'PFR_contrastive_simsiam': #contrastive loss between new and old task samples
        model, loss, optimizer = train_PFR_contrastive_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'LRD_infomax': #contrastive loss between new and old task samples
        model, loss, optimizer = train_LRD_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)    
    elif args.appr == 'ering_simsiam': #ERING
        model, loss, optimizer = train_ering_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args, transform, transform_prime)            
    else:
        raise Exception('Approach does not exist in this repo')

    #Test Linear classification acc
    print("Starting Classifier Training..")
    lin_epoch = 100
    if args.dataset == 'cifar10':
        classifier = LinearClassifier(num_classes = 10).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper
        # in_optimizer = torch.optim.SGD(classifier.parameters(), 0.1, momentum=0.9) # Infomax: no weight decay, epoch 100, cosine scheduler
        # lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=2e-4) #scheduler + values ref: infomax paper
    elif args.dataset == 'cifar100':
        classifier = LinearClassifier(num_classes = 100).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper

    test_loss, test_acc1, test_acc5, classifier = linear_evaluation(model, train_data_loaders_linear_all[0],
                                                                    test_data_loaders_all[0],lin_optimizer, classifier, 
                                                                    lin_scheduler, epochs=lin_epoch, device=device) 

    #T-SNE Plot
    print("Starting T-SNE Plot..")
    get_t_SNE_plot(test_data_loaders_all[0], model, classifier, device)


    file_name = './checkpoints/checkpoint_' + str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split) + 'acc_' + str(test_acc1) +'.pth.tar' 
    # save your encoder network
    save_checkpoint({
                    'arch': 'resnet18',
                    'lr': args.pretrain_base_lr,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss': loss,
                    'encoder': model.encoder.backbone.state_dict(),
                    'classifier': classifier.state_dict(),
                }, is_best=False, filename= file_name)




