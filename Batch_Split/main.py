import os
import sys
import time
import wandb
import argparse
from PIL import Image, ImageFilter, ImageOps
import torch
import torchvision.transforms as T
import torchvision
from torchvision import transforms as T, utils

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
# from dataloaders.dataloader_cifar10 import get_cifar10
from utils.eval_metrics import linear_evaluation, get_t_SNE_plot, linear_evaluation_task_confusion
from models.linear_classifer import LinearClassifier
from models.ssl import  SimSiam, Siamese, Encoder, Predictor
# from models.simsiam import Encoder, Predictor, SimSiam, InfoMax, BarlowTwins
# from trainers.train_dist_ering import 
from trainers.train import train_infomax, train_barlow, train_simsiam, train_byol
from trainers.train_sup import train_sup
from trainers.train_concat import train_concate
from models.resnet import resnetc18
from models.resnet_org import resnetc18_bn
# from transform import get_transform
import random
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


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
    # parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')

    parser.add_argument('-d','--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    parser.add_argument('-dl_type','--dataset_type', type=str, default='class_incremental', help='cifar10, cifar100')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--knn_report_freq', type=int, default=1)
    # parser.add_argument('--proj_hidden', type=int, default=2048)
    # parser.add_argument('--proj_out', type=int, default=64)

    parser.add_argument('-gpu','--cuda_device', type=int, default=0, metavar='N', help='device id')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='num of workers')
    parser.add_argument('--algo', type=str, default='simsiam', help='ssl algorithm')
    parser.add_argument('--exp_type', type=str, default='basic',help='concat, basic')

    # Infomax Args
    # parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    # parser.add_argument('--sim_loss_weight', type=float, default=1000.0)
    # parser.add_argument('--info_loss', type=str, default='invariance', help='infomax loss')
    # parser.add_argument('--R_eps_weight', type=float, default=1e-8)

    parser.add_argument('-cs', '--class_split', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-vcs', '--val_class_split', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--cov_loss_weight', type=float, default=1.0)
    parser.add_argument('--sim_loss_weight', type=float, default=250.0)
    parser.add_argument('--info_loss', type=str, default='invariance',
                        help='infomax loss')
    parser.add_argument('--R_eps_weight', type=float, default=1e-8)
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)
    parser.add_argument('--appr', type=str, default='basic', help='Approach name, basic, PFR') #approach

    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)
    parser.add_argument('--normalization', type=str, default='batch', help='normalization method: batch, group or none')
    parser.add_argument('--weight_standard', action='store_true', default=False, help='weight standard for conv layers')

    parser.add_argument("--normalize_on", action="store_true", help='l2 normalization after projection MLP')
    parser.add_argument('--scale_loss', type=float, default=0.025)
    parser.add_argument("--is_debug", action="store_true", help='debug or not')

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
    assert sum(args.val_class_split) == num_classes
    # assert len(args.class_split) == len(args.epochs)

    # assert sum(args.class_split) == 10
    num_worker = int(10/len(args.class_split))
    if len(args.class_split) == 10:
        num_worker = 1

    batch_size = args.pretrain_batch_size
    # for k in range(len(args.class_split)):
    #     batch_size.append(args.pretrain_batch_size)
    #device
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print(device)
    #wandb init
    wandb.init(project="CSSL", entity="yavuz-team",
                # mode="disabled",
                config=args,
                name="BatchSplit" + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split) + '-algo' + str(args.algo)+'-groupnorm')

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

    #https://github.com/The-AI-Summer/byol-cifar10/blob/main/AI_Summer_BYOL_in_CIFAR10.ipynb
    if 'byol' in args.appr: # SImCLR augmentation (ref: https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py)
        transform = T.Compose([
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(p=0.5),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.5),
            T.RandomResizedCrop((32, 32)),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),])

        transform_prime = T.Compose([
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(p=0.5),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.5),
            T.RandomResizedCrop((32, 32)),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),])


    if 'barlow' in args.appr: #ref: they do not have Gaussian Blur https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/augmentations/asymmetric.yaml
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.0), 
                T.RandomSolarize(0.51, p=0.0), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=32, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.0),
                T.RandomSolarize(0.51, p=0.2),
                T.Normalize(mean=mean, std=std)])


    
    if args.algo == 'supervised':
        transform = None
        transform_prime = None

    #Dataloaders
    print("Creating Dataloaders..")
    #Class Based

    # print(num_worker)
    train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, _, train_data_loaders_generic = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker, dl_type = args.dataset_type)


    #Create Model
    if 'simsiam' in args.appr or 'byol' in args.appr:
        print("Creating Model for Simsiam or BYOL..")
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        pred_hidden = args.pred_hidden
        pred_out = args.pred_out

        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr)
        predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
        model = SimSiam(encoder, predictor)
        if 'byol' in args.appr:
            model.initialize_EMA(0.99, 1.0, len(train_data_loaders[0])*len(args.class_split)*args.epochs)
        model.to(device) #automatically detects from model
    if 'infomax' in args.appr or 'barlow' in args.appr:
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr)
        model = Siamese(encoder)
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
            if 'infomax' in args.appr: 
                model, loss, optimizer = train_infomax(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)
            elif 'barlow' in args.appr:
                model, loss, optimizer = train_barlow(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)
            elif 'simsiam' in args.appr:
                model, loss, optimizer = train_simsiam(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)
            elif 'byol' in args.appr:
                model, loss, optimizer = train_byol(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)
        else:
            model, loss, optimizer = train_concate(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)

    #Test Linear classification acc
    print("Starting Classifier Training..")
    if args.is_debug:
        lin_epoch = 1
    else:
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


    _, _, test_data_loaders_all, _, train_data_loaders_linear_all, _, _ = get_dataloaders(transform, transform_prime, \
                                        classes=[num_classes], valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker)


    test_loss, test_acc1, test_acc5, classifier = linear_evaluation(model, train_data_loaders_linear_all[0],
                                                                    test_data_loaders_all[0],lin_optimizer, classifier, 
                                                                    lin_scheduler, epochs=lin_epoch, device=device)
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
                }, is_best=False, filename='./checkpoints/checkpoint_{:04f}_algo_{}_cs_{}_bs_{}.pth.tar'.format(args.pretrain_base_lr, args.appr, args.class_split, args.pretrain_batch_size))

    args.class_split = args.val_class_split
    wp, tp = linear_evaluation_task_confusion(model, classifier, test_data_loaders, args, device)


    print(' Linear Acc '+str(test_acc1))
    print(" Linear WP "+str(wp))
    print(" Linear TP "+str(tp))
    print(" wp* tp "+str(wp*tp*100))

    #T-SNE Plot
    print("Starting T-SNE Plot..")
    get_t_SNE_plot(test_data_loaders_all[0], model, classifier, device)







