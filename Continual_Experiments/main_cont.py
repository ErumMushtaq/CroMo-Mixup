import os
import sys
import wandb
import argparse
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageFilter, ImageOps
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
import torch
import torchvision.transforms as T
import torchvision
from torchvision import transforms as T, utils
# import data_utils
# from SSL.corinfomax_ssl.cifar10_tiny.data_utils import make_data

from dataloaders.dataloader_cifar10 import get_cifar10
from dataloaders.dataloader_cifar100 import get_cifar100
from dataloaders.dataloader_tinyImagenet2 import get_tinyImagenet


from utils.eval_metrics import linear_evaluation, get_t_SNE_plot, Knn_Validation, linear_evaluation_task_confusion
from models.linear_classifer import LinearClassifier
from models.ssl import  SimSiam, Siamese, Encoder, Predictor
from models.infomax_model import CovModel

from models.gaussian_diffusion.basic_unet import UNet_conditional
from models.gaussian_diffusion.openai_unet import UNetModel
from models.gaussian_diffusion.openai_utils.gaussian_diffusion import GaussianDiffusion
from models.gaussian_diffusion.openai_utils import gaussian_diffusion as gd
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers import UNet2DModel

from trainers.train_basic import train_simsiam, train_barlow, train_infomax, train_simclr, train_byol


from trainers.train_PFR import train_PFR_simsiam,train_PFR_barlow,train_PFR_infomax
from trainers.train_cassle import train_cassle_simsiam,train_cassle_barlow,train_cassle_infomax, train_cassle_simclr, train_cassle_byol

from trainers.train_cassle_noise import train_cassle_noise_barlow

from trainers.train_cassle_linear import train_cassle_linear_barlow, train_cassle_linear_infomax, train_cassle_linear_simsiam

from trainers.train_cassle_linear2 import train_cassle_linear_barlow2

from trainers.train_cassle_contrastive import train_cassle_contrastive_v1_barlow,train_cassle_contrastive_v2_barlow, train_cassle_contrastive_v3_barlow

from trainers.train_PFR_ering import train_PFR_ering_infomax
from trainers.train_LRD import train_LRD_infomax,train_LRD_barlow
from trainers.train_LRD_scale import train_LRD_scale_infomax, train_LRD_scale_barlow
from trainers.train_LRD_cross import  train_LRD_cross_barlow
from trainers.train_LRD_replay import train_LRD_replay_infomax, train_LRD_replay_barlow
from trainers.train_PFR_contrastive import train_PFR_contrastive_simsiam
from trainers.train_contrastive import train_contrastive_simsiam
from trainers.train_ering import train_ering_simsiam,train_ering_infomax,train_ering_barlow, train_ering_simclr, train_ering_byol
from trainers.train_dist_ering import train_dist_ering_infomax
from trainers.train_cassle_ering import train_cassle_barlow_ering, train_cassle_ering_simclr, train_cassle_ering_infomax,  train_cassle_ering_byol
# from trainers.train_cassle_contrast import train_infomax_iomix, train_cassle_infomax_mixed_distillation, train_cassle_barlow_ering_contrast, train_cassle_barlow_mixed_distillation, train_cassle_barlow_principled_iomix, train_cassle_barlow_iomixup, train_cassle_barlow_inputmixup
from trainers.train_cassle_inversion import train_cassle_barlow_inversion
from trainers.train_cassle_cosine import train_cassle_cosine_barlow
from trainers.train_cassle_cosine_linear import train_cassle_cosine_linear_barlow
from trainers.train_cosine_ering import train_cosine_ering_barlow
from trainers.train_GPM import train_gpm_barlow
from trainers.train_GPM_cosine import train_gpm_cosine_barlow
from trainers.train_ddpm import train_diffusion
from trainers.train_cddpm import train_barlow_diffusion
from trainers.train_lump import train_lump_barlow
from trainers.train_iomix import train_infomax_iomix, train_cassle_barlow_iomixup, train_simclr_iomix, train_iomix_byol, train_cassle_barlow_mixup
from trainers.train_mixed_distillation import train_cassle_infomax_mixed_distillation, train_cassle_barlow_mixed_distillation,  train_simclr_mixed_distillation, train_mixed_distillation_byol
from trainers.train_cassle_contrast import  train_cassle_barlow_ering_contrast,  train_cassle_barlow_principled_iomix, train_cassle_barlow_inputmixup



# from torchsummary import summary
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

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

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

    parser.add_argument('--lambdap', type=float, default=1.0)# should it be 1?
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
    parser.add_argument('--R_eps_weight', type=float, default=1e-8)
    parser.add_argument('--la_mu', type=float, default=0.1)
    parser.add_argument('--la_R', type=float, default=0.1)


    #LRD parameters
    parser.add_argument('--lambda_norm', type=float, default=0.0)
    parser.add_argument('--subspace_rate', type=float, default=0.99)
    parser.add_argument('--scale', type=float, default=1.0)

    # Barlow Twins Args
    parser.add_argument('--lambda_param', type=float, default=5e-3)
    parser.add_argument('--scale_loss', type=float, default=0.025)

    #simclr args
    parser.add_argument('--temperature', type=float, default=0.07)

    #Ering parameters
    parser.add_argument('--bsize', type=int, default=250, help='For Ering, number of samples that are sampled for each batch')
    parser.add_argument('--msize', type=int, default=60, help='For Ering, number of samples that are stored for each task (memory sample for each task)')

    #Architecture parameters
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)

    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)

    parser.add_argument('--contrastive_hidden', type=int, default=512)

    parser.add_argument("--normalize_on", action="store_true", help='l2 normalization after projection MLP')

    #Cassle+ering parameters
    parser.add_argument('--cur_dist', type=int, default=1)
    parser.add_argument('--old_dist', type=int, default=1)
    parser.add_argument('--start_chkpt', type=int, default=0)

    parser.add_argument('--cross_lambda', type=float, default=1.0)

    #Cosine+ering parameters
    parser.add_argument('--apply_ering', type=int, default=1)
    parser.add_argument('--apply_cosine', type=int, default=1)
    parser.add_argument('--lambdacs', type=float, default=1.0)

    #GPM parameter
    parser.add_argument('--epsilon', type=float, default=0.9)

    #Diffusion Model
    #num_train_timesteps, beta_schedule='squaredcos_cap_v2', num_inference_steps, timestep_spacing="linspace"
    parser.add_argument('--unet_model', type=str, default='basic', help='basic, openai')
    parser.add_argument('--noise_scheduler', type=str, default='DDPM', help='DDPM, DDIM')
    parser.add_argument('--beta_scheduler', type=str, default='squaredcos_cap_v2', help='squaredcos_cap_v2, linear')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--num_inference_steps', type=int, default=250, help='250, 1000')
    parser.add_argument('--timestep_spacing', type=str, default='linspace', help='needed for diffusers DDIM')
    parser.add_argument("--calculate_fid", action="store_true", help='calculate_fid or not')
    parser.add_argument('--diff_train_lr', type=float, default=1e-4, help='diffusion lr such as 1e-4')
    parser.add_argument('--diff_weight_decay', type=float, default=5e-4)
    parser.add_argument('-de', '--diff_epochs', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--sample_bs', type=int, default=100)
    parser.add_argument('--diff_train_bs', type=int, default=100)
    parser.add_argument('--replay_bs', type=int, default=128)
    parser.add_argument("--class_condition", action="store_true", help='calculate_fid or not')
    parser.add_argument("--augment_seq", action="store_true", help='augmentation type')
    parser.add_argument("--clustering_label", action="store_true", help='calculate_fid or not')
    parser.add_argument("--is_debug", action="store_true", help='debug or not')
    parser.add_argument('--image_report_freq', type=int, default=10)
    parser.add_argument('--cond_dim', type=int, default=1000, help='512, 1000, 5000')

    parser.add_argument('--temp_proj', type=str, default='nonlinear', help='nonlinear, identity')
    parser.add_argument('--transform', type=str, default='original', help='original, basic')
    parser.add_argument('--alpha', type=float, default=1.0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    if args.dataset == "cifar10":
        get_dataloaders = get_cifar10
        img_size = 32
        num_classes=10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    elif args.dataset == "cifar100":
        get_dataloaders = get_cifar100
        img_size = 32
        num_classes=100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif args.dataset == "tinyImagenet":
        get_dataloaders = get_tinyImagenet
        img_size = 64
        num_classes=200
        # mean = (0.4802, 0.4480, 0.3975)
        # std = (0.2770, 0.2691, 0.2821)
        mean = (0.485, 0.456, 0.406)  #from infomax
        std = (0.229, 0.224, 0.225)
    assert sum(args.class_split) == num_classes
    assert len(args.class_split) == len(args.epochs)
    
    num_worker = args.num_workers

    #device
    print(torch.cuda.is_available())
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    print(device)
    #wandb init
    wandb.init(project="CSSL",  entity="yavuz-team",
                #mode="disabled",
                config=args,
                name= str(args.dataset) + '-algo' + str(args.appr) + "-e" + str(args.epochs) + "-b" 
                + str(args.pretrain_batch_size) + "-lr" + str(args.pretrain_base_lr)+"-CS"+str(args.class_split))

    transform2 = []
    transform2_prime = []
    if 'simsiam' in args.appr:
        #augmentations
        transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.Normalize(mean=mean, std=std)])
    if 'infomax' in args.appr:
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=1.0), 
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.1),
                # T.RandomApply([GaussianBlur()], p=0.5), 
                T.Normalize(mean=mean, std=std)])
    if 'barlow' in args.appr: #ref: they do not have Gaussian Blur https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/augmentations/asymmetric.yaml
        min_scale = 0.08
        transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.0), #0.0
                T.RandomSolarize(0.51, p=0.0), 
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur()], p=0.0),
                T.RandomSolarize(0.51, p=0.2),
                T.Normalize(mean=mean, std=std)])

    

        if 'inputmix' in args.appr: #https://github.com/divyam3897/UCL/blob/cfaa81d1af867afa9f35ff5d27d05404e212811b/datasets/seq_cifar100.py#L47
            cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
            print('LUMP transform')
            transform = T.Compose(
                [T.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.Normalize(*cifar_norm)])
            transform_prime = T.Compose(
                [T.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.Normalize(*cifar_norm)])
        # if args.transform == 'basic':
        transform2 = T.Compose([T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),), T.RandomHorizontalFlip(0.5), T.Normalize(mean=mean, std=std)])
        transform2_prime = T.Compose([T.RandomResizedCrop(size=img_size, scale=(min_scale, 1.0),), T.RandomHorizontalFlip(0.5),T.Normalize(mean=mean, std=std)])

    if 'simclr' in args.appr:
        #augmentations
        transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                #T.RandomApply([GaussianBlur()], p=1.0),#it is definitely applied: https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
                T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
                T.RandomGrayscale(p=0.2),
                #T.RandomApply([GaussianBlur()], p=1.0),#it is definitely applied
                T.Normalize(mean=mean, std=std)])

    #https://github.com/The-AI-Summer/byol-cifar10/blob/main/AI_Summer_BYOL_in_CIFAR10.ipynb
    # if 'byol' in args.appr: # SImCLR augmentation (ref: https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py)
    #     transform = T.Compose([
    #         RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.8), #0.3
    #         T.RandomGrayscale(p=0.2),
    #         T.RandomHorizontalFlip(p=0.5),
    #         RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.5), #0.2, 
    #         T.RandomResizedCrop((32, 32)),
    #         T.Normalize(mean=mean, std=std)])

    #     transform_prime = T.Compose([
    #         RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.8),
    #         T.RandomGrayscale(p=0.2),
    #         T.RandomHorizontalFlip(p=0.5),
    #         RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.5),
    #         T.RandomResizedCrop((32, 32)),
    #         T.Normalize(mean=mean, std=std)])

    #https://github.com/DonkeyShot21/cassle/blob/main/bash_files/continual/cifar/byol.sh
    # https://github.com/DonkeyShot21/essential-BYOL/blob/main/data_utils/transforms.py
    if 'byol' in args.appr:
        transform = T.Compose([
            T.RandomResizedCrop((32, 32)),
            T.RandomHorizontalFlip(),
            RandomApply(T.ColorJitter(0.4, 0.4, 0.2, 0.1), p = 0.8), #0.3
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.0), #0.2,
            T.ToPILImage(),
            RandomApply(Solarization(),p = 0.0), 
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

        transform_prime = T.Compose([
            T.RandomResizedCrop((32, 32)),
            T.RandomHorizontalFlip(),
            RandomApply(T.ColorJitter(0.4, 0.4, 0.2, 0.1), p = 0.8), #0.3
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.0), #0.2,
            T.ToPILImage(),
            RandomApply(Solarization(),p = 0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
  

    #Dataloaders
    print("Creating Dataloaders..")

    batch_size = args.pretrain_batch_size
    if 'aug' in args.appr or 'mix' in args.appr or 'lump' in args.appr:
        org_data = True
    else:
        org_data = False

    # #Class Based
    #train_data_loaders, train_data_loaders_knn, test_data_loaders, validation_data_loaders, train_data_loaders_linear, train_data_loaders_pure
    train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, train_data_loaders_pure, train_data_loaders_generic = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker, org_data = org_data)

    #Create Model
    ##!!! Make these model arguments
    if 'simsiam' in args.appr or 'byol' in args.appr:
        print("Creating Model for Simsiam..")
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        pred_hidden = args.pred_hidden
        pred_out = args.pred_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr, dataset=args.dataset)
        predictor = Predictor(input_dim=proj_out, hidden_dim=pred_hidden, output_dim=pred_out)
        model = SimSiam(encoder, predictor)
        if 'byol' in args.appr:
            model.initialize_EMA(0.99, 1.0, len(train_data_loaders[0])*sum(args.epochs))
        model.to(device) #automatically detects from model
    if 'infomax' in args.appr or 'barlow' in args.appr or 'simclr' in args.appr:
        proj_hidden = args.proj_hidden
        proj_out = args.proj_out
        encoder = Encoder(hidden_dim=proj_hidden, output_dim=proj_out, normalization = args.normalization, weight_standard = args.weight_standard, appr_name = args.appr, dataset=args.dataset)
        model = Siamese(encoder)
        model.to(device) #automatically detects from model
    if 'diffusion' in args.appr:
        #1. DataLoader for Diffusion Model
        diffusion_tr   = T.Compose([
            T.Resize(args.image_size + int(.25*args.image_size)),  # args.img_size + 1/4 *args.img_size
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # features are normalized from 0 to 1
        diffusion_tr_prime  = None
        val_transforms = T.Compose([
            T.Resize(args.image_size),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if args.dataset == 'cifar10':
            data_normalize_mean = (0.4914, 0.4822, 0.4465)
            data_normalize_std = (0.247, 0.243, 0.261)
            transform_knn = T.Compose( [   
                T.Normalize(data_normalize_mean, data_normalize_std),
            ])
        elif args.dataset == 'cifar100':
            data_normalize_mean = (0.5071, 0.4865, 0.4409)
            data_normalize_std = (0.2673, 0.2564, 0.2762)
            transform_knn = T.Compose( [   
                T.Normalize(data_normalize_mean, data_normalize_std),
            ])
        #Dataloaders
        print("Creating Diffusion Dataloaders..")
        batch_size = args.diff_train_bs
        train_data_loaders_diffusion, _, test_data_loaders_diffusion, _, _, _, _= get_dataloaders(diffusion_tr, diffusion_tr_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker,  valid_transform=val_transforms)
        if args.unet_model == 'basic':
            diffusion_model = UNet_conditional(c_in=3, c_out=3, num_classes=num_classes)
        elif args.unet_model == 'openai': # for now using the default params for cifar100 from OpenAI's repository.
            attention_resolutions=   '16,8,4' #32,16,8
            attention_ds = []
            for res in attention_resolutions.split(","):
                attention_ds.append(args.image_size // int(res))
            model_channels= 128 #192
            #Openai-improved diffusion github
            if args.clustering_label:
                class_numbers = 200*len(args.class_split)
            else:
                class_numbers = num_classes
            diffusion_model = UNetModel(image_size=args.image_size, in_channels=3, model_channels=128,out_channels=3,num_res_blocks=3,attention_resolutions=tuple(attention_ds), dropout=0.1,channel_mult= (1, 2, 3, 4),num_classes=class_numbers, use_checkpoint=False, use_fp16=False, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=True, cond_dim=args.cond_dim)
        elif args.unet_model == 'unet_fast':   
            #self-guided diffusion model -> unet_fast
            attention_resolutions= '4' #32,16,8
            attention_ds = []
            if args.clustering_label:
                class_numbers = 200*len(args.class_split)
            else:
                class_numbers = num_classes
            model_channels= 128
            for res in attention_resolutions.split(","):
                attention_ds.append(args.image_size // int(res))
            diffusion_model = UNetModel(image_size=args.image_size, in_channels=3, model_channels=128,out_channels=3,num_res_blocks=2,attention_resolutions=tuple(attention_ds), dropout=0.1,channel_mult= (1, 2,  4),num_classes=class_numbers, use_checkpoint=False, use_fp16=False, num_head_channels=8, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=True, cond_dim=args.cond_dim)

        if args.noise_scheduler == 'DDPM':
            noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_scheduler)
        elif args.noise_scheduler == 'DDIM':
            noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_scheduler,  timestep_spacing="trailing")#rescale_betas_zero_snr=True
            noise_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=device)

    #Training
    print("Starting Training..")
    if args.appr == 'barlow_diffusion' or args.appr == 'basic_dino':
        print(args.appr)
        model, loss, optimizer = train_barlow_diffusion(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, diffusion_model, noise_scheduler, train_data_loaders_diffusion, test_data_loaders_diffusion, transform, transform_prime, diffusion_tr, transform_knn )
    elif args.appr == 'diffusion':
        trainer = train_diffusion(model, noise_scheduler, train_data_loaders[0], test_data_loaders[0], device,args,train_batch_size = batch_size,train_lr = args.pretrain_base_lr, train_epochs = args.epochs, gradient_accumulate_every = 2, ema_decay = 0.995, amp = True, calculate_fid = True)
        trainer.train()
        exit()
    elif args.appr == 'basic_simsiam': #baseline setup
        model, loss, optimizer = train_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'basic_infomax': #baseline setup
        model, loss, optimizer = train_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'basic_barlow': #baseline setup
        model, loss, optimizer = train_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'basic_simclr': #baseline setup
        model, loss, optimizer = train_simclr(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'basic_byol':
        model, loss, optimizer = train_byol(model, train_data_loaders, test_data_loaders, train_data_loaders_knn, train_data_loaders_linear, device, args)
    elif args.appr == 'PFR_simsiam': #CVPR paper
        model, loss, optimizer = train_PFR_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'PFR_infomax': #CVPR paper + NeurIPS Paper
        model, loss, optimizer = train_PFR_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'PFR_ering_infomax': #CVPR paper + NeurIPS Paper
        model, loss, optimizer = train_PFR_ering_infomax(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime)
    elif args.appr == 'PFR_barlow': #CVPR Workshop paper
        model, loss, optimizer = train_PFR_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'cassle_barlow': #CVPR main paper
        model, loss, optimizer = train_cassle_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_simclr': #CVPR main paper
        model, loss, optimizer = train_cassle_simclr(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_byol':
        model, loss, optimizer = train_cassle_byol(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_cosine_barlow': #CVPR main paper
        model, loss, optimizer = train_cassle_cosine_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, transform, transform_prime,  device, args)
    elif args.appr == 'cassle_cosine_linear_barlow': #CVPR main paper
        model, loss, optimizer = train_cassle_cosine_linear_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, transform, transform_prime,  device, args)
    elif args.appr == 'cassle_linear_barlow': #CVPR main paper
        model, loss, optimizer = train_cassle_linear_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_linear_infomax': #CVPR main paper
        model, loss, optimizer = train_cassle_linear_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_linear_simsiam': #CVPR main paper
        model, loss, optimizer = train_cassle_linear_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_linear_barlow2': #CVPR main paper
        model, loss, optimizer = train_cassle_linear_barlow2(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_barlow_inversion': #CVPR main paper
        model, loss, optimizer = train_cassle_barlow_inversion(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear,  device, args)
    elif args.appr == 'cassle_noise_barlow': #CVPR main paper
        model, loss, optimizer = train_cassle_noise_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, transform, transform_prime, device, args)
    elif args.appr == 'cassle_simsiam': #CVPR main paper
        model, loss, optimizer = train_cassle_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'cassle_infomax': #CVPR main paper
        model, loss, optimizer = train_cassle_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args)
    elif args.appr == 'infomax_dist_ering':
        model, loss, optimizer = train_dist_ering_infomax(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime)
    elif args.appr == 'contrastive_simsiam': #contrastive loss between new and old task samples
        model, loss, optimizer = train_contrastive_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'PFR_contrastive_simsiam': #contrastive loss between new and old task samples
        model, loss, optimizer = train_PFR_contrastive_simsiam(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'LRD_infomax': #contrastive loss between new and old task samples
        model, loss, optimizer = train_LRD_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'LRD_barlow': #contrastive loss between new and old task samples
        model, loss, optimizer = train_LRD_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'LRD_scale_infomax': #contrastive loss between new and old task samples
        model, loss, optimizer = train_LRD_scale_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'LRD_scale_barlow': #contrastive loss between new and old task samples
        model, loss, optimizer = train_LRD_scale_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)      
    elif args.appr == 'ering_infomax': #ERING + NeurIPS
        model, loss, optimizer = train_ering_infomax(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime) 
    elif args.appr == 'ering_simsiam': #ERING
        model, loss, optimizer = train_ering_simsiam(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime)  
    elif args.appr == 'ering_barlow': #ERING
        model, loss, optimizer = train_ering_barlow(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime) 
    elif args.appr == 'ering_simclr': #ERING
        model, loss, optimizer = train_ering_simclr(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime) 
    elif args.appr == 'ering_byol':
        model, loss, optimizer = train_ering_byol(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime)
    elif args.appr == 'LRD_replay_infomax': #LRD + Replay + infomax
        model, loss, optimizer = train_LRD_replay_infomax(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime)  
    elif args.appr == 'LRD_replay_barlow': #LRD + Replay + barlow
        model, loss, optimizer = train_LRD_replay_barlow(model, train_data_loaders, train_data_loaders_knn, train_data_loaders_pure, test_data_loaders, device, args, transform, transform_prime) 
    elif args.appr == 'LRD_cross_barlow': #LRD + Replay + barlow
        model, loss, optimizer = train_LRD_cross_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, device, args)
    elif args.appr == 'cassle_contrastive_v1_barlow': #LRD + Replay + barlow
        model, loss, optimizer = train_cassle_contrastive_v1_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, transform, transform_prime, device, args)         
    elif args.appr == 'cassle_contrastive_v2_barlow': #LRD + Replay + barlow
        model, loss, optimizer = train_cassle_contrastive_v2_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, transform, transform_prime, device, args)  
    elif args.appr == 'cassle_contrastive_v3_barlow': #LRD + Replay + barlow
        model, loss, optimizer = train_cassle_contrastive_v3_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, transform, transform_prime, device, args)    
    elif args.appr == 'cassle_ering_barlow': #cassle + ering + barlow
        model, loss, optimizer = train_cassle_barlow_ering(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime) 
    elif args.appr == 'cosine_ering_barlow': #cosine + ering + barlow
        model, loss, optimizer = train_cosine_ering_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime)
    elif args.appr == 'byol_cassle_ering':
        model, loss, optimizer =  train_cassle_ering_byol(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime)
    elif args.appr == 'gpm_barlow': #gpm+barlow
        model, loss, optimizer = train_gpm_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args) 
    elif args.appr == 'barlow_ering_contrast' or args.appr == 'barlow_cassle_ering' or args.appr == 'barlow_ering_negcontrast' or args.appr == 'barlow_ering_augcontrast' or args.appr == 'barlow_ering_inputmixcontrast':
        model, loss, optimizer = train_cassle_barlow_ering_contrast(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_inputmix':
        model, loss, optimizer = train_cassle_barlow_inputmixup(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_iomix':
        model, loss, optimizer = train_cassle_barlow_iomixup(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_mixup':
            model, loss, optimizer = train_cassle_barlow_mixup(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_lump':
        model, loss, optimizer = train_lump_barlow(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime)
    elif args.appr == 'infomax_iomix':
        model, loss, optimizer = train_infomax_iomix(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'simclr_iomix':
        model, loss, optimizer = train_simclr_iomix(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'byol_iomix':
        model, loss, optimizer = train_iomix_byol(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'simclr_mixed_distillation':
        model, loss, optimizer =  train_simclr_mixed_distillation(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'byol_mixed_distillation':
        model, loss, optimizer =  train_mixed_distillation_byol(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'simclr_cassle_ering':
        model, loss, optimizer =   train_cassle_ering_simclr(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'infomax_cassle_ering':
        model, loss, optimizer =  train_cassle_ering_infomax(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'infomax_mixed_distillation':
        model, loss, optimizer = train_cassle_infomax_mixed_distillation(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_principled_iomix':
        model, loss, optimizer = train_cassle_barlow_principled_iomix(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'barlow_mixed_distillation':
        model, loss, optimizer = train_cassle_barlow_mixed_distillation(model, train_data_loaders, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime, transform2, transform2_prime) 
    elif args.appr == 'gpm_cosine_barlow': #gpm+barlow
        model, loss, optimizer = train_gpm_cosine_barlow(model, train_data_loaders_generic, train_data_loaders_knn, test_data_loaders, train_data_loaders_linear, device, args, transform, transform_prime) 
    else:
        raise Exception('Approach does not exist in this repo')

    #Test Linear classification acc
    _, _, test_data_loaders_all, _, train_data_loaders_linear_all, _, _ = get_dataloaders(transform, transform_prime, \
                                        classes=[num_classes], valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker)
    print("Starting Classifier Training..")
    lin_epoch = 200
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
    elif args.dataset == 'tinyImagenet':
        classifier = LinearClassifier(features_dim=2048, num_classes = 200).to(device)
        lin_optimizer = torch.optim.SGD(classifier.parameters(), 0.2, momentum=0.9, weight_decay=0) # Infomax: no weight decay, epoch 100, cosine scheduler
        lin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lin_optimizer, lin_epoch, eta_min=0.002) #scheduler + values ref: infomax paper

    test_loss, test_acc1, test_acc5, classifier = linear_evaluation(model, train_data_loaders_linear_all[0],
                                                                    test_data_loaders_all[0],lin_optimizer, classifier, 
                                                                    lin_scheduler, epochs=lin_epoch, device=device) 

    #T-SNE Plot
    # print("Starting T-SNE Plot..")
    # get_t_SNE_plot(test_data_loaders_all[0], model, classifier, device)
    if len(args.class_split) == 1: # Offline, just hard core the cases you need values for
        if args.dataset == 'cifar10':
            args.class_split = [5,5]
        elif args.dataset == 'cifar100':
            args.class_split = [20,20,20,20,20]
        train_data_loaders, train_data_loaders_knn, test_data_loaders, _, train_data_loaders_linear, train_data_loaders_pure, train_data_loaders_generic = get_dataloaders(transform, transform_prime, \
                                        classes=args.class_split, valid_rate = 0.00, batch_size=batch_size, seed = 0, num_worker= num_worker, org_data = org_data)



    wp, tp = linear_evaluation_task_confusion(model, classifier, test_data_loaders, args, device)
    wandb.log({" Linear Layer Test - TP Acc": tp})
    wandb.log({" Linear Layer Test - WP Acc": wp})


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




