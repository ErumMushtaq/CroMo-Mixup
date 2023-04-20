# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
import torch.nn.functional as F
from models.resnet import resnetc18 
from models.resnet_org import resnetc18_bn
from loss import invariance_loss,CovarianceLoss, BarlowTwinsLoss, BarlowTwinsLoss_ema

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()


class Encoder(nn.Module):

    def __init__(self, hidden_dim=None, output_dim=2048):
        super().__init__()
        # resnet = resnetc18_bn()
        resnet = resnetc18()
        input_dim = resnet.fc.in_features
        if hidden_dim is None:
            hidden_dim = output_dim
        resnet_headless = nn.Sequential(*list(resnet.children())[:-1])
        resnet_headless.output_dim = input_dim
        self.backbone = resnet_headless
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        out = self.backbone(x).squeeze()
        out = self.projector(out)
        return out


class Predictor(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return out


class SimSiam(nn.Module):

    def __init__(self, encoder, predictor, augment_fn = None,
        augment_fn2 = None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.augment1 = augment_fn
        self.augment2 = augment_fn2
        
    def forward(self, x1, x2=None):
        device = next(self.parameters()).device
        if self.training:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            loss_one = loss_fn(p1, z2.detach())
            loss_two = loss_fn(p2, z1.detach())

            loss = 0.5*loss_one + 0.5*loss_two
            return loss.mean()
        else:
            out = self.encoder.backbone(x1)
            out = out.squeeze()
            return out


class InfoMax(nn.Module):
    def __init__(self, encoder, project_dim =64, sim_loss_weight=250.0,cov_loss_weight = 1.0 
    , augment_fn = None,augment_fn2 = None,device='cpu', la_mu=0.01, la_R=0.01):
        super().__init__()
        self.encoder = encoder
        self.augment1 = augment_fn
        self.augment2 = augment_fn2
        self.cov_loss = CovarianceLoss(project_dim,device=device, la_mu=la_mu,la_R=la_R)
        self.sim_loss_weight = sim_loss_weight
        self.cov_loss_weight = cov_loss_weight 
        
    def forward(self, x1, x2=None):
        device = next(self.parameters()).device
        if self.training:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)
            
            cov_loss =  self.cov_loss(z1, z2)
            sim_loss =  invariance_loss(z1, z2) 
            loss = (self.sim_loss_weight * sim_loss) + (self.cov_loss_weight * cov_loss) 
            return loss
        else:
            out = self.encoder.backbone(x1)
            out = out.squeeze()
            return out


class BarlowTwins(nn.Module):
    def __init__(self, encoder, project_dim, lambda_param = 5e-3, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.cross_loss = BarlowTwinsLoss_ema(project_dim)
        # self.cross_loss = BarlowTwinsLoss_ema(project_dim, device=device, lambda_param=lambda_param)
        
    def forward(self, x1, x2=None):
        device = next(self.parameters()).device
        if self.training:
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC
            loss =  self.cross_loss(z1, z2)
            return loss
        else:
            out = self.encoder.backbone(x1)
            out = out.squeeze()
            return out
        
        
class LinearClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = x.squeeze()
        out = F.softmax(self.fc(out), dim=1)
        return out