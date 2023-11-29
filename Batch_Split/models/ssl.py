# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import copy
import math
from torch import nn, optim
import torch.nn.functional as F
from models.resnet import resnetc18
from loss import invariance_loss,CovarianceLoss, BarlowTwinsLoss,ErrorCovarianceLoss

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()

class EMA():
    def __init__(self, base_momentum, final_momentum, max_steps):
        super().__init__()
        self.alpha = base_momentum
        self.final_alpha = final_momentum
        self.base_alpha = base_momentum
        self.max_steps = max_steps

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new

    def update_alpha(self, current_step):
        self.alpha = self.final_alpha - \
            (self.final_alpha - self.base_alpha) * \
            (math.cos(math.pi * current_step / self.max_steps) + 1) / 2
        # print(self.alpha)



class Encoder(nn.Module):

    def __init__(self, hidden_dim=None, output_dim=2048, normalization = 'batch', weight_standard = False, appr_name = 'simsiam'):
        super().__init__()
        resnet = resnetc18(normalization = normalization, weight_standard = weight_standard)
        input_dim = resnet.fc.in_features
        if hidden_dim is None:
            hidden_dim = output_dim
        resnet_headless = nn.Sequential(*list(resnet.children())[:-1])
        resnet_headless.output_dim = input_dim
        self.backbone = resnet_headless
        #Keeping it reference: ref: https://github.com/Lightning-Universe/lightning-bolts/blob/2dfe45a4cf050f120d10981c45cfa2c785a1d5e6/pl_bolts/models/self_supervised/simsiam/simsiam_module.py
        # self.projector = nn.Sequential( 
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.BatchNorm1d(output_dim)
        # )
        # #Simsiam Projector ref: https://github.com/lucidrains/byol-pytorch/blob/6717204748c2a4f4f44b991d4c59ce5b99995582/byol_pytorch/byol_pytorch.py#L86
        if 'simsiam' in appr_name or 'barlow' in appr_name:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim))
        if 'infomax' in appr_name:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),)
        if 'byol' in appr_name:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),)


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

    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.temporal_projector = None
        self.teacher_model = copy.deepcopy(self.encoder).requires_grad_(False)
        # self.target_ema_updater = EMA(0.996)

    def initialize_EMA(self, base_momentum, final_momentum, max_steps):
        self.target_ema_updater = EMA(base_momentum, final_momentum, max_steps)

    @torch.no_grad()
    def _get_teacher(self):
        return self.teacher_model


    @torch.no_grad()
    def update_moving_average(self, step):
        assert self.teacher_model is not None, 'target encoder has not been created yet'

        for student_params, teacher_params in zip(self.encoder.parameters(), self.teacher_model.parameters()):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        self.target_ema_updater.update_alpha(step)

    

        
    def forward(self, x1, x2=None, projector = False):
        device = next(self.parameters()).device
        if projector == True:
            out = self.encoder(x1)
            out = out.squeeze()
            return out
        if self.training:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return z1,z2,p1,p2
        else:
            out = self.encoder.backbone(x1)
            out = out.squeeze()
            return out


class Siamese(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temporal_projector = None

    def forward(self, x1, x2=None):
        device = next(self.parameters()).device
        if self.training:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC
            
            return z1, z2
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