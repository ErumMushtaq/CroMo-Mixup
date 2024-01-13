# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
import torch.nn.functional as F
from models.resnet import resnetc18, resnetc50

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()


class Encoder(nn.Module):

    def __init__(self, hidden_dim=None, output_dim=2048, normalization = 'batch', weight_standard = False, appr_name = 'simsiam', dataset="cifar10"):
        super().__init__()
        if "cifar" in dataset:
            resnet = resnetc18(normalization = normalization, weight_standard = weight_standard)
        else:
            resnet = resnetc50(normalization = normalization, weight_standard = weight_standard)
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
                nn.Linear(hidden_dim, output_dim,bias=False),)

        if 'simclr' in appr_name:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim,bias=False),)
            #self.projector = nn.Sequential(
            #    nn.Linear(input_dim, hidden_dim),
            #    nn.ReLU(inplace=True),
            #    nn.Linear(hidden_dim, output_dim))

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