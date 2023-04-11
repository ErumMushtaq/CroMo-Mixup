# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
import torch.nn.functional as F
from models.resnet import resnetc18
from infomax_loss import invariance_loss,CovarianceLoss
from copy import deepcopy

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  - (x * y).sum(dim=-1).mean()


def contrastive_loss(z, centers):
    z = F.normalize(z, dim=-1, p=2)
    centers = F.normalize(centers, dim=-1, p=2)
    return (z.unsqueeze(1) * centers.unsqueeze(0)).sum(dim=-1).max(axis=1)[0].mean()#we want to decrease similarity


class Encoder(nn.Module):

    def __init__(self, hidden_dim=None, output_dim=2048):
        super().__init__()
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


class SimSiam_PFR_contrastive(nn.Module):

    def __init__(self, encoder, predictor, augment_fn = None,
        augment_fn2 = None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.augment1 = augment_fn
        self.augment2 = augment_fn2

        ### Continual learning parameters ###
        self.task_id = 0
        self.oldModel = deepcopy(self.encoder.backbone)
        # self.oldProjector = deepcopy(self.temporal_projector)
        self.oldModelFull = None
        self._task_classifiers = None
        self.lamb = None
        self.lambdap = 0.0
        self.scale_loss = 0.025
        self.criterion = nn.CosineSimilarity(dim=1)
        self.temporal_projector = Predictor(512, 256, 512) #in PFR 256 but in SimSiam 256.
        self.oldProjector = deepcopy(self.encoder.projector)

        
    def forward(self, x1, x2=None, centers = None,projector = False):
        device = next(self.parameters()).device
        if self.training:
            x1, x2 = x1.to(device), x2.to(device)
            f1 = self.encoder.backbone(x1).squeeze() # NxC
            f2 = self.encoder.backbone(x2).squeeze() # NxC
            z1 = self.encoder.projector(f1) # NxC
            z2 = self.encoder.projector(f2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            if centers != None:
                loss_contrastive = contrastive_loss(torch.cat((z1,z2)), centers.to(device))
            else:
                loss_contrastive = torch.tensor(0)

            loss_one = loss_fn(p1, z2.detach())
            loss_two = loss_fn(p2, z1.detach())

            loss = 0.5*loss_one + 0.5*loss_two

            if self.task_id == 0:
                return loss.mean(), loss_contrastive
            else: #compute KD loss for p2_f
                f1Old = self.oldModel(x1).squeeze().detach()
                f2Old = self.oldModel(x2).squeeze().detach()
                p2_1 = self.temporal_projector(f1)
                p2_2 = self.temporal_projector(f2)
                lossKD = self.lambdap * (-(self.criterion(p2_1, f1Old).mean() * 0.5
                                            + self.criterion(p2_2, f2Old).mean()* 0.5))
                # lossKD = self.lambdap01 * (torch.dist(f1Old, f1)+torch.dist(f2Old, f2))
                return loss.mean() + lossKD, loss_contrastive
        else:
            if projector:
                out = self.encoder(x1)
            else:
                out = self.encoder.backbone(x1)
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