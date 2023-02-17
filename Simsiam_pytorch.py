# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=64, hidden_proj_size = 2048, pred_dim=2048, augment_fn = None,
        augment_fn2 = None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        # self.encoder = base_encoder
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.augment1 = augment_fn
        self.augment2 = augment_fn2

        hidden_proj_size = 2048
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_proj_size, bias=False),
                                        nn.BatchNorm1d(hidden_proj_size),
                                        nn.ReLU(inplace=True), # first layer #[BS,hidden_proj_size]
                                        nn.Linear(hidden_proj_size, hidden_proj_size, bias=False),
                                        nn.BatchNorm1d(hidden_proj_size),
                                        nn.ReLU(inplace=True), # second layer ##[BS,hidden_proj_size]
                                        nn.Linear(hidden_proj_size, dim, bias=False),
                                        # self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False) ##[BS, dim]
                                        ) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        #TODO: double check predictor's hidden dimensions may be from ssfl
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer ##[BS, dim]??
    
        # output embedding of avgpool
        self.encoder.avgpool.register_forward_hook(self._get_avg_output())
        self.embedding = None

    def _get_avg_output(self):
        def hook(model, input, output):
            self.embedding = output.detach()
        return hook

    def forward(self, x):
        """
        Input:
            x
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        if self.training:
            x1, x2 = self.augment1(x), self.augment2(x)
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            loss_one = loss_fn(p1, z2.detach())
            loss_two = loss_fn(p2, z1.detach())

            loss = loss_one + loss_two
            return loss.mean()
        else:
            _ = self.encoder(x)
            return self.embedding.squeeze()

        # return p1, p2, z1.detach(), z2.detach()