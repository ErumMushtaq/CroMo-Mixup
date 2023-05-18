import os
from itertools import product

# lrs = [1e-3, 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5)]
R_eps_weights = [1e-7, 1e-6,  1e-5, 1e-4, 1e-3]
for R_eps_weight in R_eps_weights:
    cmd = "python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4"
    cmd += f" --R_eps_weight {R_eps_weight}"
    print(cmd)
    os.system(cmd)
