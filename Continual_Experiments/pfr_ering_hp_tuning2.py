import os
from itertools import product

# lrs = [1e-3, 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5)]
# R_eps_weights = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
# m_sizes = [300, 450, 600, 750, 1000]
# m_sizes = [96, 128]
# m_sizes = [160, 192]
m_sizes = [224, 256]
for msize in m_sizes:
    cmd = "python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 7 --appr 'PFR_ering_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --lambdap 10.0 --msize 500"
    cmd += f" --bsize {msize}"
    print(cmd)
    os.system(cmd)