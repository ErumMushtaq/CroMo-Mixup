import os
from itertools import product

# lrs = [1e-3, 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5)]
lrs = [ 0.5e-1, 0.5e-2, 0.5e-3,  1e-1, 1e-2, 1e-3]

for lr in lrs:
    cmd = "python main.py -cs 10 --epochs 1000 --pretrain_batch_size 512 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --algo infomax"
    cmd += f" --pretrain_base_lr {lr}"
    print(cmd)
    os.system(cmd)
