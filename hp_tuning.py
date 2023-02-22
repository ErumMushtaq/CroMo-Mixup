import os
from itertools import product

lrs = [1e-3, 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5)]
for lr in lrs:
    cmd = "python main.py"
    cmd += f" --lr {lr}"
    print(cmd)
    os.system(cmd)
