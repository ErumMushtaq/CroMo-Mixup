

#Suprvised
python main.py --epochs 200 -bs 128 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100 -cs 100 -gpu 4
python main.py --epochs 200 -bs 32 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100 -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 32 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 128 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 64 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 5
#SSL
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100    -cs 100            -n group --proj_hidden 4096 --proj_out 128 -gpu 3 
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100    -cs 20,20,20,20,20 -n group --proj_hidden 4096 --proj_out 128 -gpu 3
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100sup -cs 20,20,20,20,20 -n group --proj_hidden 4096 --proj_out 128 -gpu 3

python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 250.0  -d cifar10  -cs 10  -n batch --proj_hidden 2048 --proj_out 64 -gpu 3



