

#Suprvised
python main.py --epochs 200 -bs 128 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100 -cs 100 -gpu 4
python main.py --epochs 200 -bs 32 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100 -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 32 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 128 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 4
python main.py --epochs 200 -bs 64 -we 0 -lr 0.1 --pretrain_weight_decay 5e-4 --algo supervised -n group -d cifar100sup -cs 20,20,20,20,20 -gpu 5
#SSL Infomax
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100    -cs 100            -n group --proj_hidden 4096 --proj_out 128 -gpu 3 
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100    -cs 20,20,20,20,20 -n group --proj_hidden 4096 --proj_out 128 -gpu 3
python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 1000.0 -d cifar100sup -cs 20,20,20,20,20 -n group --proj_hidden 4096 --proj_out 128 -gpu 3

python main.py --epochs 1000 -bs 512 -we 10 -wlr 3e-3 -lr 0.5 --algo infomax --sim_loss_weight 250.0  -d cifar10  -cs 10  -n batch --proj_hidden 2048 --proj_out 64 -gpu 3

#### CIFAR100 ####
# SSL Infomax
python main.py -cs 100 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 0 --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --weight_standard --algo infomax --appr infomax --exp_type basic --knn_report_freq 10 -vcs 10,10,10,10,10,10,10,10,10,10
python main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 0 --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --weight_standard --algo infomax --appr infomax --exp_type basic --knn_report_freq 10 -vcs 10,10,10,10,10,10,10,10,10,10
# SSL Barlow
python main.py -cs 100 --epochs 10,10,10,10,10,10,10,10,10,10 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr barlow --pretrain_base_lr 0.1 --pretrain_batch_size 256  --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 --exp_type basic
python main.py -cs 100 --epochs 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr barlow --pretrain_base_lr 0.1 --pretrain_batch_size 256  --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 --exp_type basic



