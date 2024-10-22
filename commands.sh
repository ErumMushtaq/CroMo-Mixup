python main.py -cs 10 --epochs 1000 --pretrain_batch_size 512 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --algo infomax --pretrain_base_lr 0.05

python main.py -cs 5,5 --cuda_device 5 --epochs 1000 --pretrain_batch_size 512 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --algo infomax --pretrain_base_lr 0.05

python main.py -cs 2,2,2,2,2 --cuda_device 3 --epochs 1000 --pretrain_batch_size 512 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --algo infomax --pretrain_base_lr 0.05 



python main.py -cs 10 --epochs 1000 --pretrain_batch_size 512 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --algo infomax --pretrain_base_lr 0.03 --info_loss error_cov --sim_loss_weight 1.0