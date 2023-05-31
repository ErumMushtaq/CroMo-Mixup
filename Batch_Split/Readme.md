python3 main.py -cs 2,2,2,2,2 --epochs 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --exp_type basic --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --algo infomax --appr infomax


python3 geo_distance.py --pretrain_batch_size 512 --pretrain_base_lr 0.5 -cs 2,2,2,2,2 --proj_hidden 2048 --proj_out 64 --appr infomax --normalization group --weight_standard --cuda_device 0 --dataset cifar10