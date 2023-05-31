python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --R_eps_weight 1e-8

python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 2 --lambda_norm 1.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr 'infomax_dist_ering' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4  --bsize 256 --msize 1250 --resume_checkpoint

#CIFAR100 infomax ering
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4  --weight_standard --bsize 256 --msize 1250
main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'ering_infomax' --pretrain_base_lr 0.3 --pretrain_batch_size 256 --weight_standard --bsize 256 --msize 1250

