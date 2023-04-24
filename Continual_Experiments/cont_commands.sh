python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'basic_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'basic_barlow'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'PFR_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'ering_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'PFR_contrastive_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'contrastive_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --cuda_device 0 --appr 'basic_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --cuda_device 4 --appr 'PFR_simsiam'



## Cifar100 Experiments
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --same_lr --cuda_device 0 --appr 'basic_simsiam'
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100  --cuda_device 2 --appr 'basic_simsiam'

python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --cuda_device 3 --appr 'basic_simsiam' --pretrain_base_lr 0.06

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 1.0 --same_lr --cuda_device 7 --appr 'PFR_simsiam'

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 10.0 --lambda_norm 0.1 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 10.0 --lambda_norm 0.001 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --cuda_device 5 --appr 'basic_infomax' --pretrain_base_lr 0.10 

##HP SEARCH
python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.0 --lambda_norm 0.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --cuda_device 5 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 1 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.0 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 0.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.1 --lambda_norm 0.1 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 0.1 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 7 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.1 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0 --lambda_norm 10.0 --same_lr --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 0 --appr 'LRD_infomax' --pretrain_base_lr 0.10


python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0  --lambda_norm 10.0 --subspace_rate 0.95 --resume_checkpoint  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0 --lambda_norm 1.0 --subspace_rate 0.95 --resume_checkpoint  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10