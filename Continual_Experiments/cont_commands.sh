python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'basic_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'basic_barlow'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'PFR_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'ering_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'PFR_contrastive_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --cuda_device 0 --appr 'contrastive_simsiam'

python3 main_cont.py -cs 5,5 -e 500,500 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --cuda_device 0 --appr 'basic_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --cuda_device 4 --appr 'PFR_simsiam'

## Cifar10 infomax Experiments
# python main_traintest.py --epochs 1000 --batch_size 512 --lin_epochs 100 --lin_batch_size 256 --R_ini 1.0  --learning_rate 0.5 --cov_loss_weight 1.0 --sim_loss_weight 250.0 --la_R 0.01 --la_mu 0.01 --projector 2048-2048-64 
python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4

## Cifar10 Barlow Experiments
python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'basic_barlow' --pretrain_base_lr 0.3 
## Cifar100 Barlow Experiments
python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'basic_barlow' --pretrain_base_lr 0.3 


## Cifar100 Experiments
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --same_lr --cuda_device 0 --appr 'basic_simsiam'
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100  --cuda_device 2 --appr 'basic_simsiam'

python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --cuda_device 3 --appr 'basic_simsiam' --pretrain_base_lr 0.06

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 1.0 --same_lr --cuda_device 7 --appr 'PFR_simsiam'

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 1.0 --lambda_norm 0.1 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.90 --lambdap 1.0 --lambda_norm 0.001 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr 'LRD_infomax' --pretrain_base_lr 0.10 




# python main_traintest.py --epochs 1000 --batch_size 512 --lin_epochs 100 --lin_batch_size 256 --R_ini 1.0  --learning_rate 0.5 --cov_loss_weight 1.0 --sim_loss_weight 1000.0 --la_R 0.01 --la_mu 0.01 --projector 4096-4096-128 --R_eps_weight 1e-8 --w_decay 1e-4 --lin_warmup_epochs 5 --pre_optimizer SGD --pre_scheduler lin_warmup_cos --lin_optimizer SGD --lin_learning_rate 0.2 --lin_w_decay 0 --lin_scheduler cos  --n_workers 4  --dataset cifar100 --lin_dataset cifar100 --con_name cov_cifar100_best_rerun --model_name resnet18 --normalize_on --min_lr 1e-6 --lin_min_lr 0.002
python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 5 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4

##HP SEARCH
python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.0 --lambda_norm 0.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --cuda_device 5 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 1 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.0 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 0.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.1 --lambda_norm 0.1 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 1.0 --lambda_norm 0.1 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 7 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 0.1 --lambda_norm 1.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr 'LRD_infomax' --pretrain_base_lr 0.10 

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0 --lambda_norm 10.0 --same_lr --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 0 --appr 'LRD_infomax' --pretrain_base_lr 0.10


python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0  --lambda_norm 10.0 --subspace_rate 0.95 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 5,5 -e 500,500 --lambdap 10.0 --lambda_norm 1.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_infomax' --pretrain_base_lr 0.10

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.90 --lambdap 10.0 --lambda_norm 0.0 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --appr LRD_infomax --pretrain_base_lr 0.10 --cuda_device 3

#Distillation experiments on CIFAR10
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 1.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
#Finettuningargs.lambdap for KD
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 1000.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --weight_standard --cuda_device 2 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
# python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 0.0 --lambda_norm 0.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
#LRD, no distillation lambbdap 0.0 
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 0.0 --lambda_norm 1.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
#LRD, no norm loss lambda_norm 0.0
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 0.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 6 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4
#LRD, orth contrastive loss
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 0.0 --lambda_norm 0.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 7 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4

#PFR+infomax:
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --weight_standard --cuda_device 4 --appr 'PFR_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --lambdap 10.0

python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 0.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 7 --appr PFR_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4

#CIFAR100 (working)
main_cont.py -cs 100 -e 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr basic_barlow --pretrain_base_lr 0.3 --pretrain_batch_size 256
#CIFAR100 Experiments Barlow Twins Baseline (SSL)
python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'basic_barlow' --pretrain_base_lr 0.3 
#(SSL 3)
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'basic_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256
#(SSL 6)
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'PFR_barlow' --pretrain_base_lr 0.3 --lambdap 10.0 --pretrain_batch_size 256


# Ering + Infomax Experiments
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'ering_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --lambdap 10.0 --bsize 200 --msize 300 

#Ering + PFR + infomax 
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 7 --appr 'PFR_ering_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --lambdap 10.0 --bsize 64 --msize 150


#CIFAR100 Balow Twins
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'basic_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'PFR_barlow' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --lambdap 15.0 

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'LRD_barlow' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 10.0 --lambda_norm 0.1  

python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'ering_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --lambdap 10.0 --bsize 256 --msize 25000 



#CIFAR100 infomax non-continual

python3 main_cont.py -cs 100 -e 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --cuda_device 6 --appr 'basic_infomax' --pretrain_base_lr 0.10 



#Hp search LRD+Replay
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 100.0 --lambda_norm 10.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 0 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 10.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 1 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 0.1 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300


python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 100.0 --lambda_norm 10.0 --subspace_rate 0.98  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 10.0 --subspace_rate 0.98  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 1.0 --subspace_rate 0.98  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300
python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --lambda_norm 0.1 --subspace_rate 0.98  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 6 --appr LRD_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --msize 300


#Hp search LRD+Replay and PFR Cifar100

python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.90 --lambdap 10.0 --lambda_norm 0.1 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 4 --appr 'LRD_replay_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.90 --lambdap 10.0 --lambda_norm 1.0 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr 'LRD_replay_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.90 --lambdap 10.0 --lambda_norm 0.0 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 2 --appr 'LRD_replay_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --subspace_rate 0.85 --lambdap 10.0 --lambda_norm 0.0 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 3 --appr 'LRD_replay_infomax' --pretrain_base_lr 0.10 
python3 main_cont.py -cs 25,25,25,25 -e 500,500,500,500 --dataset cifar100 --lambdap 10.0 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096  --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 6 --appr 'PFR_infomax' --pretrain_base_lr 0.10 


#LRD Scale cifar10 Hp search


#python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --subspace_rate 0.85  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr LRD_scale_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --resume_checkpoint --scale 1.0

python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --subspace_rate 0.85  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr LRD_scale_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --resume_checkpoint --scale 2.0


python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --subspace_rate 0.85  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 6 --appr LRD_scale_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --resume_checkpoint --scale 5.0


python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 10.0 --subspace_rate 0.85  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 7 --appr LRD_scale_infomax --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --resume_checkpoint --scale 10.0


##Cassle Hp search Barlow cifar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 0 --appr 'cassle_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_barlow' --pretrain_base_lr 0.15 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_barlow' --pretrain_base_lr 0.20 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_barlow' --pretrain_base_lr 0.30 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 5 --appr 'cassle_barlow' --pretrain_base_lr 0.35 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 6 --appr 'cassle_barlow' --pretrain_base_lr 0.08 --pretrain_batch_size 256 --same_lr


#CIFAR100 Balow Twins
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.80 --lambdap 10.0  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.80 --lambdap 1.0  --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.70 --lambdap 10.0  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.70 --lambdap 1.0 --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 10.0 --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 1.0 --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.95 --lambdap 10.0 --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 7 --appr 'LRD_replay_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.80 --lambdap 10.0 --lambda_norm 0.1 --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_replay_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.80 --lambdap 10.0 --same_lr


python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 7 --appr 'LRD_scale_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 10.0 --same_lr --scale 2.0

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'LRD_scale_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 10.0 --same_lr --scale 10.0

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'LRD_scale_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.90 --lambdap 10.0 --same_lr --scale 50.0


#updated LRD cifar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 1.0  --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.98 --lambdap 10.0  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.98 --lambdap 1.0  --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.99 --lambdap 10.0  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.99 --lambdap 1.0  --same_lr


python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0 --lambda_norm 0.1  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0 --lambda_norm 0.5  --same_lr



#cross LRD cifar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 0.001
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 0.01
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 0.1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 1.0
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 10.0


