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

#CIFAR100 simclr non-continual
python3 main_cont.py -cs 100 -e 1000 --dataset cifar100  --proj_out 128 --proj_hidden 2048 --cuda_device 2 --pretrain_batch_size 512 --appr 'basic_simclr' --pretrain_base_lr 0.6  --min_lr 1e-3 --temperature 0.5

#CIFAR10 simclr non-continual
python3 main_cont.py -cs 10 -e 1000 --dataset cifar10  --proj_out 128 --proj_hidden 2048 --cuda_device 0 --pretrain_batch_size 512 --appr 'basic_simclr' --pretrain_base_lr 0.6  --min_lr 1e-3 --temperature 0.5


#CIFAR100 simclr ering
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100  --proj_out 128 --proj_hidden 2048 --cuda_device 7 --pretrain_batch_size 512 --appr 'ering_simclr' --pretrain_base_lr 0.6  --min_lr 1e-3 --temperature 0.5 --msize 500


#CIFAR100 simclr cassle
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100  --proj_out 128 --proj_hidden 2048 --cuda_device 7 --pretrain_batch_size 512 --appr 'cassle_simclr' --pretrain_base_lr 0.6  --min_lr 1e-3 --temperature 0.5


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
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_barlow' --pretrain_base_lr 0.15 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_barlow' --pretrain_base_lr 0.20 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 5 --appr 'cassle_barlow' --pretrain_base_lr 0.30 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 6 --appr 'cassle_barlow' --pretrain_base_lr 0.35 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 7 --appr 'cassle_barlow' --pretrain_base_lr 0.08 --pretrain_batch_size 256 --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr --lambdap 1.0
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_barlow' --pretrain_base_lr 0.30 --pretrain_batch_size 256  --same_lr --lambdap 1.0
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_barlow' --pretrain_base_lr 0.10 --pretrain_batch_size 256  --same_lr --lambdap 1.0

##Cassle contrastive v1 Barlow cifar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 0 --appr 'cassle_contrastive_v1_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr 
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_contrastive_v2_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr 
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_contrastive_v3_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_barlow' --pretrain_base_lr 0.25 --pretrain_batch_size 256  --same_lr  

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_contrastive_v1_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr 
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 5 --appr 'cassle_contrastive_v2_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr 
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_contrastive_v3_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr  



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
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 0.1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 1.0
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 10.0



## 10 tasks benchmark

python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 1.0
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 0.1
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'LRD_cross_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr --contrastive_ratio 10.0


python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'LRD_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --subspace_rate 0.97 --lambdap 10.0  --same_lr


python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 0 --appr 'cassle_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0



#Cassle+Ering+Barlow CIfar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 0 --appr 'cassle_ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr  --cur_dist 1 --old_dist 1 --lambdap 1.0  --start_chkpt 1
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr  --cur_dist 1 --old_dist 1 --lambdap 1.0  --start_chkpt 0



#cassle noise
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_noise_barlow' --pretrain_base_lr 0.30 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --cross_lambda 1.0
 

#Cassle+Inversion+Barlow CIfar100
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_barlow_inversion' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr  --lambdap 1.0  --start_chkpt 1
 


 #cassle linear
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 5 --appr 'cassle_linear_barlow' --pretrain_base_lr 0.10 --pretrain_batch_size 256  --same_lr --lambdap 1.0


python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_linear_barlow' --pretrain_base_lr 0.05 --pretrain_batch_size 256  --same_lr --lambdap 1.0

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_linear_barlow' --pretrain_base_lr 0.15 --pretrain_batch_size 256  --same_lr --lambdap 1.0

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_linear_barlow' --pretrain_base_lr 0.20 --pretrain_batch_size 256  --same_lr --lambdap 1.0


python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'cassle_linear_barlow' --pretrain_base_lr 0.10 --pretrain_batch_size 256  --same_lr --lambdap 1.0


#Ering + Balow Twins
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256 --msize 60 

 #cassle linear
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_linear_barlow2' --pretrain_base_lr 0.10 --pretrain_batch_size 256  --same_lr --lambdap 1.0


 #cassle cosine
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_cosine_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 1.0 --start_chkpt 1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_cosine_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --start_chkpt 1

python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350  --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_cosine_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 1.0 --start_chkpt 0
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 350,350,350,350,350,350,350,350,350,350  --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 3 --appr 'cassle_cosine_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --start_chkpt 0


 #cassle cosine_linear barlow
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_cosine_linear_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 1.0 --start_chkpt 1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'cassle_cosine_linear_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --start_chkpt 1

python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 0 --appr 'cassle_cosine_linear_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 1.0 --start_chkpt 0
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 1 --appr 'cassle_cosine_linear_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 2.5 --start_chkpt 0


#cosine + ering
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --appr 'cosine_ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --apply_ering 1 --apply_cosine 1 --cuda_device 1 --start_chkpt 1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --appr 'cosine_ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --apply_ering 0 --apply_cosine 1 --cuda_device 2 --start_chkpt 1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --appr 'cosine_ering_barlow' --pretrain_base_lr 0.3 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --lambdacs 5.0 --apply_ering 1 --apply_cosine 0 --cuda_device 3 --start_chkpt 1\



#cassle simsiam

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 0 --appr 'cassle_simsiam' --pretrain_base_lr 0.03 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'cassle_simsiam' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --same_lr
#best
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 3 --appr 'cassle_simsiam' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr #normally 500-500-500

python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --normalization batch  --cuda_device 3 --appr 'cassle_simsiam' --pretrain_base_lr 0.03 --pretrain_batch_size 256  --same_lr --lambdap 1.0
#best
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --normalization batch  --cuda_device 4 --appr 'cassle_simsiam' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --same_lr --lambdap 1.0
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --normalization batch  --cuda_device 0 --appr 'cassle_simsiam' --pretrain_base_lr 0.10 --pretrain_batch_size 256  --same_lr --lambdap 1.0



#cassle infomax
#best
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 2 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr

python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 2 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.05  --same_lr
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 3 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.20  --same_lr

#best
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --normalization batch  --cuda_device 3 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr

python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --normalization batch  --cuda_device 0 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.05  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'cassle_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.20  --same_lr


#gpm barlow
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --appr 'gpm_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --epsilon 0.9 --cuda_device 2 --start_chkpt 1
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --appr 'gpm_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr --lambdap 1.0 --epsilon 0.93 --cuda_device 3 --start_chkpt 1



#cassle linear infomax
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'cassle_linear_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --normalization batch  --cuda_device 2 --appr 'cassle_linear_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr

#cassle linear simsiam
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 3 --appr 'cassle_linear_simsiam' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --normalization batch  --cuda_device 2 --appr 'cassle_linear_simsiam' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --same_lr --lambdap 1.0



#basic simsiam
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'basic_simsiam' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350  --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'basic_simsiam' --pretrain_base_lr 0.06 --pretrain_batch_size 256  --same_lr --lambdap 1.0


#basic infomax

python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'basic_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --normalization batch  --cuda_device 1 --appr 'basic_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr

# basic barlow
python3 main_cont.py -cs 20,20,20,20,20 -e 750,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'basic_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 2 --appr 'basic_barlow' --pretrain_base_lr 0.1 --pretrain_batch_size 256  --same_lr

OMP_NUM_THREADS=8

python3 main_cont.py -cs 5,5 -e 1,1 -de 1,1 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'barlow_diffusion' --pretrain_base_lr 0.1 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai  --msize 10000 --image_report_freq 49 --knn_report_freq 1 --class_condition --replay_bs 128 --is_debug 

python3 main_cont.py -cs 5,5 -e 500,500 -de 500,500 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'barlow_diffusion' --pretrain_base_lr 0.1 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai  --msize 10000 --image_report_freq 49 --knn_report_freq 10 --clustering_label --replay_bs 128


#tiny imagenet
python3 main_cont.py -cs 20,20,20,20,20,20,20,20,20,20 -e 2,2,2,2,2,2,2,2,2,2 --dataset tinyImagenet --normalization batch  --cuda_device 5 --appr 'basic_infomax' --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3  --pretrain_batch_size 256 --pretrain_base_lr 0.10  --same_lr
python3 main_cont.py -cs 200 -e 800 --dataset tinyImagenet --sim_loss_weight 500.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 6 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --pretrain_batch_size 512 --la_R 0.1 --la_mu 0.1
python3 main_cont.py -cs 200 -e 800 --dataset tinyImagenet --sim_loss_weight 500.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 6 --appr 'basic_infomax' --pretrain_base_lr 0.01 --pretrain_weight_decay 1e-4 --pretrain_batch_size 512 --la_R 0.1 --la_mu 0.1
python3 main_cont.py -cs 200 -e 800 --dataset tinyImagenet --sim_loss_weight 500.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 6 --appr 'basic_infomax' --pretrain_base_lr 0.1  --pretrain_weight_decay 1e-4 --pretrain_batch_size 512 --la_R 0.1 --la_mu 0.1
