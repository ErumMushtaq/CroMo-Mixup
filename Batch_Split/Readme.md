# Task Confusion Experiments for Cotrastive and Supevised Learning

## Contrastive Learning Baselines:
###  Barlow twins
CIL Experiments
```
python3 main.py -cs 100 --epochs 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr barlow --pretrain_base_lr 0.1 --pretrain_batch_size 256 --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 --exp_type basic --algo barlow --appr barlow

python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr barlow --pretrain_base_lr 0.1 --pretrain_batch_size 256 --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 --exp_type basic --algo barlow --appr barlow
```

DIL Experiments
```
python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 -dl_type data_incremental --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr barlow --pretrain_base_lr 0.1 --pretrain_batch_size 256 --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 --exp_type basic --algo barlow --appr barlow 
```

### BYOL 
CIL Experiments

```
python3 main.py -cs 100 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-6 --normalization batch --cuda_device 2 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-6 --algo byol --appr byol --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --pretrain_batch_size 1024

python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-6 --normalization group --weight_standard --cuda_device 3 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-6 --algo byol --appr byol --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --pretrain_batch_size 1024

```
DIL Experiments
```
python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 -dl_type data_incremental --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-6 --normalization group --weight_standard --cuda_device 4 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-6 --algo byol --appr byol --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --pretrain_batch_size 1024 
```

###  SImCLR
CIL Experiments
```
python main.py -cs 100 --epochs 1000 --dataset cifar100 --proj_out 128 --proj_hidden 2048 --cuda_device 5 --pretrain_batch_size 512 --appr simclr --algo simclr --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --knn_report_freq 25 --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 

python main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 --proj_out 128 --proj_hidden 2048 --cuda_device 0 --pretrain_batch_size 512 --appr simclr --algo simclr --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --knn_report_freq 1 --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10

```
DIL Experiments
```
python main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 -dl_type data_incremental --proj_out 128 --proj_hidden 2048 --cuda_device 1 --pretrain_batch_size 512 --appr simclr --algo simclr --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --knn_report_freq 25 --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 

```
### CorInfomax
CIL Experiments
```
python3 main.py -cs 100 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 0 --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --weight_standard --algo infomax --appr infomax --exp_type basic -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10

python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 1 --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --weight_standard --algo infomax --appr infomax --exp_type basic --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10

```
DIL Experiments
```
python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 1000 --dataset cifar100 -dl_type data_incremental --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 2 --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --weight_standard --algo infomax --appr infomax --exp_type basic --knn_report_freq 100 -vcs 10,10,10,10,10,10,10,10,10,10 
```

## Supervised Learning
CIL Experiments
```
python3 main.py -cs 100 --epochs 200 --dataset cifar100 --min_lr 1e-6 --pretrain_warmup_epochs 0 --pretrain_warmup_lr 3e-3 --pretrain_base_lr 0.075 --pretrain_weight_decay 5e-4 --normalization group --cuda_device 2 --pretrain_batch_size 128 --algo supervised --weight_standard -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --sup_type type2

python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 200 --dataset cifar100 --min_lr 1e-6 --pretrain_warmup_epochs 0 --pretrain_warmup_lr 3e-3 --pretrain_base_lr 0.075 --pretrain_weight_decay 5e-4 --normalization group --cuda_device 1 --pretrain_batch_size 128 --algo supervised --weight_standard -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --sup_type type2
```
DIL Experiments
```
python3 main.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 200 --dataset cifar100 --min_lr 1e-6 --pretrain_warmup_epochs 0 --pretrain_warmup_lr 3e-3 --pretrain_base_lr 0.075 --pretrain_weight_decay 5e-4 --normalization group --cuda_device 6 --pretrain_batch_size 128 --algo supervised --weight_standard -vcs 10,10,10,10,10,10,10,10,10,10 --knn_report_freq 10 --sup_type type2  -dl_type data_incremental
```
Note: for supervised learning, you might need to use the respective test trasform from the cfar100 dataloader if interested in evaluating the test accuracy.