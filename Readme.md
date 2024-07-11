# CroMo-Mixup
This repository will hold the official code of CroMo-Mixup framework introduced in our ECCV2024 paper "CroMo-Mixup: Augmenting Cross-Model Representations for Continual Self-Supervised Learning".

![](/utils/CroMo-Mixup.jpg)

## Task Confusion Experimets:
To run Task Confusion Experiments, follow the [readme.md](/Batch_Split/Readme.md) file in the Batch_Split directory.

## CroMo-Mixup Experiments:

### CIFAR10
The following are the  commands to run CroMo-Mixup with Barlow Twins, CorInfomax, BYOL, and SimCLR.
```
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --appr barlow_mixed_distillation --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2  --pretrain_base_lr 0.25 --msize 500 --knn_report_freq 25 --replay_bs 64 --pretrain_batch_size 256 --lambdap 1.0 --start_chkpt 0 --num_workers 8
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --appr infomax_mixed_distillation --sim_loss_weight 250.0 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5  --pretrain_base_lr 0.1 --msize 500 --knn_report_freq 25 --replay_bs 64 --pretrain_batch_size 256 --lambdap 1.0 --start_chkpt 0 --num_workers 8 --same_lr
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --appr byol_mixed_distillation --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-5 --cuda_device 3 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-5  --knn_report_freq 25 --pretrain_batch_size 256 --normalization group --weight_standard --msize 500 --bsize 64 --start_chkpt 1
python3 main_cont.py -cs 5,5 -e 500,500 --dataset cifar10 --appr simclr_mixed_distillation --proj_out 128 --proj_hidden 2048 --cuda_device 2 --pretrain_batch_size 512  --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --msize 500 --replay_bs 64 --knn_report_freq 25
```
### CIFAR100
The following are the  commands to run CroMo-Mixup with Barlow Twins, CorInfomax, BYOL, and SimCLR on  5 task setting.

```
python3 main_cont.py -cs 20,20,20,20,20 -e 750,750,750,750,750 --dataset cifar100 --appr barlow_mixed_distillation --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 6  --pretrain_base_lr 0.25 --pretrain_batch_size 256 --same_lr --msize 500 --replay_bs 64 --knn_report_freq 50 --lambdap 1.0 --start_chkpt 0 --num_workers 8
python3 main_cont.py -cs 20,20,20,20,20 -e 750,750,750,750,750 --dataset cifar100 --appr infomax_mixed_distillation --sim_loss_weight 250.0 --scale_loss 0.1 --normalization batch --cuda_device 2 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_weight_decay 1e-4 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_batch_size 256 --pretrain_base_lr 0.10 --same_lr  --msize 100 --replay_bs 64 --knn_report_freq 25 --start_chkpt 0 --num_workers 8 --lambdap 1.0
python3 main_cont.py -cs 20,20,20,20,20 -e 750,750,750,750,750 --dataset cifar100 --appr byol_mixed_distillation  --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-5 --cuda_device 6 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-5  --knn_report_freq 25 --pretrain_batch_size 256 --normalization group --weight_standard --msize 500 --bsize 64 --start_chkpt 1 --lambdap 1
python3 main_cont.py -cs 20,20,20,20,20 -e 750,750,750,750,750 --dataset cifar100 --appr simclr_mixed_distillation --proj_out 128 --proj_hidden 2048 --cuda_device 6 --pretrain_batch_size 512  --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --msize 500 --replay_bs 64 --knn_report_freq 25
```
The following are the  commands to run CroMo-Mixup with Barlow Twins, CorInfomax, BYOL, and SimCLR on 10 task setting.

```
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --appr barlow_mixed_distillation --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4  --pretrain_base_lr 0.25 --pretrain_batch_size 256 --same_lr --lambdap 1.0 --msize 100 --replay_bs 64 --knn_report_freq 50 --start_chkpt 0 --num_workers 8
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --appr infomax_mixed_distillation --sim_loss_weight 250.0 --scale_loss 0.1 --normalization batch --cuda_device 3 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_weight_decay 1e-4 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_batch_size 256 --pretrain_base_lr 0.10 --same_lr  --msize 100 --replay_bs 64 --knn_report_freq 25 --start_chkpt 0 --num_workers 8 --lambdap 1.0
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 --epochs 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --appr byol_mixed_distillation --sim_loss_weight 1000.0 --proj_out 256 --proj_hidden 4096 --pred_hidden 4096 --pred_out 256 --min_lr 1e-3 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-5 --cuda_device 7 --pretrain_base_lr 1.0 --pretrain_weight_decay 1e-5  --knn_report_freq 25 --pretrain_batch_size 256 --normalization group --weight_standard --msize 100 --bsize 64 --start_chkpt 1
python3 main_cont.py -cs 10,10,10,10,10,10,10,10,10,10 -e 600,350,350,350,350,350,350,350,350,350 --dataset cifar100 --appr simclr_mixed_distillation --proj_out 128 --proj_hidden 2048 --cuda_device 5 --pretrain_batch_size 512  --pretrain_base_lr 0.6 --min_lr 1e-3 --temperature 0.5 --msize 25 --replay_bs 64 --knn_report_freq 25
```
### Tiny-ImageNet:
The following are the  commands to run CroMo-Mixup with Barlow Twins, CorInfomax, BYOL, and SimCLR on 10 task setting.

```
python3 main_cont.py -cs 20,20,20,20,20,20,20,20,20,20 -e 500,350,350,350,350,350,350,350,350,350 --dataset tinyImagenet --appr barlow_mixed_distillation --normalization batch --cuda_device 3  --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_batch_size 256 --pretrain_base_lr 0.10 --pretrain_weight_decay 1e-4 --same_lr --msize 100 --replay_bs 64 --knn_report_freq 50 --num_workers 8
python3 main_cont.py -cs 20,20,20,20,20,20,20,20,20,20 -e 500,350,350,350,350,350,350,350,350,350 --dataset tinyImagenet --appr infomax_mixed_distillation --normalization batch --cuda_device 4  --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_batch_size 256 --pretrain_base_lr 0.10 --sim_loss_weight 500.0 --pretrain_weight_decay 1e-4 --la_R 0.1 --la_mu 0.1 --msize 100 --same_lr --scale_loss 0.1 --replay_bs 64 --knn_report_freq 50 --num_workers 8
python3 main_cont.py -cs 20,20,20,20,20,20,20,20,20,20 -e 500,350,350,350,350,350,350,350,350,350 --dataset tinyImagenet --appr byol_mixed_distillation --proj_out 4096 --proj_hidden 4096 --pred_hidden 4096 --pred_out 4096 --cuda_device 7 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-6 --pretrain_batch_size 256  --normalization group --weight_standard --pretrain_base_lr 0.3 --min_lr 1e-6 --same_lr --lambdap 0.0
python3 main_cont.py -cs 20,20,20,20,20,20,20,20,20,20 -e 500,350,350,350,350,350,350,350,350,350 --dataset tinyImagenet  --appr simclr_mixed_distillation--normalization batch --cuda_device 5 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_batch_size 256 --pretrain_base_lr 0.3 --temperature 0.5 --msize 100 --replay_bs 64 --knn_report_freq 50 --same_lr

```

## Other Baselines Experiments:
1. To run finetune baseline, set --appr arg to basic_barlow or basic_infomax or basic_smclr or basic_byol in the  above mentioned run commands.
2. To run ER baseline, set --appr arg to ering_barlow or ering_infomax or ering_smclr or ering_byol in the  above mentioned run commands
3. To run CaSSLe baseline, set --appr arg to cassle_barlow or cassle_infomax or cassle_smclr or cassle_byol in the  above mentioned run commands
4. To run CaSSLe+ baseline, set --appr arg to infomax_cassle_ering or simclr_cassle_ering or byol_cassle_ering or barlow_cassle_ering in the  above mentioned run commands.