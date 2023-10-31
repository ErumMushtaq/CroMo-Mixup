python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --R_eps_weight 1e-8

python3 main_cont.py -cs 5,5 --dataset cifar10  -e 500,500 --lambdap 2 --lambda_norm 1.0 --subspace_rate 0.95  --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --normalization group --weight_standard --cuda_device 5 --appr 'infomax_dist_ering' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4  --bsize 256 --msize 1250 --resume_checkpoint

#CIFAR100 infomax ering
python3 main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --sim_loss_weight 1000.0 --proj_out 128 --proj_hidden 4096 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch  --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4  --weight_standard --bsize 256 --msize 1250
main_cont.py -cs 20,20,20,20,20 -e 500,500,500,500,500 --dataset cifar100 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'ering_infomax' --pretrain_base_lr 0.3 --pretrain_batch_size 256 --weight_standard --bsize 256 --msize 1250

python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10  --epochs 100 --pretrain_batch_size 128 --cuda_device 1 --unet_model basic --noise_scheduler DDPM --beta_scheduler 'squaredcos_cap_v2' --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4

python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10  --epochs 100 --pretrain_batch_size 128 --cuda_device 2 --unet_model openai
 --noise_scheduler DDPM --beta_scheduler 'squaredcos_cap_v2' --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4

python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10  --epochs 100 --pretrain_batch_size 128 --cuda_device 3 --unet_model openai --noise_scheduler DDIM --beta_scheduler 'squaredcos_cap_v2' --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4

python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10 --epochs 200 --pretrain_batch_size 256 --cuda_device 2 --unet_model diffusers --noise_scheduler DDPM --beta_scheduler squaredcos_cap_v2 --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4 --pretrain_weight_decay 5e-4

python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10 --epochs 400 --pretrain_batch_size 256 --cuda_device 2 --unet_model diffusers --noise_scheduler DDPM --beta_scheduler squaredcos_cap_v2 --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4 --pretrain_weight_decay 5e-4 [ddpm6]
python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10 --epochs 400 --pretrain_batch_size 256 --cuda_device 3 --unet_model diffusers --noise_scheduler DDPM --beta_scheduler squaredcos_cap_v2 --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4 --pretrain_weight_decay 0.0 [ddpm]
python3 main_cont.py --appr diffusion --dataset cifar10 -cs 10 --epochs 400 --pretrain_batch_size 256 --cuda_device 4 --unet_model diffusers --noise_scheduler DDPM --beta_scheduler squaredcos_cap_v2 --image_size 32 --num_train_timesteps 1000 --pretrain_base_lr 1e-4 --pretrain_weight_decay 1.0 [ddpm7]

python3 main_cont.py -cs 10 -e 1000 --dataset cifar10 --sim_loss_weight 250.0 --proj_out 64 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization group --cuda_device 4 --appr 'basic_infomax' --pretrain_base_lr 0.5 --pretrain_weight_decay 1e-4 --R_eps_weight 1e-8

python3 main_cont.py -cs 5,5 -e 500,500 -de 200,200 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 1 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai --msize 10000



python3 main_cont.py -cs 5,5 -e 500,500 -de 200,200 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 7 --appr 'barlow_diffusion' --pretrain_base_lr 0.1 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai  --msize 10000 --image_report_freq 24 --knn_report_freq 10 --class_condition

python3 main_cont.py -cs 5,5 -e 500,500 -de 200,200 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 2 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai --class_condition --msize 10000

python3 main_cont.py -cs 5,5 -e 500,500 -de 200,200 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 0 --appr 'basic_barlow' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai

python3 main_cont.py -cs 5,5 -e 200,200 -de 100,100 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 4 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai

python3 main_cont.py -cs 5,5 -e 200,200 -de 100,100 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai --class_condition

python3 main_cont.py -cs 5,5 -e 1,1 -de 1,1 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 3 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai --class_condition

python3 main_cont.py -cs 5,5 -e 1,1 -de 1,1 --dataset cifar10 --lambda_param 5e-3 --scale_loss 0.1 --proj_out 2048 --proj_hidden 2048 --min_lr 1e-6 --pretrain_warmup_epochs 10 --pretrain_warmup_lr 3e-3 --pretrain_weight_decay 1e-4 --normalization batch --cuda_device 5 --appr 'barlow_diffusion' --pretrain_base_lr 0.3 --diff_train_bs 128 --sample_bs 64 --diff_weight_decay 5e-4 --diff_train_lr 1e-4 --unet_model openai --class_condition --image_report_freq 1 --knn_report_freq 1 --clustering_label

