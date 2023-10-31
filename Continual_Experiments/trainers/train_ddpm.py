''' File taken from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py '''


import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from fastprogress import progress_bar

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, AdamW


from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
import tqdm
# from tqdm.auto import tqdm
from models.gaussian_diffusion.basic_unet import EMA
import wandb
import torch.nn as nn
import numpy as np
from diffusers.optimization import get_cosine_schedule_with_warmup
# from accelerate import Accelerator


from models.gaussian_diffusion.eval_utils.fid_evaluation import FIDEvaluation
# diffusion_models.ddpm.fid_evaluation import FIDEvaluation
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
class train_diffusion(object):
    def __init__(
        self,
        diffusion_model,
        noise_scheduler,
        dl, #folder 
        test_dl,
        device,
        args,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_epochs = 1000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 10000,
        save_best_and_latest_only =True
    ):
        super().__init__()

        # let's keep it simple for smaller dataset for now?
        # accelerator
        # self.accelerator = Accelerator(
        #     split_batches = split_batches,
        #     mixed_precision = mixed_precision_type if amp else 'no'
        # )
        self.args = args
        self.results_folder = './results_wd_'+str(self.args.pretrain_weight_decay)+'_model_'+str(self.args.unet_model)

        # model
        self.device = device
        self.model = diffusion_model
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.noise_scheduler = noise_scheduler
        self.noise_steps = args.num_train_timesteps
        # self.channels = diffusion_model.channels
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'
        
        self.epochs = train_epochs
        # self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm
        self.dl = dl #train dataloader

        # optimizer
        
        # self.optimizer = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(0.995)
        # self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        # self.ema.to(self.device)

        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.mse = nn.MSELoss()

        # step counter state

        # self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        # self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation
        self.calculate_fid = args.calculate_fid
            # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
        #                                          steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.optimizer = AdamW(self.model.parameters(), lr=train_lr,  weight_decay=args.pretrain_weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(
                            optimizer=self.optimizer,
                            num_warmup_steps=50,
                            num_training_steps=(len(self.dl) * args.epochs[0]),)
        self.scaler = torch.cuda.amp.GradScaler()

        # if self.calculate_fid:
        #     if not is_ddim_sampling:
        #         self.accelerator.print(
        #             "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
        #             "Consider using DDIM sampling to save time."
        #         )
        self.test_dl = test_dl
        self.fid_scorer = FIDEvaluation(
            batch_size=self.batch_size,
            dl=test_dl, #TODO: Train or eval dataloader?
            sampler=self.ema_model,
            channels=3,
            accelerator=None,
            stats_dir=results_folder,
            device=self.device,
            num_fid_samples=num_fid_samples,
            inception_block_idx=inception_block_idx
        )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    # def device(self):
    #     return self.accelerator.device

    def save(self):
        # if not self.accelerator.is_local_main_process:
        #     return

        data = {
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': '1.9.0'
        }

        torch.save(data, str(self.args.pretrain_weight_decay)+'.pt')

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def train(self):
        cfg_scale = 1
        # accelerator = self.accelerator
        # device = accelerator.device
        device = self.device
        self.model.to(device)
        self.ema_model.to(device)

        for x1, x2, y in self.dl:
            x1 = (x1 / 2 + 0.5).clamp(0, 1)
            sampled_images = (x1 * 255).type(torch.uint8) # To plot
            wandb.log({"Original Images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
            utils.save_image(x1,  str(self.results_folder / 'Original Images.png'), nrow = int(8))
            sampled_images = []
            break

        for epoch in range(self.args.epochs[0]):
            loss_ = []
            self.model.train()
            for images, images2, labels in self.dl:
                self.optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                timesteps = self.sample_timesteps(images.shape[0]).to(self.device)
                # timesteps = torch.randint(0, self.noise_steps-1, (images.shape[0],), device=self.device).long()
                # timesteps = torch.linspace(0, self.noise_steps-1, images.shape[0]).long().to(self.device)
                noise = torch.randn_like(images)
                x_t = self.noise_scheduler.add_noise(images, noise, timesteps)
                # x_t, noise = self.noise_images(images, t)

                # if np.random.random() < 0.1:
                #     labels = None

                if self.args.unet_model == 'diffusers':
                    predicted_noise = self.model(x_t, timesteps, labels, return_dict=False)[0]
                else:
                    predicted_noise = self.model(x_t, timesteps, labels)
                loss = self.mse(noise, predicted_noise)
                self.scaler.scale(loss).backward()
                # # Unscales the gradients of optimizer's assigned params in-place
                # self.scaler.unscale_(self.optimizer)

                # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # loss.backward()


                # # self.scaler.clip_grad_norm_(self.model.parameters(), 1.0)
                # self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.model)
                self.scheduler.step()
                loss_.append(loss.item())
                
            
            wandb.log({"train_mse": np.mean(loss_),
                            "learning_rate": self.scheduler.get_last_lr()[0]})
            print(f'Epoch {epoch:2d}  | Loss: {np.mean(loss_):.4f}')

            #if eval
            if (epoch+1) % self.args.knn_report_freq == 0:
                print("sampling")
                wandb.log({" train loss ":np.mean(loss_), "epoch": epoch})
                self.ema_model.eval()
                # labels = torch.arange(self.num_classes).long().to(self.device)
                n = 80
                if n > 10:
                    num = n/10
                    labels = torch.tensor([[i]*8 for i in range(10)]).flatten().to(self.device) #sample 8 Images per class
                # Sample
                with torch.inference_mode():
                    x = torch.randn((n, 3, self.args.image_size, self.args.image_size)).to(self.device)
                    for i, t in enumerate(tqdm.tqdm(self.noise_scheduler.timesteps)): #timesteps in reverse order
                        with torch.no_grad():
                            if self.args.unet_model == 'diffusers':
                                predicted_noise = self.ema_model(x, t, labels).sample
                            else:
                                ts = (torch.ones(n) * t).long().to(self.device)
                                predicted_noise = self.ema_model(x, ts, labels)
                        x = self.noise_scheduler.step(predicted_noise, t, x).prev_sample
                    # x = (x / 2 + 0.5).clamp(0, 1)
                    x = (x.clamp(-1, 1) + 1) / 2 #unnormalize
                    utils.save_image(x,  str(self.results_folder / 'emasample-.png'), nrow = int(math.sqrt(n)))

                    sampled_images = (x * 255).type(torch.uint8) # To plot
                    wandb.log({"sampled_images (ema model)":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

                    # if (epoch+1) % 1 == 0:
                    

                    self.model.eval()
                    x = torch.randn((n, 3, self.args.image_size, self.args.image_size)).to(self.device)
                    for i, t in enumerate(tqdm.tqdm(self.noise_scheduler.timesteps)):
                        with torch.no_grad():
                            if self.args.unet_model == 'diffusers':
                                predicted_noise = self.model(x, t, labels).sample
                            else:
                                ts = (torch.ones(n) * t).long().to(self.device)
                                predicted_noise = self.model(x, ts, labels)
                        x = self.noise_scheduler.step(predicted_noise, t, x).prev_sample
                    # x = (x / 2 + 0.5).clamp(0, 1)
                    x = (x.clamp(-1, 1) + 1) / 2 #unnormalize
                    utils.save_image(x,  str(self.results_folder / 'sample-.png'), nrow = int(math.sqrt(n)))

                    sampled_images = (x * 255).type(torch.uint8) # To plot
                    wandb.log({"sampled_images (model)":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
                    #'/sample-'+str(epoch)+'.png'
                    # if (epoch+1) % 1 == 0:
                    #     utils.save_image(sampled_images.long(), str(self.results_folder / 'sample-.png'), nrow = int(math.sqrt(n)))

                if self.calculate_fid:
                    print("calculating fid")
                    # all_images = []
                    for i in range(20):
                        n = 500
                        with torch.inference_mode():
                            labels = torch.tensor([[i]*50 for i in range(10)]).flatten().to(self.device)
                            x = torch.randn((n, 3, self.args.image_size, self.args.image_size)).to(self.device)
                            for i, t in enumerate(tqdm.tqdm(self.noise_scheduler.timesteps)):
                                with torch.no_grad():
                                    if self.args.unet_model == 'diffusers':
                                        predicted_noise = self.ema_model(x, t, labels).sample
                                    else:
                                        ts = (torch.ones(n) * t).long().to(self.device)
                                        predicted_noise = self.ema_model(x, ts, labels)
                                x = self.noise_scheduler.step(predicted_noise, t, x).prev_sample
                            # x = (x / 2 + 0.5).clamp(0, 1)
                            x = x.clamp(-1, 1)
                            # sampled_images = (x * 255).type(torch.uint8) # To plot
                            if i == 0:
                                all_images = sampled_images
                            else:
                                all_images = torch.cat((all_images, sampled_images), dim = 0)
                    print(len(all_images))
                    fid_score = self.fid_scorer.fid_score(all_images, n)
                    # accelerator.print(f'fid_score: {fid_score}')
                    wandb.log({"FID Score ": fid_score, "epoch ": epoch})
                    print(fid_score)
                    # wandb.log({"FID Score ": total_loss, "Step ": fid_score})
                    if self.save_best_and_latest_only:
                        if self.best_fid > fid_score:
                            self.best_fid = fid_score
                #         self.save("best")
                #     self.save("latest")
                # else:
        # self.save()
        data = {
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': '1.9.0'
        }
        torch.save(data, './results/wd_'+str(self.args.pretrain_weight_decay)+'lr_'+str(self.args.pretrain_base_lr)+'model_'+str(self.args.unet_model)+'.pt')






        # with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
        #     while self.step < self.train_num_steps:

        #         total_loss = 0.
        #         dataloader_iterator = iter(self.dl)
        #         for _ in range(self.gradient_accumulate_every):
        #             try:
        #                 data = next(dataloader_iterator)[0].to(device)
        #             except StopIteration:
        #                 dataloader_iterator = iter(old_data_loader)
        #                 data = next(dataloader_iterator)[0].to(device)
        #             # print(data)

        #             with self.accelerator.autocast():

        #                 loss = self.model(data)
        #                 # print(loss)
        #                 loss = loss / self.gradient_accumulate_every
        #                 total_loss += loss.item()

        #             self.accelerator.backward(loss)

        #         pbar.set_description(f'loss: {total_loss:.4f}')

        #         accelerator.wait_for_everyone()
        #         accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        #         self.opt.step()
        #         self.opt.zero_grad()

        #         accelerator.wait_for_everyone()

        #         self.step += 1
        #         if accelerator.is_main_process:
        #             self.ema.update()

        #             if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
        #                 self.ema.ema_model.eval()

        #                 with torch.inference_mode():
        #                     milestone = self.step // self.save_and_sample_every
        #                     batches = num_to_groups(self.num_samples, self.batch_size)
        #                     all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

        #                 all_images = torch.cat(all_images_list, dim = 0)

        #                 utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

        #                 # whether to calculate fid

        #                 if self.calculate_fid:
        #                     fid_score = self.fid_scorer.fid_score()
        #                     accelerator.print(f'fid_score: {fid_score}')
        #                     wandb.log({"FID Score ": total_loss, "Step ": fid_score})
        #                 if self.save_best_and_latest_only:
        #                     if self.best_fid > fid_score:
        #                         self.best_fid = fid_score
        #                         self.save("best")
        #                     self.save("latest")
        #                 else:
        #                     self.save(milestone)

        #         pbar.update(1)
        #         wandb.log({"Train Loss ": total_loss, "Step ": self.step})
                



        # accelerator.print('training complete')




                # #call sample fuction and return images
                # all_images = x
                # wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
               

                    #backup
                    # for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                    # print()
                        # t = (torch.ones(n) * i).long().to(self.device)
                        # predicted_noise = self.model(x, t, labels)
                        # if cfg_scale > 0:
                        #     uncond_predicted_noise = self.model(x, t, None)
                        #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                        # predicted_noise = predicted_noise.to(self.device)
                        # # t = (torch.ones(n) * i).long().to('cpu')
                        # x = self.noise_scheduler.step(predicted_noise, t[0], x).prev_sample
                
                # x = (x / 2 + 0.5).clamp(0, 1)
                # # x = (x.clamp(-1, 1) + 1) / 2 #unnormalize
                # sampled_images = (x * 255).type(torch.uint8) # To plot

                # #call sample fuction and return images
                # all_images = x
                # wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
                #  #wandb plot: utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                
                