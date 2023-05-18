import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import wandb

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Attraction factor of CorInfoMax Loss: MSE loss calculation from outputs of the projection network, z1 (NXD) from 
    the first branch and z2 (NXD) from the second branch. Returns loss part comes from attraction factor (mean squared error).
    """
    return F.mse_loss(z1, z2)



class ErrorCovarianceLoss(nn.Module):
    """Big-bang factor of CorInfoMax Loss: loss calculation from outputs of the projection network,
    z1 (NXD) from the first branch and z2 (NXD) from the second branch. Returns loss part comes from bing-bang factor.
    """
    def __init__(self, project_dim,device='cpu'):
        super(ErrorCovarianceLoss, self).__init__()
        proj_output_dim = project_dim
        la_R = 0.01
        R_ini = 1.0
        R_eps_weight = 1e-8
        self.Re = R_ini*torch.eye(proj_output_dim , dtype=torch.float64, requires_grad=False).to(device)
        self.new_Re = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,  requires_grad=False).to(device)
       

        self.la_R = la_R

        self.R_eps_weight = R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,device = None) -> torch.Tensor:
        la_R = self.la_R
        N, D = z1.size()
        # covariance matrix estimation
        ze_hat =  z1 - z2
   
        Re_update = (ze_hat.T @ ze_hat) / N
    
        self.new_Re = la_R*(self.Re) + (1-la_R)*(Re_update)
      

        # loss calculation 
        cov_err_loss =  (torch.logdet(self.new_Re + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.Re = self.new_Re.detach()

        return cov_err_loss 




class CovarianceLoss(nn.Module):
    """Big-bang factor of CorInfoMax Loss: loss calculation from outputs of the projection network,
    z1 (NXD) from the first branch and z2 (NXD) from the second branch. Returns loss part comes from bing-bang factor.
    """
    def __init__(self, project_dim,device='cpu',la_mu=0.01, la_R=0.01, R_eps_weight=1e-8):
        super(CovarianceLoss, self).__init__()
        proj_output_dim = project_dim
        la_R = la_R
        la_mu = la_mu

        R_ini = 1.0
        R_eps_weight = R_eps_weight
        self.R1 = R_ini*torch.eye(proj_output_dim , dtype=torch.float64, requires_grad=False).to(device)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.R2 = R_ini*torch.eye(proj_output_dim , dtype=torch.float64,  requires_grad=False).to(device)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,  requires_grad=False).to(device)
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device) 
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,  requires_grad=False).to(device) 
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device)
        self.la_R = la_R
        self.la_mu = la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,device = None) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

    

        N, D = z1.size()

        # mean estimation
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        # covariance matrix estimation
        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        # loss calculation 
        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        # print(torch.max(torch.real(self.R_eigs)))
        # print(torch.min(torch.real(self.R_eigs)))
        # print(torch.min(self.R_eigs))

        return cov_loss 

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
        return R_eig_arr 

    def plot_eigs(self, epoch_counter) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
            wandb.log({" Max Eig Value ": np.max(R_eig_arr), " Epoch ": epoch_counter})  
            wandb.log({" Min Eig Value ":  np.min(R_eig_arr), " Epoch ": epoch_counter})
            wandb.log({" Avg Eig Value ":  np.mean(R_eig_arr), " Epoch ": epoch_counter})
        # return R_eig_arr 

class BarlowTwinsLoss(torch.nn.Module):
    #Ref: https://github.com/lightly-ai/lightly/blob/master/lightly/loss/barlow_twins_loss.py
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.0323
    """

    def __init__(self, lambda_param: float = 5e-3, scale_loss = 0.025, gather_distributed: bool = False):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.scale_loss = scale_loss
        self.gather_distributed = gather_distributed

        # moving average code


    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = self.scale_loss*c_diff.sum()

        return loss


class BarlowTwinsLoss_ema(torch.nn.Module):
    #Ref: https://github.com/lightly-ai/lightly/blob/master/lightly/loss/barlow_twins_loss.py
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.0323
    """

    def __init__(self, project_dim, device, lambda_param: float = 5e-3, gather_distributed: bool = False):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss_ema, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

        # moving average code
        proj_output_dim = project_dim
        # la_R = 0
        la_R = 0.01
        # la_mu = 0
        la_mu = 0.01
        R_ini = 1.0
        R_eps_weight = 1e-8
        self.c = R_ini*torch.eye(proj_output_dim , dtype=torch.float64, requires_grad=False).to(device)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.std1 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.std2 = torch.zeros(proj_output_dim, dtype=torch.float64, requires_grad=False).to(device)
        self.new_c = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64,  requires_grad=False).to(device)
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device) 
        self.new_std1 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device) 
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device)
        self.new_std2 = torch.zeros(proj_output_dim, dtype=torch.float64,  requires_grad=False).to(device)
        self.la_R = la_R
        self.la_mu = la_mu
        

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        device = z_a.device
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z_a.size()

        # mean estimation
        mu_update1 = torch.mean(z_a, 0)
        mu_update2 = torch.mean(z_b, 0)
        std_update1 = torch.std(z_a, 0)
        std_update2 = torch.std(z_b, 0)

        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        self.new_std1 = la_mu*(self.std1) + (1-la_mu)*(std_update1)
        self.new_std2 = la_mu*(self.std2) + (1-la_mu)*(std_update2)

        # covariance matrix estimation
        z1_hat =  (z_a - self.new_mu1)/self.new_std1
        z2_hat =  (z_b - self.new_mu2)/self.new_std2


        c_update = (z1_hat.T @ z2_hat) / N
        self.new_c = la_R*(self.c) + (1-la_R)*(c_update)

        # loss calculation 
        c_diff = (self.new_c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        # cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.c = self.new_c.detach()
        self.mu1 = self.new_mu1.detach()
        self.mu2 = self.new_mu2.detach()
        self.std1 = self.new_std1.detach()
        self.std2 = self.new_std2.detach()
        


        # # normalize repr. along the batch dimension
        # z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        # z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        # N = z_a.size(0)
        # D = z_a.size(1)

        # # cross-correlation matrix
        # c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD

        # # sum cross-correlation matrix between multiple gpus
        # if self.gather_distributed and dist.is_initialized():
        #     world_size = dist.get_world_size()
        #     if world_size > 1:
        #         c = c / world_size
        #         dist.all_reduce(c)

        # # loss
        # c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # # multiply off-diagonal elems of c_diff by lambda
        # c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        # loss = c_diff.sum()

        return loss