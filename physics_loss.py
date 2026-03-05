import torch
import torch.nn as nn
import numpy as np

from fd_module import ShallowTimeFD, ShallowWaterFD

class PhysicsLoss(nn.Module):
    def __init__(self, dt: float, dx: float, g: float = 9.81):
        super().__init__()

        self.dt = dt
        self.dx = dx
        self.g = g

        self.time_fd = ShallowTimeFD(dx=dx, dt=dt)
        self.spatial_fd = ShallowWaterFD(dx=dx, dt=dt)


    def forward(self, u, h, u_mean, h_mean, u_std, h_std):

        u = (u * u_std) + u_mean
        h = (h * h_std) + h_mean
        du_dx, dh_dx = self.spatial_fd(u, h)
        du_dt, dh_dt = self.time_fd(u, h)

        mass_residual = dh_dt + (u * dh_dx) + (h * du_dx)
        momentum_residual = du_dt + (u * du_dx) + (self.g * dh_dx)

        #print(f"mass residual : {mass_residual} | momentum_residual{momentum_residual}")
        residual_loss = torch.mean(mass_residual ** 2) + torch.mean(momentum_residual ** 2)

        return residual_loss






    

