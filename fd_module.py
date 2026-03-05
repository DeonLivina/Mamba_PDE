import torch
import torch.nn as nn

class FDModule(nn.Module):
    def __init__(self, dx: float, dt: float):
        super().__init__()
        self.dx = dx
        self.dt = dt


    def spatial_fd_better(self, u: torch.Tensor) -> torch.Tensor:
        du_dx = torch.zeros_like(u)
        # center
        du_dx[:,1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * self.dx)

        #left
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / self.dx

        #right
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / self.dx

        
        return du_dx

    def time_fd(self, u: torch.Tensor) -> torch.Tensor:
        du_dt = torch.zeros_like(u)
        # center
        du_dt[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)

        #left
        du_dt[:, 0] = (u[:, 1] - u[:, 0]) / self.dt

        #right
        du_dt[:, -1] = (u[:, -1] - u[:, -2]) / self.dt

        return du_dt


class ShallowWaterFD(FDModule):
    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        du_dx = self.spatial_fd_better(u)
        dh_dx = self.spatial_fd_better(h)

        return du_dx, dh_dx

class ShallowTimeFD(FDModule):
    def forward(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        du_dt = self.time_fd(u)
        dh_dt = self.time_fd(h)
        return du_dt, dh_dt