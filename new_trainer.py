from statistics import harmonic_mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import warnings
import os
import numpy as np

from fd_module import FDModule, ShallowWaterFD
from physics_loss import PhysicsLoss

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LR = 1e-4
DX = 0.1
DT = 0.005
class MyTrainer:
    def __init__(self, model: nn.Module, device: str, loss_fn: nn.Module):
        self.model = model.to(device)
        self.loss_fn = nn.MSELoss()
        self.physics_criterion = PhysicsLoss(dx=DX, dt=DT, g= 9.81) 
        self.device = device
        self.fdmodel = ShallowWaterFD(dx=DX, dt=DT)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )

    def train_epoch(self, Loader: DataLoader) -> float:
        self.model.train()

        total_loss = 0.0
        total_data_loss = 0.0
        total_seq_loss = 0.0
        total_physics = 0.0
        n_batches = 0 

        pbar = tqdm(Loader, desc="Train", leave=False)
        for context, targets in pbar:
            context = context.to(self.device)
            targets = targets.to(self.device)

            B, N, T, F = context.shape
            _, _, T_out, F_out = targets.shape

            context_flat = context.reshape(B*N, T, F)
            du_dx, dh_dx = self.fdmodel(context_flat[:, :, 0], context_flat[:, :, 1])
            
            # avoid blowup while preserving shock signals 
            du_dx = torch.sign(du_dx) * torch.log1p(du_dx.abs())
            dh_dx = torch.sign(dh_dx) * torch.log1p(dh_dx.abs())

            context_flat = torch.cat([context_flat, du_dx.unsqueeze(-1), dh_dx.unsqueeze(-1)], dim=2)
            
            target_flat = targets.reshape(B*N, T_out, F_out)

            # Two predictions
            context_1 = context_flat
            context_2 = context_flat[:, 1:, :]

            pred1 = self.model(context_1)
            pred2 = self.model(context_2)

            data_loss = self.loss_fn(pred1[:, 1:, :], target_flat[:, 1:, :])

            overlap1 = pred1[:, 1:, :]      # drop first
            overlap2 = pred2[:, :-1, :]     # drop last

            seq_loss = self.loss_fn(overlap1, overlap2)


            # Physics loss
            u = pred1[:, :, 0]
            h = pred1[:, :, 1]
            u_mean = 0.7404
            h_mean = 2.0291
            u_std = 1.1907
            h_std = 1.0724
            
            pde_loss = self.physics_criterion(u, h, u_mean, h_mean, u_std, h_std)
            
             # weights
            lam_seq = 0.1
            lam_physics = 0.002 

            loss =  data_loss + lam_seq * seq_loss #+ lam_physics * pde_loss


            # backpropagation
            self.optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_data_loss += data_loss.item()
            total_seq_loss += seq_loss.item()
            total_loss += loss.item()
            total_physics += pde_loss.item()
            n_batches += 1

            pbar.set_postfix({
            "data_loss": f"{data_loss.item():.4f}",
            "seq_loss": f"{seq_loss.item():.4f}",
            "physics_loss": f"{pde_loss.item():.4f}"
            })

        return {
            "total" : total_loss/n_batches,
            "data" : total_data_loss/n_batches,
            "seq" : total_seq_loss/n_batches,
            "physics" : total_physics/n_batches
                }
    def validation_epoch(self, Loader: DataLoader) -> float:
        
        self.model.eval()

        total_valid_loss = 0.0
        n_batches = 0 

        dx = 0.1
        dt = 0.005

        pbar = tqdm(Loader, desc="Val", leave=False)

        with torch.no_grad():
            for context, targets in pbar:
                context = context.to(self.device)
                targets = targets.to(self.device)

                B, N, T, F = context.shape
                _, _, T_out, F_out = targets.shape

                context_flat = context.reshape(B*N, T, F)
                du_dx, dh_dx = self.fdmodel(context_flat[:, :, 0], context_flat[:, :, 1])
                du_dx = torch.sign(du_dx) * torch.log1p(du_dx.abs())
                dh_dx = torch.sign(dh_dx) * torch.log1p(dh_dx.abs())
                context_flat = torch.cat([context_flat, du_dx.unsqueeze(-1), dh_dx.unsqueeze(-1)], dim=2)
                target_flat = targets.reshape(B*N, T_out, F_out)

                pred = self.model(context_flat)
            
                loss = self.loss_fn(pred, target_flat)

                total_valid_loss += loss.item()
                n_batches += 1

                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        return total_valid_loss/n_batches
