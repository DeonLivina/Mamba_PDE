import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import os

from fd_module import ShallowWaterFD
from physics_loss import PhysicsLoss

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

LR = 1e-4
DX = 0.1
DT = 0.005


class MyTrainer:
    def __init__(self, model: nn.Module, device: str, loss_fn: nn.Module):
        self.model = model.to(device)
        self.loss_fn = nn.MSELoss()
        self.physics_criterion = PhysicsLoss(dx=DX, dt=DT, g=9.81) 
        self.device = device
        self.fdmodel = ShallowWaterFD(dx=DX, dt=DT)
        
        # L-BFGS Setup
        # Note: lr=1 is standard for L-BFGS; max_iter controls internal re-evaluations
        self.optimizer = torch.optim.LBFGS(
            model.parameters(), 
            lr=1, 
            max_iter=20, 
            history_size=10, 
            line_search_fn="strong_wolfe"
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )

    def train_epoch(self, Loader: DataLoader) -> dict:
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

            # Prepare inputs outside the closure to save compute
            context_flat = context.reshape(B*N, T, F)
            du_dx, dh_dx = self.fdmodel(context_flat[:, :, 0], context_flat[:, :, 1])
            du_dx = torch.sign(du_dx) * torch.log1p(du_dx.abs())
            dh_dx = torch.sign(dh_dx) * torch.log1p(dh_dx.abs())
            context_flat = torch.cat([context_flat, du_dx.unsqueeze(-1), dh_dx.unsqueeze(-1)], dim=2)
            
            target_flat = targets.reshape(B*N, T_out, F_out)
            context_1 = context_flat
            context_2 = context_flat[:, 1:, :]

            # Local variables to capture loss for logging
            # (Closure needs to return a single loss, but we want to track sub-losses)
            step_losses = {"data": 0, "seq": 0, "phys": 0, "total": 0}

            def closure():
                self.optimizer.zero_grad()
                
                # Predictions
                pred1 = self.model(context_1)
                pred2 = self.model(context_2)

                # Losses
                d_loss = self.loss_fn(pred1[:, 1:, :], target_flat[:, 1:, :])
                
                overlap1 = pred1[:, 1:, :]
                overlap2 = pred2[:, :-1, :]
                s_loss = self.loss_fn(overlap1, overlap2)

                u, h = pred1[:, :, 0], pred1[:, :, 1]
                u_mean, h_mean, u_std, h_std = 0.7404, 2.0291, 1.1907, 1.0724
                p_loss = self.physics_criterion(u, h, u_mean, h_mean, u_std, h_std)

                lam_seq, lam_physics = 0.01, 0.2 
                combined_loss = 0.70 * d_loss + lam_seq * s_loss + lam_physics * p_loss

                combined_loss.backward()
                
                # Store values for logging outside closure
                step_losses["data"] = d_loss.item()
                step_losses["seq"] = s_loss.item()
                step_losses["phys"] = p_loss.item()
                step_losses["total"] = combined_loss.item()
                
                return combined_loss

            # L-BFGS step requires the closure function
            self.optimizer.step(closure)

            # Accumulate stats
            total_data_loss += step_losses["data"]
            total_seq_loss += step_losses["seq"]
            total_physics += step_losses["phys"]
            total_loss += step_losses["total"]
            n_batches += 1

            pbar.set_postfix({
                "data": f"{step_losses['data']:.4f}",
                "phys": f"{step_losses['phys']:.4f}"
            })

        return {
            "total": total_loss/n_batches,
            "data": total_data_loss/n_batches,
            "seq": total_seq_loss/n_batches,
            "physics": total_physics/n_batches
        }

    # validation_epoch remains identical to your original code
    def validation_epoch(self, Loader: DataLoader) -> float:
        self.model.eval()
        total_valid_loss = 0.0
        n_batches = 0 
        pbar = tqdm(Loader, desc="Val", leave=False)
        with torch.no_grad():
            for context, targets in pbar:
                context, targets = context.to(self.device), targets.to(self.device)
                B, N, T, F = context.shape
                T_out, F_out = targets.shape[2], targets.shape[3]
                
                context_flat = context.reshape(B*N, T, F)
                du_dx, dh_dx = self.fdmodel(context_flat[:, :, 0], context_flat[:, :, 1])
                du_dx = torch.sign(du_dx) * torch.log1p(du_dx.abs())
                dh_dx = torch.sign(dh_dx) * torch.log1p(dh_dx.abs())
                context_flat = torch.cat([context_flat, du_dx.unsqueeze(-1), dh_dx.unsqueeze(-1)], dim=2)
                
                pred = self.model(context_flat)
                loss = self.loss_fn(pred, targets.reshape(B*N, T_out, F_out))
                total_valid_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        return total_valid_loss/n_batches