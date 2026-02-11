import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import wandb
import matplotlib.pyplot as plt

# Import your models and dataset
from burgers_data import create_burgers_dataloaders

from model import PDE_model, Mamba


class TrainerBurgers:
    """
    Trainer for Mamba PDE model on Burgers equation
    """
    
    def __init__(self, model, device='cpu', checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (u_t, u_next) in enumerate(train_loader):
            u_t = u_t.to(self.device)
            u_next = u_next.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            dx = 2 * np.pi / u_t.shape[-1]
            u_pred = self.model(u_t, dx)
            
            # Compute loss
            loss = criterion(u_pred, u_next)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}: loss = {loss.item():.6f}")
        
        return total_loss / n_batches
    
    def validate(self, val_loader, criterion):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for u_t, u_next in val_loader:
                u_t = u_t.to(self.device)
                u_next = u_next.to(self.device)
                
                dx = 2 * np.pi / u_t.shape[-1]
                u_pred = self.model(u_t, dx)
                
                loss = criterion(u_pred, u_next)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def fit(self, train_loader, val_loader, 
            n_epochs=100, 
            lr=1e-3,
            weight_decay=1e-5,
            use_wandb=False):
        """
        Full training loop
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        if use_wandb:
            wandb.init(project="mamba-pde", name="burgers-1d")
        
        print(f"\n{'='*60}")
        print(f"Training Mamba PDE Model on Burgers Equation")
        print(f"{'='*60}")
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            print(f"  Train loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            print(f"  Val loss: {val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss)
                print(f"  ✓ New best model saved!")
            
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0]
                })
        
        if use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        path = self.checkpoint_dir / f'best_model.pt'
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint
    
    def test_rollout(self, u0, dx, n_steps, device=None):
        """
        Test the model's ability to rollout multiple steps
        
        Parameters:
        -----------
        u0 : torch.Tensor (n_spatial,)
            Initial condition
        dx : float
            Spatial step
        n_steps : int
            Number of steps to rollout
        
        Returns:
        --------
        trajectory : torch.Tensor (n_steps+1, n_spatial)
        """
        self.model.eval()
        device = device or self.device
        
        u0 = u0.to(device)
        trajectory = [u0.unsqueeze(0)]
        u = u0
        
        with torch.no_grad():
            for step in range(n_steps):
                u = self.model(u, dx)
                trajectory.append(u.unsqueeze(0))
        
        return torch.cat(trajectory, dim=0)


def main():
    """Main training script"""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nCreating Burgers dataset...")
    train_loader, val_loader, info = create_burgers_dataloaders(
        n_samples=50,          # 50 trajectories
        n_spatial=64,          # 64 spatial points
        n_time_steps=50,       # 50 time steps per trajectory
        nu=0.01,               # Viscosity
        dt=0.01,               # Time step
        batch_size=16,
        train_split=0.8
    )
    
    print("\nDataset Info:")
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    # Create model
    print("\nCreating Mamba PDE model...")
    from model import PDE_model  # Import your model
    
    model = PDE_model(
        n_spatial=info['n_spatial'],
        d_model=128,
        n_layers=4,
        d_state=16,
        expand=2,
        n_input_features=3  # [u, du/dx, d2u/dx2]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = TrainerBurgers(model, device=device)
    trainer.fit(
        train_loader, 
        val_loader,
        n_epochs=50,
        lr=1e-3,
        weight_decay=1e-5,
        use_wandb=False  # Set to True if using wandb
    )
    
    # Test rollout
    print("\n" + "="*60)
    print("Testing model rollout capability")
    print("="*60)
    
    # Get a test sample
    u_t_batch, _ = next(iter(val_loader))
    u0 = u_t_batch[0]  # First sample
    
    dx = info['dx']
    n_steps = 20
    
    traj = trainer.test_rollout(u0, dx, n_steps)
    print(f"\nRollout trajectory shape: {traj.shape}")
    print(f"  Initial: min={traj[0].min():.4f}, max={traj[0].max():.4f}")
    print(f"  Final: min={traj[-1].min():.4f}, max={traj[-1].max():.4f}")


if __name__ == "__main__":
    main()