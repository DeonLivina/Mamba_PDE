import torch
import torch.nn as nn
from mamba_blocks import MambaBlock, MambaConfig


class TemporalMamba(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 32,
        n_layers: int = 4,
        expand: int = 2,
        n_input_features: int = 3,
        n_output_features: int = 2,
        n_future: int = 10,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_future = n_future
        self.n_output_features = n_output_features

        # Input projection
        self.input_proj = nn.Linear(n_input_features, d_model)
        #Output projection
        self.output_proj = nn.Linear(d_model, n_future * n_output_features)
        # Optimized Mamba blocks (with pscan)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(MambaConfig(
                d_model=d_model,
                d_state=d_state,
                n_layers=1,
                expand_factor=expand,
                pscan=True,
            ))
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, temporal_seq: torch.Tensor) -> torch.Tensor:

        x = self.input_proj(temporal_seq)  # (B, n_context, d_model)
        
        # Process through optimized Mamba blocks
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Normalize and project
        x = self.norm(x)
        h_final = x[:, -1, :]  # (B, d_model)
        u_future = self.output_proj(h_final)  # (B, n_future * n_output_features)
        u_future = u_future.reshape(-1, self.n_future, self.n_output_features)
        return u_future


class TemporalMambaWithSpatial(nn.Module):
    
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 32,
        n_layers: int = 4,
        expand: int = 2,
        n_input_features: int = 3,
        n_future: int = 10,
        n_output_features: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_future = n_future
        self.n_output_features = n_output_features
        
        self.temporal_mamba = TemporalMamba(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            expand=expand,
            n_input_features=n_input_features,
            n_future=n_future,
            n_output_features=n_output_features
        )
    
    def forward(self, temporal_seq: torch.Tensor) -> torch.Tensor:

        input_shape = temporal_seq.shape
        
        if len(input_shape) == 3:
            return self.temporal_mamba(temporal_seq)
        
        elif len(input_shape) == 4:
            B, N, T, F = input_shape
            temporal_seq_flat = temporal_seq.reshape(B*N, T, F)
            u_future_flat = self.temporal_mamba(temporal_seq_flat)
            return u_future_flat.reshape(B, N, self.n_future)
        
        else:
            raise ValueError(f"Expected 3D or 4D input, got {len(input_shape)}D")
