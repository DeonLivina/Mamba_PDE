"""
mamba_blocks_optimized.py

Official Mamba implementation with optimized parallel selective scan (pscan).
This is the full official Mamba code with all optimizations for efficient training and inference.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mambapy.pscan import pscan
except ImportError:
    print("Warning: mambapy not installed. Install with: pip install mambapy")
    # Fallback to sequential scan if pscan unavailable
    def pscan(deltaA, deltaB_x):
        """Fallback sequential scan implementation"""
        # This won't be used if mambapy is installed
        raise ImportError(
            "mambapy is required for optimized pscan. "
            "Install with: pip install mambapy"
        )


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False

    mup: bool = False
    mup_base_width: float = 128

    pscan: bool = True  # Use parallel scan (highly optimized)
    use_cuda: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    """Full Mamba model with stacked residual blocks."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        x : (B, L, D) → y : (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        """Single step inference with state caching."""
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches

    def chunk_step(self, x_seq, caches):
        """Chunk-based inference (between full sequence and step-by-step)."""
        for i, layer in enumerate(self.layers):
            x_seq, caches[i] = layer.chunk_step(x_seq, caches[i])
        return x_seq, caches


class ResidualBlock(nn.Module):
    """Residual block: output = mamba(norm(input)) + input"""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        """x : (B, L, D) → output : (B, L, D)"""
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        """Single step with cache."""
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

    def chunk_step(self, x_seq, cache):
        """Chunk step with cache."""
        y_seq, cache = self.mixer.chunk_step(self.norm(x_seq), cache)
        y_seq = y_seq + x_seq
        return y_seq, cache


class MambaBlock(nn.Module):
    """
    Core Mamba block with selective scan SSM.
    
    Input-dependent state dynamics:
    - h[t] = A(x[t]) * h[t-1] + B(x[t]) * x[t]
    - y[t] = C(x[t]) @ h[t]
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Input projection: D → 2*ED
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # Depthwise 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1
        )

        # SSM parameters
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A matrix (S4D real)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D feedthrough
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Output projection: ED → D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Optional inner layer norms
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        """
        Forward pass (training mode).
        x : (B, L, D) → y : (B, L, D)
        Uses optimized pscan for parallel computation.
        """
        _, L, _ = x.shape

        # Split into two branches
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)

        # x branch: conv + activation + SSM
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)  # Uses optimized pscan

        # z branch
        z = F.silu(z)

        # Combine
        output = y * z
        output = self.out_proj(output)

        return output

    def ssm(self, x):
        """
        Selective Scan SSM (training).
        Uses OPTIMIZED parallel scan (pscan) for efficiency.
        
        x : (B, L, ED) → y : (B, L, ED)
        """
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        # Compute input-dependent parameters
        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        delta, B, C = self._apply_layernorms(delta, B, C)

        # Project delta
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ ...
        delta = delta.transpose(1, 2)  # (B, L, ED)
        delta = F.softplus(delta + self.dt_proj.bias)

        # OPTIMIZED: Use parallel scan
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, x, delta, A, B, C, D):
        """
        OPTIMIZED selective scan using parallel scan (pscan).
        
        This is O(L log L) instead of O(L) sequential, but much faster in practice
        due to better GPU utilization.
        
        x : (B, L, ED)
        delta : (B, L, ED)
        A : (ED, N)
        B : (B, L, N)
        C : (B, L, N)
        D : (ED)
        
        y : (B, L, ED)
        """
        # Precompute state transitions
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        # Element-wise product
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        # OPTIMIZED: Parallel scan over time dimension
        hs = pscan(deltaA, BX)  # (B, L, ED, N)

        # Output projection
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        y = y + D * x

        return y

    # ========================= Inference Methods =========================

    def step(self, x, cache):
        """
        Single step inference (constant time, constant memory).
        
        x : (B, D)
        cache : (h, inputs)
        """
        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)

        # Convolution with cached inputs
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]
        x = F.silu(x)

        # SSM step
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        # Update cache
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        """
        SSM single step (for autoregressive inference).
        
        x : (B, ED)
        h : (B, ED, N)
        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank + 2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX  # (B, ED, N)
        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED)
        y = y + D * x

        return y, h

    def chunk_step(self, x_seq, cache):
        """
        Chunk-based inference (compromise between speed and memory).
        Process sequence in chunks while maintaining state.
        
        x_seq : (B, L, D)
        cache : (h0, inputs)
        """
        B, L, _ = x_seq.shape
        h0, inputs = cache

        xz = self.in_proj(x_seq)  # (B, L, 2*ED)
        x_pre, z = xz.chunk(2, dim=-1)

        # Conv with history
        k = max(self.config.d_conv - 1, 0)
        if k > 0:
            x_cat = torch.cat([inputs, x_pre.transpose(1, 2)], dim=2)
            x_conv_full = self.conv1d(x_cat)
            x_conv = x_conv_full[:, :, k : k + L]
            x_act = F.silu(x_conv.transpose(1, 2))
        else:
            x_act = self.conv1d(x_pre.transpose(1, 2))[:, :, :L]
            x_act = F.silu(x_act.transpose(1, 2))

        # SSM chunk
        y_seq, hs = self.ssm_chunk_step(x_act, h0=h0)

        # z branch
        z = F.silu(z)
        y_seq = self.out_proj(y_seq * z)

        # Update cache
        h_T = hs[:, -1]
        if k > 0:
            pre_full = torch.cat([inputs.transpose(1, 2), x_pre], dim=1)
            inputs_last = pre_full[:, -k:, :].transpose(1, 2).contiguous()
        else:
            inputs_last = torch.zeros(B, self.config.d_inner, 0, device=x_seq.device, dtype=x_seq.dtype)

        return y_seq, (h_T, inputs_last)

    def ssm_chunk_step(self, x, h0=None):
        """
        SSM for chunk processing.
        Uses optimized pscan for the chunk.
        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        delta, B, C = self._apply_layernorms(delta, B, C)

        delta = self.dt_proj.weight @ delta.transpose(1, 2)
        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

        if h0 is not None:
            BX[:, 0] = BX[:, 0] + deltaA[:, 0] * h0

        # OPTIMIZED: Use pscan for chunk
        hs = pscan(deltaA, BX)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        y = y + D * x

        return y, hs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()
        self.use_mup = use_mup
        self.eps = eps

        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if not self.use_mup:
            return output * self.weight
        else:
            return output
