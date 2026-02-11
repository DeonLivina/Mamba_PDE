import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange



def selective_scan(u, delta, A, B, C, D):
    """
    Selective scan implementation
    
    u: (B, L, D) - input
    delta: (B, L, D) - step sizes
    A: (D, N) - state transition matrix
    B: (B, L, N) - input projection
    C: (B, L, N) - output projection
    D: (D,) - skip connection
    
    Returns: (B, L, D)
    """
    batch, seq_len, dim = u.shape
    n_state = A.shape[1]
    
    # Discretize continuous parameters
    # deltaA: (B, L, D, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    
    # deltaB: (B, L, D, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
    
    # Initialize hidden state
    h = torch.zeros(batch, dim, n_state, device=u.device, dtype=u.dtype)
    
    outputs = []
    
    for t in range(seq_len):
        # h: (B, D, N)
        # deltaA[:, t]: (B, D, N)
        # deltaB[:, t]: (B, D, N)
        # u[:, t]: (B, D)
        
        # State update: h = deltaA * h + deltaB * u
        h = deltaA[:, t] * h + deltaB[:, t] * u[:, t:t+1].transpose(1, 2)
        
        # Output: y = sum over N dimension of (h * C)
        # h: (B, D, N)
        # C[:, t]: (B, N)
        # Result: (B, D)
        y = (h * C[:, t].unsqueeze(1)).sum(dim=2)
        
        outputs.append(y)
    
    # Stack outputs: list of (B, D) -> (B, L, D)
    y = torch.stack(outputs, dim=1)
    
    # Skip connection
    y = y + u * D
    
    return y
