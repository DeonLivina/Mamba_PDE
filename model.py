import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from sel_scan import selective_scan


class S6(nn.Module):
  def __init__(self, d_model, d_state = 16, dt_rank = 'auto', dt_min=0.001, dt_max = 0.1):
    super().__init__()

    self.d_model = d_model # dimention of the model
    self.d_state = d_state # dimention of the state

    self.dt_rank = math.ceil(d_model/16) if dt_rank == 'auto' else dt_rank

    # Create learnable parameters

    self.A_log = nn.Parameter(torch.randn(d_model, d_state)) #nn.Parameter = should be unpdated during training
    self.D = nn.Parameter(torch.randn(d_model))

    self.x_proj = nn.Linear(d_model, self.dt_rank + d_state*2, bias = False)
    self.dt_proj = nn.Linear(self.dt_rank, d_model, bias = True)

    dt_init_std = self.dt_rank**(-0.5) # standard deviation

    nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

    dt  =  torch.exp(torch.rand(d_model)*(math.log(dt_max)-math.log(dt_min)) + math.log(dt_min))

    inv_dt = dt + torch.log(-torch.expm1(-dt))

    with torch.no_grad():
      self.dt_proj.bias.copy_(inv_dt) # _ means "in place"


  def forward(self, x):
    """
    x: Input tensor (B, L, D)
    """
    batch, swq_len, dim = x.shape

    #get matrix A
    A = -torch.exp(self.A_log.float())
    x_proj = self.x_proj(x) # (B, L, dt_rank + 2*d_state)

    #Split projection to delta, B, C
    delta_rank = x_proj[:, :, :self.dt_rank]
    B = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
    C = x_proj[:, :, self.dt_rank + self.d_state:]

    delta = F.softplus(self.dt_proj(delta_rank))

    y = selective_scan(x, delta, A, B, C, self.D)

    return y


class MambaBlock(nn.Module):
  def __init__(self, d_model, d_state, expand = 2):
    super().__init__()

    self.d_model = d_model
    self.d_inner = int(expand * d_model)

    self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias = False)

    self.conv1d = nn.Conv1d(
        in_channels = self.d_inner,
        out_channels = self.d_inner,
        kernel_size = 4,
        padding = 3,
        groups = self.d_inner
    )

    self.s6 = S6(self.d_inner, d_state)
    # creat the S6 layer

    self.out_proj = nn.Linear(self.d_inner, d_model, bias = False)
    #project back to oridinal dimentiosn

    self.norm = nn.LayerNorm(d_model)
    # LayerNorm normalizes across features


  def forward(self, x):
    batch, seq_len, dim = x.shape

      # Save residual connection
    residual = x

      # Normalize
    x = self.norm(x)

      #Project and split

    x_and_res = self.in_proj(x)
    x, res = x_and_res.split(self.d_inner, dim = -1)

    x = rearrange(x, 'b l d -> b d l')
    x = self.conv1d(x)[:, :, :seq_len]

    x = rearrange(x, 'b d l -> b l d')

    x = F.silu(x)  # x * sigmoid(x) smooth nonlinearity

      #Apply SSM
    x = self.s6(x)

      # gated multiplicative unit
    x = x * F.silu(res)

      #Project back to og dim
    x = self.out_proj(x)

      # Residual connection
    return x + residual

class Mamba(nn.Module):
  def __init__(self, d_model=256, n_layers=4, d_state=16, expand=2, vocab_size=10000):
    super().__init__()

    # Embedding layer: converts token IDs to vectors
    self.embedding = nn.Embedding(vocab_size, d_model)

    # Create multiple Mamba blocks
    self.layers = nn.ModuleList([
        MambaBlock(d_model, d_state, expand)
        for _ in range(n_layers)
    ])

    self.norm_f = nn.LayerNorm(d_model)

    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


  def forward(self, input_ids):
    """
    input_ids: (B, L)
    """

    # convert token IDs to embeddings
    x = self.embedding(input_ids)

    # Pass through all layers
    for layer in self.layers:
      x = layer(x)

    x = self.norm_f(x)

    logits = self.lm_head(x)

    return logits

class PDE_model(nn.Module):
    """
    Mamba model for PDEs - predicts u(t+1) directly from u(t)
    
    The model LEARNS the time evolution operator
    Input: [u(t), ∂u/∂x, ∂²u/∂x²] at time t
    Output: u(t+1) 
    """
    def __init__(self, 
                 n_spatial=64,       # Number of spatial grid points
                 d_model=128,        # Hidden dimension
                 n_layers=4,         # Number of Mamba blocks
                 d_state=16,         # SSM state dimension
                 expand=2,           # Expansion factor
                 n_input_features=3):  # [u, ∂u/∂x, ∂²u/∂x²]
        super().__init__()
        
        self.n_spatial = n_spatial
        self.d_model = d_model
        self.n_input_features = n_input_features
        
        # Project input features to model dimension
        self.input_proj = nn.Linear(n_input_features, d_model)
        
        # Stack of Mamba blocks 
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand)
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm_f = nn.LayerNorm(d_model)
        
        # Output: predict u(t+1) directly (1 value per spatial point)
        self.output_proj = nn.Linear(d_model, 1)
    
    def compute_spatial_derivatives_periodic(self, u, dx):
        """
        Compute spatial derivatives using finite differences
        WITH PERIODIC BOUNDARY CONDITIONS
        
        Parameters:
        -----------
        u : torch.Tensor (batch, n_spatial)
            Current velocity field
        dx : float
            Spatial step size
        
        Returns:
        --------
        features : torch.Tensor (batch, n_spatial, 3)
            Stacked [u, ∂u/∂x, ∂²u/∂x²]
        """
        if u.dim() == 2:
            u = u.unsqueeze(-1)  # (batch, n_spatial, 1)
        
        # Periodic BC: use torch.roll
        u_right = torch.roll(u, shifts=-1, dims=1)  # u[i+1]
        u_left = torch.roll(u, shifts=1, dims=1)    # u[i-1]
        
        # First derivative: ∂u/∂x (central difference)
        du_dx = (u_right - u_left) / (2 * dx)
        
        # Second derivative: ∂²u/∂x² (central difference)
        d2u_dx2 = (u_right - 2*u + u_left) / (dx**2)
        
        # Stack features: [u, ∂u/∂x, ∂²u/∂x²]
        features = torch.cat([u, du_dx, d2u_dx2], dim=-1)  # (batch, n_spatial, 3)
        
        return features
    
    def forward(self, u, dx):
        """
        Predict u(t+1) directly given u(t)
        
        The time operator is LEARNED by the network
        
        Parameters:
        -----------
        u : torch.Tensor (batch, n_spatial)
            Current velocity field at time t
        dx : float
            Spatial step size
        
        Returns:
        --------
        u_next : torch.Tensor (batch, n_spatial)
            Velocity field at time t+1 (DIRECTLY, not derivative)
        """
        # Compute spatial derivatives
        features = self.compute_spatial_derivatives_periodic(u, dx)
        
        # Project to model dimension
        x = self.input_proj(features)  # (batch, n_spatial, d_model)
        
        # Pass through Mamba blocks
        
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Project to output (u at next timestep)
        u_next = self.output_proj(x)  # (batch, n_spatial, 1)
        u_next = u_next.squeeze(-1)   # (batch, n_spatial)
        
        return u_next
    
    def rollout(self, u0, dx, n_steps):
        """
        Roll out the solution for multiple time steps
        
        Parameters:
        -----------
        u0 : torch.Tensor (batch, n_spatial)
            Initial condition at t=0
        dx : float
            Spatial step size
        n_steps : int
            Number of time steps to predict
        
        Returns:
        --------
        trajectory : torch.Tensor (batch, n_steps+1, n_spatial)
            Complete trajectory from t=0 to t=n_steps
        """
        # Store trajectory
        trajectory = [u0.unsqueeze(1)]  # (batch, 1, n_spatial)
        
        # Current state
        u = u0
        
        # Step forward in time
        for step in range(n_steps):
            # Predict next step directly
            u = self.forward(u, dx)
            trajectory.append(u.unsqueeze(1))
        
        # Stack all time steps
        trajectory = torch.cat(trajectory, dim=1)  # (batch, n_steps+1, n_spatial)
        
        return trajectory


