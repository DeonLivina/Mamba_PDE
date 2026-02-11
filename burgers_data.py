import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BurgersEquation1D:
    """
    Solves 1D Burgers equation using finite differences
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    """
    
    def __init__(self, n_spatial=64, dx=None, nu=0.01, dt=0.01):
        """
        Parameters:
        -----------
        n_spatial : int
            Number of spatial grid points
        dx : float
            Spatial step size. If None, computed from domain
        nu : float
            Viscosity coefficient
        dt : float
            Time step size
        """
        self.n_spatial = n_spatial
        self.nu = nu
        self.dt = dt
        
        # Domain: [0, 2π]
        self.L = 2 * np.pi
        self.dx = self.L / n_spatial if dx is None else dx
        self.x = np.linspace(0, self.L, n_spatial, endpoint=False)
        
        # CFL condition: dt <= dx² / (4*nu) for stability
        self.dt_max = self.dx**2 / (4 * self.nu)
        if self.dt > self.dt_max:
            print(f"Warning: dt={self.dt} exceeds CFL limit {self.dt_max:.4f}")
            print(f"Using dt={self.dt_max * 0.9:.4f}")
            self.dt = self.dt_max * 0.9
    
    def rhs(self, u):
        """
        Compute right-hand side: ∂u/∂t = -u ∂u/∂x + ν ∂²u/∂x²
        
        Parameters:
        -----------
        u : np.ndarray (n_spatial,)
            Current velocity field
        
        Returns:
        --------
        dudt : np.ndarray (n_spatial,)
            Time derivative
        """
        # Periodic boundary conditions
        u_right = np.roll(u, -1)  # u[i+1]
        u_left = np.roll(u, 1)    # u[i-1]
        
        # Advection: -u ∂u/∂x (central difference)
        du_dx = (u_right - u_left) / (2 * self.dx)
        advection = -u * du_dx
        
        # Diffusion: ν ∂²u/∂x² (central difference)
        d2u_dx2 = (u_right - 2*u + u_left) / (self.dx**2)
        diffusion = self.nu * d2u_dx2
        
        # Total time derivative
        dudt = advection + diffusion
        
        return dudt
    
    def step_rk4(self, u):
        """
        Fourth-order Runge-Kutta time stepping
        
        Parameters:
        -----------
        u : np.ndarray (n_spatial,)
            Current velocity field
        
        Returns:
        --------
        u_next : np.ndarray (n_spatial,)
            Velocity field at next time step
        """
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5 * self.dt * k1)
        k3 = self.rhs(u + 0.5 * self.dt * k2)
        k4 = self.rhs(u + self.dt * k3)
        
        u_next = u + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return u_next
    
    def solve(self, u0, n_steps):
        """
        Solve Burgers equation for multiple time steps
        
        Parameters:
        -----------
        u0 : np.ndarray (n_spatial,)
            Initial condition
        n_steps : int
            Number of time steps to solve
        
        Returns:
        --------
        trajectory : np.ndarray (n_steps+1, n_spatial)
            Solution trajectory [u(t0), u(t1), ..., u(tn_steps)]
        """
        trajectory = [u0.copy()]
        u = u0.copy()
        
        for step in range(n_steps):
            u = self.step_rk4(u)
            trajectory.append(u.copy())
        
        return np.array(trajectory)


class BurgersDataset1D(Dataset):
    """
    PyTorch Dataset for 1D Burgers equation
    Generates random initial conditions and solves to create training pairs
    """
    
    def __init__(self, 
                 n_samples=100,
                 n_spatial=64,
                 n_time_steps=50,
                 nu=0.01,
                 dt=0.01,
                 initial_condition_type='gaussian_mixture'):
        """
        Parameters:
        -----------
        n_samples : int
            Number of training samples (trajectories)
        n_spatial : int
            Number of spatial grid points
        n_time_steps : int
            Length of each trajectory
        nu : float
            Viscosity coefficient
        dt : float
            Time step size
        initial_condition_type : str
            How to generate initial conditions
            - 'gaussian_mixture': random mixture of Gaussians
            - 'sin_sum': sum of sinusoids
            - 'random': random noise
        """
        self.n_samples = n_samples
        self.n_spatial = n_spatial
        self.n_time_steps = n_time_steps
        self.nu = nu
        self.dx = 2 * np.pi / n_spatial
        self.initial_condition_type = initial_condition_type
        
        # Initialize Burgers solver
        self.solver = BurgersEquation1D(n_spatial=n_spatial, nu=nu, dt=dt)
        
        # Generate dataset
        self.data = []
        self._generate_dataset()
    
    def _generate_initial_condition(self):
        """Generate a random initial condition"""
        x = self.solver.x
        
        if self.initial_condition_type == 'gaussian_mixture':
            # Random mixture of Gaussians
            n_peaks = np.random.randint(2, 5)
            u = np.zeros_like(x)
            for _ in range(n_peaks):
                center = np.random.uniform(0, 2*np.pi)
                width = np.random.uniform(0.3, 1.0)
                amplitude = np.random.uniform(-1.0, 1.0)
                u += amplitude * np.exp(-((x - center)**2) / (2 * width**2))
            # Normalize
            u = u / (np.max(np.abs(u)) + 1e-6)
            
        elif self.initial_condition_type == 'sin_sum':
            # Sum of random sinusoids
            u = np.zeros_like(x)
            n_modes = np.random.randint(2, 6)
            for _ in range(n_modes):
                k = np.random.randint(1, 8)
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(-1.0, 1.0)
                u += amplitude * np.sin(k * x + phase)
            # Normalize
            u = u / (np.max(np.abs(u)) + 1e-6)
            
        elif self.initial_condition_type == 'random':
            # Random noise + smooth
            u = np.random.randn(self.n_spatial)
            # Smooth with Gaussian filter
            from scipy.ndimage import gaussian_filter1d
            u = gaussian_filter1d(u, sigma=2.0)
            u = u / (np.max(np.abs(u)) + 1e-6)
        
        return u.astype(np.float32)
    
    def _generate_dataset(self):
        """Generate all trajectories"""
        print(f"Generating {self.n_samples} Burgers equation trajectories...")
        
        for sample_idx in range(self.n_samples):
            # Generate random initial condition
            u0 = self._generate_initial_condition()
            
            # Solve to get trajectory
            trajectory = self.solver.solve(u0, self.n_time_steps)
            
            # Store all consecutive pairs (u_t, u_{t+1})
            for t in range(trajectory.shape[0] - 1):
                self.data.append({
                    'u_t': torch.from_numpy(trajectory[t]).float(),
                    'u_next': torch.from_numpy(trajectory[t+1]).float()
                })
            
            if (sample_idx + 1) % max(1, self.n_samples // 10) == 0:
                print(f"  Generated {sample_idx + 1}/{self.n_samples} samples")
        
        print(f"Total training pairs: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return (u_t, u_{t+1}) pair"""
        return self.data[idx]['u_t'], self.data[idx]['u_next']


def create_burgers_dataloaders(n_samples=100,
                               n_spatial=64,
                               n_time_steps=50,
                               nu=0.01,
                               dt=0.01,
                               batch_size=16,
                               train_split=0.8,
                               num_workers=0):
    """
    Create train and validation DataLoaders for Burgers equation
    
    Parameters:
    -----------
    n_samples : int
        Number of trajectories to generate
    n_spatial : int
        Spatial resolution
    n_time_steps : int
        Length of each trajectory
    nu : float
        Viscosity
    dt : float
        Time step
    batch_size : int
        Batch size
    train_split : float
        Fraction of data for training
    num_workers : int
        Number of workers for data loading
    
    Returns:
    --------
    train_loader, val_loader : DataLoaders
    dataset_info : dict with metadata
    """
    
    # Create full dataset
    dataset = BurgersDataset1D(
        n_samples=n_samples,
        n_spatial=n_spatial,
        n_time_steps=n_time_steps,
        nu=nu,
        dt=dt,
        initial_condition_type='gaussian_mixture'
    )
    
    # Split into train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    dataset_info = {
        'n_samples': n_samples,
        'n_spatial': n_spatial,
        'n_time_steps': n_time_steps,
        'nu': nu,
        'dt': dt,
        'dx': dataset.dx,
        'n_train': n_train,
        'n_val': n_val,
        'n_total_pairs': len(dataset),
    }
    
    return train_loader, val_loader, dataset_info


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("1D Burgers Equation Dataset Generator")
    print("=" * 60)
    
    # Create dataset
    train_loader, val_loader, info = create_burgers_dataloaders(
        n_samples=10,  # Start small for testing
        n_spatial=64,
        n_time_steps=50,
        nu=0.01,
        dt=0.01,
        batch_size=4
    )
    
    print("\nDataset Info:")
    for key, val in info.items():
        print(f"  {key}: {val}")
    
    # Test loading a batch
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    for batch_idx, (u_t, u_next) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  u_t shape: {u_t.shape}")
        print(f"  u_next shape: {u_next.shape}")
        print(f"  u_t range: [{u_t.min():.4f}, {u_t.max():.4f}]")
        print(f"  u_next range: [{u_next.min():.4f}, {u_next.max():.4f}]")
        
        if batch_idx == 0:
            break