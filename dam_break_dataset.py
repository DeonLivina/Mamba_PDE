import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os


def shallow_water_fvm(
    h0: np.ndarray,
    u0: np.ndarray,
    g: float = 9.81,
    dx: float = 0.1,
    dt: float = 0.005,
    n_steps: int = 500,
    nu: float = 0.001,
) -> tuple:
   
    h  = h0.copy().astype(np.float64)
    hu = (h0 * u0).astype(np.float64)  # momentum = mass (via h) × velocity

    h_history  = [h.copy()]
    u_history  = [(hu / np.maximum(h, 1e-6)).copy()]

    for step in range(n_steps - 1):

        # Recover velocity from momentum
        u = hu / np.maximum(h, 1e-6)

        # Recover velocity from momentum: u = (momentum) / (mass) = hu / h
        if step % 50 == 0:
            c_max = np.max(np.sqrt(g * np.maximum(h, 1e-6)))  # wave speed
            u_max = np.max(np.abs(u))
            cfl = (u_max + c_max) * dt / dx
            if cfl > 0.9:
                print(f"CFL={cfl:.3f} at step {step}, aborting.")
                return None, None

        # Wave speeds (for HLL solver)
        c = np.sqrt(g * np.maximum(h, 1e-6))

       
        # Left and right states at each interface
        hL  = h[:-1];   hR  = h[1:]
        huL = hu[:-1];  huR = hu[1:]
        uL  = u[:-1];   uR  = u[1:]
        cL  = c[:-1];   cR  = c[1:]

        # HLL wave speed estimates
        sL = np.minimum(uL - cL, uR - cR)  # fastest left-moving wave
        sR = np.maximum(uL + cL, uR + cR)  # fastest right-moving wave
        denom = np.maximum(sR - sL, 1e-8)

        # Physical fluxes in left and right states
        # Mass flux: F_h = hu (momentum, which is mass flux)
        # Momentum flux: F_hu = hu*u + (1/2)*g*h^2 (momentum × velocity + pressure)
        fhL  = huL
        fhR  = huR
        fhuL = huL * uL + 0.5 * g * hL**2
        fhuR = huR * uR + 0.5 * g * hR**2

        # HLL flux selection (weighted average based on wave speeds)
        f_h = np.where(
            sL >= 0, fhL,
            np.where(sR <= 0, fhR,
                (sR * fhL - sL * fhR + sL * sR * (hR - hL)) / denom)
        )
        f_hu = np.where(
            sL >= 0, fhuL,
            np.where(sR <= 0, fhuR,
                (sR * fhuL - sL * fhuR + sL * sR * (huR - huL)) / denom)
        )

        h_new  = h.copy()
        hu_new = hu.copy()

        # Conservative update: U_new = U_old - (dt/dx) * (F_right - F_left)
        h_new[1:-1]  = h[1:-1]  - (dt/dx) * (f_h[1:]  - f_h[:-1])
        hu_new[1:-1] = hu[1:-1] - (dt/dx) * (f_hu[1:] - f_hu[:-1])


        h_new[1:-1]  += nu * (dt/dx**2) * (h[2:]  - 2*h[1:-1]  + h[:-2])
        hu_new[1:-1] += nu * (dt/dx**2) * (hu[2:] - 2*hu[1:-1] + hu[:-2])


        # BOUNDARY CONDITIONS: REFLECTIVE WALLS

        # Left wall (x=0): water bounces back
        h_new[0]   = h_new[1]
        hu_new[0]  = -hu_new[1]
        # Right wall (x=L): water bounces back
        h_new[-1]  = h_new[-2]
        hu_new[-1] = -hu_new[-2]

        h_new = np.maximum(h_new, 1e-6)

        if np.isnan(h_new).any()  or np.isinf(h_new).any() or \
           np.isnan(hu_new).any() or np.isinf(hu_new).any():
            print(f"  ⚠ FVM unstable at step {step}, aborting.")
            return None, None

        h  = h_new
        hu = hu_new

        u_out = hu / np.maximum(h, 1e-6)
        h_history.append(h.copy())
        u_history.append(u_out.copy())

    return (
        np.array(h_history, dtype=np.float32),
        np.array(u_history, dtype=np.float32),
    )



def compute_spatial_derivatives(arr: torch.Tensor, dx: float) -> torch.Tensor:
    
    d = torch.zeros_like(arr)
    
    # Central differences: (f[i+1] - f[i-1]) / (2*dx)
    d[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0 * dx)
    
    # Left boundary: forward difference
    d[:, 0] = (arr[:, 1] - arr[:, 0]) / dx
    
    # Right boundary: backward difference
    d[:, -1] = (arr[:, -1] - arr[:, -2]) / dx
    
    return d


# Initial conditions

def create_dam_break_ic(n_spatial: int, h_left: float, h_right: float, 
                        dam_position: int = None) -> tuple:
    """
    Returns:
        h0: (n_spatial,) initial water height
        u0: (n_spatial,) initial velocity (all zeros)
    """
    if dam_position is None:
        dam_position = n_spatial // 2
    
    h0 = np.zeros(n_spatial, dtype=np.float32)
    h0[:dam_position] = h_left      # left side: deep
    h0[dam_position:] = h_right     # right side: shallow
    
    u0 = np.zeros(n_spatial, dtype=np.float32)  # everything at rest
    
    return h0, u0



# DATASET CLASS


class DamBreakDataset(Dataset):
    def __init__(
        self,
        n_spatial: int = 32,
        n_time: int = 500,
        dx: float = 0.1,
        dt: float = 0.005,
        g: float = 9.81,
        n_context: int = 40,
        n_future: int = 70,
        n_simulations: int = 50,
        nu: float = 0.001,
        use_robust_normalization: bool = True,
        robust_percentile: float = 98.0,
    ):
        self.dx = dx
        self.dt = dt
        self.n_context = n_context
        self.n_future = n_future
        self.use_robust_normalization = use_robust_normalization
        self.robust_percentile = robust_percentile

        print(f"\n{'='*70}")
        
        print(f"{'='*70}")
        print(f"Domain:")
        print(f"  Grid cells     : {n_spatial}")
        print(f"  Physical length: {n_spatial * dx:.2f} m")
        print(f"  Spatial step   : dx = {dx} m")
        print(f"  Time step      : dt = {dt} s")
        print(f"  Total time     : {n_time * dt:.3f} s")
        print(f"  Viscosity      : nu = {nu} (for stability)")
        print(f"\nDataset:")
        print(f"  # Simulations  : {n_simulations}")
        print(f"  Context window : {n_context} timesteps")
        print(f"  Predict window : {n_future} timesteps")
        print(f"  Features       : [h, u, dh/dx, du/dx]  (4 features per cell)")
        print(f"\nNormalization:")
        if use_robust_normalization:
            print(f"  Method: Robust (using percentiles, handles shocks)")
            print(f"  For h, u: standard mean/std")
            print(f"  For dh/dx, du/dx: {robust_percentile}th percentile")
        else:
            print(f"  Method: Standard (mean/std)")
        print(f"{'='*70}\n")

    
        print("Step 1: Running FVM simulations...")
        raw_simulations = []
        success = 0
        attempts = 0

        while success < n_simulations:
            attempts += 1
            if attempts > n_simulations * 5:
                print(f"Only {success} stable sims after {attempts} attempts.")
                break

            # Random dam configuration
            h_left = np.random.uniform(0.5, 5.0)    # 1.5 to 3.0 meters
            h_right = np.random.uniform(0.1, 2.0)   # 0.3 to 1.2 meters
            dam_pos = np.random.randint(n_spatial // 4, 3 * n_spatial // 4)

            # Create initial conditions
            h0, u0 = create_dam_break_ic(n_spatial, h_left, h_right, dam_pos)

            # Solve
            h_sol, u_sol = shallow_water_fvm(
                h0, u0, g=g, dx=dx, dt=dt, n_steps=n_time, nu=nu
            )

            # Check for solver failure
            if h_sol is None:
                continue

            # Check for NaN/Inf in output
            if np.isnan(h_sol).any() or np.isinf(h_sol).any() or \
               np.isnan(u_sol).any() or np.isinf(u_sol).any():
                continue

            # Check physical validity
            if h_sol.min() < 0:
                continue

            raw_simulations.append((h_sol, u_sol))
            success += 1
            if success % 10 == 0:
                print(f" {success}/{n_simulations} simulations complete")

        print(f"Generated {success} stable simulations\n")

        
        print("Computing spatial derivatives...")
        processed = []

        for sim_idx, (h_sol, u_sol) in enumerate(raw_simulations):
            h_t = torch.from_numpy(h_sol)
            u_t = torch.from_numpy(u_sol)

            # Compute derivatives
            dh_dx = compute_spatial_derivatives(h_t, dx)
            du_dx = compute_spatial_derivatives(u_t, dx)

            # Validate
            if torch.isnan(dh_dx).any() or torch.isinf(dh_dx).any() or \
               torch.isnan(du_dx).any() or torch.isinf(du_dx).any():
                print(f"Bad derivatives in sim {sim_idx}, skipping.")
                continue

            processed.append((
                h_sol,
                u_sol,
                dh_dx.numpy(),
                du_dx.numpy(),
            ))

        print(f"{len(processed)} simulations passed derivative check\n")

        # Statistics

        print("Computing normalization statistics")
        
        all_h = np.concatenate([s[0] for s in processed], axis=0)
        all_u = np.concatenate([s[1] for s in processed], axis=0)
        all_dh_dx = np.concatenate([s[2] for s in processed], axis=0)
        all_du_dx = np.concatenate([s[3] for s in processed], axis=0)

        def compute_stats(arr, name, use_robust=False, percentile=98.0):
            """Compute mean and scale for normalization"""
            if use_robust:
                # For heavy-tailed distributions (like derivatives with shocks)
                q_low = (100 - percentile) / 2
                q_high = 100 - q_low
                p_low = np.percentile(arr, q_low)
                p_high = np.percentile(arr, q_high)
                
                mean = float(np.median(arr))
                scale = float((p_high - p_low) / 2)
                scale = max(scale, 1e-6)
                
                print(f"  {name:10s}: median={mean:8.4f}, scale={scale:8.4f} "
                      f"[p{q_low:.0f}={p_low:7.3f}, p{q_high:.0f}={p_high:7.3f}]")
            else:
                # For well-behaved distributions
                mean = float(arr.mean())
                std = float(arr.std())
                scale = max(std, 1e-6)
                
                print(f"  {name:10s}: mean={mean:8.4f}, std={scale:8.4f} "
                      f"[min={arr.min():7.3f}, max={arr.max():7.3f}]")
            
            return mean, scale

        # h and u are well-behaved
        h_mean, h_std = compute_stats(all_h, "h", False)
        u_mean, u_std = compute_stats(all_u, "u", False)
        
        # Derivatives have heavy tails from shocks → use robust normalization
        dh_dx_mean, dh_dx_std = compute_stats(
            all_dh_dx, "dh_dx", use_robust_normalization, robust_percentile
        )
        du_dx_mean, du_dx_std = compute_stats(
            all_du_dx, "du_dx", use_robust_normalization, robust_percentile
        )

        self.stats = {
            'h_mean': h_mean,       'h_std': h_std,
            'u_mean': u_mean,       'u_std': u_std,
            'dh_dx_mean': dh_dx_mean, 'dh_dx_std': dh_dx_std,
            'du_dx_mean': du_dx_mean, 'du_dx_std': du_dx_std,
        }

        
        print("Normalizing and validating...")
        self.sim_data = []
        dropped = 0

        for sim_idx, (h_sol, u_sol, dh_sol, du_sol) in enumerate(processed):
            # Normalize each feature
            h_n = (h_sol - h_mean) / h_std
            u_n = (u_sol - u_mean) / u_std
            dh_n = (dh_sol - dh_dx_mean) / dh_dx_std
            du_n = (du_sol - du_dx_mean) / du_dx_std

            # Validation: check for extreme values
            bad = False
            for name, arr, threshold in [
                ('h_n', h_n, 10.0),     # height should be normal
                ('u_n', u_n, 10.0),     # velocity should be normal
                ('dh_n', dh_n, 20.0),   # derivatives can be larger (shocks)
                ('du_n', du_n, 20.0),   # derivatives can be larger (shocks)
            ]:
                if np.isnan(arr).any() or np.isinf(arr).any():
                    print(f" Sim {sim_idx}: NaN/Inf in {name}, dropping.")
                    bad = True
                    break
                if np.abs(arr).max() > threshold:
                    print(f" Sim {sim_idx}: extreme {name} (max={np.abs(arr).max():.2f}), dropping.")
                    bad = True
                    break

            if bad:
                dropped += 1
                continue

            self.sim_data.append((
                h_n.astype(np.float32),
                u_n.astype(np.float32),
                dh_n.astype(np.float32),
                du_n.astype(np.float32),
            ))

        print(f"{len(self.sim_data)} kept, {dropped} dropped\n")

        #
        print("Verifying normalization...")
        for feat_idx, name in enumerate(['h', 'u', 'dh_dx', 'du_dx']):
            arr = np.concatenate([s[feat_idx] for s in self.sim_data], axis=0)
            print(f"  {name:10s}: mean={arr.mean():6.4f}, std={arr.std():6.4f}, "
                  f"range=[{arr.min():7.3f}, {arr.max():7.3f}]")

        
        print("Building sample index map...")
        self.indices = []
        for sim_idx, (h_n, _, _, _) in enumerate(self.sim_data):
            T = h_n.shape[0]
            n_windows = T - n_context - n_future
            if n_windows <= 0:
                print(f"  ⚠ Sim {sim_idx} too short, skipping.")
                continue
            for t in range(n_windows):
                self.indices.append((sim_idx, t))

        print(f"\n{'='*70}")
        print(f"Dataset Summary:")
        print(f"  Simulations  : {len(self.sim_data)}")
        print(f"  Total samples: {len(self.indices)}")
        if len(self.sim_data) > 0:
            print(f"  Avg samples/sim: {len(self.indices) // len(self.sim_data)}")
        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Return one training sample.
        
        Returns:
            context: (n_context, n_spatial, 4) - 4 features
            target: (n_future, n_spatial, 2) - 2 targets (h, u)
        """
        sim_idx, t_start = self.indices[idx]
        h_n, u_n, dh_n, du_n = self.sim_data[sim_idx]

        t_ctx_end = t_start + self.n_context
        t_tgt_end = t_ctx_end + self.n_future

        # Context: all 4 features
        h_ctx = torch.from_numpy(h_n[t_start:t_ctx_end])
        u_ctx = torch.from_numpy(u_n[t_start:t_ctx_end])
        dh_ctx = torch.from_numpy(dh_n[t_start:t_ctx_end])
        du_ctx = torch.from_numpy(du_n[t_start:t_ctx_end])

        # Target: just h and u (what model predicts)
        h_tgt = torch.from_numpy(h_n[t_ctx_end:t_tgt_end])
        u_tgt = torch.from_numpy(u_n[t_ctx_end:t_tgt_end])

        # Stack features: (n_spatial, n_context, 4)
        context = torch.stack(
            [h_ctx, u_ctx], dim=-1
        ).permute(1, 0, 2)

        # Stack targets: (n_spatial, n_future, 2)
        target = torch.stack(
            [h_tgt, u_tgt], dim=-1
        ).permute(1, 0, 2)

        return context, target

    def get_normalization_stats(self):
        return self.stats


if __name__ == "__main__":

    SAVE_PATH = "dam_break_dataset_clean.pkl"

    dataset = DamBreakDataset(
        n_spatial=32,
        n_time=150,
        dx=0.1,
        dt=0.005,
        g=9.81,
        n_context=40,
        n_future=70,
        n_simulations=3,
        nu=0.005,  # Lower viscosity → sharper shocks
        use_robust_normalization=True,
        robust_percentile=98.0,
    )

    print(f"Saving to '{SAVE_PATH}'...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    size_mb = os.path.getsize(SAVE_PATH) / 1e6
    print(f"Saved: {len(dataset)} samples, {size_mb:.1f} MB")

    print(f"\n{'='*70}")
    print("Dataset Features:")
    print(f"{'='*70}")
    print("Context features (model INPUT):")
    print("  • h:     water height")
    print("  • u:     velocity")
    # print("  • dh/dx: height gradient (steep = piling up)")
    # print("  • du/dx: velocity gradient (steep = flow changing)")
    print("\nTarget features (model PREDICTS):")
    print("  • h:     water height at future times")
    print("  • u:     velocity at future times")
    print("\nModel learns: How gradients evolve into shocks!")
    print(f"{'='*70}\n")
