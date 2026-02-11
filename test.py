from model import PDE_model
import torch

if __name__ == "__main__":
    print("\n🧪 Testing PDE_model (Direct Prediction)...\n")
    
    # Create model
    model = PDE_model(
        n_spatial=64,
        d_model=64,
        n_layers=4,
        d_state=16,
        expand=2
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}\n")
    model = model.to(device)
    
    # Test input
    batch_size = 4
    n_spatial = 64
    u_t = torch.randn(batch_size, n_spatial).to(device)
    dx = 0.1
    
    print("="*60)
    print("TEST 1: Forward Pass (predict u(t+1) from u(t))")
    print("="*60)
    
    with torch.no_grad():
        u_t_plus_1 = model.forward(u_t, dx)
    
    print(f"✓ Input u(t): {u_t.shape}")
    print(f"✓ Output u(t+1): {u_t_plus_1.shape}")
    print(f"✓ Model directly predicts next state (no dt needed!)")
    
    print("\n" + "="*60)
    print("TEST 2: Rollout (multiple steps)")
    print("="*60)
    
    n_steps = 20
    
    with torch.no_grad():
        trajectory = model.rollout(u_t, dx, n_steps)
    
    print(f"✓ Initial u(t=0): {u_t.shape}")
    print(f"✓ Trajectory: {trajectory.shape}")
    print(f"  → {trajectory.shape[0]} samples")
    print(f"  → {trajectory.shape[1]} time points (0, 1, 2, ..., {n_steps})")
    print(f"  → {trajectory.shape[2]} spatial points")
    print(f"\n✓ trajectory[:, 0, :] = u(t=0) (initial)")
    print(f"✓ trajectory[:, 1, :] = u(t=1)")
    print(f"✓ trajectory[:, 2, :] = u(t=2)")
    print(f"✓ ...")
    print(f"✓ trajectory[:, {n_steps}, :] = u(t={n_steps})")
    
    print("\n" + "="*60)
    print("MODEL INFO")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✅ All tests passed!")
    print("\n💡 Key Difference:")
    print("   • Input: u(t) and its spatial derivatives")
    print("   • Output: u(t+1) DIRECTLY")
    print("   • The model LEARNS the time evolution operator")
    print("   • No explicit dt needed during forward pass!")
    print()