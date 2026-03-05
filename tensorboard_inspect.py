import torch
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.tensorboard import SummaryWriter
from temporal_model import TemporalMambaWithSpatial

# ── Config (mirror your main script exactly) ────────────────────────────────
D_MODEL       = 128
D_STATE       = 64
N_LAYERS      = 4
EXPAND        = 2
N_FUTURE      = 70
N_INPUT_FEAT  = 4      # u, h, du/dx, dh/dx
N_OUTPUT_FEAT = 2

T_CONTEXT     = 20     # number of context timesteps your dataset uses
B             = 2      # small dummy batch size, just for graph tracing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────

def main():
    model = TemporalMambaWithSpatial(
        d_model          = D_MODEL,
        d_state          = D_STATE,
        n_layers         = N_LAYERS,
        expand           = EXPAND,
        n_input_features = N_INPUT_FEAT,
        n_output_features= N_OUTPUT_FEAT,
        n_future         = N_FUTURE,
    ).to(DEVICE)

    model.eval()

    # Dummy input: (B*N, T, F) — 3D path through TemporalMamba
    # Matches what your trainer sends after reshape + FD concat
    dummy_input = torch.zeros(B, T_CONTEXT, N_INPUT_FEAT).to(DEVICE)

    writer = SummaryWriter(log_dir="runs/mamba_inspect")

    # ── Graph ────────────────────────────────────────────────────────────────
    print("Logging model graph...")
    try:
        writer.add_graph(model, dummy_input)
        print("  ✓ Graph logged")
    except Exception as e:
        print(f"  ✗ Graph logging failed: {e}")
        print("    (Mamba's custom CUDA kernels sometimes block tracing — see note below)")

    # ── Parameter count ──────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameter summary:")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Log param counts as scalars so they appear in TensorBoard
    writer.add_scalar("Model/total_params",     total_params,     0)
    writer.add_scalar("Model/trainable_params", trainable_params, 0)

    # ── Per-layer breakdown ──────────────────────────────────────────────────
    print(f"\nPer-layer breakdown:")
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"  {name:<50s} {params:>10,}")

    # ── Weight histograms (logged to TensorBoard HISTOGRAMS tab) ─────────────
    print("\nLogging weight histograms...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(name, param.data.cpu(), global_step=0)
    print("  ✓ Histograms logged")

    # ── Dummy forward pass ───────────────────────────────────────────────────
    print("\nRunning dummy forward pass...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Input shape : {dummy_input.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected    : ({B}, {N_FUTURE}, {N_OUTPUT_FEAT})")

    writer.close()
    print("\nDone. Launch TensorBoard with:")
    print("  tensorboard --logdir=runs")
    print("  then open http://localhost:6006")
    print("\nNOTE: If graph logging failed, it's likely because MambaBlock uses")
    print("  custom CUDA ops (pscan) that aren't traceable by torch.jit.")
    print("  The histograms and scalars will still be available.")


if __name__ == "__main__":
    main()
