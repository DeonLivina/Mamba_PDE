import torch
import pickle
import warnings
import os
from torch.utils.data import DataLoader, random_split

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')

from dam_break_dataset import DamBreakDataset
from temporal_model import TemporalMambaWithSpatial
from fd_module import ShallowWaterFD
from plotting_utils import (
    plot_prediction_comparison,
    plot_spatial_temporal_heatmaps,
    plot_multiple_points,
    plot_error_analysis
)

DATASET_PATH    = "dam_break_dataset_clean.pkl"
CHECKPOINT_PATH = "best_new_model.pt"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 12
D_MODEL       = 128
D_STATE       = 32
N_LAYERS      = 4
EXPAND        = 2
N_FUTURE      = 70
N_INPUT_FEAT  = 2      # u, h, du/dx, dh/dx
N_OUTPUT_FEAT = 2
DX            = 0.1
DT            = 0.005



def preprocess(context: torch.Tensor, fdmodel: ShallowWaterFD, device: str) -> torch.Tensor:
    """
    Apply FD preprocessing to match training pipeline exactly.
    context: (B, N, T, 2)  →  returns (B*N, T, 4)
    """
    B, N, T, F = context.shape
    context_flat = context.reshape(B * N, T, F).to(device)

    #du_dx, dh_dx = fdmodel(context_flat[:, :, 0], context_flat[:, :, 1])

    # Log-compression to avoid blowup (same as trainer)
    #du_dx = torch.sign(du_dx) * torch.log1p(du_dx.abs())
    #dh_dx = torch.sign(dh_dx) * torch.log1p(dh_dx.abs())

    #context_flat = torch.cat(
     #   [context_flat, du_dx.unsqueeze(-1), dh_dx.unsqueeze(-1)], dim=-1
    #)  # (B*N, T, 4)

    return context_flat


class PreprocessedLoader:
    """
    Wraps a DataLoader and applies FD preprocessing on the fly.
    Plotting utils expect a DataLoader-like object that yields
    (context, targets) where context is already (B*N, T, 4).
    """
    def __init__(self, loader: DataLoader, fdmodel: ShallowWaterFD, device: str):
        self.loader  = loader
        self.fdmodel = fdmodel
        self.device  = device
        self.dataset = loader.dataset   # expose .dataset for plot_spatial_temporal_heatmaps

    def __iter__(self):
        for context, targets in self.loader:
            context_flat = preprocess(context, self.fdmodel, self.device)
            targets      = targets.to(self.device)
            yield context_flat, targets

    def __len__(self):
        return len(self.loader)


class PreprocessedDataset:
    """
    Wraps a Dataset and applies FD preprocessing per sample.
    Used by plot_spatial_temporal_heatmaps and plot_multiple_points
    which call dataset[idx] directly.
    """
    def __init__(self, dataset, fdmodel: ShallowWaterFD, device: str):
        self.dataset = dataset
        self.fdmodel = fdmodel
        self.device  = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        context, target = self.dataset[idx]             # (N, T, 2), (N, n_future, 2)
        context = context.unsqueeze(0)                  # (1, N, T, 2)
        context_flat = preprocess(context, self.fdmodel, self.device)  # (N, T, 4)
        return context_flat, target


with open(DATASET_PATH, "rb") as f:
    full_dataset = pickle.load(f)

torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
_, val_ds  = random_split(full_dataset, [train_size, val_size])

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        drop_last=False, num_workers=2, pin_memory=True)


fdmodel = ShallowWaterFD(dx=DX, dt=DT)

preprocessed_loader  = PreprocessedLoader(val_loader, fdmodel, DEVICE)
preprocessed_dataset = PreprocessedDataset(val_ds, fdmodel, DEVICE)

model = TemporalMambaWithSpatial(
    d_model          = D_MODEL,
    d_state          = D_STATE,
    n_layers         = N_LAYERS,
    expand           = EXPAND,
    n_input_features = N_INPUT_FEAT,
    n_output_features= N_OUTPUT_FEAT,
    n_future         = N_FUTURE,
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print(f"✓ Loaded checkpoint from '{CHECKPOINT_PATH}'")

print("\nGenerating plots...")

plot_prediction_comparison(model, preprocessed_loader, DEVICE,
                           save_path='test_pred_comparison.png')

plot_spatial_temporal_heatmaps(model, preprocessed_dataset, DEVICE,
                               save_path='test_spatial_temporal.png')

plot_multiple_points(model, preprocessed_dataset, DEVICE, n_points=4,
                     save_path='test_multiple_points.png')

plot_error_analysis(model, preprocessed_loader, DEVICE,
                    save_path='test_error_analysis.png')

print("\n✓ All plots saved.")