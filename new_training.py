import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import warnings
import os
import numpy as np
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use GPU 1

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dam_break_dataset import DamBreakDataset
from temporal_model import TemporalMamba, TemporalMambaWithSpatial
from plotting_utils import plot_losses_clean
from new_trainer import MyTrainer


DATASET_PATH  = "dam_break_dataset_clean.pkl"
CHECKPOINT_PATH = "best_model.pt"

BATCH_SIZE    = 5
LR            = 1e-3
N_EPOCHS      = 100

# model config
D_MODEL       = 128
D_STATE       = 32
N_LAYERS      = 4
EXPAND        = 2
N_FUTURE      = 70
N_INPUT_FEAT  = 2     # u, h, du/dx, dh/dx
N_OUTPUT_FEAT = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DX = 1.0
DT = 0.01
LR = 1e-4
LAMBDA_PHYS = 0.5
g = 9.81

# load dataset

with open(DATASET_PATH, "rb") as f:
    full_dataset = pickle.load(f)

torch.manual_seed(42)

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=2, pin_memory=True)

model = TemporalMambaWithSpatial(
    d_model = D_MODEL,
    d_state = D_STATE,
    n_layers = N_LAYERS,
    expand = EXPAND,
    n_input_features = N_INPUT_FEAT,
    n_output_features= N_OUTPUT_FEAT,
    n_future = N_FUTURE 
)

trainer = MyTrainer(
    model = model,
    loss_fn = torch.nn.MSELoss(),
    device = DEVICE,
)

best_val = float("inf")

for epoch in range(1, N_EPOCHS + 1):
    train_metrics = trainer.train_epoch(train_loader)
    val_loss = trainer.validation_epoch(val_loader)
    
    #Step scheduler
    trainer.scheduler.step(val_loss)

    print(
        f"Epoch {epoch:03d} | "
        f"Train: {train_metrics['total']:.6f} | "
        f"Val: {val_loss:.6f}"
        f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_new_model.pt")