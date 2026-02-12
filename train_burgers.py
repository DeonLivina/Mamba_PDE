import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from burgers_data import create_burgers_dataloaders
from model import PDE_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataloaders
train_loader, val_loader, info = create_burgers_dataloaders(
    n_samples=2000,
    n_spatial=128,
    n_time_steps=100,
    nu=0.01,
    dt=0.00001,
    batch_size=16
)

# Create model
model = PDE_model(
    n_spatial=info['n_spatial'],
    d_model=256,
    n_layers=8,
    d_state=128,
    expand=2,
    n_input_features=3
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for u_t, u_next in train_loader:
        u_t, u_next = u_t.to(device), u_next.to(device)
        dx = info['dx']

        optimizer.zero_grad()
        u_pred = model(u_t, dx)
        loss = criterion(u_pred, u_next)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for u_t, u_next in val_loader:
            u_t, u_next = u_t.to(device), u_next.to(device)
            dx = info['dx']
            u_pred = model(u_t, dx)
            loss = criterion(u_pred, u_next)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

# -----------------------------
# Plot rollout
# -----------------------------
model.eval()
with torch.no_grad():
    # Take the first sample from validation
    u0, _ = next(iter(val_loader))
    u0 = u0[0].to(device)  # shape: (n_spatial,)
    
    n_steps = 20
    trajectory_pred = [u0.cpu().numpy()]
    u = u0
    for _ in range(n_steps):
        u = model(u.unsqueeze(0), info['dx'])  # add batch dim
        u = u.squeeze(0)
        trajectory_pred.append(u.cpu().numpy())
    
    trajectory_pred = np.array(trajectory_pred)

    # True trajectory (from dataset)
    # Take consecutive pairs from val_loader
    u_true = []
    u_curr = u0.cpu().numpy()
    for step in range(n_steps):
        u_next = next(iter(val_loader))[1][0].numpy()
        u_true.append(u_curr)
        u_curr = u_next
    u_true = np.array(u_true)

# Plot predicted vs true
plt.figure(figsize=(10,6))
plt.imshow(trajectory_pred.T, origin='lower', aspect='auto', extent=[0, n_steps, 0, info['n_spatial']])
plt.colorbar(label='u(x,t)')
plt.xlabel('Time step')
plt.ylabel('Spatial index')
plt.title('Predicted solution trajectory (heatmap)')
plt.show()
