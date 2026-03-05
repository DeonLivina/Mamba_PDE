import torch
import matplotlib.pyplot as plt
import numpy as np


def _get_dataset(dataset_or_loader):
    """Return the underlying dataset whether given a Dataset or DataLoader."""
    if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
        return dataset_or_loader.dataset
    return dataset_or_loader


def _flatten_batch(context, targets):
    
    if len(context.shape) == 4:
        B, N, T, F = context.shape
        context_flat = context.reshape(B * N, T, F)
        targets_flat = targets.reshape(B * N, -1, 2)
    elif len(context.shape) == 3:
        context_flat = context
        if len(targets.shape) == 4:
            B, N, T_out, F_out = targets.shape
            targets_flat = targets.reshape(B * N, T_out, F_out)
        else:
            targets_flat = targets
    else:
        raise ValueError(f"Unexpected context shape: {context.shape}")

    return context_flat, targets_flat


def plot_losses_clean(train_losses, val_losses, save_path='loss_dam_break.png'):
    """Plot training curves with clean styling."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=1.5, alpha=0.8)
    ax.plot(epochs, val_losses,   'r-', label='Val',   linewidth=1.5, alpha=0.8)

    best_val_epoch = int(np.argmin(val_losses)) + 1
    best_val = val_losses[best_val_epoch - 1]
    ax.axvline(x=best_val_epoch, color='tomato', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.scatter([best_val_epoch], [best_val], color='tomato', zorder=5, s=60,
               label=f'Best Val: {best_val:.6f} (epoch {best_val_epoch})')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss',  fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to '{save_path}'")
    print(f"  Best val loss: {best_val:.6f} at epoch {best_val_epoch}")
    plt.close()


def plot_prediction_comparison(model, data_loader, device, save_path='predictions_comparison.png'):
    
    model.eval()

    with torch.no_grad():
        for context, targets in data_loader:
            context = context.to(device)
            targets = targets.to(device)

            context_flat, targets_flat = _flatten_batch(context, targets)

            pred = model(context_flat)  # (B*N, n_future, 2)

            pred_np    = pred[0].detach().cpu().numpy()          # (n_future, 2)
            targets_np = targets_flat[0].detach().cpu().numpy()  # (n_future, 2)

            n_timesteps     = pred_np.shape[0]
            marker_interval = max(1, n_timesteps // 20)
            tick_interval   = max(1, n_timesteps // 10)
            timesteps       = np.arange(n_timesteps)

            feature_names = ['u (velocity)', 'h (depth)']
            colors_gt   = ['blue',      'green']
            colors_pred = ['lightblue', 'lightgreen']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for feat_idx, (ax, feat_name, color_gt, color_pred) in enumerate(
                zip(axes, feature_names, colors_gt, colors_pred)
            ):
                ax.plot(timesteps, targets_np[:, feat_idx],
                        color=color_gt, linestyle='-', linewidth=1.0,
                        label='Ground Truth', alpha=0.8,
                        marker='o', markersize=3, markevery=marker_interval)
                ax.plot(timesteps, pred_np[:, feat_idx],
                        color=color_pred, linestyle='--', linewidth=1.0,
                        label='Prediction', alpha=0.8,
                        marker='s', markersize=3, markevery=marker_interval)

                ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
                ax.set_ylabel(feat_name,  fontsize=12, fontweight='bold')
                ax.set_title(f'{feat_name}: Prediction vs Ground Truth ({n_timesteps} steps ahead)',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=11, loc='best')
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.set_xticks(range(0, n_timesteps, tick_interval))

            plt.suptitle(f'Dam Break: Prediction vs Ground Truth ({n_timesteps} steps ahead)',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to '{save_path}' ({n_timesteps} timesteps)")
            plt.close()
            break


def plot_spatial_temporal_heatmaps(model, dataset, device, save_path='spatial_temporal.png'):
    
    model.eval()
    dataset = _get_dataset(dataset)

    n_samples = min(5, len(dataset))
    ground_truth_all = []
    predictions_all  = []

    with torch.no_grad():
        for idx in range(n_samples):
            context, target = dataset[idx]

            # Handle both (N, T, F) and (B*N, T, F) from preprocessed dataset
            if len(context.shape) == 2:
                # Shouldn't happen but guard anyway
                context = context.unsqueeze(0)

            if len(context.shape) == 3:
                # Already flat: (N, T, F) — treat N as batch
                context_flat = context.to(device)
                N = context_flat.shape[0]
            elif len(context.shape) == 4:
                B, N, T, F = context.shape
                context_flat = context.reshape(B * N, T, F).to(device)

            pred = model(context_flat)           # (N, n_future, 2)
            pred = pred.unsqueeze(0)             # (1, N, n_future, 2)

            if len(target.shape) == 3:           # (N, n_future, 2)
                ground_truth_all.append(target.numpy())
            else:
                ground_truth_all.append(target[0].numpy())

            predictions_all.append(pred[0].detach().cpu().numpy())

    gt_avg   = np.mean(ground_truth_all, axis=0)  # (N, n_future, 2)
    pred_avg = np.mean(predictions_all,  axis=0)  # (N, n_future, 2)
    error_avg = np.abs(gt_avg - pred_avg)

    feature_names = ['u (velocity)', 'h (depth)']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for feat_idx, feat_name in enumerate(feature_names):
        gt_data   = gt_avg[:, :, feat_idx]
        pred_data = pred_avg[:, :, feat_idx]
        err_data  = error_avg[:, :, feat_idx]

        vmin = min(gt_data.min(), pred_data.min())
        vmax = max(gt_data.max(), pred_data.max())

        im0 = axes[feat_idx, 0].imshow(gt_data, aspect='auto', cmap='RdBu_r',
                                        origin='lower', vmin=vmin, vmax=vmax)
        axes[feat_idx, 0].set_xlabel('Timestep',       fontsize=11, fontweight='bold')
        axes[feat_idx, 0].set_ylabel('Spatial Point',  fontsize=11, fontweight='bold')
        axes[feat_idx, 0].set_title(f'{feat_name} - Ground Truth', fontsize=12, fontweight='bold')
        plt.colorbar(im0, ax=axes[feat_idx, 0]).set_label(feat_name, fontsize=10)

        im1 = axes[feat_idx, 1].imshow(pred_data, aspect='auto', cmap='RdBu_r',
                                        origin='lower', vmin=vmin, vmax=vmax)
        axes[feat_idx, 1].set_xlabel('Timestep',      fontsize=11, fontweight='bold')
        axes[feat_idx, 1].set_ylabel('Spatial Point', fontsize=11, fontweight='bold')
        axes[feat_idx, 1].set_title(f'{feat_name} - Prediction', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[feat_idx, 1]).set_label(feat_name, fontsize=10)

        im2 = axes[feat_idx, 2].imshow(err_data, aspect='auto', cmap='hot', origin='lower')
        axes[feat_idx, 2].set_xlabel('Timestep',      fontsize=11, fontweight='bold')
        axes[feat_idx, 2].set_ylabel('Spatial Point', fontsize=11, fontweight='bold')
        axes[feat_idx, 2].set_title(f'{feat_name} - Absolute Error', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[feat_idx, 2]).set_label('Error', fontsize=10)

    plt.suptitle('Spatial-Temporal Heatmaps: u and h', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to '{save_path}'")
    plt.close()

    print(f"\nError Statistics:")
    for feat_idx, feat_name in enumerate(feature_names):
        err = error_avg[:, :, feat_idx]
        print(f"  {feat_name}: Mean={err.mean():.6f}, Max={err.max():.6f}, Std={err.std():.6f}")


def plot_multiple_points(model, dataset, device, n_points=3, save_path='multiple_points.png'):
   
    model.eval()
    dataset = _get_dataset(dataset)

    context, target = dataset[0]

    if len(context.shape) == 3:
        # Already flat: (N, T, F)
        context_flat = context.to(device)
        N = context_flat.shape[0]
    elif len(context.shape) == 4:
        B, N, T, F = context.shape
        context_flat = context.reshape(B * N, T, F).to(device)

    pred = model(context_flat)               # (N, n_future, 2)

    gt_np   = target.numpy()                 # (N, n_future, 2)
    pred_np = pred.detach().cpu().numpy()    # (N, n_future, 2)

    point_indices = np.linspace(0, N - 1, n_points, dtype=int)

    feature_names = ['u (velocity)', 'h (depth)']
    timesteps       = np.arange(pred_np.shape[1])
    marker_interval = max(1, len(timesteps) // 20)

    fig, axes = plt.subplots(n_points, 2, figsize=(10, 3 * n_points))
    if n_points == 1:
        axes = axes.reshape(1, -1)

    for row_idx, point_idx in enumerate(point_indices):
        for col_idx, feat_name in enumerate(feature_names):
            ax = axes[row_idx, col_idx]

            ax.plot(timesteps, gt_np[point_idx, :, col_idx],
                    'b-', label='Ground Truth', linewidth=1.2, alpha=0.8,
                    marker='o', markersize=3, markevery=marker_interval)
            ax.plot(timesteps, pred_np[point_idx, :, col_idx],
                    'r--', label='Prediction', linewidth=1.2, alpha=0.8,
                    marker='s', markersize=3, markevery=marker_interval)

            if row_idx == 0:
                ax.set_title(f'{feat_name}', fontsize=11, fontweight='bold')

            ax.set_ylabel(f'Point {point_idx}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Timestep', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')

    plt.suptitle(f'Multiple Spatial Points: u and h ({n_points} locations)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to '{save_path}'")
    plt.close()


def plot_error_analysis(model, data_loader, device, save_path='error_analysis.png'):
    
    model.eval()

    all_pred   = []
    all_target = []

    with torch.no_grad():
        for context, targets in data_loader:
            context = context.to(device)
            targets = targets.to(device)

            context_flat, targets_flat = _flatten_batch(context, targets)

            pred = model(context_flat)  # (B*N, n_future, 2)

            all_pred.append(pred.detach().cpu().numpy())
            all_target.append(targets_flat.detach().cpu().numpy())

    all_pred   = np.vstack(all_pred)    # (total, n_future, 2)
    all_target = np.vstack(all_target)

    n_future      = all_pred.shape[1]
    feature_names = ['u (velocity)', 'h (depth)']
    timesteps     = np.arange(n_future)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for feat_idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        errors = np.abs(all_pred[:, :, feat_idx] - all_target[:, :, feat_idx])
        mse_per_timestep = np.mean(errors ** 2, axis=0)
        mae_per_timestep = np.mean(errors,      axis=0)

        ax2   = ax.twinx()
        line1 = ax.plot(timesteps,  mse_per_timestep, 'b-', linewidth=1.5,
                        marker='o', markersize=5, alpha=0.8, label='MSE')
        line2 = ax2.plot(timesteps, mae_per_timestep, 'r-', linewidth=1.5,
                         marker='s', markersize=5, alpha=0.8, label='MAE')

        ax.set_xlabel('Timestep', fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold', color='b')
        ax2.set_ylabel('MAE', fontsize=11, fontweight='bold', color='r')
        ax.set_title(f'{feat_name} - Error Metrics', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.2, linestyle='--')

        lines = line1 + line2
        ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=9)

    plt.suptitle('Error Analysis: u and h', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to '{save_path}'")
    plt.close()

    print(f"\nError Summary (all timesteps, all spatial points):")
    for feat_idx, feat_name in enumerate(feature_names):
        errors = np.abs(all_pred[:, :, feat_idx] - all_target[:, :, feat_idx])
        print(f"  {feat_name}:")
        print(f"    Mean MAE: {errors.mean():.6f}")
        print(f"    Mean MSE: {(errors**2).mean():.6f}")
        print(f"    Max error: {errors.max():.6f}")