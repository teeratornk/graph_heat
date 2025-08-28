"""
Testing script for Graph-DeepONet with result visualizations.

Evaluates the best model checkpoint on the test set, saves predictions,
and generates comprehensive visualizations.
"""

import argparse
import json
import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Try to import seaborn for better plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sampler import GraphDeepONetDataset, GraphDeepONetSampler, SamplerConfig
from model import GraphDeepONet


def get_timestamp() -> str:
    """Get timestamp string for test results directory."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: str, obj: Any) -> None:
    """Save object as JSON."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def load_scalers(scalers_path: str) -> Dict[str, Any]:
    """Load scalers from JSON file."""
    with open(scalers_path, 'r') as f:
        return json.load(f)


def denormalize_temperature(normalized: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
    """Denormalize temperature values."""
    temp_min = scalers['target']['temperature']['min']
    temp_max = scalers['target']['temperature']['max']
    return normalized * (temp_max - temp_min) + temp_min


def denormalize_coords(coords: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
    """Denormalize coordinate values (t, x, y, z)."""
    denorm_coords = coords.copy()
    
    # Time: coords[:, 0] is already normalized to [0, 1], no need to denormalize
    # The original time steps are 1-120, but we keep it as [0, 1] for display
    
    # Note: The coordinates in the normalized dataset are already in original units
    # The scalers show the min/max of the original data, not normalization parameters
    # So we don't need to denormalize spatial coordinates
    
    return denorm_coords


def evaluate_all_test_samples(
    model: GraphDeepONet,
    dataset: GraphDeepONetDataset,
    test_config: SamplerConfig,
    split_path: str,
    device: torch.device
) -> Dict[str, Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]]:
    """
    Evaluate model on all test samples in one pass.
    
    Returns:
        Dictionary mapping sample_id to (metrics, preds, targets, coords)
    """
    model.eval()
    
    # Create sampler
    test_sampler = GraphDeepONetSampler(dataset, test_config, split_path)
    
    # Initialize storage for each sample
    results = {}
    for sample_id in test_sampler.mode_sample_ids:
        results[sample_id] = {
            'preds': [],
            'targets': [],
            'coords': [],
            'node_idx': [],
            'total_loss': 0.0,
            'total_pairs': 0
        }
    
    with torch.no_grad():
        # Process all batches
        for batch_list in test_sampler:
            for b in batch_list:
                sample_id = b["sample_id"]
                
                # Move to device
                edge_index = b["edge_index"].to(device)
                edge_attr_r = b["edge_attr_r"].to(device)
                k = b["k"].to(device)
                node_pos = b["node_pos"].to(device)
                I_T = b["I_T"].to(device)
                coords = b["coords"].to(device)
                node_idx = b["node_idx"].to(device)
                targets = b["targets"].to(device)
                
                q = b.get("q")
                s = b.get("s")
                if q is not None:
                    q = q.to(device)
                if s is not None:
                    s = s.to(device)
                
                # Forward pass
                cache = model.encode_case(
                    edge_index, edge_attr_r, k, node_pos, I_T, q=q, s=s
                )
                pred = model.predict_with_cache(cache, coords, node_idx)
                
                # Compute loss
                loss = nn.functional.mse_loss(pred, targets)
                
                # Accumulate for this sample
                S_b = pred.shape[0]
                results[sample_id]['total_loss'] += loss.item() * S_b
                results[sample_id]['total_pairs'] += S_b
                
                # Store results
                results[sample_id]['preds'].append(to_numpy(pred))
                results[sample_id]['targets'].append(to_numpy(targets))
                results[sample_id]['coords'].append(to_numpy(coords))
                results[sample_id]['node_idx'].append(to_numpy(node_idx))
    
    # Process results for each sample
    final_results = {}
    
    for sample_id, data in results.items():
        # Concatenate all chunks
        if data['preds']:
            preds = np.concatenate(data['preds'], axis=0)
            targets = np.concatenate(data['targets'], axis=0)
            coords = np.concatenate(data['coords'], axis=0)
            node_indices = np.concatenate(data['node_idx'], axis=0)
        else:
            preds = np.array([])
            targets = np.array([])
            coords = np.array([])
            node_indices = np.array([])
        
        # Compute metrics
        if len(preds) > 0:
            errors = preds.flatten() - targets.flatten()
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            mse = data['total_loss'] / max(1, data['total_pairs'])
            
            # Compute relative error if targets are not close to zero
            mask = np.abs(targets.flatten()) > 1e-6
            if np.any(mask):
                rel_errors = np.abs(errors[mask] / targets.flatten()[mask])
                mape = np.mean(rel_errors) * 100
            else:
                mape = np.nan
        else:
            mae = rmse = mse = mape = np.nan
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mse": float(mse),
            "mape": float(mape) if not np.isnan(mape) else None,
            "pairs": int(data['total_pairs']),
            "batches": len(data['preds'])
        }
        
        # Reshape to full format
        N = dataset.N
        T = dataset.Tlen
        
        preds_full = np.full((T, N, 1), np.nan, dtype=np.float32)
        targets_full = np.full((T, N, 1), np.nan, dtype=np.float32)
        coords_full = np.full((T, N, 4), np.nan, dtype=np.float32)
        
        if len(preds) > 0:
            for i in range(len(preds)):
                t_idx = int(coords[i, 0] * (T - 1))
                n_idx = int(node_indices[i])
                preds_full[t_idx, n_idx] = preds[i]
                targets_full[t_idx, n_idx] = targets[i]
                coords_full[t_idx, n_idx] = coords[i]
        
        final_results[sample_id] = (metrics, preds_full, targets_full, coords_full)
    
    return final_results


def compute_hotspot_error(
    preds: np.ndarray,
    targets: np.ndarray,
    scalers: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Compute error metrics for temperature hotspots.
    
    Args:
        preds: Predictions array (T, N, 1)
        targets: Ground truth array (T, N, 1)
        scalers: Optional scalers for denormalization
    
    Returns:
        Dictionary with hotspot-specific metrics
    """
    T, N, _ = targets.shape
    
    # Denormalize if scalers provided
    if scalers is not None:
        preds = denormalize_temperature(preds, scalers)
        targets = denormalize_temperature(targets, scalers)
    
    hotspot_preds = []
    hotspot_targets = []
    
    for t in range(T):
        # Skip if all NaN
        if np.all(np.isnan(targets[t])):
            continue
            
        # Find hotspot in target
        target_slice = targets[t, :, 0]
        valid_mask = ~np.isnan(target_slice)
        
        if np.any(valid_mask):
            hotspot_idx = np.nanargmax(target_slice)
            hotspot_targets.append(target_slice[hotspot_idx])
            hotspot_preds.append(preds[t, hotspot_idx, 0])
    
    if hotspot_targets:
        hotspot_targets = np.array(hotspot_targets)
        hotspot_preds = np.array(hotspot_preds)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(hotspot_targets) | np.isnan(hotspot_preds))
        hotspot_targets = hotspot_targets[valid_mask]
        hotspot_preds = hotspot_preds[valid_mask]
        
        if len(hotspot_targets) > 0:
            hotspot_mae = np.mean(np.abs(hotspot_preds - hotspot_targets))
            hotspot_rmse = np.sqrt(np.mean((hotspot_preds - hotspot_targets)**2))
            
            return {
                "hotspot_mae": float(hotspot_mae),
                "hotspot_rmse": float(hotspot_rmse),
                "hotspot_points": len(hotspot_targets)
            }
    
    return {
        "hotspot_mae": None,
        "hotspot_rmse": None,
        "hotspot_points": 0
    }


def plot_error_histogram(
    all_errors: List[float],
    save_path: str,
    title: str = "Prediction Error Distribution",
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Plot histogram of prediction errors with optional KDE."""
    plt.figure(figsize=(10, 6))
    
    errors = np.array(all_errors)
    abs_errors = np.abs(errors)
    
    # Plot histogram
    plt.hist(abs_errors, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    
    # Add KDE if seaborn available
    if HAS_SEABORN:
        sns.kdeplot(abs_errors, color='red', linewidth=2, label='KDE')
    
    # Add statistics
    plt.axvline(np.mean(abs_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(abs_errors):.2f}')
    plt.axvline(np.median(abs_errors), color='green', linestyle='--',
                label=f'Median: {np.median(abs_errors):.2f}')
    
    xlabel = 'Absolute Error'
    if scalers is not None:
        xlabel += ' (K)'  # Temperature in Kelvin
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pred_vs_true(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    save_path: str,
    max_points: int = 50000,
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Create scatter plot of predictions vs true values."""
    plt.figure(figsize=(10, 10))
    
    # Flatten arrays
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(preds_flat) | np.isnan(targets_flat))
    preds_flat = preds_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]
    
    # Downsample if too many points
    if len(preds_flat) > max_points:
        idx = np.random.choice(len(preds_flat), max_points, replace=False)
        preds_flat = preds_flat[idx]
        targets_flat = targets_flat[idx]
    
    # Create scatter plot
    plt.scatter(targets_flat, preds_flat, alpha=0.5, s=1, c='blue')
    
    # Add perfect prediction line
    min_val = min(np.min(targets_flat), np.min(preds_flat))
    max_val = max(np.max(targets_flat), np.max(preds_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Calculate R²
    correlation = np.corrcoef(targets_flat, preds_flat)[0, 1]
    r_squared = correlation ** 2
    
    xlabel = 'True Values'
    ylabel = 'Predictions'
    if scalers is not None:
        xlabel += ' (K)'
        ylabel += ' (K)'
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Predictions vs True Values (R² = {r_squared:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hotspot_curves(
    sample_id: str,
    preds: np.ndarray,
    targets: np.ndarray,
    save_path: str,
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Plot temporal curves for hotspot and cold nodes."""
    T, N, _ = targets.shape
    
    # Find hotspot node (highest mean temperature)
    mean_temps = np.nanmean(targets[:, :, 0], axis=0)
    valid_nodes = ~np.isnan(mean_temps)
    
    if not np.any(valid_nodes):
        return
    
    hotspot_idx = np.argmax(mean_temps)
    coldspot_idx = np.argmin(mean_temps[valid_nodes])
    
    # Extract time series
    times = np.arange(T)
    hotspot_true = targets[:, hotspot_idx, 0]
    hotspot_pred = preds[:, hotspot_idx, 0]
    coldspot_true = targets[:, coldspot_idx, 0]
    coldspot_pred = preds[:, coldspot_idx, 0]
    
    # Denormalize if scalers provided
    if scalers is not None:
        hotspot_true = denormalize_temperature(hotspot_true.reshape(-1, 1), scalers).flatten()
        hotspot_pred = denormalize_temperature(hotspot_pred.reshape(-1, 1), scalers).flatten()
        coldspot_true = denormalize_temperature(coldspot_true.reshape(-1, 1), scalers).flatten()
        coldspot_pred = denormalize_temperature(coldspot_pred.reshape(-1, 1), scalers).flatten()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ylabel = 'Temperature'
    if scalers is not None:
        ylabel += ' (K)'
    
    # Hotspot plot
    ax1.plot(times, hotspot_true, 'b-', label='True', linewidth=2)
    ax1.plot(times, hotspot_pred, 'r--', label='Predicted', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{sample_id} - Hotspot Node {hotspot_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Coldspot plot
    ax2.plot(times, coldspot_true, 'b-', label='True', linewidth=2)
    ax2.plot(times, coldspot_pred, 'r--', label='Predicted', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{sample_id} - Cold Node {coldspot_idx}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spatial_snapshots(
    sample_id: str,
    node_pos: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    times: List[int],
    save_dir: str,
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Plot spatial temperature distribution at selected time points."""
    # Note: node_pos from the dataset is already in original units (meters)
    # The scalers.json shows the min/max values but the data is not normalized
    node_pos_display = node_pos
    
    for t in times:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get values at time t
        target_vals = targets[t, :, 0]
        pred_vals = preds[t, :, 0]
        
        # Denormalize if scalers provided
        if scalers is not None:
            target_vals_plot = denormalize_temperature(target_vals.reshape(-1, 1), scalers).flatten()
            pred_vals_plot = denormalize_temperature(pred_vals.reshape(-1, 1), scalers).flatten()
            error_vals = np.abs(pred_vals_plot - target_vals_plot)
        else:
            target_vals_plot = target_vals
            pred_vals_plot = pred_vals
            error_vals = np.abs(pred_vals - target_vals)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(target_vals) | np.isnan(pred_vals))
        
        if not np.any(valid_mask):
            plt.close()
            continue
        
        # Determine color scale
        vmin = min(np.nanmin(target_vals_plot), np.nanmin(pred_vals_plot))
        vmax = max(np.nanmax(target_vals_plot), np.nanmax(pred_vals_plot))
        
        # Use 2D projection if 3D
        if node_pos_display.shape[1] >= 2:
            x = node_pos_display[valid_mask, 0]
            y = node_pos_display[valid_mask, 1]
        else:
            # Fall back to node index
            x = np.arange(len(node_pos_display))[valid_mask]
            y = np.zeros_like(x)
        
        # True values
        scatter1 = axes[0].scatter(x, y, c=target_vals_plot[valid_mask], 
                                  cmap='hot', s=20, vmin=vmin, vmax=vmax)
        axes[0].set_title(f'True Temperature (t={t})')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        if scalers:
            cbar1.set_label('Temperature (K)')
        
        # Predicted values
        scatter2 = axes[1].scatter(x, y, c=pred_vals_plot[valid_mask], 
                                  cmap='hot', s=20, vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Predicted Temperature (t={t})')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        if scalers:
            cbar2.set_label('Temperature (K)')
        
        # Error
        scatter3 = axes[2].scatter(x, y, c=error_vals[valid_mask], 
                                  cmap='Blues', s=20)
        axes[2].set_title(f'Absolute Error (t={t})')
        axes[2].set_xlabel('X (m)')
        axes[2].set_ylabel('Y (m)')
        cbar3 = plt.colorbar(scatter3, ax=axes[2])
        if scalers:
            cbar3.set_label('Error (K)')
        
        plt.suptitle(f'{sample_id} - Spatial Distribution at Time {t}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sample_id}_snapshot_t{t}.png'), dpi=150)
        plt.close()


def plot_sample_metrics_bar(
    per_sample_metrics: Dict[str, Dict[str, float]],
    save_path: str,
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Create bar plot of metrics per sample."""
    samples = list(per_sample_metrics.keys())
    
    # Check if we have denormalized metrics
    if scalers and all('mae_denorm' in per_sample_metrics[s] for s in samples):
        mae_values = [per_sample_metrics[s]['mae_denorm'] for s in samples]
        rmse_values = [per_sample_metrics[s]['rmse_denorm'] for s in samples]
        ylabel = 'Error (K)'
        title = 'Test Set Metrics by Sample (Temperature in K)'
    else:
        mae_values = [per_sample_metrics[s]['mae'] for s in samples]
        rmse_values = [per_sample_metrics[s]['rmse'] for s in samples]
        ylabel = 'Error (Normalized)'
        title = 'Test Set Metrics by Sample'
    
    x = np.arange(len(samples))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
    bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
    
    ax.set_xlabel('Sample ID')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            format_str = f'{height:.2f}' if scalers else f'{height:.4f}'
            ax.annotate(format_str,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_visualizations(
    metrics_path: str,
    predictions_path: str,
    test_dir: str,
    dataset: Optional[GraphDeepONetDataset] = None,
    scalers: Optional[Dict[str, Any]] = None
) -> None:
    """Generate all visualization plots from saved results."""
    print("\nGenerating visualizations...")
    
    # Create figures directory
    figures_dir = os.path.join(test_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    per_sample_metrics = metrics['per_sample']
    
    # Load predictions
    all_preds = []
    all_targets = []
    
    with h5py.File(predictions_path, 'r') as h5f:
        for sample_id in per_sample_metrics.keys():
            if sample_id in h5f:
                preds = h5f[f'{sample_id}/preds'][:]
                targets = h5f[f'{sample_id}/targets'][:]
                
                # Collect non-NaN values
                valid_mask = ~(np.isnan(preds) | np.isnan(targets))
                
                if scalers is not None:
                    # Denormalize before collecting
                    preds_denorm = denormalize_temperature(preds[valid_mask], scalers)
                    targets_denorm = denormalize_temperature(targets[valid_mask], scalers)
                    all_preds.extend(preds_denorm.tolist())
                    all_targets.extend(targets_denorm.tolist())
                else:
                    all_preds.extend(preds[valid_mask].tolist())
                    all_targets.extend(targets[valid_mask].tolist())
                
                # Generate per-sample plots
                plot_hotspot_curves(
                    sample_id, preds, targets,
                    os.path.join(figures_dir, f'{sample_id}_hotspot_curve.png'),
                    scalers
                )
                
                # Generate spatial snapshots if we have node positions
                if dataset is not None:
                    # Get node positions
                    if dataset.has_shared:
                        try:
                            node_pos = dataset.get_shared_node_pos().numpy()
                        except:
                            node_pos = dataset.get_sample_node_pos(sample_id).numpy()
                    else:
                        node_pos = dataset.get_sample_node_pos(sample_id).numpy()
                    
                    # Select time points (first, middle, last)
                    T = preds.shape[0]
                    times = [0, T//2, T-1]
                    plot_spatial_snapshots(
                        sample_id, node_pos, preds, targets, times, figures_dir, scalers
                    )
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_errors = all_preds - all_targets
    
    # Generate aggregate plots
    plot_error_histogram(
        all_errors.tolist(),
        os.path.join(figures_dir, 'error_hist.png'),
        title='Test Set Absolute Error Distribution',
        scalers=scalers
    )
    
    plot_pred_vs_true(
        all_preds, all_targets,
        os.path.join(figures_dir, 'pred_vs_true.png'),
        scalers=scalers
    )
    
    plot_sample_metrics_bar(
        per_sample_metrics,
        os.path.join(figures_dir, 'sample_metrics_bar.png'),
        scalers
    )
    
    # Additional plot: Temperature distribution comparison
    if scalers is not None:
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Temperature distribution
        ax1.hist(all_targets, bins=50, alpha=0.5, label='True', color='blue', density=True)
        ax1.hist(all_preds, bins=50, alpha=0.5, label='Predicted', color='red', density=True)
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Density')
        ax1.set_title('Temperature Distribution: True vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error vs Temperature
        ax2.hexbin(all_targets, all_errors, gridsize=50, cmap='Blues')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('True Temperature (K)')
        ax2.set_ylabel('Prediction Error (K)')
        ax2.set_title('Prediction Error vs Temperature')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'temperature_analysis.png'), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to: {figures_dir}")


def main():
    """Main testing script."""
    parser = argparse.ArgumentParser(description='Test Graph-DeepONet on test set')
    
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--h5', default='processed_data/normalized_full_dataset.hdf5',
                        help='Path to normalized HDF5 dataset')
    parser.add_argument('--split', default='processed_data/train_val_test_split.json',
                        help='Path to train/val/test split JSON')
    parser.add_argument('--scalers', default='processed_data/scalers.json',
                        help='Path to scalers JSON')
    parser.add_argument('--batch_nodes', type=int, default=500,
                        help='Number of nodes per chunk for test evaluation')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--viz_only', action='store_true',
                        help='Only generate visualizations from existing results')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load scalers
    scalers = None
    if os.path.exists(args.scalers):
        print(f"Loading scalers from: {args.scalers}")
        scalers = load_scalers(args.scalers)
    
    # Determine test directory
    if args.viz_only:
        # Find existing test results
        if args.outdir:
            test_dir = args.outdir
        else:
            # Look for most recent test results in run_dir
            test_dirs = [d for d in os.listdir(args.run_dir) if d.startswith('test_results_')]
            if not test_dirs:
                raise ValueError("No test results found in run directory")
            test_dir = os.path.join(args.run_dir, sorted(test_dirs)[-1])
        
        print(f"Regenerating visualizations from: {test_dir}")
        
        # Check for required files
        metrics_path = os.path.join(test_dir, 'metrics.json')
        predictions_path = os.path.join(test_dir, 'predictions.h5')
        
        if not os.path.exists(metrics_path) or not os.path.exists(predictions_path):
            raise FileNotFoundError("Required files not found in test directory")
        
        # Load dataset for node positions
        dataset = GraphDeepONetDataset(
            h5_path=args.h5,
            include_qs=False,
            use_shared_if_available=True
        )
        
        # Generate visualizations
        generate_visualizations(metrics_path, predictions_path, test_dir, dataset, scalers)
        return
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.run_dir, 'best.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model args
    model_args = checkpoint['args']
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    print(f"Checkpoint from epoch {epoch} with val_loss {val_loss:.6f}")
    
    # Rebuild model
    print("Building model...")
    model = GraphDeepONet(
        q_dim=model_args['q_dim'],
        trunk_hidden=model_args['trunk_hidden'],
        trunk_depth=model_args['trunk_depth'],
        glob_hidden=model_args['glob_hidden'],
        graph_hidden=model_args['graph_hidden'],
        graph_layers=model_args['graph_layers'],
        use_fourier_time=model_args['use_fourier_time'],
        time_bands=model_args['time_bands'],
        use_qs_features=model_args['use_qs_features'],
        film_global=model_args['film_global'],
        head_bias_mlp=model_args['head_bias_mlp']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Create dataset
    print("Loading dataset...")
    dataset = GraphDeepONetDataset(
        h5_path=args.h5,
        include_qs=model_args['use_qs_features'],
        use_shared_if_available=True
    )
    
    # Load split info
    with open(args.split, 'r') as f:
        split_info = json.load(f)
    
    test_ids = split_info['test_ids']
    print(f"Test samples: {test_ids}")
    
    # Create output directory
    if args.outdir:
        test_dir = args.outdir
    else:
        test_dir = os.path.join(args.run_dir, f'test_results_{get_timestamp()}')
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test results directory: {test_dir}")
    
    # Create test config
    test_config = SamplerConfig(
        h5_path=args.h5,
        split_json=args.split,
        mode="test",
        rng_seed=args.seed,
        deterministic=True,
        test_chunk_nodes=args.batch_nodes,
        include_qs=model_args['use_qs_features'],
    )
    
    # Evaluate all test samples in one pass
    print("\nEvaluating all test samples...")
    all_results = evaluate_all_test_samples(
        model, dataset, test_config, args.split, device
    )
    
    # Initialize storage
    per_sample_metrics = {}
    all_errors = []
    
    # Create HDF5 file for predictions
    h5_path = os.path.join(test_dir, 'predictions.h5')
    with h5py.File(h5_path, 'w') as h5f:
        
        # Process each test sample result
        for sample_id in test_ids:
            if sample_id not in all_results:
                print(f"Warning: {sample_id} not found in results")
                continue
                
            print(f"\nEvaluating {sample_id}...")
            
            metrics, preds, targets, coords = all_results[sample_id]
            
            # Compute hotspot metrics with denormalized values
            hotspot_metrics = compute_hotspot_error(preds, targets, scalers)
            metrics.update(hotspot_metrics)
            
            # Compute denormalized metrics
            if scalers is not None:
                # Denormalize for metric computation
                preds_denorm = denormalize_temperature(preds, scalers)
                targets_denorm = denormalize_temperature(targets, scalers)
                
                valid_mask = ~(np.isnan(preds_denorm.flatten()) | np.isnan(targets_denorm.flatten()))
                if np.any(valid_mask):
                    errors_denorm = preds_denorm.flatten()[valid_mask] - targets_denorm.flatten()[valid_mask]
                    mae_denorm = np.mean(np.abs(errors_denorm))
                    rmse_denorm = np.sqrt(np.mean(errors_denorm**2))
                    
                    metrics['mae_denorm'] = float(mae_denorm)
                    metrics['rmse_denorm'] = float(rmse_denorm)
            
            # Store metrics
            per_sample_metrics[sample_id] = metrics
            
            # Collect errors for histogram
            valid_mask = ~(np.isnan(preds.flatten()) | np.isnan(targets.flatten()))
            if np.any(valid_mask):
                errors = (preds.flatten()[valid_mask] - targets.flatten()[valid_mask]).tolist()
                all_errors.extend(errors)
            
            # Save predictions to HDF5
            grp = h5f.create_group(sample_id)
            grp.create_dataset('preds', data=preds, compression='gzip')
            grp.create_dataset('targets', data=targets, compression='gzip')
            grp.create_dataset('coords', data=coords, compression='gzip')
            
            # Print sample metrics
            print(f"  MAE (normalized): {metrics['mae']:.6f}")
            print(f"  RMSE (normalized): {metrics['rmse']:.6f}")
            if scalers and 'mae_denorm' in metrics:
                print(f"  MAE (K): {metrics['mae_denorm']:.2f}")
                print(f"  RMSE (K): {metrics['rmse_denorm']:.2f}")
            if metrics.get('mape') is not None:
                print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Pairs evaluated: {metrics['pairs']:,}")
            if metrics.get('hotspot_mae') is not None:
                print(f"  Hotspot MAE: {metrics['hotspot_mae']:.2f} K" if scalers else f"  Hotspot MAE: {metrics['hotspot_mae']:.6f}")
    
    # Compute aggregate metrics
    aggregate_metrics = {}
    
    # Weighted averages by number of pairs
    total_pairs = sum(m['pairs'] for m in per_sample_metrics.values())
    
    for metric in ['mae', 'rmse', 'mse']:
        weighted_sum = sum(
            m[metric] * m['pairs'] 
            for m in per_sample_metrics.values() 
            if not np.isnan(m[metric])
        )
        aggregate_metrics[metric] = weighted_sum / total_pairs if total_pairs > 0 else np.nan
    
    # Simple average for MAPE (percentage metric)
    mape_values = [m['mape'] for m in per_sample_metrics.values() if m.get('mape') is not None]
    if mape_values:
        aggregate_metrics['mape'] = np.mean(mape_values)
    
    # Hotspot metrics
    hotspot_mae_values = [
        m['hotspot_mae'] for m in per_sample_metrics.values() 
        if m.get('hotspot_mae') is not None
    ]
    if hotspot_mae_values:
        aggregate_metrics['hotspot_mae'] = np.mean(hotspot_mae_values)
        aggregate_metrics['hotspot_rmse'] = np.mean([
            m['hotspot_rmse'] for m in per_sample_metrics.values() 
            if m.get('hotspot_rmse') is not None
        ])
    
    aggregate_metrics['total_pairs'] = total_pairs
    aggregate_metrics['n_samples'] = len(per_sample_metrics)
    
    # Save metrics
    all_metrics = {
        'per_sample': per_sample_metrics,
        'aggregate': aggregate_metrics,
        'checkpoint': {
            'path': checkpoint_path,
            'epoch': epoch,
            'val_loss': val_loss
        }
    }
    
    save_json(os.path.join(test_dir, 'metrics.json'), all_metrics)
    
    # Save summary CSV
    summary_data = []
    for sample_id, metrics in per_sample_metrics.items():
        row_data = {
            'sample_id': sample_id,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics.get('mape'),
            'pairs': metrics['pairs'],
            'hotspot_mae': metrics.get('hotspot_mae'),
            'hotspot_rmse': metrics.get('hotspot_rmse')
        }
        
        # Add denormalized metrics if available
        if 'mae_denorm' in metrics:
            row_data['mae_K'] = metrics['mae_denorm']
            row_data['rmse_K'] = metrics['rmse_denorm']
        
        summary_data.append(row_data)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(test_dir, 'summary.csv'), index=False)
    
    # Generate visualizations
    generate_visualizations(
        os.path.join(test_dir, 'metrics.json'),
        h5_path,
        test_dir,
        dataset,
        scalers
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TEST EVALUATION COMPLETE")
    print("="*60)
    print(f"Test samples evaluated: {len(per_sample_metrics)}")
    print(f"Total prediction pairs: {aggregate_metrics['total_pairs']:,}")
    print(f"\nAggregate Metrics:")
    print(f"  MAE (normalized):  {aggregate_metrics['mae']:.6f}")
    print(f"  RMSE (normalized): {aggregate_metrics['rmse']:.6f}")
    print(f"  MSE (normalized):  {aggregate_metrics['mse']:.6f}")
    
    # Compute aggregate denormalized metrics if available
    if scalers and any('mae_denorm' in m for m in per_sample_metrics.values()):
        mae_denorm_values = [m['mae_denorm'] for m in per_sample_metrics.values() if 'mae_denorm' in m]
        rmse_denorm_values = [m['rmse_denorm'] for m in per_sample_metrics.values() if 'rmse_denorm' in m]
        if mae_denorm_values:
            # Weighted average for denormalized metrics
            total_pairs = sum(m['pairs'] for m in per_sample_metrics.values())
            mae_denorm_weighted = sum(m['mae_denorm'] * m['pairs'] for m in per_sample_metrics.values() if 'mae_denorm' in m) / total_pairs
            rmse_denorm_weighted = sum(m['rmse_denorm'] * m['pairs'] for m in per_sample_metrics.values() if 'rmse_denorm' in m) / total_pairs
            
            print(f"  MAE (K):  {mae_denorm_weighted:.2f}")
            print(f"  RMSE (K): {rmse_denorm_weighted:.2f}")
            
            # Also add to aggregate metrics
            aggregate_metrics['mae_denorm'] = mae_denorm_weighted
            aggregate_metrics['rmse_denorm'] = rmse_denorm_weighted
    
    if 'mape' in aggregate_metrics:
        print(f"  MAPE: {aggregate_metrics['mape']:.2f}%")
    if 'hotspot_mae' in aggregate_metrics:
        unit = " K" if scalers else ""
        print(f"  Hotspot MAE:  {aggregate_metrics['hotspot_mae']:.2f}{unit}")
        print(f"  Hotspot RMSE: {aggregate_metrics['hotspot_rmse']:.2f}{unit}")
    
    # Update saved metrics with denormalized values
    all_metrics = {
        'per_sample': per_sample_metrics,
        'aggregate': aggregate_metrics,
        'checkpoint': {
            'path': checkpoint_path,
            'epoch': epoch,
            'val_loss': val_loss
        }
    }
    
    save_json(os.path.join(test_dir, 'metrics.json'), all_metrics)
    
    print(f"\nResults saved to: {test_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
