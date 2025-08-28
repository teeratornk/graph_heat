"""
Train script for lightweight Graph-DeepONet (CPU-friendly).

Trains the model on normalized CFD simulation data with proper train/val splits.
"""

import argparse
import json
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt

from sampler import GraphDeepONetDataset, GraphDeepONetSampler, SamplerConfig
from model import GraphDeepONet


def get_timestamp() -> str:
    """Get timestamp string for run directory."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(path: str, obj: Any) -> None:
    """Save object as JSON."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def copy_file(src: str, dst: str) -> None:
    """Copy file from src to dst."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    use_cosine: bool,
    base_lr: float
) -> Optional[Union[LinearLR, SequentialLR]]:
    """Create learning rate scheduler with warmup and optional cosine decay."""
    if warmup_steps <= 0:
        return None
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    if use_cosine and total_steps > warmup_steps:
        # Create cosine scheduler for after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=base_lr * 0.01
        )
        # Combine them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        # Just use warmup
        scheduler = warmup_scheduler
    
    return scheduler


def train_one_epoch(
    model: GraphDeepONet,
    optimizer: optim.Optimizer,
    scheduler: Optional[Union[LinearLR, SequentialLR]],
    train_sampler: GraphDeepONetSampler,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pairs = 0
    
    for step, batch_list in enumerate(train_sampler):
        loss_step = 0.0
        pairs_step = 0
        
        # Process each sample in the batch
        for b in batch_list:
            # Move to device (CPU)
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
            
            # Forward pass with caching
            cache = model.encode_case(
                edge_index, edge_attr_r, k, node_pos, I_T, q=q, s=s
            )
            pred = model.predict_with_cache(cache, coords, node_idx)
            
            # Compute loss
            loss = nn.functional.mse_loss(pred, targets)
            
            # Accumulate
            S_b = pred.shape[0]  # number of pairs
            loss_step += loss.item() * S_b
            pairs_step += S_b
            
            # Backward
            loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Running totals
        total_loss += loss_step
        total_pairs += pairs_step
    
    return total_loss / max(1, total_pairs)


def eval_one_epoch(
    model: GraphDeepONet,
    val_sampler: GraphDeepONetSampler,
    device: torch.device
) -> float:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    
    with torch.no_grad():
        for batch_list in val_sampler:
            for b in batch_list:
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
                
                # Accumulate
                S_b = pred.shape[0]
                total_loss += loss.item() * S_b
                total_pairs += S_b
    
    return total_loss / max(1, total_pairs)


def plot_losses(train_losses: list, val_losses: list, save_path: str) -> None:
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Graph-DeepONet on CFD data')
    
    # Data
    parser.add_argument('--h5', default='processed_data/normalized_full_dataset.hdf5',
                        help='Path to normalized HDF5 dataset')
    parser.add_argument('--split', default='processed_data/train_val_test_split.json',
                        help='Path to train/val/test split JSON')
    parser.add_argument('--scalers', default='processed_data/scalers.json',
                        help='Path to scalers JSON')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model
    parser.add_argument('--q_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--trunk_hidden', type=int, default=128, help='Trunk hidden size')
    parser.add_argument('--trunk_depth', type=int, default=2, help='Trunk depth')
    parser.add_argument('--glob_hidden', type=int, default=64, help='Global branch hidden size')
    parser.add_argument('--graph_hidden', type=int, default=64, help='Graph branch hidden size')
    parser.add_argument('--graph_layers', type=int, default=1, help='Number of GCN layers')
    parser.add_argument('--use_fourier_time', action='store_true', help='Use Fourier time features')
    parser.add_argument('--time_bands', type=int, default=0, help='Number of Fourier bands')
    parser.add_argument('--use_qs_features', action='store_true', help='Use q and s features')
    parser.add_argument('--film_global', action='store_true', help='Use FiLM conditioning')
    parser.add_argument('--head_bias_mlp', action='store_true', help='Use MLP head instead of linear')
    
    # Optimization
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup steps')
    parser.add_argument('--cosine', action='store_true', help='Use cosine decay after warmup')
    parser.add_argument('--print_every', type=int, default=1, help='Print frequency')
    
    # Sampling
    parser.add_argument('--samples_per_step', type=int, default=1, help='Samples per training step')
    parser.add_argument('--t_subsample_frac', type=float, default=0.2, help='Time subsample fraction')
    parser.add_argument('--node_subsample_frac', type=float, default=0.2, help='Node subsample fraction')
    parser.add_argument('--target_pairs_cap', type=int, default=30000, help='Max pairs per sample')
    parser.add_argument('--val_node_frac', type=float, default=0.25, help='Validation node fraction')
    parser.add_argument('--test_chunk_nodes', type=int, default=0, help='Test chunk size (0=no chunking)')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (keep 0 for h5py)')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seed(args.seed, args.deterministic)
    
    # Create run directory
    timestamp = get_timestamp()
    run_dir = os.path.join(args.outdir, f'train_results_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save configuration
    save_json(os.path.join(run_dir, 'args.json'), vars(args))
    copy_file(args.split, os.path.join(run_dir, 'train_val_test_split.json'))
    copy_file(args.scalers, os.path.join(run_dir, 'scalers.json'))
    
    # Create dataset
    print("Loading dataset...")
    dataset = GraphDeepONetDataset(
        h5_path=args.h5,
        include_qs=args.use_qs_features,
        use_shared_if_available=True
    )
    print(f"Dataset: {len(dataset.sample_ids)} samples, {dataset.N} nodes, {dataset.E} edges")
    
    # Create samplers
    print("Creating samplers...")
    train_config = SamplerConfig(
        h5_path=args.h5,
        split_json=args.split,
        mode="train",
        rng_seed=args.seed,
        samples_per_step=args.samples_per_step,
        t_subsample_frac=args.t_subsample_frac,
        node_subsample_frac=args.node_subsample_frac,
        target_pairs_cap=None if args.target_pairs_cap <= 0 else args.target_pairs_cap,
        deterministic=args.deterministic,
        include_qs=args.use_qs_features,
        val_node_frac=args.val_node_frac,
    )
    train_sampler = GraphDeepONetSampler(dataset, train_config, args.split)
    
    val_config = SamplerConfig(
        h5_path=args.h5,
        split_json=args.split,
        mode="val",
        rng_seed=args.seed,
        samples_per_step=1,
        t_subsample_frac=1.0,
        node_subsample_frac=args.val_node_frac,
        target_pairs_cap=None,
        deterministic=True,
        include_qs=args.use_qs_features,
        val_node_frac=args.val_node_frac,
    )
    val_sampler = GraphDeepONetSampler(dataset, val_config, args.split)
    
    # Build model
    print("Building model...")
    model = GraphDeepONet(
        q_dim=args.q_dim,
        trunk_hidden=args.trunk_hidden,
        trunk_depth=args.trunk_depth,
        glob_hidden=args.glob_hidden,
        graph_hidden=args.graph_hidden,
        graph_layers=args.graph_layers,
        use_fourier_time=args.use_fourier_time,
        time_bands=args.time_bands,
        use_qs_features=args.use_qs_features,
        film_global=args.film_global,
        head_bias_mlp=args.head_bias_mlp
    )
    device = torch.device("cpu")
    model.to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = None
    if args.warmup_steps > 0:
        # Estimate total steps
        steps_per_epoch = len(train_sampler.mode_sample_ids) // args.samples_per_step
        total_steps = steps_per_epoch * args.epochs
        scheduler = create_scheduler(
            optimizer, args.warmup_steps, total_steps, args.cosine, args.lr
        )
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_sampler, device, args.grad_clip
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = eval_one_epoch(model, val_sampler, device)
        val_losses.append(val_loss)
        
        # Check if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
        
        # Print progress
        if epoch % args.print_every == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")
        
        # Save checkpoints
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'args': vars(args)
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(run_dir, 'best.pt'))
        torch.save(checkpoint, os.path.join(run_dir, 'last.pt'))
        
        # Reset sampler for next epoch
        train_sampler = GraphDeepONetSampler(dataset, train_config, args.split)
        val_sampler = GraphDeepONetSampler(dataset, val_config, args.split)
    
    # Save history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val': best_val_loss,
        'best_epoch': best_epoch
    }
    save_json(os.path.join(run_dir, 'history.json'), history)
    
    # Plot losses
    plot_losses(train_losses, val_losses, os.path.join(run_dir, 'loss_curve.png'))
    
    # Final summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Total parameters: {n_params:,}")
    print(f"Results saved to: {run_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
