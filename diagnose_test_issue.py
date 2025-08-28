"""
Diagnostic script to investigate why test evaluation fails for specific samples.

Traces through the sampler and evaluation process step by step.
"""

import argparse
import json
import torch
import numpy as np
from typing import Dict, Any

from sampler import GraphDeepONetDataset, GraphDeepONetSampler, SamplerConfig
from model import GraphDeepONet


def diagnose_sampler(sample_id: str, args: argparse.Namespace) -> None:
    """Diagnose sampler behavior for a specific sample."""
    print(f"\nDiagnosing sampler for {sample_id}")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = GraphDeepONetDataset(
        h5_path=args.h5,
        include_qs=False,
        use_shared_if_available=True
    )
    print(f"Dataset loaded: {len(dataset.sample_ids)} samples")
    print(f"Sample IDs: {dataset.sample_ids}")
    print(f"Dimensions: N={dataset.N}, E={dataset.E}, T={dataset.Tlen}")
    
    # Load split info
    with open(args.split, 'r') as f:
        split_info = json.load(f)
    
    # Check if sample is in test set
    print(f"\nChecking split assignment...")
    if sample_id in split_info['train_ids']:
        print(f"{sample_id} is in TRAIN set")
    elif sample_id in split_info['test_ids']:
        print(f"{sample_id} is in TEST set")
    else:
        print(f"{sample_id} is NOT in any split!")
        return
    
    # Create test sampler
    print(f"\nCreating test sampler...")
    test_config = SamplerConfig(
        h5_path=args.h5,
        split_json=args.split,
        mode="test",
        rng_seed=42,
        deterministic=True,
        test_chunk_nodes=args.batch_nodes,
        include_qs=False,
    )
    
    print(f"Test config:")
    print(f"  mode: {test_config.mode}")
    print(f"  test_chunk_nodes: {test_config.test_chunk_nodes}")
    print(f"  deterministic: {test_config.deterministic}")
    
    sampler = GraphDeepONetSampler(dataset, test_config, args.split)
    print(f"\nSampler created:")
    print(f"  mode_sample_ids: {sampler.mode_sample_ids}")
    print(f"  current_idx: {sampler.current_idx}")
    
    # Find the sample in the sampler
    print(f"\nSearching for {sample_id} in sampler...")
    initial_idx = sampler.current_idx
    found = False
    
    for i in range(len(sampler.mode_sample_ids)):
        current_sample = sampler.mode_sample_ids[sampler.current_idx]
        print(f"  Position {sampler.current_idx}: {current_sample}")
        
        if current_sample == sample_id:
            found = True
            print(f"  Found {sample_id} at position {sampler.current_idx}")
            break
        
        sampler.current_idx += 1
        if sampler.current_idx >= len(sampler.mode_sample_ids):
            print("  Reached end of samples")
            break
    
    if not found:
        print(f"ERROR: {sample_id} not found in test sampler!")
        return
    
    # Try to get a batch
    print(f"\nAttempting to get batch for {sample_id}...")
    print(f"Current sampler state:")
    print(f"  current_idx: {sampler.current_idx}")
    print(f"  test_node_offset: {sampler.test_node_offset}")
    
    try:
        batch_list = next(iter(sampler))
        print(f"\nSuccessfully got batch list with {len(batch_list)} batches")
        
        for i, batch in enumerate(batch_list):
            print(f"\nBatch {i}:")
            print(f"  sample_id: {batch['sample_id']}")
            for key, tensor in batch.items():
                if key != 'sample_id' and isinstance(tensor, torch.Tensor):
                    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                    # Check for NaN values
                    if tensor.dtype in [torch.float32, torch.float64]:
                        has_nan = torch.isnan(tensor).any().item()
                        if has_nan:
                            nan_count = torch.isnan(tensor).sum().item()
                            print(f"    WARNING: Contains {nan_count} NaN values!")
                    # Show first few values
                    if tensor.numel() > 0:
                        flat = tensor.flatten()
                        if len(flat) > 5:
                            print(f"    First 5 values: {flat[:5].tolist()}")
                        else:
                            print(f"    Values: {flat.tolist()}")
            
            # Check coords specifically
            if 'coords' in batch:
                coords = batch['coords']
                print(f"\n  Coords analysis:")
                print(f"    Shape: {coords.shape}")
                if coords.shape[0] > 0:
                    print(f"    Time range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
                    print(f"    X range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
                    print(f"    Y range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
                    print(f"    Z range: [{coords[:, 3].min():.3f}, {coords[:, 3].max():.3f}]")
                else:
                    print(f"    WARNING: Empty coords tensor!")
            
            # Check targets
            if 'targets' in batch:
                targets = batch['targets']
                print(f"\n  Targets analysis:")
                print(f"    Shape: {targets.shape}")
                if targets.shape[0] > 0:
                    print(f"    Range: [{targets.min():.6f}, {targets.max():.6f}]")
                    print(f"    Mean: {targets.mean():.6f}")
                else:
                    print(f"    WARNING: Empty targets tensor!")
                    
    except StopIteration:
        print("ERROR: Sampler raised StopIteration - no more batches available")
    except Exception as e:
        print(f"ERROR: Exception while getting batch: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Check the underlying data directly
    print(f"\n\nChecking raw data for {sample_id}...")
    try:
        # Get temperature data
        temperature = dataset.get_temperature(sample_id)
        print(f"Temperature shape: {temperature.shape}")
        print(f"Temperature range: [{temperature.min():.6f}, {temperature.max():.6f}]")
        print(f"Has NaN: {torch.isnan(temperature).any().item()}")
        
        # Get topology data
        topology = dataset.get_topology(sample_id)
        print(f"Topology shape: {topology.shape}")
        print(f"Topology has NaN: {torch.isnan(topology).any().item()}")
        
        # Check time indices
        print(f"\nChecking time indices for test mode...")
        print(f"Dataset Tlen: {dataset.Tlen}")
        print(f"Test should use all timesteps: 0 to {dataset.Tlen-1}")
        
    except Exception as e:
        print(f"ERROR accessing raw data: {e}")


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description='Diagnose test evaluation issues')
    parser.add_argument('--h5', default='processed_data/normalized_full_dataset.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--split', default='processed_data/train_val_test_split.json',
                        help='Path to split JSON')
    parser.add_argument('--sample', type=str, default='S006',
                        help='Sample ID to diagnose')
    parser.add_argument('--batch_nodes', type=int, default=500,
                        help='Number of nodes per batch')
    
    args = parser.parse_args()
    
    print(f"Diagnosing test evaluation issue for sample: {args.sample}")
    diagnose_sampler(args.sample, args)


if __name__ == "__main__":
    main()
