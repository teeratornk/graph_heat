import h5py
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Any, Optional
import warnings


def load_split(split_path: str) -> Dict[str, Any]:
    """Load train/val/test split from JSON file.
    
    Args:
        split_path: Path to train_val_test_split.json
        
    Returns:
        Dictionary containing split information
    """
    with open(split_path, 'r') as f:
        split = json.load(f)
    
    # Validate time indices
    train_time_idx = split['train_time_idx']
    val_time_idx = split['val_time_idx']
    
    # Check range
    all_time_idx = train_time_idx + val_time_idx
    if any(idx < 0 or idx >= 120 for idx in all_time_idx):
        raise ValueError("Time indices must be in range [0, 120)")
    
    # Check no overlap
    if set(train_time_idx).intersection(set(val_time_idx)):
        raise ValueError("Train and validation time indices must not overlap")
    
    return split


def compute_train_stats(hdf5_path: str, split: Dict[str, Any], include_q_s: bool = False) -> Dict[str, Any]:
    """Compute min/max statistics on training data only.
    
    Args:
        hdf5_path: Path to full_dataset.hdf5
        split: Split dictionary
        include_q_s: Whether to include q and s features
        
    Returns:
        Dictionary of min/max statistics for each feature
    """
    train_ids = split['train_ids']
    train_time_idx = split['train_time_idx']
    
    stats = {
        'topology': {},
        'node': {},
        'edge': {},
        'globals': {},
        'target': {}
    }
    
    with h5py.File(hdf5_path, 'r') as f:
        # 1. Topology stats
        # t: min/max from train_time_idx (these are 0-119 after dropping first timestep)
        # Since topology stores t values starting from 1, we need to add 1
        t_values = np.array(train_time_idx) + 1  # Convert to actual t values
        stats['topology']['t'] = {'min': float(np.min(t_values)), 'max': float(np.max(t_values))}
        
        # x, y, z: from node positions
        if 'shared' in f and 'node_pos' in f['shared']:
            node_pos = f['shared/node_pos'][:]
        else:
            # Aggregate from train samples
            node_pos_list = []
            for train_id in train_ids:
                if f'{train_id}/node_pos' in f:
                    node_pos_list.append(f[f'{train_id}/node_pos'][:])
            node_pos = np.concatenate(node_pos_list, axis=0) if node_pos_list else None
        
        if node_pos is not None:
            stats['topology']['x'] = {'min': float(np.min(node_pos[:, 0])), 'max': float(np.max(node_pos[:, 0]))}
            stats['topology']['y'] = {'min': float(np.min(node_pos[:, 1])), 'max': float(np.max(node_pos[:, 1]))}
            stats['topology']['z'] = {'min': float(np.min(node_pos[:, 2])), 'max': float(np.max(node_pos[:, 2]))}
        
        # 2. Node features
        # k
        if 'shared' in f and 'k' in f['shared']:
            k_values = f['shared/k'][:]
        else:
            k_list = []
            for train_id in train_ids:
                if f'{train_id}/k' in f:
                    k_list.append(f[f'{train_id}/k'][:])
            k_values = np.concatenate(k_list, axis=0) if k_list else None
        
        if k_values is not None:
            stats['node']['k'] = {'min': float(np.min(k_values)), 'max': float(np.max(k_values))}
        
        # Optional q, s
        if include_q_s:
            for feature in ['q', 's']:
                feature_list = []
                for train_id in train_ids:
                    if f'{train_id}/{feature}' in f:
                        feature_list.append(f[f'{train_id}/{feature}'][:])
                if feature_list:
                    feature_values = np.concatenate(feature_list, axis=0)
                    stats['node'][feature] = {'min': float(np.min(feature_values)), 'max': float(np.max(feature_values))}
        
        # 3. Edge features
        # edge_attr_r
        if 'shared' in f and 'edge_attr_r' in f['shared']:
            r_values = f['shared/edge_attr_r'][:]
        else:
            r_list = []
            for train_id in train_ids:
                if f'{train_id}/edge_attr_r' in f:
                    r_list.append(f[f'{train_id}/edge_attr_r'][:])
            r_values = np.concatenate(r_list, axis=0) if r_list else None
        
        if r_values is not None:
            stats['edge']['r'] = {'min': float(np.min(r_values)), 'max': float(np.max(r_values))}
        
        # 4. Globals (I, T)
        I_T = f['I_T'][:]
        train_indices = [int(train_id[1:]) for train_id in train_ids]  # Extract numeric indices
        train_I_T = I_T[train_indices]
        
        stats['globals']['I'] = {'min': float(np.min(train_I_T[:, 0])), 'max': float(np.max(train_I_T[:, 0]))}
        stats['globals']['T'] = {'min': float(np.min(train_I_T[:, 1])), 'max': float(np.max(train_I_T[:, 1]))}
        
        # 5. Target (temperature)
        temp_list = []
        for train_id in train_ids:
            temperature = f[f'{train_id}/temperature'][:]
            # Only use train timesteps
            temp_list.append(temperature[train_time_idx].flatten())
        
        all_temps = np.concatenate(temp_list)
        stats['target']['temperature'] = {'min': float(np.min(all_temps)), 'max': float(np.max(all_temps))}
    
    return stats


def normalize_value(x: np.ndarray, min_val: float, max_val: float, epsilon: float = 1e-12) -> np.ndarray:
    """Apply min-max normalization with epsilon for zero-range.
    
    Args:
        x: Values to normalize
        min_val: Minimum value from training data
        max_val: Maximum value from training data
        epsilon: Small value to prevent division by zero
        
    Returns:
        Normalized values clipped to [0, 1]
    """
    denom = max(epsilon, max_val - min_val)
    if denom == epsilon:
        warnings.warn(f"Zero range detected: min={min_val}, max={max_val}. Outputting zeros.")
    
    x_norm = (x - min_val) / denom
    return np.clip(x_norm, 0, 1).astype(np.float32)


def normalize_dataset(
    input_path: str,
    output_path: str,
    scalers: Dict[str, Any],
    split: Dict[str, Any],
    include_q_s: bool = False
) -> None:
    """Create normalized copy of the dataset.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output normalized HDF5 file
        scalers: Dictionary of min/max scalers
        split: Split dictionary
        include_q_s: Whether to normalize q and s features
    """
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Copy structure and normalize data
        
        # 1. Metadata group
        meta_grp = f_out.create_group('metadata')
        if 'metadata' in f_in:
            for key, value in f_in['metadata'].attrs.items():
                meta_grp.attrs[key] = value
        meta_grp.attrs['normalized'] = True
        
        # Store split and scalers in metadata
        f_out.create_group('meta')
        f_out['meta'].attrs['suggested_split'] = json.dumps(split)
        f_out['meta'].attrs['scalers'] = json.dumps(scalers)
        
        # 2. Shared tensors
        if 'shared' in f_in:
            shared_grp = f_out.create_group('shared')
            
            # edge_index - copy as-is (no normalization)
            if 'edge_index' in f_in['shared']:
                shared_grp.create_dataset('edge_index', data=f_in['shared/edge_index'][:], compression='gzip')
            
            # node_pos - normalize x, y, z
            if 'node_pos' in f_in['shared']:
                node_pos = f_in['shared/node_pos'][:]
                node_pos_norm = np.zeros_like(node_pos, dtype=np.float32)
                node_pos_norm[:, 0] = normalize_value(node_pos[:, 0], scalers['topology']['x']['min'], scalers['topology']['x']['max'])
                node_pos_norm[:, 1] = normalize_value(node_pos[:, 1], scalers['topology']['y']['min'], scalers['topology']['y']['max'])
                node_pos_norm[:, 2] = normalize_value(node_pos[:, 2], scalers['topology']['z']['min'], scalers['topology']['z']['max'])
                shared_grp.create_dataset('node_pos', data=node_pos_norm, compression='gzip')
            
            # k - normalize
            if 'k' in f_in['shared']:
                k = f_in['shared/k'][:]
                k_norm = normalize_value(k, scalers['node']['k']['min'], scalers['node']['k']['max'])
                shared_grp.create_dataset('k', data=k_norm, compression='gzip')
            
            # edge_attr_r - normalize
            if 'edge_attr_r' in f_in['shared']:
                r = f_in['shared/edge_attr_r'][:]
                r_norm = normalize_value(r, scalers['edge']['r']['min'], scalers['edge']['r']['max'])
                shared_grp.create_dataset('edge_attr_r', data=r_norm, compression='gzip')
        
        # 3. Global I_T - normalize each column
        if 'I_T' in f_in:
            I_T = f_in['I_T'][:]
            I_T_norm = np.zeros_like(I_T, dtype=np.float32)
            I_T_norm[:, 0] = normalize_value(I_T[:, 0], scalers['globals']['I']['min'], scalers['globals']['I']['max'])
            I_T_norm[:, 1] = normalize_value(I_T[:, 1], scalers['globals']['T']['min'], scalers['globals']['T']['max'])
            f_out.create_dataset('I_T', data=I_T_norm, compression='gzip')
        
        # 4. Per-sample data
        sample_ids = [key for key in f_in.keys() if key.startswith('S')]
        for sample_id in sample_ids:
            sample_grp = f_out.create_group(sample_id)
            
            # temperature - normalize
            if f'{sample_id}/temperature' in f_in:
                temperature = f_in[f'{sample_id}/temperature'][:]
                temp_norm = normalize_value(temperature, scalers['target']['temperature']['min'], scalers['target']['temperature']['max'])
                sample_grp.create_dataset('temperature', data=temp_norm, compression='gzip')
            
            # topology - normalize t, x, y, z
            if f'{sample_id}/topology' in f_in:
                topology = f_in[f'{sample_id}/topology'][:]
                topology_norm = np.zeros_like(topology, dtype=np.float32)
                topology_norm[:, :, 0] = normalize_value(topology[:, :, 0], scalers['topology']['t']['min'], scalers['topology']['t']['max'])
                topology_norm[:, :, 1] = normalize_value(topology[:, :, 1], scalers['topology']['x']['min'], scalers['topology']['x']['max'])
                topology_norm[:, :, 2] = normalize_value(topology[:, :, 2], scalers['topology']['y']['min'], scalers['topology']['y']['max'])
                topology_norm[:, :, 3] = normalize_value(topology[:, :, 3], scalers['topology']['z']['min'], scalers['topology']['z']['max'])
                sample_grp.create_dataset('topology', data=topology_norm, compression='gzip')
            
            # Copy non-shared tensors if they exist
            for key in ['edge_index', 'node_pos', 'k', 'edge_attr_r']:
                if f'{sample_id}/{key}' in f_in and key not in f_out.get('shared', {}):
                    data = f_in[f'{sample_id}/{key}'][:]
                    if key == 'edge_index':
                        # Don't normalize
                        sample_grp.create_dataset(key, data=data, compression='gzip')
                    elif key == 'node_pos':
                        # Normalize x, y, z
                        data_norm = np.zeros_like(data, dtype=np.float32)
                        data_norm[:, 0] = normalize_value(data[:, 0], scalers['topology']['x']['min'], scalers['topology']['x']['max'])
                        data_norm[:, 1] = normalize_value(data[:, 1], scalers['topology']['y']['min'], scalers['topology']['y']['max'])
                        data_norm[:, 2] = normalize_value(data[:, 2], scalers['topology']['z']['min'], scalers['topology']['z']['max'])
                        sample_grp.create_dataset(key, data=data_norm, compression='gzip')
                    elif key == 'k':
                        data_norm = normalize_value(data, scalers['node']['k']['min'], scalers['node']['k']['max'])
                        sample_grp.create_dataset(key, data=data_norm, compression='gzip')
                    elif key == 'edge_attr_r':
                        data_norm = normalize_value(data, scalers['edge']['r']['min'], scalers['edge']['r']['max'])
                        sample_grp.create_dataset(key, data=data_norm, compression='gzip')
            
            # Optional q, s
            if include_q_s:
                for feature in ['q', 's']:
                    if f'{sample_id}/{feature}' in f_in and feature in scalers['node']:
                        data = f_in[f'{sample_id}/{feature}'][:]
                        data_norm = normalize_value(data, scalers['node'][feature]['min'], scalers['node'][feature]['max'])
                        sample_grp.create_dataset(feature, data=data_norm, compression='gzip')
        
        # 5. Copy split group if exists
        if 'split' in f_in:
            split_grp = f_out.create_group('split')
            for key in f_in['split'].keys():
                split_grp.create_dataset(key, data=f_in[f'split/{key}'][:])
            for attr_key, attr_val in f_in['split'].attrs.items():
                split_grp.attrs[attr_key] = attr_val


def print_summary(input_path: str, output_path: str, scalers: Dict[str, Any], split: Dict[str, Any]) -> None:
    """Print normalization summary and sanity checks.
    
    Args:
        input_path: Path to original dataset
        output_path: Path to normalized dataset
        scalers: Dictionary of scalers used
        split: Split dictionary
    """
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    
    # Print which arrays were normalized
    print("\nNormalized arrays:")
    print("  - topology: t, x, y, z")
    print("  - node features: k" + (", q, s" if 'q' in scalers.get('node', {}) else ""))
    print("  - edge features: r")
    print("  - globals: I, T")
    print("  - target: temperature")
    
    # Print scalers
    print("\nScalers (fitted on training data only):")
    for category, features in scalers.items():
        if features and category != 'meta':
            print(f"\n  {category}:")
            for feature, stats in features.items():
                print(f"    {feature}: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # Sanity checks on training data
    print("\nSanity checks on normalized training data:")
    train_ids = split['train_ids']
    train_time_idx = split['train_time_idx']
    
    with h5py.File(output_path, 'r') as f:
        # Check temperature range on train samples
        temp_values = []
        for train_id in train_ids:
            if f'{train_id}/temperature' in f:
                temp = f[f'{train_id}/temperature'][:]
                temp_values.append(temp[train_time_idx].flatten())
        
        if temp_values:
            all_temps = np.concatenate(temp_values)
            print(f"  Temperature (train): min={np.min(all_temps):.6f}, max={np.max(all_temps):.6f}")
        
        # Check node_pos if shared
        if 'shared/node_pos' in f:
            node_pos = f['shared/node_pos'][:]
            print(f"  Node positions x: min={np.min(node_pos[:, 0]):.6f}, max={np.max(node_pos[:, 0]):.6f}")
            print(f"  Node positions y: min={np.min(node_pos[:, 1]):.6f}, max={np.max(node_pos[:, 1]):.6f}")
            print(f"  Node positions z: min={np.min(node_pos[:, 2]):.6f}, max={np.max(node_pos[:, 2]):.6f}")
        
        # Check edge_index unchanged
        if 'shared/edge_index' in f:
            with h5py.File(input_path, 'r') as f_orig:
                orig_ei = f_orig['shared/edge_index'][:]
                norm_ei = f['shared/edge_index'][:]
                print(f"  Edge index unchanged: {np.array_equal(orig_ei, norm_ei)}")
                print(f"  Number of edges: {orig_ei.shape[1]}")
    
    print("="*60 + "\n")


def main():
    """Main normalization pipeline."""
    parser = argparse.ArgumentParser(description='Normalize GIS dataset using train-only statistics')
    parser.add_argument('--in', dest='input', default='processed_data/full_dataset.hdf5',
                        help='Input HDF5 file path')
    parser.add_argument('--out', default='processed_data/normalized_full_dataset.hdf5',
                        help='Output normalized HDF5 file path')
    parser.add_argument('--split', default='processed_data/train_val_test_split.json',
                        help='Train/val/test split JSON file')
    parser.add_argument('--report', default='processed_data/processing_report.json',
                        help='Processing report JSON file')
    parser.add_argument('--scalers', default='processed_data/scalers.json',
                        help='Output scalers JSON file')
    parser.add_argument('--include-q-s', action='store_true',
                        help='Include q and s features in normalization')
    
    args = parser.parse_args()
    
    print(f"Loading split from {args.split}...")
    split = load_split(args.split)
    
    print(f"Computing training statistics from {args.input}...")
    stats = compute_train_stats(args.input, split, args.include_q_s)
    
    # Prepare scalers dictionary with metadata
    scalers = stats.copy()
    scalers['meta'] = {
        'fit_on': 'train_only',
        'train_ids': split['train_ids'],
        'train_time_idx_range': [min(split['train_time_idx']), max(split['train_time_idx'])]
    }
    
    # Save scalers
    print(f"Saving scalers to {args.scalers}...")
    with open(args.scalers, 'w') as f:
        json.dump(scalers, f, indent=2)
    
    # Normalize dataset
    print(f"Normalizing dataset and saving to {args.out}...")
    normalize_dataset(args.input, args.out, scalers, split, args.include_q_s)
    
    # Print summary
    print_summary(args.input, args.out, scalers, split)
    
    print("Normalization complete!")


if __name__ == "__main__":
    main()
