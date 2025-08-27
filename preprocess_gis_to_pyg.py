"""
Preprocess GIS HDF5 files to PyTorch Geometric-ready dataset.
Filters out node_type==2, deduplicates shared tensors, and creates consolidated HDF5.
"""

import h5py
import numpy as np
import json
import os
import re
import glob
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import sys

# Import train/val/test split functions
from train_val_test_split import generate_train_val_test_split, print_split_summary


@dataclass
class ProcessingStats:
    """Statistics for a single processed file."""
    filename: str
    original_num_nodes: int
    original_num_edges: int
    filtered_num_nodes: int
    filtered_num_edges: int
    nodes_removed: int
    edges_removed: int
    current_I: float
    temperature_T: float


@dataclass
class SharedTensorDecisions:
    """Decisions about which tensors can be shared across samples."""
    edge_index_shared: bool
    node_pos_shared: bool
    k_shared: bool
    edge_attr_r_shared: bool
    reasons: Dict[str, str]


def parse_filename(filename: str) -> Tuple[float, float]:
    """Extract I and T parameters from filename using regex.
    
    Args:
        filename: HDF5 filename
        
    Returns:
        Tuple of (I, T) values
        
    Raises:
        ValueError: If pattern doesn't match
    """
    pattern = r"I=(?P<I>[-+]?\d+(?:\.\d+)?)_T=(?P<T>[-+]?\d+(?:\.\d+)?)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Could not parse I and T from filename: {filename}")
    
    I = float(match.group('I'))
    T = float(match.group('T'))
    return I, T


def load_file(filepath: str) -> Dict[str, np.ndarray]:
    """Load all datasets from an HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary of array name -> numpy array
    """
    arrays = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arrays[key] = f[key][()]
    return arrays


def filter_nodes_and_edges(
    raw_data: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Filter out node_type==2 and all derived structures.
    
    Args:
        raw_data: Dictionary of raw arrays from HDF5
        
    Returns:
        Tuple of:
        - Filtered arrays dictionary
        - old_to_new mapping array
        - Metadata dictionary
    """
    # Create node mask (keep only types 0 and 1)
    node_types = raw_data['node_types']
    keep_nodes = np.isin(node_types, [0, 1])
    
    # Create old to new node index mapping
    old_to_new = np.full(len(node_types), -1, dtype=np.int64)
    new_indices = np.arange(np.sum(keep_nodes))
    old_to_new[keep_nodes] = new_indices
    
    # Filter node arrays
    filtered = {}
    node_arrays = ['e', 'k', 'node_components', 'node_pos', 'node_types', 
                   'node_volumes', 'q', 's']
    
    for arr_name in node_arrays:
        if arr_name in raw_data:
            filtered[arr_name] = raw_data[arr_name][keep_nodes]
    
    # Filter temperature array (shape: 121, N, 1)
    temperature = raw_data['temperature']
    filtered['temperature'] = temperature[:, keep_nodes, :]
    
    # Filter edges
    edge_src = raw_data['edge_src']
    edge_dst = raw_data['edge_dst']
    
    # Keep only edges where both nodes are kept
    keep_edges = keep_nodes[edge_src] & keep_nodes[edge_dst]
    
    # Remap edge indices
    edge_src_new = old_to_new[edge_src[keep_edges]]
    edge_dst_new = old_to_new[edge_dst[keep_edges]]
    
    # Create edge_index for PyG (2, E)
    filtered['edge_index'] = np.stack([edge_src_new, edge_dst_new], axis=0).astype(np.int64)
    
    # Filter edge attributes
    filtered['r'] = raw_data['r'][keep_edges]
    
    # Metadata - ensure all values are Python native types
    metadata = {
        'original_num_nodes': int(len(node_types)),
        'original_num_edges': int(len(edge_src)),
        'filtered_num_nodes': int(np.sum(keep_nodes)),
        'filtered_num_edges': int(np.sum(keep_edges)),
        'nodes_removed': int(len(node_types) - np.sum(keep_nodes)),
        'edges_removed': int(len(edge_src) - np.sum(keep_edges))
    }
    
    return filtered, old_to_new, metadata


def check_arrays_equal(arrays: List[np.ndarray], rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """Check if all arrays in list are equal using np.allclose.
    
    Args:
        arrays: List of numpy arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if all arrays are equal
    """
    if len(arrays) < 2:
        return True
    
    reference = arrays[0]
    for arr in arrays[1:]:
        if arr.shape != reference.shape:
            return False
        if not np.allclose(reference, arr, rtol=rtol, atol=atol):
            return False
    return True


def build_topology(t_len: int, node_pos: np.ndarray, start_t: int = 1) -> np.ndarray:
    """Build topology array [t, x, y, z] for each timestep and node.
    
    Args:
        t_len: Number of timesteps (120)
        node_pos: Node positions (N, 3)
        start_t: Starting timestep index (1 since we drop t=0)
        
    Returns:
        Array of shape (t_len, N, 4) with [t, x, y, z]
    """
    num_nodes = node_pos.shape[0]
    
    # Create time indices
    t_values = np.arange(start_t, start_t + t_len, dtype=np.float32).reshape(-1, 1)
    
    # Create topology array
    topology = np.zeros((t_len, num_nodes, 4), dtype=np.float32)
    
    # Fill time values (broadcast to all nodes)
    topology[:, :, 0] = t_values
    
    # Fill spatial coordinates (broadcast to all timesteps)
    topology[:, :, 1:4] = node_pos[np.newaxis, :, :]
    
    return topology


def check_and_record_shared(
    all_filtered: List[Dict[str, np.ndarray]]
) -> Tuple[SharedTensorDecisions, Dict[str, np.ndarray]]:
    """Check which tensors can be shared across samples.
    
    Args:
        all_filtered: List of filtered data dictionaries for each sample
        
    Returns:
        Tuple of (decisions, shared_tensors)
    """
    # Extract arrays for comparison
    edge_indices = [d['edge_index'] for d in all_filtered]
    node_positions = [d['node_pos'] for d in all_filtered]
    k_values = [d['k'] for d in all_filtered]
    r_values = [d['r'] for d in all_filtered]
    
    # Check equality
    edge_index_equal = check_arrays_equal(edge_indices)
    node_pos_equal = check_arrays_equal(node_positions)
    k_equal = check_arrays_equal(k_values)
    r_equal = check_arrays_equal(r_values)
    
    # Create decisions object
    decisions = SharedTensorDecisions(
        edge_index_shared=edge_index_equal,
        node_pos_shared=node_pos_equal,
        k_shared=k_equal,
        edge_attr_r_shared=r_equal,
        reasons={
            'edge_index': 'identical' if edge_index_equal else 'differs across samples',
            'node_pos': 'identical' if node_pos_equal else 'differs across samples',
            'k': 'identical' if k_equal else 'differs across samples',
            'edge_attr_r': 'identical' if r_equal else 'differs across samples'
        }
    )
    
    # Collect shared tensors
    shared_tensors = {}
    if edge_index_equal:
        shared_tensors['edge_index'] = edge_indices[0].astype(np.int64)
    if node_pos_equal:
        shared_tensors['node_pos'] = node_positions[0].astype(np.float32)
    if k_equal:
        shared_tensors['k'] = k_values[0].astype(np.float32)
    if r_equal:
        shared_tensors['edge_attr_r'] = r_values[0].astype(np.float32)
    
    return decisions, shared_tensors


def write_full_hdf5(
    output_path: str,
    shared_tensors: Dict[str, np.ndarray],
    per_sample_data: List[Dict[str, np.ndarray]],
    metadata: Dict[str, Any],
    decisions: SharedTensorDecisions,
    I_T_array: np.ndarray
) -> None:
    """Write consolidated HDF5 file with shared and per-sample data.
    
    Args:
        output_path: Path for output HDF5 file
        shared_tensors: Dictionary of shared arrays
        per_sample_data: List of per-sample data dictionaries
        metadata: Global metadata
        decisions: Sharing decisions
        I_T_array: Array of [I, T] values for each sample
    """
    with h5py.File(output_path, 'w') as f:
        # Write metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['num_samples'] = len(per_sample_data)
        meta_grp.attrs['original_num_nodes'] = metadata['original_num_nodes']
        meta_grp.attrs['original_num_edges'] = metadata['original_num_edges']
        meta_grp.attrs['filtered_num_nodes'] = metadata['filtered_num_nodes']
        meta_grp.attrs['filtered_num_edges'] = metadata['filtered_num_edges']
        meta_grp.attrs['dropped_first_timestep'] = True
        meta_grp.attrs['equality_checks'] = json.dumps(asdict(decisions))
        
        # Write shared tensors
        if shared_tensors:
            shared_grp = f.create_group('shared')
            for name, tensor in shared_tensors.items():
                shared_grp.create_dataset(name, data=tensor, compression='gzip')
        
        # Write global I_T array
        f.create_dataset('I_T', data=I_T_array, compression='gzip')
        
        # Write per-sample data
        for idx, sample_data in enumerate(per_sample_data):
            sample_grp = f.create_group(f'S{idx:03d}')
            
            # Always store temperature and topology per sample
            sample_grp.create_dataset('temperature', data=sample_data['temperature'], 
                                    compression='gzip')
            sample_grp.create_dataset('topology', data=sample_data['topology'], 
                                    compression='gzip')
            
            # Store non-shared tensors
            if not decisions.edge_index_shared:
                sample_grp.create_dataset('edge_index', data=sample_data['edge_index'], 
                                        compression='gzip')
            if not decisions.node_pos_shared:
                sample_grp.create_dataset('node_pos', data=sample_data['node_pos'], 
                                        compression='gzip')
            if not decisions.k_shared:
                sample_grp.create_dataset('k', data=sample_data['k'], 
                                        compression='gzip')
            if not decisions.edge_attr_r_shared:
                sample_grp.create_dataset('edge_attr_r', data=sample_data['r'], 
                                        compression='gzip')
        
        # Add suggested split scaffold
        f.attrs['suggested_split'] = json.dumps({
            "train_ids": [],
            "val_ids": [],
            "test_ids": []
        })


def generate_report(
    all_stats: List[ProcessingStats],
    decisions: SharedTensorDecisions,
    shared_tensors: Dict[str, np.ndarray],
    per_sample_data: List[Dict[str, np.ndarray]],
    I_T_array: np.ndarray
) -> Dict[str, Any]:
    """Generate comprehensive processing report.
    
    Args:
        all_stats: List of processing statistics per file
        decisions: Sharing decisions
        shared_tensors: Dictionary of shared tensors
        per_sample_data: List of per-sample data
        I_T_array: Array of [I, T] values
        
    Returns:
        Report dictionary
    """
    # Collect temperature statistics
    all_temps = []
    for sample in per_sample_data:
        all_temps.append(sample['temperature'].flatten())
    all_temps = np.concatenate(all_temps)
    
    # Edge index validation
    edge_validation = {}
    if 'edge_index' in shared_tensors:
        ei = shared_tensors['edge_index']
        num_nodes = shared_tensors['node_pos'].shape[0] if 'node_pos' in shared_tensors else per_sample_data[0]['node_pos'].shape[0]
        edge_validation = {
            'bounds_valid': bool(np.all((ei >= 0) & (ei < num_nodes))),
            'self_loops': int(np.sum(ei[0] == ei[1])),
            'num_edges': int(ei.shape[1])  # Convert to int
        }
    
    # Get node positions (from shared or first sample)
    if 'node_pos' in shared_tensors:
        node_pos = shared_tensors['node_pos']
    else:
        node_pos = per_sample_data[0]['node_pos']
    
    # Get k values (from shared or first sample)
    if 'k' in shared_tensors:
        k_values = shared_tensors['k']
    else:
        k_values = per_sample_data[0]['k']
    
    # Get r values (from shared or first sample)
    if 'edge_attr_r' in shared_tensors:
        r_values = shared_tensors['edge_attr_r']
    else:
        r_values = per_sample_data[0]['r']
    
    # Calculate statistics for each coordinate
    x_coords = node_pos[:, 0]
    y_coords = node_pos[:, 1]
    z_coords = node_pos[:, 2]
    
    # Time range (we use timesteps 1-120 after dropping first)
    t_values = np.arange(1, 121, dtype=np.float32)
    
    # Calculate comprehensive parameter ranges with statistics
    parameter_ranges = {
        'I': {
            'min': float(np.min(I_T_array[:, 0])),
            'max': float(np.max(I_T_array[:, 0])),
            'mean': float(np.mean(I_T_array[:, 0])),
            'std': float(np.std(I_T_array[:, 0]))
        },
        'T': {
            'min': float(np.min(I_T_array[:, 1])),
            'max': float(np.max(I_T_array[:, 1])),
            'mean': float(np.mean(I_T_array[:, 1])),
            'std': float(np.std(I_T_array[:, 1]))
        },
        't': {
            'min': float(np.min(t_values)),
            'max': float(np.max(t_values)),
            'mean': float(np.mean(t_values)),
            'std': float(np.std(t_values))
        },
        'x': {
            'min': float(np.min(x_coords)),
            'max': float(np.max(x_coords)),
            'mean': float(np.mean(x_coords)),
            'std': float(np.std(x_coords))
        },
        'y': {
            'min': float(np.min(y_coords)),
            'max': float(np.max(y_coords)),
            'mean': float(np.mean(y_coords)),
            'std': float(np.std(y_coords))
        },
        'z': {
            'min': float(np.min(z_coords)),
            'max': float(np.max(z_coords)),
            'mean': float(np.mean(z_coords)),
            'std': float(np.std(z_coords))
        },
        'k': {
            'min': float(np.min(k_values)),
            'max': float(np.max(k_values)),
            'mean': float(np.mean(k_values)),
            'std': float(np.std(k_values))
        },
        'r': {
            'min': float(np.min(r_values)),
            'max': float(np.max(r_values)),
            'mean': float(np.mean(r_values)),
            'std': float(np.std(r_values))
        }
    }
    
    # Get first sample for additional shape information
    first_sample = per_sample_data[0]
    
    # Build per_sample shapes based on what's actually stored per-sample
    per_sample_shapes = {
        'temperature': [int(x) for x in first_sample['temperature'].shape],
        'topology': [int(x) for x in first_sample['topology'].shape]
    }
    
    # Add non-shared tensors to per_sample shapes
    if not decisions.edge_index_shared:
        per_sample_shapes['edge_index'] = [int(x) for x in first_sample['edge_index'].shape]
    if not decisions.node_pos_shared:
        per_sample_shapes['node_pos'] = [int(x) for x in first_sample['node_pos'].shape]
    if not decisions.k_shared:
        per_sample_shapes['k'] = [int(x) for x in first_sample['k'].shape]
    if not decisions.edge_attr_r_shared:
        per_sample_shapes['edge_attr_r'] = [int(x) for x in first_sample['r'].shape]
    
    # Comprehensive shapes dictionary including all processed variables
    all_shapes = {
        # Shared tensors
        'shared': {
            name: [int(x) for x in tensor.shape]
            for name, tensor in shared_tensors.items()
        },
        # Per-sample tensors (what's actually stored per-sample)
        'per_sample': per_sample_shapes,
        # Global tensors
        'global': {
            'I_T': [int(x) for x in I_T_array.shape]
        },
        # All processed variables (including those that were filtered)
        'all_variables': {
            'edge_index': [int(x) for x in (shared_tensors.get('edge_index', first_sample['edge_index'])).shape],
            'node_pos': [int(x) for x in node_pos.shape],
            'k': [int(x) for x in k_values.shape],
            'edge_attr_r': [int(x) for x in r_values.shape],
            'temperature': [int(x) for x in first_sample['temperature'].shape],
            'topology': [int(x) for x in first_sample['topology'].shape],
            'I_T': [int(x) for x in I_T_array.shape],
            # Additional arrays from the first sample
            'e': [int(x) for x in first_sample['e'].shape] if 'e' in first_sample else None,
            'node_components': [int(x) for x in first_sample['node_components'].shape] if 'node_components' in first_sample else None,
            'node_types': [int(x) for x in first_sample['node_types'].shape] if 'node_types' in first_sample else None,
            'node_volumes': [int(x) for x in first_sample['node_volumes'].shape] if 'node_volumes' in first_sample else None,
            'q': [int(x) for x in first_sample['q'].shape] if 'q' in first_sample else None,
            's': [int(x) for x in first_sample['s'].shape] if 's' in first_sample else None
        }
    }
    
    # Remove None values from all_variables
    all_shapes['all_variables'] = {k: v for k, v in all_shapes['all_variables'].items() if v is not None}
    
    report = {
        'processing_summary': {
            'num_files_processed': len(all_stats),
            'total_nodes_removed': sum(s.nodes_removed for s in all_stats),
            'total_edges_removed': sum(s.edges_removed for s in all_stats),
            'consistent_graph_structure': all(
                s.filtered_num_nodes == all_stats[0].filtered_num_nodes 
                for s in all_stats
            )
        },
        'shapes': all_shapes,
        'deduplication_decisions': asdict(decisions),
        'temperature_stats': {
            'min': float(np.min(all_temps)),
            'max': float(np.max(all_temps)),
            'mean': float(np.mean(all_temps)),
            'std': float(np.std(all_temps))
        },
        'parameter_ranges': parameter_ranges,
        'sanity_checks': {
            'edge_index_validation': edge_validation,
            'temperature_time_length': int(per_sample_data[0]['temperature'].shape[0]),
            'expected_time_length': 120,
            'time_length_correct': per_sample_data[0]['temperature'].shape[0] == 120
        },
        'file_statistics': [asdict(s) for s in all_stats]
    }
    
    return report


def print_summary_table(report: Dict[str, Any]) -> None:
    """Print a concise summary table to console."""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    
    ps = report['processing_summary']
    print(f"Files processed: {ps['num_files_processed']}")
    print(f"Consistent graph structure: {ps['consistent_graph_structure']}")
    print(f"Total nodes removed: {ps['total_nodes_removed']}")
    print(f"Total edges removed: {ps['total_edges_removed']}")
    
    print("\nSHARED TENSOR SHAPES:")
    for name, shape in report['shapes']['shared'].items():
        print(f"  {name}: {shape}")
    
    print("\nPER-SAMPLE TENSOR SHAPES:")
    for name, shape in report['shapes']['per_sample'].items():
        print(f"  {name}: {shape}")
    
    print("\nDEDUPLICATION DECISIONS:")
    dd = report['deduplication_decisions']
    print(f"  edge_index: {'shared' if dd['edge_index_shared'] else 'per-sample'}")
    print(f"  node_pos: {'shared' if dd['node_pos_shared'] else 'per-sample'}")
    print(f"  k: {'shared' if dd['k_shared'] else 'per-sample'}")
    print(f"  edge_attr_r: {'shared' if dd['edge_attr_r_shared'] else 'per-sample'}")
    
    print("\nTEMPERATURE STATISTICS:")
    ts = report['temperature_stats']
    print(f"  Range: [{ts['min']:.2f}, {ts['max']:.2f}]")
    print(f"  Mean: {ts['mean']:.2f} ± {ts['std']:.2f}")
    
    print("\nPARAMETER RANGES:")
    pr = report['parameter_ranges']
    print(f"  I: {pr['I']}")
    print(f"  T: {pr['T']}")
    
    print("="*60 + "\n")


def update_hdf5_with_split(hdf5_path: str, split_dict: Dict[str, Any]) -> None:
    """Update the HDF5 file with train/val/test split information.
    
    Args:
        hdf5_path: Path to the HDF5 file
        split_dict: Dictionary containing split information
    """
    with h5py.File(hdf5_path, 'a') as f:
        # Update the suggested_split attribute with actual split
        f.attrs['suggested_split'] = json.dumps(split_dict)
        
        # Also create a dedicated split group for easier access
        if 'split' in f:
            del f['split']
        split_grp = f.create_group('split')
        
        # Store split information as datasets for easier programmatic access
        split_grp.create_dataset('train_ids', data=np.array(split_dict['train_ids'], dtype='S4'))
        split_grp.create_dataset('test_ids', data=np.array(split_dict['test_ids'], dtype='S4'))
        split_grp.create_dataset('val_time_idx', data=np.array(split_dict['val_time_idx'], dtype=np.int32))
        split_grp.create_dataset('train_time_idx', data=np.array(split_dict['train_time_idx'], dtype=np.int32))
        
        # Store metadata as attributes
        split_grp.attrs['protocol'] = split_dict['protocol']
        split_grp.attrs['seed'] = split_dict['seed']


def main():
    """Main preprocessing pipeline."""
    # Configuration
    input_dir = "data"
    output_dir = "processed_data"
    output_hdf5 = os.path.join(output_dir, "full_dataset.hdf5")
    report_json = os.path.join(output_dir, "processing_report.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    pattern = os.path.join(input_dir, "*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No HDF5 files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(files)} HDF5 files to process")
    
    # Process all files
    all_filtered = []
    all_stats = []
    I_T_list = []
    
    for idx, filepath in enumerate(files):
        print(f"\nProcessing {idx+1}/{len(files)}: {os.path.basename(filepath)}")
        
        # Parse filename
        filename = os.path.basename(filepath)
        I, T = parse_filename(filename)
        I_T_list.append([I, T])
        
        # Load and filter
        raw_data = load_file(filepath)
        filtered_data, old_to_new, metadata = filter_nodes_and_edges(raw_data)
        
        # Create processing stats
        stats = ProcessingStats(
            filename=filename,
            original_num_nodes=metadata['original_num_nodes'],
            original_num_edges=metadata['original_num_edges'],
            filtered_num_nodes=metadata['filtered_num_nodes'],
            filtered_num_edges=metadata['filtered_num_edges'],
            nodes_removed=metadata['nodes_removed'],
            edges_removed=metadata['edges_removed'],
            current_I=I,
            temperature_T=T
        )
        all_stats.append(stats)
        
        # Drop first timestep from temperature
        filtered_data['temperature'] = filtered_data['temperature'][1:, :, :].astype(np.float32)
        
        # Build topology
        filtered_data['topology'] = build_topology(
            t_len=120,
            node_pos=filtered_data['node_pos'],
            start_t=1
        )
        
        # Convert to appropriate dtypes
        filtered_data['edge_index'] = filtered_data['edge_index'].astype(np.int64)
        filtered_data['node_pos'] = filtered_data['node_pos'].astype(np.float32)
        filtered_data['k'] = filtered_data['k'].astype(np.float32)
        filtered_data['r'] = filtered_data['r'].astype(np.float32)
        
        all_filtered.append(filtered_data)
        
        print(f"  Kept {stats.filtered_num_nodes}/{stats.original_num_nodes} nodes")
        print(f"  Kept {stats.filtered_num_edges}/{stats.original_num_edges} edges")
    
    # Check for shared tensors
    print("\nChecking for shared tensors across samples...")
    decisions, shared_tensors = check_and_record_shared(all_filtered)
    
    # Prepare I_T array
    I_T_array = np.array(I_T_list, dtype=np.float32)
    
    # Use first sample's metadata for global values
    global_metadata = {
        'original_num_nodes': all_stats[0].original_num_nodes,
        'original_num_edges': all_stats[0].original_num_edges,
        'filtered_num_nodes': all_stats[0].filtered_num_nodes,
        'filtered_num_edges': all_stats[0].filtered_num_edges
    }
    
    # Write HDF5 file
    print(f"\nWriting consolidated dataset to {output_hdf5}...")
    write_full_hdf5(
        output_hdf5,
        shared_tensors,
        all_filtered,
        global_metadata,
        decisions,
        I_T_array
    )
    
    # Generate and save report
    report = generate_report(
        all_stats,
        decisions,
        shared_tensors,
        all_filtered,
        I_T_array
    )
    
    # Print summary
    print_summary_table(report)
    
    # Generate train/val/test split
    print("\nGenerating train/val/test split...")
    split_dict = generate_train_val_test_split(
        num_samples=len(all_filtered),
        num_timesteps=all_filtered[0]['temperature'].shape[0]  # Should be 120
    )
    
    # Add split to report
    report['train_val_test_split'] = split_dict
    
    # Update HDF5 file with split information
    update_hdf5_with_split(output_hdf5, split_dict)
    
    # Print split summary
    print_split_summary(split_dict)
    
    # Save JSON report (including split information)
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to {report_json}")
    
    # Also save split as separate JSON for convenience
    split_json = os.path.join(output_dir, "train_val_test_split.json")
    with open(split_json, 'w') as f:
        json.dump(split_dict, f, indent=2)
    print(f"Split configuration saved to {split_json}")
    
    print(f"\nProcessing complete! Dataset saved to {output_hdf5}")


if __name__ == "__main__":
    main()
