"""Analyze HDF5 file datasets and generate comprehensive metadata."""
import h5py
import numpy as np
import json
import os
import sys
import argparse
from typing import Dict, Any, List, Union, Optional


def compute_statistics(
    data: np.ndarray,
    axis: Optional[int] = None
) -> Dict[str, Union[float, int]]:
    """Compute statistics for a numpy array.
    
    Args:
        data: Numpy array to analyze
        axis: Axis along which to compute statistics
        
    Returns:
        Dictionary containing min, max, mean, std, num_nan, num_outliers
    """
    stats = {}
    
    # Handle non-numeric data
    if not np.issubdtype(data.dtype, np.number):
        return {
            'min': 'N/A',
            'max': 'N/A',
            'mean': 'N/A',
            'std': 'N/A',
            'num_nan': 0,
            'num_outliers': 0
        }
    
    # Compute basic statistics
    if axis is not None:
        stats['min'] = float(np.nanmin(data, axis=axis))
        stats['max'] = float(np.nanmax(data, axis=axis))
        stats['mean'] = float(np.nanmean(data, axis=axis))
        stats['std'] = float(np.nanstd(data, axis=axis))
    else:
        stats['min'] = float(np.nanmin(data))
        stats['max'] = float(np.nanmax(data))
        stats['mean'] = float(np.nanmean(data))
        stats['std'] = float(np.nanstd(data))
    
    # Count NaN values (only for float types)
    if np.issubdtype(data.dtype, np.floating):
        stats['num_nan'] = int(np.sum(np.isnan(data)))
    else:
        stats['num_nan'] = 0
    
    # Outlier detection (values more than 3 std from mean)
    if stats['std'] > 0:
        # Avoid NaN in outlier detection
        valid_data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            outliers = np.abs(valid_data - mean) > 3 * std
            stats['num_outliers'] = int(np.sum(outliers))
        else:
            stats['num_outliers'] = 0
    else:
        stats['num_outliers'] = 0
    
    return stats


def analyze_dataset(
    name: str,
    dataset: h5py.Dataset
) -> Dict[str, Any]:
    """Analyze a single HDF5 dataset.
    
    Args:
        name: Dataset name
        dataset: HDF5 dataset object
        
    Returns:
        Dictionary containing dataset metadata and statistics
    """
    print(f"\nAnalyzing dataset: {name}")
    
    # Get basic metadata
    metadata = {
        'shape': list(dataset.shape),
        'dtype': str(dataset.dtype),
        'size': int(np.prod(dataset.shape))
    }
    
    # Load data into memory (be careful with large datasets)
    try:
        data = dataset[()]
    except Exception as e:
        print(f"  Error loading dataset {name}: {e}")
        metadata['error'] = str(e)
        return metadata
    
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    
    # GLOBAL STATISTICS = statistics for ALL values in this dataset within THIS FILE
    global_stats = compute_statistics(data.flatten())
    metadata['global_stats'] = global_stats
    
    print(f"  Global statistics (entire dataset in this file):")
    print(f"    Min: {global_stats['min']:.6f}, Max: {global_stats['max']:.6f}")
    print(f"    Mean: {global_stats['mean']:.6f}, Std: {global_stats['std']:.6f}")
    print(f"    NaN count: {global_stats['num_nan']}, Outliers: {global_stats['num_outliers']}")
    
    # For multidimensional arrays, also compute statistics along last axis
    if len(data.shape) > 1:
        # Reshape to 2D: (all_other_dims, last_dim)
        reshaped = data.reshape(-1, data.shape[-1])
        features = {}
        
        for i in range(data.shape[-1]):
            feature_data = reshaped[:, i]
            stats = compute_statistics(feature_data)
            features[f'dim_{i}'] = stats
            
            print(f"  Dimension {i}:")
            print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"    NaN count: {stats['num_nan']}, Outliers: {stats['num_outliers']}")
        
        metadata['features'] = features
        
        # Add percentile information for global stats
        if np.issubdtype(data.dtype, np.number):
            percentiles = [0, 25, 50, 75, 100]
            global_percentiles = {}
            for p in percentiles:
                global_percentiles[f'p{p}'] = float(np.nanpercentile(data.flatten(), p))
            metadata['global_percentiles'] = global_percentiles
    else:
        # For 1D arrays, global stats are the only stats
        # Add percentile information
        if np.issubdtype(data.dtype, np.number):
            percentiles = [0, 25, 50, 75, 100]
            global_percentiles = {}
            for p in percentiles:
                global_percentiles[f'p{p}'] = float(np.nanpercentile(data, p))
            metadata['global_percentiles'] = global_percentiles
    
    # Add additional info for specific datasets
    if name == 'temperature' and len(data.shape) == 3:
        # Temperature evolution statistics
        initial_temp = data[0, :, :].flatten()
        final_temp = data[-1, :, :].flatten()
        metadata['temperature_evolution'] = {
            'initial_mean': float(np.mean(initial_temp)),
            'initial_std': float(np.std(initial_temp)),
            'final_mean': float(np.mean(final_temp)),
            'final_std': float(np.std(final_temp)),
            'max_temp_reached': float(np.max(data)),
            'min_temp_reached': float(np.min(data)),
            'temp_rise': float(np.mean(final_temp) - np.mean(initial_temp))
        }
        
        # Time evolution statistics
        time_mean_temps = np.mean(data.reshape(data.shape[0], -1), axis=1)
        metadata['time_evolution'] = {
            'mean_temps_over_time': time_mean_temps.tolist(),
            'steady_state_estimate': float(time_mean_temps[-1]),
            'max_rate_of_change': float(np.max(np.abs(np.diff(time_mean_temps))))
        }
    
    # Special handling for edge data
    if name in ['edge_src', 'edge_dst']:
        unique_nodes = np.unique(data)
        metadata['edge_info'] = {
            'num_unique_nodes': int(len(unique_nodes)),
            'min_node_id': int(np.min(unique_nodes)),
            'max_node_id': int(np.max(unique_nodes)),
            'num_edges': int(len(data))
        }
    
    # Node position statistics
    if name == 'node_pos' and data.shape[1] == 3:
        metadata['spatial_bounds'] = {
            'x_min': float(np.min(data[:, 0])),
            'x_max': float(np.max(data[:, 0])),
            'y_min': float(np.min(data[:, 1])),
            'y_max': float(np.max(data[:, 1])),
            'z_min': float(np.min(data[:, 2])),
            'z_max': float(np.max(data[:, 2])),
            'centroid': [float(np.mean(data[:, i])) for i in range(3)],
            'spatial_extent': [
                float(np.max(data[:, i]) - np.min(data[:, i])) 
                for i in range(3)
            ]
        }
    
    # Special handling for node_types - count categories
    if name == 'node_types' and np.issubdtype(data.dtype, np.integer):
        unique_types, counts = np.unique(data, return_counts=True)
        node_type_counts = {}
        for node_type, count in zip(unique_types, counts):
            node_type_counts[int(node_type)] = int(count)
        
        metadata['node_type_distribution'] = {
            'counts': node_type_counts,
            'percentages': {int(k): float(v / len(data) * 100) for k, v in node_type_counts.items()},
            'num_unique_types': len(unique_types)
        }
        
        print(f"  Node type distribution:")
        for node_type, count in sorted(node_type_counts.items()):
            percentage = count / len(data) * 100
            type_desc = {
                0: "interior node",
                1: "boundary node", 
                2: "interface/special material node"
            }.get(node_type, f"type {node_type}")
            print(f"    Type {node_type} ({type_desc}): {count} nodes ({percentage:.2f}%)")
    
    return metadata


def analyze_hdf5_file(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Analyze all datasets in an HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary containing metadata for all datasets
    """
    print(f"Analyzing HDF5 file: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / (1024**2):.2f} MB")
    
    results = {}
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nFound {len(f.keys())} datasets:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Analyze each dataset
        for name, dataset in f.items():
            if isinstance(dataset, h5py.Dataset):
                results[name] = analyze_dataset(name, dataset)
            elif isinstance(dataset, h5py.Group):
                print(f"\nSkipping group: {name}")
                # If needed, recursively analyze groups
    
    return results


def save_metadata(
    metadata: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """Save metadata to JSON file.
    
    Args:
        metadata: Dictionary containing all dataset metadata
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python native types
    def convert_types(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    clean_metadata = convert_types(metadata)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {output_path}")


def main():
    """Main function to analyze HDF5 file."""
    parser = argparse.ArgumentParser(
        description='Analyze HDF5 file datasets and generate metadata'
    )
    parser.add_argument(
        'filepath',
        help='Path to HDF5 file to analyze'
    )
    parser.add_argument(
        '-o', '--output',
        default='metadata.json',
        help='Output JSON file path (default: metadata.json)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    
    # Analyze the file
    try:
        metadata = analyze_hdf5_file(args.filepath)
        
        # Save results
        save_metadata(metadata, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total datasets analyzed: {len(metadata)}")
        
        for name, info in metadata.items():
            print(f"\n{name}:")
            print(f"  Shape: {info['shape']}")
            print(f"  Size: {info['size']:,} elements")
            
            if 'global_stats' in info:
                g_stats = info['global_stats']
                print(f"  Global range: [{g_stats['min']:.3f}, {g_stats['max']:.3f}]")
                print(f"  Global mean: {g_stats['mean']:.3f} ± {g_stats['std']:.3f}")
            
            if 'features' in info:
                print(f"  Features: {len(info['features'])}")
                # Print per-dimension statistics
                for dim_key, dim_stats in info['features'].items():
                    dim_idx = int(dim_key.split('_')[1]) if 'dim_' in dim_key else dim_key
                    print(f"    Dimension {dim_idx}: range [{dim_stats['min']:.3f}, {dim_stats['max']:.3f}], "
                          f"mean: {dim_stats['mean']:.3f} ± {dim_stats['std']:.3f}")
            
            if 'temperature_evolution' in info:
                evo = info['temperature_evolution']
                print(f"  Temperature range: {evo['min_temp_reached']:.2f} - {evo['max_temp_reached']:.2f}")
                print(f"  Temperature rise: {evo['temp_rise']:.2f}°C")
            
            if 'spatial_bounds' in info:
                bounds = info['spatial_bounds']
                print(f"  Spatial extent: X={bounds['spatial_extent'][0]:.3f}, "
                      f"Y={bounds['spatial_extent'][1]:.3f}, Z={bounds['spatial_extent'][2]:.3f}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()