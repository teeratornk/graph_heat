"""Batch analyze multiple HDF5 files and create a summary report."""
import os
import glob
import json
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from analyze_hdf5_metadata import analyze_hdf5_file, save_metadata


def extract_parameters_from_filename(filename: str) -> Dict[str, Union[int, float, str]]:
    """Extract parameters from filename pattern.
    
    Args:
        filename: Filename like 'graph_module_short_I=4500_T=40_theta30_knode.hdf5'
        
    Returns:
        Dictionary with extracted parameters
    """
    params = {}
    
    # Extract current (I)
    current_match = re.search(r'I=(\d+(?:\.\d+)?)', filename)
    if current_match:
        params['current_I'] = float(current_match.group(1))
    
    # Extract temperature (T)
    temp_match = re.search(r'T=(\d+(?:\.\d+)?)', filename)
    if temp_match:
        params['temperature_T'] = float(temp_match.group(1))
    
    # Extract theta if present
    theta_match = re.search(r'theta(\d+)', filename)
    if theta_match:
        params['theta'] = int(theta_match.group(1))
    
    # Extract any other pattern you might have
    # Add more patterns as needed
    
    return params


def compute_cross_file_statistics(
    all_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Compute statistics across all files for each dataset.
    
    Args:
        all_metadata: Dictionary with filename as key and metadata as value
        
    Returns:
        Dictionary with dataset names as keys and cross-file statistics as values
    """
    print("\n" + "="*60)
    print("Computing Cross-File Statistics")
    print("="*60)
    
    # Collect all dataset names across all files
    dataset_names = set()
    valid_files = {}
    
    for filename, file_metadata in all_metadata.items():
        if isinstance(file_metadata, dict) and "error" not in file_metadata:
            # Remove the file parameters from dataset names
            dataset_names.update(k for k in file_metadata.keys() if k != '_file_parameters')
            valid_files[filename] = file_metadata
    
    cross_file_stats = {}
    
    for dataset_name in sorted(dataset_names):
        print(f"\nAnalyzing '{dataset_name}' across files...")
        
        # Collect global statistics from each file
        file_stats = []
        shapes = []
        dtypes = set()
        
        # Collections for per-dimension statistics
        feature_stats_by_dim = {}
        
        # Special collections for specific datasets
        temperature_evolutions = []
        spatial_bounds_list = []
        node_type_distributions = []  # Add this for node_types
        
        for filename, file_metadata in valid_files.items():
            if dataset_name in file_metadata:
                ds_meta = file_metadata[dataset_name]
                
                # Skip if there was an error loading this dataset
                if "error" in ds_meta:
                    continue
                
                shapes.append(tuple(ds_meta["shape"]))
                dtypes.add(ds_meta["dtype"])
                
                # Collect global stats if available
                if "global_stats" in ds_meta:
                    stats = ds_meta["global_stats"].copy()
                    stats["filename"] = filename
                    file_stats.append(stats)
                
                # Collect per-dimension statistics if available
                if "features" in ds_meta:
                    for dim_key, dim_stats in ds_meta["features"].items():
                        if dim_key not in feature_stats_by_dim:
                            feature_stats_by_dim[dim_key] = []
                        dim_stats_copy = dim_stats.copy()
                        dim_stats_copy["filename"] = filename
                        feature_stats_by_dim[dim_key].append(dim_stats_copy)
                
                # Collect special data
                if "temperature_evolution" in ds_meta:
                    temp_evo = ds_meta["temperature_evolution"].copy()
                    temp_evo["filename"] = filename
                    temperature_evolutions.append(temp_evo)
                
                if "spatial_bounds" in ds_meta:
                    bounds = ds_meta["spatial_bounds"].copy()
                    bounds["filename"] = filename
                    spatial_bounds_list.append(bounds)
                
                # Collect node type distributions
                if "node_type_distribution" in ds_meta:
                    dist = ds_meta["node_type_distribution"].copy()
                    dist["filename"] = filename
                    node_type_distributions.append(dist)
        
        # Compute cross-file statistics
        cross_stats = {
            "found_in_files": len(file_stats),
            "total_files": len(valid_files),
            "unique_shapes": list(set(shapes)),
            "dtypes": list(dtypes)
        }
        
        if file_stats and all(isinstance(s.get("min"), (int, float)) for s in file_stats):
            # Extract numeric values
            all_mins = [s["min"] for s in file_stats if isinstance(s.get("min"), (int, float))]
            all_maxs = [s["max"] for s in file_stats if isinstance(s.get("max"), (int, float))]
            all_means = [s["mean"] for s in file_stats if isinstance(s.get("mean"), (int, float))]
            all_stds = [s["std"] for s in file_stats if isinstance(s.get("std"), (int, float))]
            
            # Cross-file statistics
            cross_stats["cross_file_stats"] = {
                "global_min": float(np.min(all_mins)),
                "global_max": float(np.max(all_maxs)),
                "mean_of_means": float(np.mean(all_means)),
                "std_of_means": float(np.std(all_means)),
                "mean_of_stds": float(np.mean(all_stds)),
                "range_of_means": [float(np.min(all_means)), float(np.max(all_means))],
                "range_of_stds": [float(np.min(all_stds)), float(np.max(all_stds))]
            }
            
            print(f"  Global range across all files: [{cross_stats['cross_file_stats']['global_min']:.3f}, "
                  f"{cross_stats['cross_file_stats']['global_max']:.3f}]")
            print(f"  Mean of file means: {cross_stats['cross_file_stats']['mean_of_means']:.3f}")
            print(f"  Std of file means: {cross_stats['cross_file_stats']['std_of_means']:.3f}")
        
        # Compute per-dimension cross-file statistics
        if feature_stats_by_dim:
            cross_stats["feature_cross_file_stats"] = {}
            
            for dim_key, dim_stats_list in feature_stats_by_dim.items():
                if dim_stats_list and all(isinstance(s.get("min"), (int, float)) for s in dim_stats_list):
                    dim_mins = [s["min"] for s in dim_stats_list if isinstance(s.get("min"), (int, float))]
                    dim_maxs = [s["max"] for s in dim_stats_list if isinstance(s.get("max"), (int, float))]
                    dim_means = [s["mean"] for s in dim_stats_list if isinstance(s.get("mean"), (int, float))]
                    dim_stds = [s["std"] for s in dim_stats_list if isinstance(s.get("std"), (int, float))]
                    
                    cross_stats["feature_cross_file_stats"][dim_key] = {
                        "global_min": float(np.min(dim_mins)),
                        "global_max": float(np.max(dim_maxs)),
                        "mean_of_means": float(np.mean(dim_means)),
                        "std_of_means": float(np.std(dim_means)),
                        "mean_of_stds": float(np.mean(dim_stds)),
                        "range_of_means": [float(np.min(dim_means)), float(np.max(dim_means))],
                        "range_of_stds": [float(np.min(dim_stds)), float(np.max(dim_stds))]
                    }
                    
                    # Extract dimension index for clearer printing
                    dim_idx = int(dim_key.split('_')[1]) if 'dim_' in dim_key else dim_key
                    print(f"  Dimension {dim_idx}: range [{cross_stats['feature_cross_file_stats'][dim_key]['global_min']:.3f}, "
                          f"{cross_stats['feature_cross_file_stats'][dim_key]['global_max']:.3f}], "
                          f"mean: {cross_stats['feature_cross_file_stats'][dim_key]['mean_of_means']:.3f}")
        
        # Temperature-specific cross-file analysis
        if temperature_evolutions:
            initial_means = [t["initial_mean"] for t in temperature_evolutions]
            final_means = [t["final_mean"] for t in temperature_evolutions]
            temp_rises = [t["temp_rise"] for t in temperature_evolutions]
            max_temps = [t["max_temp_reached"] for t in temperature_evolutions]
            
            cross_stats["temperature_cross_file"] = {
                "mean_initial_temp": float(np.mean(initial_means)),
                "mean_final_temp": float(np.mean(final_means)),
                "mean_temp_rise": float(np.mean(temp_rises)),
                "std_temp_rise": float(np.std(temp_rises)),
                "global_max_temp": float(np.max(max_temps)),
                "global_min_initial": float(np.min(initial_means)),
                "range_of_temp_rises": [float(np.min(temp_rises)), float(np.max(temp_rises))]
            }
            
            print(f"  Temperature rise range: {cross_stats['temperature_cross_file']['range_of_temp_rises']}")
            print(f"  Mean temperature rise: {cross_stats['temperature_cross_file']['mean_temp_rise']:.2f}°C")
        
        # Spatial bounds cross-file analysis
        if spatial_bounds_list:
            # Find overall bounding box
            x_mins = [b["x_min"] for b in spatial_bounds_list]
            x_maxs = [b["x_max"] for b in spatial_bounds_list]
            y_mins = [b["y_min"] for b in spatial_bounds_list]
            y_maxs = [b["y_max"] for b in spatial_bounds_list]
            z_mins = [b["z_min"] for b in spatial_bounds_list]
            z_maxs = [b["z_max"] for b in spatial_bounds_list]
            
            cross_stats["spatial_cross_file"] = {
                "global_bbox": {
                    "x_min": float(np.min(x_mins)),
                    "x_max": float(np.max(x_maxs)),
                    "y_min": float(np.min(y_mins)),
                    "y_max": float(np.max(y_maxs)),
                    "z_min": float(np.min(z_mins)),
                    "z_max": float(np.max(z_maxs))
                },
                "bbox_consistency": {
                    "x_range_std": float(np.std(x_maxs) + np.std(x_mins)),
                    "y_range_std": float(np.std(y_maxs) + np.std(y_mins)),
                    "z_range_std": float(np.std(z_maxs) + np.std(z_mins))
                }
            }
            
            print(f"  Spatial consistency (std of bounds): X={cross_stats['spatial_cross_file']['bbox_consistency']['x_range_std']:.6f}")
        
        # Node type distribution cross-file analysis
        if node_type_distributions:
            # Aggregate counts across all files
            all_type_counts = {}
            total_nodes = 0
            
            for dist in node_type_distributions:
                for node_type, count in dist['counts'].items():
                    if node_type not in all_type_counts:
                        all_type_counts[node_type] = 0
                    all_type_counts[node_type] += count
                    total_nodes += count
            
            # Calculate overall percentages
            overall_percentages = {k: (v / total_nodes * 100) for k, v in all_type_counts.items()}
            
            # Check consistency across files
            type_percentage_variations = {}
            for node_type in all_type_counts.keys():
                percentages = [dist['percentages'].get(node_type, 0) for dist in node_type_distributions]
                type_percentage_variations[node_type] = {
                    'mean_percentage': float(np.mean(percentages)),
                    'std_percentage': float(np.std(percentages)),
                    'min_percentage': float(np.min(percentages)),
                    'max_percentage': float(np.max(percentages))
                }
            
            cross_stats["node_type_cross_file"] = {
                "total_counts": all_type_counts,
                "overall_percentages": overall_percentages,
                "total_nodes_analyzed": total_nodes,
                "consistency_analysis": type_percentage_variations,
                "num_files_analyzed": len(node_type_distributions)
            }
            
            print(f"  Node type distribution across {len(node_type_distributions)} files:")
            for node_type, percentage in sorted(overall_percentages.items()):
                type_desc = {
                    0: "interior",
                    1: "boundary", 
                    2: "interface/special"
                }.get(node_type, f"type {node_type}")
                var = type_percentage_variations[node_type]
                print(f"    Type {node_type} ({type_desc}): {percentage:.2f}% (std: {var['std_percentage']:.3f}%)")
        
        cross_file_stats[dataset_name] = cross_stats
    
    return cross_file_stats


def batch_analyze_files(
    pattern: str = "data/*.hdf5",
    output_dir: str = "outputs/metadata"
) -> Dict[str, Dict[str, Any]]:
    """Analyze multiple HDF5 files matching a pattern.
    
    Args:
        pattern: Glob pattern to match HDF5 files
        output_dir: Directory to save individual metadata files
        
    Returns:
        Dictionary with filename as key and metadata as value
    """
    # Find all matching files
    files = glob.glob(pattern) + glob.glob(pattern.replace('.hdf5', '.h5'))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return {}
    
    print(f"Found {len(files)} files to analyze")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_metadata = {}
    
    for i, filepath in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(files)}")
        print(f"{'='*60}")
        
        try:
            # Analyze file
            metadata = analyze_hdf5_file(filepath)
            
            # Extract parameters from filename
            basename = os.path.basename(filepath)
            params = extract_parameters_from_filename(basename)
            
            # Add parameters to metadata
            metadata['_file_parameters'] = params
            
            # Save individual metadata
            name_without_ext = os.path.splitext(basename)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_metadata.json")
            save_metadata(metadata, output_path)
            
            # Store in combined dictionary
            all_metadata[basename] = metadata
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            all_metadata[basename] = {"error": str(e)}
    
    return all_metadata


def create_summary_report(
    all_metadata: Dict[str, Dict[str, Any]],
    cross_file_stats: Dict[str, Dict[str, Any]],
    output_path: str = "outputs/metadata/summary_report.json"
) -> None:
    """Create a summary report from all metadata including cross-file statistics.
    
    Args:
        all_metadata: Dictionary with all file metadata
        cross_file_stats: Dictionary with cross-file statistics
        output_path: Path to save summary report
    """
    # Extract file parameters for summary
    file_parameters = []
    for filename, metadata in all_metadata.items():
        if '_file_parameters' in metadata:
            params = metadata['_file_parameters'].copy()
            params['filename'] = filename
            file_parameters.append(params)
    
    summary = {
        "num_files": len(all_metadata),
        "files": list(all_metadata.keys()),
        "file_parameters": file_parameters,
        "dataset_summary": {},
        "cross_file_statistics": cross_file_stats,
        "temperature_statistics": [],
        "parameter_analysis": {}
    }
    
    # Analyze relationship between parameters and results
    if file_parameters:
        param_df = pd.DataFrame(file_parameters)
        
        # Group by parameters if we have temperature data
        temp_stats = []
        for filename, file_metadata in all_metadata.items():
            if "temperature" in file_metadata and "temperature_evolution" in file_metadata["temperature"]:
                temp_evo = file_metadata["temperature"]["temperature_evolution"].copy()
                temp_evo["filename"] = filename
                
                # Add file parameters
                params = next((p for p in file_parameters if p['filename'] == filename), {})
                temp_evo.update(params)
                temp_stats.append(temp_evo)
        
        if temp_stats:
            temp_df = pd.DataFrame(temp_stats)
            
            # Analyze correlations if we have numeric parameters
            if 'current_I' in temp_df.columns and 'temperature_T' in temp_df.columns:
                summary["parameter_analysis"] = {
                    "current_temperature_correlation": {
                        "current_vs_max_temp": float(temp_df[['current_I', 'max_temp_reached']].corr().iloc[0, 1]) if len(temp_df) > 1 else None,
                        "current_vs_temp_rise": float(temp_df[['current_I', 'temp_rise']].corr().iloc[0, 1]) if len(temp_df) > 1 else None,
                        "ambient_temp_vs_max_temp": float(temp_df[['temperature_T', 'max_temp_reached']].corr().iloc[0, 1]) if len(temp_df) > 1 else None,
                        "ambient_temp_vs_temp_rise": float(temp_df[['temperature_T', 'temp_rise']].corr().iloc[0, 1]) if len(temp_df) > 1 else None
                    },
                    "parameter_ranges": {
                        "current_I": [float(temp_df['current_I'].min()), float(temp_df['current_I'].max())] if 'current_I' in temp_df else None,
                        "temperature_T": [float(temp_df['temperature_T'].min()), float(temp_df['temperature_T'].max())] if 'temperature_T' in temp_df else None
                    }
                }
    
    # Collect statistics across all files
    dataset_names = set()
    for file_metadata in all_metadata.values():
        if isinstance(file_metadata, dict) and "error" not in file_metadata:
            # Remove the file parameters from dataset names
            dataset_names.update(k for k in file_metadata.keys() if k != '_file_parameters')
    
    # Summarize each dataset type
    for dataset_name in dataset_names:
        dataset_info = {
            "found_in_files": 0,
            "shapes": [],
            "dtypes": set()
        }
        
        for filename, file_metadata in all_metadata.items():
            if dataset_name in file_metadata:
                dataset_info["found_in_files"] += 1
                ds_meta = file_metadata[dataset_name]
                if "shape" in ds_meta:
                    dataset_info["shapes"].append(tuple(ds_meta["shape"]))
                if "dtype" in ds_meta:
                    dataset_info["dtypes"].add(ds_meta["dtype"])
        
        # Convert sets to lists for JSON serialization
        dataset_info["dtypes"] = list(dataset_info["dtypes"])
        dataset_info["unique_shapes"] = list(set(dataset_info["shapes"]))
        del dataset_info["shapes"]
        
        # Add cross-file stats if available
        if dataset_name in cross_file_stats:
            dataset_info["cross_file_analysis"] = cross_file_stats[dataset_name]
        
        summary["dataset_summary"][dataset_name] = dataset_info
    
    # Extract temperature evolution data if available
    for filename, file_metadata in all_metadata.items():
        if "temperature" in file_metadata and "temperature_evolution" in file_metadata["temperature"]:
            temp_evo = file_metadata["temperature"]["temperature_evolution"].copy()
            temp_evo["filename"] = filename
            
            # Add file parameters
            params = next((p for p in file_parameters if p['filename'] == filename), {})
            temp_evo.update(params)
            
            summary["temperature_statistics"].append(temp_evo)
    
    # Save summary report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to: {output_path}")
    
    # Create CSV summary for temperature data with parameters
    if summary["temperature_statistics"]:
        df = pd.DataFrame(summary["temperature_statistics"])
        csv_path = output_path.replace('.json', '_temperatures.csv')
        df.to_csv(csv_path, index=False)
        print(f"Temperature summary CSV saved to: {csv_path}")
    
    # Create parameter summary CSV
    if file_parameters:
        param_df = pd.DataFrame(file_parameters)
        param_csv_path = output_path.replace('.json', '_parameters.csv')
        param_df.to_csv(param_csv_path, index=False)
        print(f"Parameter summary CSV saved to: {param_csv_path}")
    
    # Create cross-file statistics CSV
    if cross_file_stats:
        cross_file_rows = []
        for dataset_name, stats in cross_file_stats.items():
            if "cross_file_stats" in stats:
                row = {"dataset": dataset_name}
                row.update(stats["cross_file_stats"])
                cross_file_rows.append(row)
        
        if cross_file_rows:
            df_cross = pd.DataFrame(cross_file_rows)
            csv_cross_path = output_path.replace('.json', '_cross_file_stats.csv')
            df_cross.to_csv(csv_cross_path, index=False)
            print(f"Cross-file statistics CSV saved to: {csv_cross_path}")


def main():
    """Main function for batch analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch analyze HDF5 files and create summary report with cross-file statistics'
    )
    parser.add_argument(
        '--pattern',
        default='data/*.hdf5',
        help='Glob pattern for HDF5 files (default: data/*.hdf5)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/metadata',
        help='Output directory for metadata files (default: outputs/metadata)'
    )
    
    args = parser.parse_args()
    
    # Analyze all files
    all_metadata = batch_analyze_files(args.pattern, args.output_dir)
    
    if all_metadata:
        # Compute cross-file statistics
        cross_file_stats = compute_cross_file_statistics(all_metadata)
        
        # Create summary report
        summary_path = os.path.join(args.output_dir, 'summary_report.json')
        create_summary_report(all_metadata, cross_file_stats, summary_path)
        
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Analyzed {len(all_metadata)} files")
        print(f"Results saved to: {args.output_dir}")
        
        # Print key cross-file findings
        print("\nKey Cross-File Findings:")
        for dataset_name, stats in cross_file_stats.items():
            if "cross_file_stats" in stats:
                cfs = stats["cross_file_stats"]
                print(f"\n{dataset_name}:")
                print(f"  Global range: [{cfs['global_min']:.3f}, {cfs['global_max']:.3f}]")
                print(f"  Mean across files: {cfs['mean_of_means']:.3f} ± {cfs['std_of_means']:.3f}")
        
        # Print parameter analysis
        print("\nParameter Analysis:")
        for filename, metadata in all_metadata.items():
            if '_file_parameters' in metadata:
                params = metadata['_file_parameters']
                print(f"\n{filename}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
    else:
        print("No files were analyzed.")


if __name__ == "__main__":
    main()