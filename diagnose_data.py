"""
Diagnostic script to investigate data issues in the dataset.

Checks for missing data, NaN values, and other potential problems.
"""

import argparse
import h5py
import numpy as np
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from pathlib import Path

from sampler import GraphDeepONetDataset


def check_sample_data(h5_file: h5py.File, sample_id: str, prefix: str = "") -> Dict[str, Any]:
    """Check a single sample for data issues."""
    results = {
        'sample_id': sample_id,
        'exists': False,
        'arrays': {}
    }
    
    sample_path = f"{prefix}{sample_id}"
    
    if sample_path not in h5_file:
        return results
    
    results['exists'] = True
    sample_grp = h5_file[sample_path]
    
    # Check each array in the sample
    for key in sample_grp.keys():
        arr = sample_grp[key][:]
        
        array_info = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'has_nan': np.any(np.isnan(arr)),
            'nan_count': np.sum(np.isnan(arr)),
            'nan_fraction': np.sum(np.isnan(arr)) / arr.size if arr.size > 0 else 0,
            'min': float(np.nanmin(arr)) if not np.all(np.isnan(arr)) else None,
            'max': float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else None,
            'mean': float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else None,
            'all_nan': np.all(np.isnan(arr))
        }
        
        # Special handling for specific arrays
        if key == 'temperature':
            # Check each timestep
            timestep_info = []
            for t in range(arr.shape[0]):
                t_slice = arr[t]
                timestep_info.append({
                    't': t,
                    'has_nan': np.any(np.isnan(t_slice)),
                    'nan_fraction': np.sum(np.isnan(t_slice)) / t_slice.size,
                    'all_nan': np.all(np.isnan(t_slice))
                })
            array_info['timestep_info'] = timestep_info
            
            # Count completely NaN timesteps
            all_nan_timesteps = sum(1 for info in timestep_info if info['all_nan'])
            array_info['all_nan_timesteps'] = all_nan_timesteps
        
        results['arrays'][key] = array_info
    
    return results


def check_shared_data(h5_file: h5py.File) -> Dict[str, Any]:
    """Check shared data for issues."""
    results = {
        'has_shared': 'shared' in h5_file,
        'arrays': {}
    }
    
    if not results['has_shared']:
        return results
    
    shared_grp = h5_file['shared']
    
    for key in shared_grp.keys():
        arr = shared_grp[key][:]
        
        array_info = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'has_nan': np.any(np.isnan(arr)),
            'nan_count': np.sum(np.isnan(arr)),
            'nan_fraction': np.sum(np.isnan(arr)) / arr.size if arr.size > 0 else 0,
            'min': float(np.nanmin(arr)) if not np.all(np.isnan(arr)) else None,
            'max': float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else None,
            'mean': float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else None
        }
        
        results['arrays'][key] = array_info
    
    return results


def plot_sample_temperature_coverage(
    h5_file: h5py.File,
    sample_id: str,
    prefix: str,
    save_path: str
) -> None:
    """Plot temperature data coverage for a sample."""
    sample_path = f"{prefix}{sample_id}"
    
    if sample_path not in h5_file:
        print(f"Sample {sample_id} not found")
        return
    
    temperature = h5_file[f'{sample_path}/temperature'][:]
    T, N, _ = temperature.shape
    
    # Create coverage matrix (1 = valid, 0 = NaN)
    coverage = ~np.isnan(temperature[:, :, 0])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(coverage, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Valid Data (Green) / NaN (Red)')
    plt.xlabel('Node Index')
    plt.ylabel('Time Step')
    plt.title(f'{sample_id} - Temperature Data Coverage')
    
    # Add statistics
    valid_fraction = np.sum(coverage) / coverage.size
    plt.text(0.02, 0.98, f'Valid data: {valid_fraction*100:.1f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description='Diagnose dataset issues')
    parser.add_argument('--h5', default='processed_data/normalized_full_dataset.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--split', default='processed_data/train_val_test_split.json',
                        help='Path to split JSON')
    parser.add_argument('--sample', type=str, help='Specific sample to investigate')
    parser.add_argument('--plot', action='store_true', help='Generate coverage plots')
    parser.add_argument('--output', default='diagnostic_report.txt', help='Output report file')
    
    args = parser.parse_args()
    
    print(f"Diagnosing dataset: {args.h5}")
    
    # Load split info
    with open(args.split, 'r') as f:
        split_info = json.load(f)
    
    # Open HDF5 file
    with h5py.File(args.h5, 'r') as h5f:
        # Determine structure
        if 'samples' in h5f:
            prefix = 'samples/'
            sample_ids = list(h5f['samples'].keys())
        else:
            prefix = ''
            sample_ids = [key for key in h5f.keys() if key.startswith('S')]
        
        # Filter to specific sample if requested
        if args.sample:
            if args.sample in sample_ids:
                sample_ids = [args.sample]
            else:
                print(f"Error: Sample {args.sample} not found")
                return
        
        # Collect diagnostic information
        report_lines = []
        report_lines.append(f"Dataset Diagnostic Report")
        report_lines.append(f"========================")
        report_lines.append(f"File: {args.h5}")
        report_lines.append(f"Total samples: {len(sample_ids)}")
        report_lines.append(f"Structure prefix: '{prefix}'")
        report_lines.append("")
        
        # Check shared data
        shared_info = check_shared_data(h5f)
        report_lines.append("Shared Data:")
        report_lines.append(f"  Has shared: {shared_info['has_shared']}")
        if shared_info['has_shared']:
            for key, info in shared_info['arrays'].items():
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Shape: {info['shape']}")
                report_lines.append(f"    Has NaN: {info['has_nan']}")
                if info['has_nan']:
                    report_lines.append(f"    NaN fraction: {info['nan_fraction']:.2%}")
        report_lines.append("")
        
        # Check each sample
        problem_samples = []
        
        for sample_id in sample_ids:
            sample_info = check_sample_data(h5f, sample_id, prefix)
            
            if not sample_info['exists']:
                report_lines.append(f"\n{sample_id}: DOES NOT EXIST")
                problem_samples.append(sample_id)
                continue
            
            # Check for problems
            has_problem = False
            problem_details = []
            
            # Check temperature array specifically
            if 'temperature' in sample_info['arrays']:
                temp_info = sample_info['arrays']['temperature']
                if temp_info['all_nan']:
                    has_problem = True
                    problem_details.append("Temperature array is completely NaN")
                elif temp_info['all_nan_timesteps'] > 0:
                    has_problem = True
                    problem_details.append(f"{temp_info['all_nan_timesteps']} timesteps are completely NaN")
                elif temp_info['nan_fraction'] > 0.5:
                    has_problem = True
                    problem_details.append(f"Temperature is {temp_info['nan_fraction']:.1%} NaN")
            
            # Report on this sample
            if has_problem or args.sample:  # Always report if specific sample requested
                report_lines.append(f"\n{sample_id}:")
                
                # Report each array
                for key, info in sample_info['arrays'].items():
                    report_lines.append(f"  {key}:")
                    report_lines.append(f"    Shape: {info['shape']}")
                    report_lines.append(f"    Data type: {info['dtype']}")
                    report_lines.append(f"    Has NaN: {info['has_nan']}")
                    
                    if info['has_nan']:
                        report_lines.append(f"    NaN count: {info['nan_count']:,} / {np.prod(info['shape']):,}")
                        report_lines.append(f"    NaN fraction: {info['nan_fraction']:.2%}")
                    
                    if not info['all_nan'] and info['min'] is not None:
                        report_lines.append(f"    Range: [{info['min']:.6f}, {info['max']:.6f}]")
                        report_lines.append(f"    Mean: {info['mean']:.6f}")
                    
                    # Special reporting for temperature
                    if key == 'temperature' and 'timestep_info' in info:
                        if info['all_nan_timesteps'] > 0:
                            report_lines.append(f"    Timesteps with all NaN: {info['all_nan_timesteps']}")
                            # List which timesteps
                            all_nan_t = [t['t'] for t in info['timestep_info'] if t['all_nan']]
                            if len(all_nan_t) <= 10:
                                report_lines.append(f"      Timesteps: {all_nan_t}")
                            else:
                                report_lines.append(f"      First 10: {all_nan_t[:10]}")
                
                if problem_details:
                    report_lines.append(f"  PROBLEMS: {'; '.join(problem_details)}")
                    problem_samples.append(sample_id)
                
                # Generate coverage plot if requested
                if args.plot:
                    plot_path = f"{sample_id}_coverage.png"
                    plot_sample_temperature_coverage(h5f, sample_id, prefix, plot_path)
                    report_lines.append(f"  Coverage plot saved to: {plot_path}")
        
        # Summary
        report_lines.append(f"\n\nSummary:")
        report_lines.append(f"========")
        report_lines.append(f"Total samples: {len(sample_ids)}")
        report_lines.append(f"Problem samples: {len(problem_samples)}")
        if problem_samples:
            report_lines.append(f"Problem sample IDs: {problem_samples}")
        
        # Check against splits
        report_lines.append(f"\nSplit Assignment:")
        for sample_id in problem_samples:
            if sample_id in split_info['train_ids']:
                report_lines.append(f"  {sample_id}: TRAIN")
            elif sample_id in split_info['test_ids']:
                report_lines.append(f"  {sample_id}: TEST")
            else:
                report_lines.append(f"  {sample_id}: NOT IN SPLITS")
    
    # Write report
    report_content = '\n'.join(report_lines)
    
    # Print to console
    print(report_content)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
