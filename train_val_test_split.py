"""
Write a Python script to generate a train/val/test split for full_dataset.hdf5.

Rules:
1. We have 10 samples: /samples/S000 … /samples/S009.
   - Choose 8 train, 2 test at the sample level (fix a random seed for reproducibility).
   - Example: train_ids = ["S000","S001","S002","S003","S004","S005","S006","S007"]
              test_ids  = ["S008","S009"]

2. For the 8 train samples:
   - There are 120 timesteps (after dropping the first).
   - Randomly pick 10% of timesteps (~12 indices) as validation indices.
   - The remaining 108 are train indices.
   - Use np.random.choice with a fixed seed (e.g. 42) to select val_time_idx.
   - Ensure no overlap between train_time_idx and val_time_idx.

3. Output:
   - Save a JSON file train_val_test_split.json with keys:
     {
       "protocol": "8_train_2_test_random_val_10pct",
       "seed": 42,
       "train_ids": [...],
       "test_ids": [...],
       "val_time_idx": [... list of 12 ints ...],
       "train_time_idx": [... list of 108 ints ...]
     }
   - Print summary to console: counts of train_ids, test_ids, val_time_idx, train_time_idx.

4. Implementation details:
   - Use Python's json and numpy modules.
   - Make sure val_time_idx and train_time_idx cover disjoint partitions of range(0,120).
   - Sort indices before saving to JSON for readability.
"""

import json
import numpy as np
import os


def generate_train_val_test_split(
    num_samples: int = 10,
    num_train: int = 8,
    num_test: int = 2,
    num_timesteps: int = 120,
    val_percentage: float = 0.1,
    seed: int = 42,
    output_dir: str = "processed_data"
) -> dict:
    """Generate train/val/test split for the dataset.
    
    Args:
        num_samples: Total number of samples (default: 10)
        num_train: Number of training samples (default: 8)
        num_test: Number of test samples (default: 2)
        num_timesteps: Total number of timesteps (default: 120)
        val_percentage: Percentage of timesteps for validation (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        output_dir: Directory to save the split file (default: "processed_data")
        
    Returns:
        Dictionary containing the split information
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate sample IDs
    sample_ids = [f"S{i:03d}" for i in range(num_samples)]
    
    # Shuffle and split samples into train and test
    shuffled_ids = np.random.permutation(sample_ids)
    train_ids = sorted(shuffled_ids[:num_train].tolist())
    test_ids = sorted(shuffled_ids[num_train:num_train + num_test].tolist())
    
    # Generate timestep indices
    all_timesteps = np.arange(num_timesteps)
    num_val = int(num_timesteps * val_percentage)
    
    # Randomly select validation timesteps
    val_time_idx = sorted([int(x) for x in np.random.choice(all_timesteps, size=num_val, replace=False)])
    
    # Train timesteps are the remaining ones
    train_time_idx = sorted([int(t) for t in all_timesteps if t not in val_time_idx])
    
    # Create split dictionary
    split_dict = {
        "protocol": "8_train_2_test_random_val_10pct",
        "seed": seed,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "val_time_idx": val_time_idx,
        "train_time_idx": train_time_idx
    }
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train_val_test_split.json")
    
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"Split saved to: {output_path}")
    
    return split_dict


def print_split_summary(split_dict: dict) -> None:
    """Print summary of the train/val/test split.
    
    Args:
        split_dict: Dictionary containing split information
    """
    print("\n" + "="*60)
    print("TRAIN/VAL/TEST SPLIT SUMMARY")
    print("="*60)
    print(f"Protocol: {split_dict['protocol']}")
    print(f"Random seed: {split_dict['seed']}")
    print(f"\nSample-level split:")
    print(f"  Train samples: {len(split_dict['train_ids'])} - {split_dict['train_ids']}")
    print(f"  Test samples: {len(split_dict['test_ids'])} - {split_dict['test_ids']}")
    print(f"\nTimestep-level split (for train samples):")
    print(f"  Train timesteps: {len(split_dict['train_time_idx'])} indices")
    print(f"  Val timesteps: {len(split_dict['val_time_idx'])} indices")
    print(f"\nValidation timestep indices: {split_dict['val_time_idx']}")
    
    # Verify coverage
    total_timesteps = len(split_dict['train_time_idx']) + len(split_dict['val_time_idx'])
    print(f"\nVerification:")
    print(f"  Total timesteps covered: {total_timesteps}")
    print(f"  Expected total: 120")
    print(f"  Coverage complete: {total_timesteps == 120}")
    
    # Check for overlap
    overlap = set(split_dict['train_time_idx']).intersection(set(split_dict['val_time_idx']))
    print(f"  No overlap between train/val timesteps: {len(overlap) == 0}")
    print("="*60 + "\n")


def main():
    """Main function to generate and save the train/val/test split."""
    # Generate the split
    split_dict = generate_train_val_test_split()
    
    # Print summary
    print_split_summary(split_dict)


if __name__ == "__main__":
    main()