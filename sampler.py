"""
Sampler utilities for Graph-DeepONet.

Provides CPU-friendly iterators for training, validation, and testing
with efficient batching and subsampling strategies.
"""

import h5py
import json
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal, Any
from pathlib import Path


@dataclass
class SamplerConfig:
    """Configuration for GraphDeepONet sampler."""
    h5_path: str
    split_json: str
    mode: Literal["train", "val", "test"]
    rng_seed: int = 42
    samples_per_step: int = 1
    t_subsample_frac: float = 0.2
    node_subsample_frac: float = 0.2
    target_pairs_cap: Optional[int] = None
    deterministic: bool = True
    use_shared_if_available: bool = True
    include_qs: bool = False
    prefer_cartesian_pairs: bool = True
    val_node_frac: float = 0.25
    test_chunk_nodes: Optional[int] = None


class GraphDeepONetDataset:
    """
    HDF5-backed dataset helper for Graph-DeepONet.
    
    Handles loading of shared/per-sample arrays with proper dtype conversion.
    """
    
    def __init__(
        self, 
        h5_path: str,
        include_qs: bool = False,
        use_shared_if_available: bool = True
    ):
        self.h5_path = h5_path
        self.include_qs = include_qs
        self.use_shared_if_available = use_shared_if_available
        
        # Open HDF5 file
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Get sample IDs - handle both structures
        if 'samples' in self.h5_file:
            # Structure with samples group
            self.sample_ids = sorted(list(self.h5_file['samples'].keys()))
            self.samples_prefix = 'samples/'
        else:
            # Structure with samples at root level (normalized dataset)
            self.sample_ids = sorted([key for key in self.h5_file.keys() if key.startswith('S')])
            self.samples_prefix = ''
        
        if not self.sample_ids:
            raise ValueError(f"No sample IDs found in {h5_path}")
            
        self.sample_id_to_idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        
        # Get dimensions from first sample
        first_sample_path = f'{self.samples_prefix}{self.sample_ids[0]}'
        first_sample = self.h5_file[first_sample_path]
        self.N = first_sample['temperature'].shape[1]  # nodes
        self.Tlen = first_sample['temperature'].shape[0]  # time steps (120)
        
        # Check for shared tensors
        self.has_shared = 'shared' in self.h5_file and use_shared_if_available
        if self.has_shared:
            self.E = self.h5_file['shared/edge_index'].shape[1]  # edges
        else:
            self.E = first_sample['edge_index'].shape[1]
    
    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
    
    def _to_tensor(self, arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert numpy array to torch tensor with specified dtype."""
        return torch.from_numpy(arr).to(dtype)
    
    def get_shared_edge_index(self) -> torch.LongTensor:
        """Get shared edge index or raise if not available."""
        if not self.has_shared or 'edge_index' not in self.h5_file['shared']:
            raise ValueError("Shared edge_index not available")
        return self._to_tensor(self.h5_file['shared/edge_index'][:], torch.long)
    
    def get_shared_edge_attr_r(self) -> torch.Tensor:
        """Get shared edge attributes or raise if not available."""
        if not self.has_shared or 'edge_attr_r' not in self.h5_file['shared']:
            raise ValueError("Shared edge_attr_r not available")
        return self._to_tensor(self.h5_file['shared/edge_attr_r'][:])
    
    def get_shared_k(self) -> torch.Tensor:
        """Get shared thermal conductivity or raise if not available."""
        if not self.has_shared or 'k' not in self.h5_file['shared']:
            raise ValueError("Shared k not available")
        return self._to_tensor(self.h5_file['shared/k'][:])
    
    def get_shared_node_pos(self) -> torch.Tensor:
        """Get shared node positions or raise if not available."""
        if not self.has_shared or 'node_pos' not in self.h5_file['shared']:
            raise ValueError("Shared node_pos not available")
        return self._to_tensor(self.h5_file['shared/node_pos'][:])
    
    def get_sample_edge_index(self, sample_id: str) -> torch.LongTensor:
        """Get edge index for specific sample."""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/edge_index'][:], torch.long)
    
    def get_sample_edge_attr_r(self, sample_id: str) -> torch.Tensor:
        """Get edge attributes for specific sample."""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/edge_attr_r'][:])
    
    def get_sample_k(self, sample_id: str) -> torch.Tensor:
        """Get thermal conductivity for specific sample."""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/k'][:])
    
    def get_sample_node_pos(self, sample_id: str) -> torch.Tensor:
        """Get node positions for specific sample."""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/node_pos'][:])
    
    def get_temperature(self, sample_id: str) -> torch.Tensor:
        """Get temperature array for sample. Shape: (120, N, 1)"""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/temperature'][:])
    
    def get_topology(self, sample_id: str) -> torch.Tensor:
        """Get topology (coordinates) array. Shape: (120, N, 4) with [t,x,y,z]"""
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/topology'][:])
    
    def get_globals(self, sample_id: str) -> torch.Tensor:
        """Get global parameters [I, T] for sample."""
        idx = self.sample_id_to_idx[sample_id]
        return self._to_tensor(self.h5_file['I_T'][idx])
    
    def get_q(self, sample_id: str) -> torch.Tensor:
        """Get heat source q for sample nodes."""
        if not self.include_qs:
            raise ValueError("Dataset not configured to include q/s")
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/q'][:])
    
    def get_s(self, sample_id: str) -> torch.Tensor:
        """Get heat capacity s for sample nodes."""
        if not self.include_qs:
            raise ValueError("Dataset not configured to include q/s")
        return self._to_tensor(self.h5_file[f'{self.samples_prefix}{sample_id}/s'][:])


class GraphDeepONetSampler:
    """
    Stateful iterator that yields batches for training/validation/testing.
    """
    
    def __init__(
        self,
        dataset: GraphDeepONetDataset,
        config: SamplerConfig,
        split_json_path: str
    ):
        self.dataset = dataset
        self.config = config
        
        # Load split info
        with open(split_json_path, 'r') as f:
            self.split_info = json.load(f)
        
        # Initialize RNGs
        if config.deterministic:
            self.rng_step = np.random.Generator(np.random.PCG64(config.rng_seed))
            self.rng_nodes = np.random.Generator(np.random.PCG64(config.rng_seed + 1))
            self.rng_pairs = np.random.Generator(np.random.PCG64(config.rng_seed + 2))
            self.rng_val_nodes = np.random.Generator(np.random.PCG64(config.rng_seed + 3))
        else:
            self.rng_step = np.random.default_rng()
            self.rng_nodes = np.random.default_rng()
            self.rng_pairs = np.random.default_rng()
            self.rng_val_nodes = np.random.default_rng()
        
        # Get sample IDs for mode
        self.mode_sample_ids = self._pick_sample_ids(config.mode)
        self.current_idx = 0
        
        # For test mode chunking
        self.test_node_offset = 0
    
    def __iter__(self):
        """Reset iterator state."""
        self.current_idx = 0
        self.test_node_offset = 0
        return self
    
    def __next__(self) -> List[Dict[str, torch.Tensor]]:
        """Get next batch."""
        if self.config.mode == "train":
            return self._next_train()
        elif self.config.mode == "val":
            return self._next_val()
        else:  # test
            return self._next_test()
    
    def _pick_sample_ids(self, mode: str) -> List[str]:
        """Get sample IDs for the given mode."""
        if mode in ["train", "val"]:
            return self.split_info["train_ids"]
        else:  # test
            return self.split_info["test_ids"]
    
    def _pick_times_train(self, train_time_idx: List[int]) -> np.ndarray:
        """Randomly select subset of training times."""
        n_times = len(train_time_idx)
        n_select = max(1, int(n_times * self.config.t_subsample_frac))
        return self.rng_step.choice(train_time_idx, size=n_select, replace=False)
    
    def _pick_nodes(self, N: int, frac: float, rng: np.random.Generator) -> np.ndarray:
        """Randomly select subset of nodes."""
        n_select = max(1, int(N * frac))
        return rng.choice(N, size=n_select, replace=False)
    
    def _build_pairs(
        self,
        times_idx: np.ndarray,
        nodes_idx: np.ndarray,
        prefer_cartesian: bool,
        target_cap: Optional[int],
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (time, node) index pairs."""
        if prefer_cartesian:
            # Cartesian product
            t_grid, n_grid = np.meshgrid(times_idx, nodes_idx, indexing='ij')
            t_flat = t_grid.flatten()
            n_flat = n_grid.flatten()
            
            # Apply cap if needed
            if target_cap and len(t_flat) > target_cap:
                indices = rng.choice(len(t_flat), size=target_cap, replace=False)
                t_flat = t_flat[indices]
                n_flat = n_flat[indices]
            
            return t_flat, n_flat
        else:
            # Random pairs
            n_pairs = len(times_idx) * len(nodes_idx)
            if target_cap:
                n_pairs = min(n_pairs, target_cap)
            
            t_indices = rng.choice(times_idx, size=n_pairs, replace=True)
            n_indices = rng.choice(nodes_idx, size=n_pairs, replace=True)
            
            return t_indices, n_indices
    
    def _gather_coords_targets(
        self,
        sample_id: str,
        t_idx_arr: np.ndarray,
        node_idx_arr: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather coordinates and target values for selected pairs."""
        topology = self.dataset.get_topology(sample_id)  # [120, N, 4]
        temperature = self.dataset.get_temperature(sample_id)  # [120, N, 1]
        
        # Index into arrays
        coords = topology[t_idx_arr, node_idx_arr]  # [S, 4]
        targets = temperature[t_idx_arr, node_idx_arr]  # [S, 1]
        
        return coords, targets
    
    def _get_graph_tensors(
        self,
        sample_id: str
    ) -> Tuple[torch.Tensor, ...]:
        """Get all graph-related tensors for a sample."""
        # Try shared first if available
        if self.dataset.has_shared:
            try:
                edge_index = self.dataset.get_shared_edge_index()
                edge_attr_r = self.dataset.get_shared_edge_attr_r()
                k = self.dataset.get_shared_k()
                node_pos = self.dataset.get_shared_node_pos()
            except ValueError:
                # Fall back to per-sample
                edge_index = self.dataset.get_sample_edge_index(sample_id)
                edge_attr_r = self.dataset.get_sample_edge_attr_r(sample_id)
                k = self.dataset.get_sample_k(sample_id)
                node_pos = self.dataset.get_sample_node_pos(sample_id)
        else:
            edge_index = self.dataset.get_sample_edge_index(sample_id)
            edge_attr_r = self.dataset.get_sample_edge_attr_r(sample_id)
            k = self.dataset.get_sample_k(sample_id)
            node_pos = self.dataset.get_sample_node_pos(sample_id)
        
        I_T = self.dataset.get_globals(sample_id)
        
        result = [edge_index, edge_attr_r, k, node_pos, I_T]
        
        # Add q, s if requested
        if self.config.include_qs:
            q = self.dataset.get_q(sample_id)
            s = self.dataset.get_s(sample_id)
            result.extend([q, s])
        
        return tuple(result)
    
    def _next_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get next training batch."""
        if self.current_idx >= len(self.mode_sample_ids):
            raise StopIteration
        
        batch_list = []
        
        # Select samples for this step
        n_samples = min(self.config.samples_per_step, 
                       len(self.mode_sample_ids) - self.current_idx)
        sample_ids = self.mode_sample_ids[self.current_idx:self.current_idx + n_samples]
        self.current_idx += n_samples
        
        # Select times (same for all samples in step)
        train_times = self.split_info["train_time_idx"]
        selected_times = self._pick_times_train(train_times)
        
        # Process each sample
        for sample_id in sample_ids:
            # Select nodes
            selected_nodes = self._pick_nodes(
                self.dataset.N,
                self.config.node_subsample_frac,
                self.rng_nodes
            )
            
            # Build pairs
            t_indices, n_indices = self._build_pairs(
                selected_times,
                selected_nodes,
                self.config.prefer_cartesian_pairs,
                self.config.target_pairs_cap,
                self.rng_pairs
            )
            
            # Gather data
            coords, targets = self._gather_coords_targets(sample_id, t_indices, n_indices)
            graph_tensors = self._get_graph_tensors(sample_id)
            
            # Build batch dict
            batch = {
                "sample_id": sample_id,
                "edge_index": graph_tensors[0],
                "edge_attr_r": graph_tensors[1],
                "k": graph_tensors[2],
                "node_pos": graph_tensors[3],
                "I_T": graph_tensors[4],
                "coords": coords,
                "node_idx": torch.from_numpy(n_indices).long(),
                "targets": targets
            }
            
            if self.config.include_qs:
                batch["q"] = graph_tensors[5]
                batch["s"] = graph_tensors[6]
            
            batch_list.append(batch)
        
        return batch_list
    
    def _next_val(self) -> List[Dict[str, torch.Tensor]]:
        """Get next validation batch."""
        if self.current_idx >= len(self.mode_sample_ids):
            raise StopIteration
        
        batch_list = []
        
        # Process one sample at a time for validation
        sample_id = self.mode_sample_ids[self.current_idx]
        self.current_idx += 1
        
        # Use validation times
        val_times = np.array(self.split_info["val_time_idx"])
        
        # Select subset of nodes (deterministic)
        selected_nodes = self._pick_nodes(
            self.dataset.N,
            self.config.val_node_frac,
            self.rng_val_nodes
        )
        
        # Build pairs (all val times × selected nodes)
        t_indices, n_indices = self._build_pairs(
            val_times,
            selected_nodes,
            True,  # Always use Cartesian for validation
            None,  # No cap for validation
            self.rng_pairs
        )
        
        # Gather data
        coords, targets = self._gather_coords_targets(sample_id, t_indices, n_indices)
        graph_tensors = self._get_graph_tensors(sample_id)
        
        # Build batch dict
        batch = {
            "sample_id": sample_id,
            "edge_index": graph_tensors[0],
            "edge_attr_r": graph_tensors[1],
            "k": graph_tensors[2],
            "node_pos": graph_tensors[3],
            "I_T": graph_tensors[4],
            "coords": coords,
            "node_idx": torch.from_numpy(n_indices).long(),
            "targets": targets
        }
        
        if self.config.include_qs:
            batch["q"] = graph_tensors[5]
            batch["s"] = graph_tensors[6]
        
        batch_list.append(batch)
        
        return batch_list
    
    def _next_test(self) -> List[Dict[str, torch.Tensor]]:
        """Get next test batch."""
        if self.current_idx >= len(self.mode_sample_ids):
            raise StopIteration
        
        batch_list = []
        
        # Process one sample at a time
        sample_id = self.mode_sample_ids[self.current_idx]
        
        # All test times
        test_times = np.arange(self.dataset.Tlen)
        
        # Handle node chunking
        if self.config.test_chunk_nodes:
            # Chunk nodes
            start_node = self.test_node_offset
            end_node = min(start_node + self.config.test_chunk_nodes, self.dataset.N)
            selected_nodes = np.arange(start_node, end_node)
            
            # Update offset
            self.test_node_offset = end_node
            if self.test_node_offset >= self.dataset.N:
                # Move to next sample
                self.test_node_offset = 0
                self.current_idx += 1
        else:
            # All nodes at once
            selected_nodes = np.arange(self.dataset.N)
            self.current_idx += 1
        
        # Build pairs (all times × selected nodes)
        t_indices, n_indices = self._build_pairs(
            test_times,
            selected_nodes,
            True,  # Cartesian
            None,  # No cap
            self.rng_pairs
        )
        
        # Gather data
        coords, targets = self._gather_coords_targets(sample_id, t_indices, n_indices)
        graph_tensors = self._get_graph_tensors(sample_id)
        
        # Build batch dict
        batch = {
            "sample_id": sample_id,
            "edge_index": graph_tensors[0],
            "edge_attr_r": graph_tensors[1],
            "k": graph_tensors[2],
            "node_pos": graph_tensors[3],
            "I_T": graph_tensors[4],
            "coords": coords,
            "node_idx": torch.from_numpy(n_indices).long(),
            "targets": targets
        }
        
        if self.config.include_qs:
            batch["q"] = graph_tensors[5]
            batch["s"] = graph_tensors[6]
        
        batch_list.append(batch)
        
        return batch_list


def main():
    """CLI for testing the sampler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Graph-DeepONet sampler")
    parser.add_argument("--h5-path", type=str, required=True,
                       help="Path to normalized HDF5 dataset")
    parser.add_argument("--split-json", type=str, required=True,
                       help="Path to train/val/test split JSON")
    parser.add_argument("--mode", choices=["train", "val", "test"], default="train",
                       help="Sampling mode")
    parser.add_argument("--include-qs", action="store_true",
                       help="Include q and s features")
    parser.add_argument("--samples-per-step", type=int, default=1,
                       help="Number of samples per batch")
    parser.add_argument("--target-pairs-cap", type=int, default=1000,
                       help="Maximum coordinate pairs per sample")
    
    args = parser.parse_args()
    
    # Create config
    config = SamplerConfig(
        h5_path=args.h5_path,
        split_json=args.split_json,
        mode=args.mode,
        include_qs=args.include_qs,
        samples_per_step=args.samples_per_step,
        target_pairs_cap=args.target_pairs_cap
    )
    
    # Create dataset and sampler
    print(f"Loading dataset from: {args.h5_path}")
    dataset = GraphDeepONetDataset(
        args.h5_path,
        include_qs=args.include_qs,
        use_shared_if_available=True
    )
    print(f"Dataset info: N={dataset.N}, E={dataset.E}, T={dataset.Tlen}")
    print(f"Samples: {len(dataset.sample_ids)}")
    
    sampler = GraphDeepONetSampler(dataset, config, args.split_json)
    print(f"\nMode: {args.mode}")
    print(f"Mode samples: {len(sampler.mode_sample_ids)}")
    
    # Get one batch
    print("\nFetching one batch...")
    batch_list = next(iter(sampler))
    
    for i, batch in enumerate(batch_list):
        print(f"\nBatch {i+1}/{len(batch_list)}:")
        print(f"  Sample ID: {batch['sample_id']}")
        for key, tensor in batch.items():
            if key == "sample_id":
                continue
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    
    print("\n✓ Sampler test complete!")


if __name__ == "__main__":
    main()
