"""
Lightweight, CPU-friendly Graph-DeepONet model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict, Tuple, Optional
import math


class GraphDeepONet(nn.Module):
    def __init__(
        self,
        q_dim: int = 128,                 # shared latent size
        trunk_hidden: int = 128,          # hidden width for trunk MLP
        trunk_depth: int = 2,             # number of hidden layers in trunk
        glob_hidden: int = 64,            # hidden width for global branch
        graph_hidden: int = 64,           # hidden width for graph branch
        graph_layers: int = 1,            # {1,2} GCN layers for CPU friendliness
        use_fourier_time: bool = False,   # if True, add Fourier features for t
        time_bands: int = 0,              # number of Fourier bands for t
        use_qs_features: bool = False,    # if True, BranchGraph also consumes q,s
        activation: str = "leaky_relu",   # activation in trunk/branches
        film_global: bool = True,         # apply FiLM conditioning
        head_bias_mlp: bool = False       # if True, add a tiny bias MLP
    ):
        super().__init__()
        
        # Store config
        self.q_dim = q_dim
        self.trunk_hidden = trunk_hidden
        self.trunk_depth = trunk_depth
        self.glob_hidden = glob_hidden
        self.graph_hidden = graph_hidden
        self.graph_layers = graph_layers
        self.use_fourier_time = use_fourier_time
        self.time_bands = time_bands
        self.use_qs_features = use_qs_features
        self.film_global = film_global
        self.head_bias_mlp = head_bias_mlp
        
        # Activation function
        self.act = self._get_activation(activation)
        
        # Build trunk network
        trunk_input_dim = 4  # [t, x, y, z]
        if use_fourier_time and time_bands > 0:
            trunk_input_dim += 2 * time_bands  # sin/cos features
        
        trunk_layers = []
        # First layer
        trunk_layers.extend([
            nn.Linear(trunk_input_dim, trunk_hidden),
            self.act,
            nn.LayerNorm(trunk_hidden)
        ])
        # Hidden layers
        for _ in range(trunk_depth - 1):
            trunk_layers.extend([
                nn.Linear(trunk_hidden, trunk_hidden),
                self.act
            ])
        # Output layer
        trunk_layers.append(nn.Linear(trunk_hidden, q_dim))
        
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Build global branch
        self.branch_global = nn.Sequential(
            nn.Linear(2, glob_hidden),
            self.act,
            nn.Linear(glob_hidden, q_dim)
        )
        
        # Build graph branch
        node_feat_dim = 1  # k
        if use_qs_features:
            node_feat_dim += 2  # +q, +s
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, graph_hidden),
            self.act
        )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(graph_layers):
            in_dim = graph_hidden
            out_dim = graph_hidden
            self.gcn_layers.append(GCNConv(in_dim, out_dim, add_self_loops=True))
        
        # Graph output projection
        self.graph_out = nn.Linear(graph_hidden, q_dim)
        
        # FiLM conditioning
        if film_global:
            self.film_net = nn.Linear(q_dim, 2 * q_dim)
        
        # Head network
        if head_bias_mlp:
            self.head = nn.Sequential(
                nn.Linear(q_dim, q_dim),
                self.act,
                nn.Linear(q_dim, 1)
            )
        else:
            self.head = nn.Linear(q_dim, 1)
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        if name == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def _fourier_features(self, t: torch.Tensor) -> torch.Tensor:
        """Create Fourier features for time coordinate."""
        features = []
        for k in range(self.time_bands):
            freq = 2**k * math.pi
            features.append(torch.sin(freq * t))
            features.append(torch.cos(freq * t))
        return torch.cat(features, dim=-1)
    
    def encode_case(
        self,
        edge_index: torch.LongTensor,     # [2, E]
        edge_attr_r: torch.Tensor,        # [E]
        k: torch.Tensor,                  # [N]
        node_pos: torch.Tensor,           # [N, 3]
        I_T: torch.Tensor,                # [2]
        q: Optional[torch.Tensor] = None, # [N]
        s: Optional[torch.Tensor] = None, # [N]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single case/sample (graph + globals) once.
        """
        # Validate inputs
        assert edge_index.dim() == 2 and edge_index.shape[0] == 2, f"edge_index must be [2, E], got {edge_index.shape}"
        assert edge_attr_r.shape[0] == edge_index.shape[1], f"edge_attr_r length must match edges, got {edge_attr_r.shape[0]} vs {edge_index.shape[1]}"
        assert k.dim() == 1, f"k must be 1D, got shape {k.shape}"
        N = k.shape[0]
        assert node_pos.shape == (N, 3), f"node_pos must be [N, 3], got {node_pos.shape}"
        assert I_T.shape == (2,), f"I_T must be [2], got {I_T.shape}"
        
        if self.use_qs_features:
            assert q is not None and s is not None, "q and s must be provided when use_qs_features=True"
            assert q.shape == (N,) and s.shape == (N,), f"q and s must be [N], got {q.shape} and {s.shape}"
        
        # Cast to correct dtypes
        edge_index = edge_index.long()
        edge_attr_r = edge_attr_r.float()
        k = k.float()
        node_pos = node_pos.float()
        I_T = I_T.float()
        if q is not None:
            q = q.float()
        if s is not None:
            s = s.float()
        
        # 1. Global branch
        z_global = self.branch_global(I_T.unsqueeze(0))  # [1, q_dim]
        z_global = z_global.squeeze(0)  # [q_dim]
        
        # 2. Graph branch
        # Prepare node features
        node_features = [k.unsqueeze(-1)]  # [N, 1]
        if self.use_qs_features:
            node_features.append(q.unsqueeze(-1))
            node_features.append(s.unsqueeze(-1))
        x = torch.cat(node_features, dim=-1)  # [N, node_feat_dim]
        
        # Node MLP
        h = self.node_mlp(x)  # [N, graph_hidden]
        
        # Normalize edge weights
        edge_weight = edge_attr_r / (edge_attr_r.max() + 1e-12)
        
        # GCN layers
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, edge_index, edge_weight)
            if i < len(self.gcn_layers) - 1:  # Skip activation on last layer
                h = self.act(h)
        
        # Project to q_dim
        node_latents = self.graph_out(h)  # [N, q_dim]
        
        # Apply FiLM conditioning
        if self.film_global:
            film_params = self.film_net(z_global)  # [2*q_dim]
            gamma = film_params[:self.q_dim].unsqueeze(0)  # [1, q_dim]
            beta = film_params[self.q_dim:].unsqueeze(0)   # [1, q_dim]
            A = gamma * node_latents + beta  # [N, q_dim]
        else:
            A = node_latents
        
        return {
            "A": A,
            "z_global": z_global,
            "node_latents": node_latents
        }
    
    def trunk(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Trunk over coordinates.
        Args:
            coords: [S, 4] = [t, x, y, z] already normalized
        Returns:
            Phi: [S, q_dim] basis values
        """
        assert coords.shape[1] == 4, f"coords must be [S, 4], got {coords.shape}"
        coords = coords.float()
        
        # Extract time for Fourier features if needed
        if self.use_fourier_time and self.time_bands > 0:
            t = coords[:, 0:1]  # [S, 1]
            fourier_feats = self._fourier_features(t)  # [S, 2*time_bands]
            trunk_input = torch.cat([coords, fourier_feats], dim=-1)
        else:
            trunk_input = coords
        
        # Pass through trunk network
        phi = self.trunk_net(trunk_input)  # [S, q_dim]
        return phi
    
    def predict_with_cache(
        self,
        cache: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        node_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Fast path for inference/training with precomputed per-node coefficients.
        """
        assert coords.shape[1] == 4, f"coords must be [S, 4], got {coords.shape}"
        assert node_idx.shape[0] == coords.shape[0], f"node_idx length must match coords rows, got {node_idx.shape[0]} vs {coords.shape[0]}"
        
        coords = coords.float()
        node_idx = node_idx.long()
        
        # Get per-node coefficients
        A = cache["A"]  # [N, q_dim]
        A_i = A[node_idx]  # [S, q_dim]
        
        # Compute trunk basis
        phi = self.trunk(coords)  # [S, q_dim]
        
        # Elementwise product
        z = A_i * phi  # [S, q_dim]
        
        # Final prediction
        T_hat = self.head(z)  # [S, 1]
        
        return T_hat
    
    def forward(
        self,
        edge_index: torch.LongTensor,
        edge_attr_r: torch.Tensor,
        k: torch.Tensor,
        node_pos: torch.Tensor,
        I_T: torch.Tensor,
        coords: torch.Tensor,
        node_idx: torch.LongTensor,
        q: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convenience end-to-end forward (no caching).
        """
        # Encode case
        cache = self.encode_case(edge_index, edge_attr_r, k, node_pos, I_T, q, s)
        
        # Predict with cache
        T_hat = self.predict_with_cache(cache, coords, node_idx)
        
        # Prepare auxiliary outputs
        phi = self.trunk(coords)
        aux = {
            "A": cache["A"],
            "z_global": cache["z_global"],
            "phi": phi,
            "node_idx": node_idx
        }
        
        return T_hat, aux


if __name__ == "__main__":
    # Self-test
    print("Running GraphDeepONet self-test...")
    
    # Create small random graph
    N = 100  # nodes
    E = 400  # edges
    S = 50   # coordinate samples
    
    # Random graph data
    edge_index = torch.randint(0, N, (2, E))
    edge_attr_r = torch.rand(E)
    k = torch.rand(N)
    node_pos = torch.rand(N, 3)
    I_T = torch.rand(2)
    q = torch.rand(N)
    s = torch.rand(N)
    
    # Random coordinates and node indices
    coords = torch.rand(S, 4)
    node_idx = torch.randint(0, N, (S,))
    targets = torch.rand(S, 1)
    
    # Test model creation
    model = GraphDeepONet(
        q_dim=128,
        graph_layers=1,
        use_fourier_time=True,
        time_bands=4,
        use_qs_features=True
    )
    
    # Test forward pass
    pred, aux = model(edge_index, edge_attr_r, k, node_pos, I_T, coords, node_idx, q, s)
    
    print(f"✓ Forward pass successful")
    print(f"  - Prediction shape: {pred.shape} (expected: [{S}, 1])")
    print(f"  - A shape: {aux['A'].shape} (expected: [{N}, 128])")
    print(f"  - z_global shape: {aux['z_global'].shape} (expected: [128])")
    print(f"  - phi shape: {aux['phi'].shape} (expected: [{S}, 128])")
    
    # Test backward pass
    loss = F.mse_loss(pred, targets)
    loss.backward()
    
    print(f"✓ Backward pass successful")
    
    # Test cache-based prediction
    cache = model.encode_case(edge_index, edge_attr_r, k, node_pos, I_T, q, s)
    pred_cached = model.predict_with_cache(cache, coords, node_idx)
    
    print(f"✓ Cache-based prediction successful")
    print(f"  - Cached prediction shape: {pred_cached.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nAll tests passed! ✨")
