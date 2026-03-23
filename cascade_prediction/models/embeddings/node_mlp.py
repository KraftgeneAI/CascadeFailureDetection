"""
Node Feature MLP
================
Encodes the 115-feature per-node-per-timestep vector produced by
CascadeDataset._build_node_features() into the shared embedding space.

Feature layout (115 total):
  [0:18]   SCADA measurements          (18)
  [18:26]  PMU measurements             ( 8)
  [26:36]  Equipment status            (10)
  [36:37]  Active power injection       ( 1)
  [37:38]  Reactive power injection     ( 1)
  [38:76]  1-step temporal deltas      (38)
  [76:114] 2-step temporal deltas      (38)
  [114]    Normalised timestep position ( 1)

The MLP is a 3-layer feed-forward network with BatchNorm and dropout,
matching the architecture used during standalone MLP training.
"""

import torch
import torch.nn as nn
from cascade_prediction.data.generator.config import Settings


class NodeFeatureMLP(nn.Module):
    """
    3-layer MLP that maps (*, 115) → (*, embedding_dim).

    Accepts any leading batch dimensions so it works for both:
      - (B, N, 115)    — single-timestep inference
      - (B, T, N, 115) — full temporal sequence (reshape → process → reshape)
    """

    def __init__(
        self,
        in_features:   int = Settings.Embedding.NODE_FEATURE_DIM,
        hidden_1:      int = Settings.Embedding.NODE_MLP_HIDDEN_1,
        hidden_2:      int = Settings.Embedding.NODE_MLP_HIDDEN_2,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM,
        dropout:       float = Settings.Embedding.DROPOUT_FC,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1 — input projection
            nn.Linear(in_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2 — compression
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3 — output projection into embedding space
            nn.Linear(hidden_2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 115) float tensor — any number of leading dimensions.

        Returns:
            (..., embedding_dim) float tensor — same leading shape.
        """
        leading = x.shape[:-1]              # e.g. (B, T, N) or (B, N)
        flat = x.reshape(-1, x.shape[-1])   # (M, 115)  — BatchNorm needs 2-D
        out  = self.net(flat)               # (M, embedding_dim)
        return out.reshape(*leading, -1)    # (..., embedding_dim)
