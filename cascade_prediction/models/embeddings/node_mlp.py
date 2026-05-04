"""
Node Feature MLP
================
Encodes the 124-feature per-node-per-timestep vector produced by
CascadeDataset._build_node_features() into the shared embedding space.

Feature layout (124 total, IMPROVED v3 — cascade susceptibility):
  [0:18]   SCADA measurements          (18)
  [18:26]  PMU measurements             ( 8)
  [26:36]  Equipment status            (10)
  [36:37]  Active power injection       ( 1)
  [37:38]  Reactive power injection     ( 1)
  [38:76]  1-step temporal deltas      (38)
  [76:114] 2-step temporal deltas      (38)
  [114]    Normalised timestep position ( 1)
  [115]    TTF voltage  (normalised steps to voltage failure)   ( 1)
  [116]    TTF temp     (normalised steps to temperature failure)( 1)
  [117]    TTF freq     (normalised steps to frequency failure)  ( 1)
  [118]    TTF loading  (normalised steps to loading failure)    ( 1)
  [119]    mean_adjacent_line_loading  (mean |flow|/limit adj edges) ( 1)
  [120]    cascade_initiation_risk     (|P_inj|/sum_thermal_limits)  ( 1)
  [121]    cascade_reception_risk      (weighted flow from neighbors)( 1)
  [122]    max_adjacent_line_loading   (max |flow|/limit adj edges)  ( 1)
  [123]    loading_x_max_line          (loading_ratio × max_adj_ll)  ( 1)

The 5 new cascade susceptibility features encode topology-informed
pre-cascade risk signals: which nodes are likely to initiate cascade
(high power injection relative to line capacity) and which are likely
to receive cascade propagation (high-loading stressed neighbors).

The MLP is a 3-layer feed-forward network with BatchNorm and dropout.
"""

import torch
import torch.nn as nn
from cascade_prediction.data.generator.config import Settings


class NodeFeatureMLP(nn.Module):
    """
    3-layer MLP that maps (*, 124) → (*, embedding_dim).

    Accepts any leading batch dimensions so it works for both:
      - (B, N, 124)    — single-timestep inference
      - (B, T, N, 124) — full temporal sequence (reshape → process → reshape)
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
            # Layer 1 — input projection (124 → 256)
            nn.Linear(in_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2 — compression (256 → 128)
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3 — output projection into embedding space (128 → embedding_dim)
            nn.Linear(hidden_2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 124) float tensor — any number of leading dimensions.

        Returns:
            (..., embedding_dim) float tensor — same leading shape.
        """
        leading = x.shape[:-1]              # e.g. (B, T, N) or (B, N)
        flat = x.reshape(-1, x.shape[-1])   # (M, 124)  — BatchNorm needs 2-D
        out  = self.net(flat)               # (M, embedding_dim)
        return out.reshape(*leading, -1)    # (..., embedding_dim)
