"""
Node Feature MLP
================
Encodes the 136-feature per-node-per-timestep vector produced by
CascadeDataset._build_node_features() into the shared embedding space.

Feature layout (136 total, v4 — Phase 3 widened SCADA 18→22):
  [0:22]   SCADA measurements          (22; 18-21 = cap/reactor status,
                                        SVC Q, AVR setpoint)
  [22:30]  PMU measurements             ( 8)
  [30:40]  Equipment status            (10)
  [40:41]  Active power injection       ( 1)
  [41:42]  Reactive power injection     ( 1)
  [42:84]  1-step temporal deltas      (42)
  [84:126] 2-step temporal deltas      (42)
  [126]    Normalised timestep position ( 1)
  [127]    TTF voltage  (normalised steps to voltage failure)   ( 1)
  [128]    TTF temp     (normalised steps to temperature failure)( 1)
  [129]    TTF freq     (normalised steps to frequency failure)  ( 1)
  [130]    TTF loading  (normalised steps to loading failure)    ( 1)
  [131]    mean_adjacent_line_loading  (mean |flow|/limit adj edges) ( 1)
  [132]    cascade_initiation_risk     (|P_inj|/sum_thermal_limits)  ( 1)
  [133]    cascade_reception_risk      (weighted flow from neighbors)( 1)
  [134]    max_adjacent_line_loading   (max |flow|/limit adj edges)  ( 1)
  [135]    loading_x_max_line          (loading_ratio × max_adj_ll)  ( 1)

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
