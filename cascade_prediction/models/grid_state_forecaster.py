"""
Grid State Forecaster
=====================
Lightweight next-step prediction model for autoregressive inference.

Only uses:
  - InfrastructureEmbedding  (SCADA + PMU + equipment_status)
  - NodeFeatureMLP            (119-feature per-node vector)

Architecture mirrors UnifiedCascadePredictionModel but replaces the
multi-task classification heads with three regression decoders that
predict the variable features of the *next* timestep.

Variable / Constant split
--------------------------
SCADA (18):
  Variable [13]: indices 0-6  (voltage, angle, generation, reactive,
                               load, temp, frequency)
               + indices 12-17 (time_ratio, stress_level,
                                voltage_ratio, temp_ratio,
                                freq_ratio, loading_ratio)
  Constant  [5]: indices 7-11 (equipment_age, equipment_condition,
                               gen_capacity, base_load, node_types)

PMU (8): all variable

Equipment status (10):
  Variable [3]: indices 2 (temp), 6 (temp_ratio), 9 (load_ratio)
  Constant [7]: indices 0,1,3,4,5,7,8

Autoregressive rollout (inference)
------------------------------------
Given a seed window of T ground-truth timesteps the caller:
  1. Calls forward() → gets next_scada_vars, next_pmu, next_equip_vars
  2. Reconstructs full SCADA / equipment tensors by merging predicted
     variables with carried-forward constants
  3. Recomputes node_features (deltas, TTF, t_pos) from the new state
  4. Appends the new step, drops the oldest → slides the window
  5. Repeats until the desired horizon is reached
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from cascade_prediction.data.generator.config import Settings
from .embeddings import InfrastructureEmbedding
from .embeddings import NodeFeatureMLP
from .layers import GraphAttentionLayer, TemporalGNNCell


# ---------------------------------------------------------------------------
# Feature-index constants
# ---------------------------------------------------------------------------
SCADA_VAR_IDX   = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17]  # 13 features
SCADA_CONST_IDX = [7, 8, 9, 10, 11]                                 #  5 features
EQUIP_VAR_IDX   = [2, 6, 9]                                         #  3 features
EQUIP_CONST_IDX = [0, 1, 3, 4, 5, 7, 8]                            #  7 features

N_SCADA_VAR  = len(SCADA_VAR_IDX)    # 13
N_PMU        = Settings.Embedding.INFRA_PMU_FEATURES   # 8
N_EQUIP_VAR  = len(EQUIP_VAR_IDX)    # 3


class GridStateForecaster(nn.Module):
    """
    Predicts the variable features of the next grid timestep.

    Input batch keys (same names as CascadeDataset):
        scada_data       [B, T, N, 18]
        pmu_sequence     [B, T, N,  8]
        equipment_status [B, T, N, 10]
        node_features    [B, T, N, 119]
        edge_index       [2, E]
        edge_attr        [B, T, E, 7]  or  [B, E, 7]  (optional)
        edge_mask        [B, T, E]     or  [B, E]      (optional)

    Output keys:
        next_scada_vars  [B, N, 13]   — variable SCADA features at t+1
        next_pmu         [B, N,  8]   — all PMU features at t+1
        next_equip_vars  [B, N,  3]   — variable equipment features at t+1
    """

    def __init__(
        self,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM,
        num_gnn_layers: int = Settings.Model.NUM_GNN_LAYERS,
        heads: int = Settings.Model.HEADS,
        dropout: float = Settings.Model.DROPOUT,
    ):
        super().__init__()

        # Two modalities → MHA over 2 tokens → reshape → fused_dim = 2 * embedding_dim
        # MHA mixes infra and node embeddings without discarding either token.
        # Output [B*N, 2, D] is reshaped to [B*N, 2D] — no mean, no compression.
        self.fused_dim = 2 * embedding_dim

        # Embeddings
        self.infra_embedding  = InfrastructureEmbedding(embedding_dim=embedding_dim)
        self.node_feature_mlp = NodeFeatureMLP(embedding_dim=embedding_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(self.fused_dim)

        # Temporal GNN (identical pattern to UnifiedCascadePredictionModel)
        self.temporal_gnn = TemporalGNNCell(
            node_features=self.fused_dim,
            hidden_dim=self.fused_dim,
            edge_dim=self.fused_dim,
            num_heads=heads,
            dropout=dropout,
        )

        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_channels=self.fused_dim,
                out_channels=self.fused_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=self.fused_dim,
            )
            for _ in range(num_gnn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.fused_dim) for _ in range(num_gnn_layers)
        ])

        self.edge_embedding = nn.Sequential(
            nn.Linear(Settings.Model.EDGE_FEATURES, self.fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Decoders — predict variable features for next timestep
        self.scada_decoder = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim // 2, N_SCADA_VAR),
        )
        self.pmu_decoder = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim // 2, N_PMU),
        )
        self.equip_decoder = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim // 2, N_EQUIP_VAR),
        )

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys described in the class docstring.

        Returns:
            dict with next_scada_vars, next_pmu, next_equip_vars.
        """
        scada_data       = batch['scada_data']        # [B, T, N, 18]
        pmu_sequence     = batch['pmu_sequence']      # [B, T, N,  8]
        equipment_status = batch['equipment_status']  # [B, T, N, 10]
        node_features    = batch['node_features']     # [B, T, N, 119]

        B, T, N, _ = scada_data.shape

        # --- Embeddings ---
        infra_emb = self.infra_embedding(
            scada_data, pmu_sequence, equipment_status
        )                                              # [B, T, N, D]
        node_emb = self.node_feature_mlp(node_features)  # [B, T, N, D]

        # --- Fusion: MHA over 2 tokens → reshape → [B, T, N, 2D] ---
        # Stack as 2-token sequence: [B*T*N, 2, D]
        BT_N = B * T * N
        tokens = torch.stack(
            [infra_emb.reshape(BT_N, -1),
             node_emb.reshape(BT_N, -1)],
            dim=1,
        )                                              # [B*T*N, 2, D]
        attended, _ = self.fusion_attention(tokens, tokens, tokens)
        # Reshape both tokens into a single 2D vector — no information dropped
        fused_sequence = self.fusion_norm(
            attended.reshape(B, T, N, self.fused_dim)
        )                                              # [B, T, N, 2D]

        # --- Edge attributes ---
        edge_attr_input = batch.get('edge_attr')
        if edge_attr_input is None:
            E = batch['edge_index'].shape[1]
            edge_attr_input = torch.zeros(
                B, E, Settings.Model.EDGE_FEATURES, device=scada_data.device
            )
        # Normalise to [B, E, F] or [B, T, E, F]
        if edge_attr_input.dim() == 2:
            edge_attr_input = edge_attr_input.unsqueeze(0).expand(B, -1, -1)

        edge_mask_input = batch.get('edge_mask')

        # --- Temporal GNN loop ---
        h_states = []
        lstm_state = None

        for t in range(T):
            x_t = fused_sequence[:, t]                        # [B, N, 2D]

            if edge_attr_input.dim() == 4:
                edge_attr_t = edge_attr_input[:, t]
            else:
                edge_attr_t = edge_attr_input                 # [B, E, F]
            edge_emb_t = self.edge_embedding(edge_attr_t)    # [B, E, 2D]

            if edge_mask_input is not None and edge_mask_input.dim() == 3:
                mask_t = edge_mask_input[:, t]
            else:
                mask_t = edge_mask_input

            h_t, lstm_state = self.temporal_gnn(
                x_t, batch['edge_index'], edge_emb_t,
                edge_mask=mask_t, h_prev=lstm_state,
            )
            h_states.append(h_t)

        # Use last timestep hidden state
        h = h_states[-1]                                       # [B, N, 2D]

        # Edge embedding for final GNN layers
        if edge_attr_input.dim() == 4:
            edge_embedded = self.edge_embedding(edge_attr_input[:, -1])
        else:
            edge_embedded = self.edge_embedding(edge_attr_input)

        # Extract final edge mask
        final_mask = (
            edge_mask_input[:, -1]
            if edge_mask_input is not None and edge_mask_input.dim() == 3
            else edge_mask_input
        )

        # --- Additional GNN layers ---
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h = layer_norm(h + gnn_layer(h, batch['edge_index'], edge_embedded, edge_mask=final_mask))

        # --- Decode next-step variable features ---
        return {
            'next_scada_vars': self.scada_decoder(h),   # [B, N, 13]
            'next_pmu':        self.pmu_decoder(h),     # [B, N,  8]
            'next_equip_vars': self.equip_decoder(h),   # [B, N,  3]
        }

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        MSE loss on each predicted output against ground-truth next-step values.

        targets must contain:
            next_scada_vars  [B, N, 13]
            next_pmu         [B, N,  8]
            next_equip_vars  [B, N,  3]
        """
        mse = nn.functional.mse_loss

        loss_scada = mse(predictions['next_scada_vars'], targets['next_scada_vars'])
        loss_pmu   = mse(predictions['next_pmu'],        targets['next_pmu'])
        loss_equip = mse(predictions['next_equip_vars'], targets['next_equip_vars'])

        total = loss_scada + loss_pmu + loss_equip

        return total, {
            'loss_scada': loss_scada.item(),
            'loss_pmu':   loss_pmu.item(),
            'loss_equip': loss_equip.item(),
            'total':      total.item(),
        }


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------

def extract_next_step_targets(
    scada: torch.Tensor,
    pmu: torch.Tensor,
    equip: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Given full ground-truth tensors [B, T, N, F], return the variable
    features of the *last* timestep as training targets.

    Use during training: call with the T+1 timestep tensors while the
    model sees only timesteps 0..T-1.
    """
    return {
        'next_scada_vars': scada[:, -1, :, :][:, :, SCADA_VAR_IDX],
        'next_pmu':        pmu[:, -1, :, :],
        'next_equip_vars': equip[:, -1, :, :][:, :, EQUIP_VAR_IDX],
    }


def assemble_full_scada(
    pred_vars: torch.Tensor,          # [B, N, 13]
    constants: torch.Tensor,          # [B, N,  5]  — constant SCADA features
) -> torch.Tensor:
    """Reconstruct a [B, N, 18] SCADA tensor from predicted variables + constants."""
    B, N, _ = pred_vars.shape
    full = torch.zeros(B, N, 18, device=pred_vars.device, dtype=pred_vars.dtype)
    for out_i, src_i in enumerate(SCADA_VAR_IDX):
        full[:, :, src_i] = pred_vars[:, :, out_i]
    for out_i, src_i in enumerate(SCADA_CONST_IDX):
        full[:, :, src_i] = constants[:, :, out_i]
    return full


def assemble_full_equip(
    pred_vars: torch.Tensor,          # [B, N,  3]
    constants: torch.Tensor,          # [B, N,  7]  — constant equipment features
) -> torch.Tensor:
    """Reconstruct a [B, N, 10] equipment tensor from predicted variables + constants."""
    B, N, _ = pred_vars.shape
    full = torch.zeros(B, N, 10, device=pred_vars.device, dtype=pred_vars.dtype)
    for out_i, src_i in enumerate(EQUIP_VAR_IDX):
        full[:, :, src_i] = pred_vars[:, :, out_i]
    for out_i, src_i in enumerate(EQUIP_CONST_IDX):
        full[:, :, src_i] = constants[:, :, out_i]
    return full
