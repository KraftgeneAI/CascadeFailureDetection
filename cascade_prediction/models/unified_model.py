"""
Unified Cascade Prediction Model
=================================
Complete model combining all components:
- Multi-modal embeddings (environmental, infrastructure, robotic)
- Graph attention networks with physics-aware message passing
- Temporal dynamics with LSTM
- Multi-task prediction heads
- Physics-informed loss

This is the main model class that orchestrates all submodules.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from cascade_prediction.data.generator.config import Settings

# Import embeddings
from .embeddings import (
    EnvironmentalEmbedding,
    InfrastructureEmbedding,
    RoboticEmbedding,
)

# Import layers
from .layers import (
    GraphAttentionLayer,
    TemporalGNNCell,
)

# Import prediction heads
from .heads import (
    FailureProbabilityHead,
    VoltageHead,
    AngleHead,
    FrequencyHead,
    TemperatureHead,
    LineFlowHead,
    ReactiveFlowHead,
    ActivePowerLineFlowHead,
    RiskHead,
    TimingHead,
)

# Import loss
from .loss import PhysicsInformedLoss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UnifiedCascadePredictionModel(nn.Module):
    """
    Complete unified model combining:
    - Multi-modal fusion (environmental, infrastructure, robotic)
    - Physics-informed GNN with graph attention
    - Temporal dynamics with multi-layer LSTM
    - Frequency dynamics modeling
    - Multi-task prediction (including direct node timing)
    """
    
    def __init__(
        self,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM,
        hidden_dim: int = Settings.Model.HIDDEN_DIM,
        num_gnn_layers: int = Settings.Model.NUM_GNN_LAYERS,
        heads: int = Settings.Model.HEADS,
        dropout: float = Settings.Model.DROPOUT
    ):
        """
        Initialize unified cascade prediction model.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of hidden representations
            num_gnn_layers: Number of GNN layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(UnifiedCascadePredictionModel, self).__init__()
        
        # Multi-modal embeddings
        self.env_embedding = EnvironmentalEmbedding(embedding_dim=embedding_dim)
        self.infra_embedding = InfrastructureEmbedding(embedding_dim=embedding_dim)
        self.robot_embedding = RoboticEmbedding(embedding_dim=embedding_dim)
        
        # Multi-modal fusion with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(embedding_dim)
        
        # Temporal GNN layers
        self.temporal_gnn = TemporalGNNCell(
            node_features=embedding_dim,
            hidden_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout
        )
        
        # Additional GNN layers for spatial processing
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=hidden_dim
            )
            for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Edge embedding - accepts Settings.Model.EDGE_FEATURES features:
        # [reactance, thermal_limits, resistance, susceptance, conductance, line_flows_p, line_flows_q]
        self.edge_embedding = nn.Sequential(
            nn.Linear(Settings.Model.EDGE_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.failure_prob_head = FailureProbabilityHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.failure_time_head = TimingHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.voltage_head = VoltageHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.angle_head = AngleHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.line_flow_head = LineFlowHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.reactive_nodes_head = ReactiveFlowHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.frequency_head = FrequencyHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.temperature_head = TemperatureHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.active_power_line_flow_head = ActivePowerLineFlowHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.risk_head = RiskHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_sequence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all modalities.
        
        Args:
            batch: Input batch dictionary containing all modalities
            return_sequence: Whether to return full sequence (not used currently)
        
        Returns:
            Dictionary of predictions for all tasks
        """
        # Multi-modal embeddings
        env_emb = self.env_embedding(
            batch['satellite_data'],
            batch['weather_sequence'],
            batch['threat_indicators']
        )
        
        infra_emb = self.infra_embedding(
            batch['scada_data'],
            batch['pmu_sequence'],
            batch['equipment_status']
        )
        
        robot_emb = self.robot_embedding(
            batch['visual_data'],
            batch['thermal_data'],
            batch['sensor_data']
        )
        
        has_temporal = env_emb.dim() == 4  # [B, T, N, D]
        
        # Prepare edge attributes
        edge_attr_input = batch.get('edge_attr')
        if edge_attr_input is None:
            E = batch['edge_index'].shape[1]
            edge_attr_input = torch.zeros(
                env_emb.shape[0], E, Settings.Model.EDGE_FEATURES,
                device=env_emb.device
            )
        
        if edge_attr_input.dim() == 2 and env_emb.dim() > 2:
            edge_attr_input = edge_attr_input.unsqueeze(0).expand(
                env_emb.shape[0], -1, -1
            )
        
        edge_mask_input = batch.get('edge_mask')
        
        # Process temporal sequences
        if has_temporal:
            B, T, N, D = env_emb.shape
            
            fused_list = []
            for t in range(T):
                # Select timestep t for robot embedding
                robot_feat_t = robot_emb[:, t, :, :]
                
                multi_modal_t = torch.stack([
                    env_emb[:, t, :, :],
                    infra_emb[:, t, :, :],
                    robot_feat_t
                ], dim=2)
                
                multi_modal_flat = multi_modal_t.reshape(B * N, 3, D)
                fused_t, _ = self.fusion_attention(
                    multi_modal_flat, multi_modal_flat, multi_modal_flat
                )
                fused_t = fused_t.mean(dim=1).reshape(B, N, D)
                fused_t = self.fusion_norm(fused_t)
                fused_list.append(fused_t)
            
            fused_sequence = torch.stack(fused_list, dim=1)
            fused = fused_sequence[:, -1, :, :]
            
            edge_embedded = self.edge_embedding(edge_attr_input)
            
            h_states = []
            lstm_state = None
            
            for t in range(T):
                x_t = fused_sequence[:, t, :, :]
                
                # Handle edge mask dimensions
                if edge_mask_input is not None and edge_mask_input.dim() == 3:
                    mask_t = edge_mask_input[:, t, :]
                else:
                    mask_t = edge_mask_input
                
                h_t, lstm_state = self.temporal_gnn(
                    x_t, batch['edge_index'], edge_embedded,
                    edge_mask=mask_t,
                    h_prev=lstm_state
                )
                h_states.append(h_t)
            
            h_stack = torch.stack(h_states, dim=2)
            
            if 'sequence_length' in batch:
                lengths = batch['sequence_length']
                # Move lengths to CPU to avoid CUDA illegal memory access
                lengths_cpu = lengths.cpu()
                h_final_list = []
                for b in range(B):
                    valid_idx = int(lengths_cpu[b]) - 1
                    
                    if valid_idx < 0:
                        valid_idx = 0
                    if valid_idx >= T:
                        valid_idx = T - 1
                    h_final_list.append(h_stack[b, :, valid_idx, :])
                h = torch.stack(h_final_list, dim=0)
            else:
                h = h_stack[:, :, -1, :]
        
        else:
            # Non-temporal processing
            B, N, D = env_emb.shape
            
            if robot_emb.dim() == 2:
                robot_emb_expanded = robot_emb.unsqueeze(1).expand(-1, N, -1)
            else:
                robot_emb_expanded = robot_emb
            
            multi_modal = torch.stack([env_emb, infra_emb, robot_emb_expanded], dim=2)
            B, N, M, D = multi_modal.shape
            multi_modal_flat = multi_modal.reshape(B * N, M, D)
            
            fused, _ = self.fusion_attention(
                multi_modal_flat, multi_modal_flat, multi_modal_flat
            )
            fused = fused.mean(dim=1).reshape(B, N, D)
            fused = self.fusion_norm(fused)
            
            edge_embedded = self.edge_embedding(edge_attr_input)
            
            h, _ = self.temporal_gnn(
                fused, batch['edge_index'], edge_embedded,
                edge_mask=edge_mask_input
            )
        
        # Extract final edge mask
        final_mask = edge_mask_input[:, -1, :] if (
            edge_mask_input is not None and edge_mask_input.dim() == 3
        ) else edge_mask_input
        
        # Apply additional GNN layers
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, batch['edge_index'], edge_embedded, edge_mask=final_mask)
            h = layer_norm(h + h_new)

        # Multi-task predictions
        failure_prob = self.failure_prob_head(h)
        temperature = self.temperature_head(h)
        failure_timing = self.failure_time_head(h)
        voltages = self.voltage_head(h)
        angles = self.angle_head(h)
        
        h_global = h.mean(dim=1, keepdim=True)
        frequency = self.frequency_head(h_global)
        
        # Edge-based predictions
        src, dst = batch['edge_index']
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        
        line_flows = self.line_flow_head(edge_features)
        active_power_line_flows = self.active_power_line_flow_head(edge_features)
        reactive_nodes = self.reactive_nodes_head(h)
        
        risk_scores = self.risk_head(h)
        
        # Check for NaN values
        if torch.isnan(failure_prob).any():
            logging.error("[ERROR] NaN detected in failure_prob!")
        
        return {
            'failure_probability': failure_prob,
            'cascade_timing': failure_timing,
            'voltages': voltages,
            'angles': angles,
            'line_flows': line_flows, #line_reactive_power
            'active_power_line_flows': active_power_line_flows,
            'temperature': temperature,
            'reactive_nodes': reactive_nodes,
            'frequency': frequency,
            'risk_scores': risk_scores,
            'node_embeddings': h,
            'env_embedding': env_emb,
            'infra_embedding': infra_emb,
            'robot_embedding': robot_emb,
            'fused_embedding': fused
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_properties: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with physics constraints.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            graph_properties: Graph properties including edge attributes
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        logging.debug("Computing total loss with physics constraints")
        loss_details = self.physics_loss(predictions, targets, graph_properties)
        logging.debug(f"Loss computation complete. Total loss: {loss_details[0].item()}")
        return loss_details
