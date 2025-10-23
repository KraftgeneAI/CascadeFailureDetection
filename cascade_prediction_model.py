"""
Cascade Failure Prediction Model
=================================
A physics-informed Graph Neural Network for predicting cascading failures
in electrical power grids.

Based on the research paper:
"AI-Driven Predictive Cascade Failure Analysis Using Multi-Modal 
Environmental-Infrastructure Data Fusion"

Author: Kraftgene AI Inc. (Implementation)
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Optional, Tuple, Dict, List
import math


# ============================================================================
# STEP 3: GNN PREDICTION LAYER
# ============================================================================

class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Layer implementing multi-head attention mechanism
    for learning importance weights between neighboring nodes.
    
    Based on Equation 3 and 6 from the paper.
    
    ** FIXED: Now properly handles batched inputs [B, N, F] **
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)  # Fixed node_dim to 0 for proper batching
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention mechanism parameters (Equation 3)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing attention-based message passing.
        
        Args:
            x: Node feature matrix [B, N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Updated node features [B, N, out_channels * heads] or [B, N, out_channels]
        """
        B, N, C = x.shape
        H, C_out = self.heads, self.out_channels
        
        # Flatten batch dimension: [B, N, C] -> [B*N, C]
        x_flat = x.view(B * N, C)
        
        # Linear transformation
        x_transformed = self.lin(x_flat).view(B * N, H, C_out)  # [B*N, H, C_out]
        
        # Create batched edge_index by offsetting node indices for each batch
        edge_index_batched = []
        for b in range(B):
            edge_index_batched.append(edge_index + b * N)
        edge_index_batched = torch.cat(edge_index_batched, dim=1)  # [2, B*E]
        
        # Add self-loops to batched edge_index
        if self.add_self_loops_flag:
            edge_index_batched, _ = add_self_loops(edge_index_batched, num_nodes=B * N)
        
        # Propagate messages
        out = self.propagate(edge_index_batched, x=x_transformed, size=(B * N, B * N))  # [B*N, H, C_out]
        
        # Reshape back to batched format: [B*N, H, C_out] -> [B, N, H, C_out]
        out = out.view(B, N, H, C_out)
        
        # Concatenate or average multi-head outputs
        if self.concat:
            out = out.view(B, N, H * C_out)
        else:
            out = out.mean(dim=2)  # [B, N, C_out]
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(
        self, 
        x_i: torch.Tensor, 
        x_j: torch.Tensor, 
        edge_index_i: torch.Tensor,
        size_i: Optional[int],
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute messages with attention weights (Equation 3).
        
        Args:
            x_i: Target node features [E, heads, out_channels]
            x_j: Source node features [E, heads, out_channels]
            edge_index_i: Target node indices
            size_i: Number of target nodes
            edge_attr: Edge attributes (optional)
            
        Returns:
            Attention-weighted messages [E, heads, out_channels]
        """
        # Compute attention coefficients (Equation 3)
        alpha_src = (x_j * self.att_src).sum(dim=-1)  # [E, H]
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)  # [E, H]
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention coefficients using softmax
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)  # [E, H]
        
        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Return attention-weighted messages (Equation 4)
        return x_j * alpha.unsqueeze(-1)  # [E, H, C_out]


class TemporalGNNCell(nn.Module):
    """
    Temporal GNN Cell combining graph attention with LSTM for temporal dynamics.
    Implements the temporal modeling described in Section 4.1.4.
    ** BATCH-AWARE & DIMENSION-FIXED **
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(TemporalGNNCell, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gat_out_channels_per_head = hidden_dim // num_heads
        self.gat_out_dim = num_heads * self.gat_out_channels_per_head
        
        # Graph attention layer for spatial aggregation
        self.gat = GraphAttentionLayer(
            in_channels=node_features,
            out_channels=self.gat_out_channels_per_head,
            heads=num_heads,
            concat=True,
            dropout=dropout
        )
        
        if self.gat_out_dim != hidden_dim:
            self.projection = nn.Linear(self.gat_out_dim, hidden_dim)
        else:
            self.projection = None
        
        # LSTM for temporal dynamics (Equation 7)
        self.lstm = nn.LSTMCell(
            input_size=hidden_dim,  # Now matches the projected dimension
            hidden_size=hidden_dim
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass combining spatial and temporal processing.
        
        Args:
            x: Node features [B, N, node_features]
            edge_index: Graph connectivity [2, E]
            h_prev: Previous hidden state [B, N, hidden_dim]
            c_prev: Previous cell state [B, N, hidden_dim]
            
        Returns:
            Tuple of (output, hidden_state, cell_state) all [B, N, hidden_dim]
        """
        B, N, _ = x.shape
        
        # Spatial aggregation via graph attention
        spatial_features = self.gat(x, edge_index)  # [B, N, gat_out_dim]
        
        if self.projection is not None:
            spatial_features = self.projection(spatial_features)  # [B, N, hidden_dim]
        
        # Initialize LSTM states if not provided
        if h_prev is None:
            h_prev = torch.zeros(B, N, self.hidden_dim, device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(B, N, self.hidden_dim, device=x.device)
        
        # Flatten for LSTMCell
        spatial_flat = spatial_features.view(B * N, -1)
        h_prev_flat = h_prev.view(B * N, -1)
        c_prev_flat = c_prev.view(B * N, -1)
        
        # Temporal update via LSTM (Equation 7)
        h_new_flat, c_new_flat = self.lstm(spatial_flat, (h_prev_flat, c_prev_flat))
        
        # Reshape back to [B, N, H]
        h_new = h_new_flat.view(B, N, self.hidden_dim)
        c_new = c_new_flat.view(B, N, self.hidden_dim)
        
        # Layer normalization
        h_new = self.layer_norm(h_new)
        
        return h_new, h_new, c_new


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal modeling with parallel streams at different resolutions.
    Implements Section 4.1.4: Multi-Scale Temporal Modeling.
    ** BATCH-AWARE **
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        num_heads: int = 4
    ):
        super(MultiScaleTemporalEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.stream_dim = hidden_dim // 3
        
        # Three parallel temporal streams
        self.fast_stream = TemporalGNNCell(node_features, self.stream_dim, num_heads)
        self.medium_stream = TemporalGNNCell(node_features, self.stream_dim, num_heads)
        self.slow_stream = TemporalGNNCell(node_features, self.stream_dim, num_heads)
        
        # Temporal attention for weighting different scales (Equation 8)
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.stream_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index: torch.Tensor,
        states: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process multi-scale temporal sequences.
        
        Args:
            x_sequence: List of node features at different timesteps [B, N, F]
            edge_index: Graph connectivity [2, E]
            states: Dictionary of previous states for each stream
            
        Returns:
            Tuple of (fused_output_sequence [B, N, T, H_fused], updated_states)
        """
        if states is None:
            states = {
                'fast': (None, None),
                'medium': (None, None),
                'slow': (None, None)
            }
        
        outputs = []
        
        # Process each timestep
        for t, x_t in enumerate(x_sequence):
            # Fast stream (every timestep)
            fast_out, fast_h, fast_c = self.fast_stream(
                x_t, edge_index, states['fast'][0], states['fast'][1]
            )
            
            # Medium stream (every 15 timesteps)
            if t % 15 == 0:
                medium_out, medium_h, medium_c = self.medium_stream(
                    x_t, edge_index, states['medium'][0], states['medium'][1]
                )
                states['medium'] = (medium_h, medium_c)
            else:
                medium_out = states['medium'][0] if states['medium'][0] is not None else torch.zeros_like(fast_out)
            
            # Slow stream (every 150 timesteps)
            if t % 150 == 0:
                slow_out, slow_h, slow_c = self.slow_stream(
                    x_t, edge_index, states['slow'][0], states['slow'][1]
                )
                states['slow'] = (slow_h, slow_c)
            else:
                slow_out = states['slow'][0] if states['slow'][0] is not None else torch.zeros_like(fast_out)
            
            # Update fast stream state
            states['fast'] = (fast_h, fast_c)
            
            # Concatenate multi-scale features
            multi_scale = torch.cat([fast_out, medium_out, slow_out], dim=-1)
            
            # Apply temporal attention (Equation 8)
            scale_weights = self.temporal_attention(multi_scale)  # [B, N, 3]
            weighted_output = (
                scale_weights[..., 0:1] * fast_out +
                scale_weights[..., 1:2] * medium_out +
                scale_weights[..., 2:3] * slow_out
            )
            
            outputs.append(weighted_output)
        
        # Stack outputs across time
        output_sequence = torch.stack(outputs, dim=1)  # [B, T, N, H_stream]
        output_sequence = output_sequence.permute(0, 2, 1, 3)  # [B, N, T, H_stream]
        
        return output_sequence, states


class CascadePredictionGNN(nn.Module):
    """
    Complete GNN architecture for cascade failure prediction.
    Implements the full prediction layer from Section 3.4 and 4.1.
    ** BATCH-AWARE **
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(CascadePredictionGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale temporal encoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            node_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        stream_dim = hidden_dim // 3
        self.temporal_projection = nn.Linear(stream_dim, hidden_dim)
        
        # Stack of GNN layers with residual connections
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout
            )
            for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Multi-task output heads (Section 3.6)
        self.failure_probability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.failure_timing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.line_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_states: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cascade prediction.
        
        Args:
            x_sequence: List of node feature tensors [B, N, node_features] for each timestep
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [B, E, edge_features]
            temporal_states: Previous temporal states
            
        Returns:
            Dictionary containing predictions
        """
        # Embed initial features
        x_embedded = [self.node_embedding(x_t) for x_t in x_sequence]
        edge_embedded = self.edge_embedding(edge_attr)
        
        # Multi-scale temporal encoding
        temporal_output, updated_states = self.temporal_encoder(
            x_embedded, edge_index, temporal_states
        )
        
        # Use the final timestep for prediction
        h = temporal_output[:, :, -1, :]  # [B, N, stream_dim]
        
        h = self.temporal_projection(h)  # [B, N, hidden_dim]
        
        # Apply GNN layers with residual connections
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn_layer(h, edge_index)  # [B, N, H]
            h = layer_norm(h + h_new)
        
        # Multi-task predictions
        failure_prob = self.failure_probability_head(h)  # [B, N, 1]
        failure_time = self.failure_timing_head(h)  # [B, N, 1]
        voltages = self.voltage_head(h)  # [B, N, 1]
        angles = self.angle_head(h)  # [B, N, 1]
        
        # Line flow prediction
        src, dst = edge_index
        h_src = h[:, src, :]  # [B, E, H]
        h_dst = h[:, dst, :]  # [B, E, H]
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # [B, E, H*2]
        line_flows = self.line_flow_head(edge_features)  # [B, E, 1]
        
        return {
            'failure_probability': failure_prob,
            'failure_timing': failure_time,
            'voltages': voltages,
            'angles': angles,
            'line_flows': line_flows,
            'node_embeddings': h,
            'temporal_states': updated_states
        }


# ============================================================================
# STEP 4: PHYSICS-INFORMED METHODOLOGY
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function integrating power system constraints.
    Implements Equation 12 and the physics constraints from Section 4.2.
    ** BATCH-AWARE **
    """
    
    def __init__(
        self,
        lambda_powerflow: float = 0.1,
        lambda_capacity: float = 0.05,
        lambda_stability: float = 0.05,
        lambda_temporal: float = 0.02
    ):
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_capacity = lambda_capacity
        self.lambda_stability = lambda_stability
        self.lambda_temporal = lambda_temporal
    
    def power_flow_loss(
        self,
        voltages: torch.Tensor,
        angles: torch.Tensor,
        edge_index: torch.Tensor,
        conductance: torch.Tensor,
        susceptance: torch.Tensor,
        power_injection: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute power flow constraint violations (Equations 10-11). (Batch-aware)
        
        Args:
            voltages: Predicted voltage magnitudes [B, N, 1]
            angles: Predicted voltage angles [B, N, 1]
            edge_index: Graph connectivity [2, E]
            conductance: Line conductance values [B, E, 1]
            susceptance: Line susceptance values [B, E, 1]
            power_injection: True power injections [B, N, 1]
            
        Returns:
            Power flow violation loss
        """
        src, dst = edge_index
        batch_size, num_nodes, _ = voltages.shape
        
        if conductance.dim() == 3 and conductance.size(-1) > 1:
            conductance = conductance[:, :, 0:1]
        elif conductance.dim() == 2:
            conductance = conductance.unsqueeze(-1)
        elif conductance.dim() == 1:
            conductance = conductance.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
            
        if susceptance.dim() == 3 and susceptance.size(-1) > 1:
            susceptance = susceptance[:, :, 0:1]
        elif susceptance.dim() == 2:
            susceptance = susceptance.unsqueeze(-1)
        elif susceptance.dim() == 1:
            susceptance = susceptance.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Gather V_i, V_j, theta_i, theta_j
        V_i = voltages[:, src, :]  # [B, E, 1]
        V_j = voltages[:, dst, :]  # [B, E, 1]
        theta_i = angles[:, src, :]  # [B, E, 1]
        theta_j = angles[:, dst, :]  # [B, E, 1]
        
        theta_ij = theta_i - theta_j  # [B, E, 1]
        
        # AC power flow equations (Equation 10)
        P_ij = V_i * V_j * (
            conductance * torch.cos(theta_ij) + 
            susceptance * torch.sin(theta_ij)
        )  # [B, E, 1]
        
        P_ij_squeezed = P_ij.squeeze(-1)  # [B, E]
        P_calc = torch.zeros(batch_size, num_nodes, device=voltages.device)  # [B, N]
        
        # Aggregate power flows at each node
        for b in range(batch_size):
            P_calc[b].index_add_(0, src, P_ij_squeezed[b])
            P_calc[b].index_add_(0, dst, -P_ij_squeezed[b])
        
        P_calc = P_calc.unsqueeze(-1)  # [B, N, 1]
        
        # Power flow violation
        power_violation = F.mse_loss(P_calc, power_injection)
        
        return power_violation
    
    def capacity_loss(
        self,
        line_flows: torch.Tensor,
        thermal_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute thermal capacity constraint violations.
        
        Args:
            line_flows: Predicted line flows [B, E, 1]
            thermal_limits: Line thermal limits [B, E, 1]
            
        Returns:
            Capacity violation loss
        """
        # Penalize flows exceeding thermal limits
        violations = F.relu(line_flows - thermal_limits)
        return torch.mean(violations ** 2)
    
    def voltage_stability_loss(
        self,
        voltages: torch.Tensor,
        voltage_min: float = 0.95,
        voltage_max: float = 1.05
    ) -> torch.Tensor:
        """
        Compute voltage stability constraint violations (Equation 13).
        
        Args:
            voltages: Predicted voltage magnitudes [B, N, 1]
            voltage_min: Minimum acceptable voltage (per-unit)
            voltage_max: Maximum acceptable voltage (per-unit)
            
        Returns:
            Voltage stability loss
        """
        # Penalize voltages outside acceptable range
        low_violations = F.relu(voltage_min - voltages)
        high_violations = F.relu(voltages - voltage_max)
        
        return torch.mean(low_violations ** 2 + high_violations ** 2)
    
    def temporal_consistency_loss(
        self,
        predictions_t: torch.Tensor,
        predictions_t_prev: torch.Tensor,
        expected_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce temporal smoothness (Equation 9).
        
        Args:
            predictions_t: Predictions at time t
            predictions_t_prev: Predictions at time t-1
            expected_delta: Expected change based on dynamics
            
        Returns:
            Temporal consistency loss
        """
        actual_delta = predictions_t - predictions_t_prev
        return F.mse_loss(actual_delta, expected_delta)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_properties: Dict[str, torch.Tensor],
        prev_predictions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss (Equation 12).
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            graph_properties: Physical properties of the grid
            prev_predictions: Previous timestep predictions (for temporal loss)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 1. Standard prediction loss
        L_prediction = F.binary_cross_entropy(
            predictions['failure_probability'],
            targets['failure_label']
        )
        
        # Add timing loss for failed nodes
        failed_mask = targets['failure_label'] > 0.5
        if failed_mask.sum() > 0:
            time_target = targets['failure_time'].unsqueeze(1).expand_as(predictions['failure_timing'])
            
            L_timing = F.mse_loss(
                predictions['failure_timing'][failed_mask],
                time_target[failed_mask]
            )
            L_prediction = L_prediction + 0.5 * L_timing
        
        # 2. Power flow physics loss (Equations 10-11)
        L_powerflow = self.power_flow_loss(
            voltages=predictions['voltages'],
            angles=predictions['angles'],
            edge_index=graph_properties['edge_index'],
            conductance=graph_properties['conductance'],
            susceptance=graph_properties['susceptance'],
            power_injection=graph_properties['power_injection']
        )
        
        # 3. Capacity constraint loss
        L_capacity = self.capacity_loss(
            line_flows=predictions['line_flows'],
            thermal_limits=graph_properties['thermal_limits']
        )
        
        # 4. Voltage stability loss (Equation 13)
        L_stability = self.voltage_stability_loss(
            voltages=predictions['voltages']
        )
        
        # 5. Temporal consistency loss (Equation 9)
        L_temporal = torch.tensor(0.0, device=predictions['voltages'].device)
        if prev_predictions is not None:
            L_temporal = self.temporal_consistency_loss(
                predictions_t=predictions['voltages'],
                predictions_t_prev=prev_predictions['voltages'],
                expected_delta=torch.zeros_like(predictions['voltages'])
            )
        
        # Total loss (Equation 12)
        L_total = (
            L_prediction +
            self.lambda_powerflow * L_powerflow +
            self.lambda_capacity * L_capacity +
            self.lambda_stability * L_stability +
            self.lambda_temporal * L_temporal
        )
        
        # Return loss components for monitoring
        loss_components = {
            'total': L_total.item(),
            'prediction': L_prediction.item(),
            'powerflow': L_powerflow.item(),
            'capacity': L_capacity.item(),
            'stability': L_stability.item(),
            'temporal': L_temporal.item()
        }
        
        return L_total, loss_components


class PhysicsInformedFeatureExtractor(nn.Module):
    """
    Extract physics-based features for domain knowledge integration.
    Implements Section 4.2.3: Domain Knowledge Integration.
    ** BATCH-AWARE **
    """
    
    def __init__(self):
        super(PhysicsInformedFeatureExtractor, self).__init__()
    
    def compute_n1_violations(
        self,
        line_flows: torch.Tensor,
        thermal_limits: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute N-1 contingency violation indicators.
        
        Args:
            line_flows: Current line flows [B, E, 1]
            thermal_limits: Line thermal limits [B, E, 1]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            N-1 violation features [B, N, 1]
        """
        loading_ratios = line_flows / (thermal_limits + 1e-6)
        
        batch_size = loading_ratios.size(0)
        num_nodes = edge_index.max().item() + 1
        
        src, dst = edge_index
        
        if loading_ratios.dim() == 3:
            batch_loading = loading_ratios.squeeze(-1)  # [B, E]
        else:
            batch_loading = loading_ratios
        
        # Compute max loading per node for each batch
        max_loading_list = []
        for b in range(batch_size):
            loading_b = batch_loading[b]  # [E]
            if loading_b.dim() > 1:
                loading_b = loading_b.squeeze()
            
            max_loading_b = torch.zeros(num_nodes, device=line_flows.device)
            
            for i in range(loading_b.size(0)):
                src_node = src[i].item()
                if loading_b[i] > max_loading_b[src_node]:
                    max_loading_b[src_node] = loading_b[i]
            
            max_loading_list.append(max_loading_b)
        
        max_loading = torch.stack(max_loading_list, dim=0).unsqueeze(-1)  # [B, N, 1]

        # Binary indicator: 1 if any N-1 contingency causes overload
        n1_violations = (max_loading > 1.0).float()
        
        return n1_violations
    
    def compute_voltage_stability_index(
        self,
        voltages: torch.Tensor,
        edge_index: torch.Tensor,
        susceptance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute voltage stability index L_i (Equation 13). (Batch-aware)
        
        Args:
            voltages: Voltage magnitudes [B, N, 1]
            edge_index: Graph connectivity [2, E]
            susceptance: Line susceptance [B, E, 1]
            
        Returns:
            Voltage stability indices [B, N, 1]
        """
        src, dst = edge_index
        batch_size, num_nodes, _ = voltages.shape

        if susceptance.dim() == 3 and susceptance.size(-1) > 1:
            susceptance = susceptance[:, :, 0:1]
        elif susceptance.dim() == 2:
            susceptance = susceptance.unsqueeze(-1)
        elif susceptance.dim() == 1:
            susceptance = susceptance.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

        # Gather V_i and V_j
        V_i = voltages[:, src, :]  # [B, E, 1]
        V_j = voltages[:, dst, :]  # [B, E, 1]

        # F_ij approximation using susceptance
        F_ij = susceptance * V_j / (V_i + 1e-6)  # [B, E, 1]

        F_ij_squeezed = F_ij.squeeze(-1)  # [B, E]
        stability_sum = torch.zeros(batch_size, num_nodes, device=voltages.device)  # [B, N]
        
        # Aggregate stability factors at each node
        for b in range(batch_size):
            stability_sum[b].index_add_(0, src, F_ij_squeezed[b])
        
        stability_sum = stability_sum.unsqueeze(-1)  # [B, N, 1]

        # L_i = |1 - sum(F_ij)|
        L_i = torch.abs(1.0 - stability_sum)
        
        return L_i
    
    def compute_dynamic_line_ratings(
        self,
        base_ratings: torch.Tensor,
        ambient_temp: torch.Tensor,
        wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic thermal limits based on weather (Equation 14).
        
        Args:
            base_ratings: Base thermal ratings [B, E, 1]
            ambient_temp: Ambient temperature [B, E, 1] in Celsius
            wind_speed: Wind speed [B, E, 1] in m/s
            
        Returns:
            Dynamic thermal limits [B, E, 1]
        """
        # Simplified dynamic rating model
        temp_factor = 1.0 - 0.01 * (ambient_temp - 25.0)
        wind_factor = 1.0 + 0.05 * wind_speed
        
        dynamic_ratings = base_ratings * temp_factor * wind_factor
        
        return torch.clamp(dynamic_ratings, min=base_ratings * 0.8, max=base_ratings * 1.3)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        graph_properties: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract and concatenate all physics-based features.
        
        Args:
            node_features: Base node features [B, N, F]
            edge_index: Graph connectivity [2, E]
            graph_properties: Dictionary of physical properties
            
        Returns:
            Augmented node features [B, N, F + physics_features]
        """
        # Extract voltages from node features
        voltages = node_features[:, :, 0:1]  # [B, N, 1]
        
        batch_size = node_features.size(0)
        num_edges = edge_index.size(1)
        
        def prep_prop(key):
            prop = graph_properties.get(key)
            if prop is None:
                return torch.zeros(batch_size, num_edges, 1, device=node_features.device)
            return prop

        line_flows = prep_prop('line_flows')
        thermal_limits = prep_prop('thermal_limits')
        susceptance = prep_prop('susceptance')
        
        # Compute physics-based features
        n1_violations = self.compute_n1_violations(
            line_flows=line_flows,
            thermal_limits=thermal_limits,
            edge_index=edge_index
        )
        
        voltage_stability = self.compute_voltage_stability_index(
            voltages=voltages,
            edge_index=edge_index,
            susceptance=susceptance
        )
        
        # Concatenate physics features with original features
        augmented_features = torch.cat([
            node_features,
            n1_violations,
            voltage_stability
        ], dim=-1)
        
        return augmented_features


# ============================================================================
# COMPLETE CASCADE PREDICTION MODEL
# ============================================================================

class CompleteCascadePredictionModel(nn.Module):
    """
    Complete cascade failure prediction system integrating GNN and physics.
    This is the full model combining Steps 3 and 4.
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        lambda_powerflow: float = 0.1,
        lambda_capacity: float = 0.05,
        lambda_stability: float = 0.05
    ):
        super(CompleteCascadePredictionModel, self).__init__()
        
        # Physics-informed feature extractor
        self.physics_extractor = PhysicsInformedFeatureExtractor()
        
        # Calculate augmented feature dimension
        augmented_node_features = node_features + 2
        
        # Core GNN prediction model
        self.gnn_model = CascadePredictionGNN(
            node_features=augmented_node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Physics-informed loss function
        self.physics_loss = PhysicsInformedLoss(
            lambda_powerflow=lambda_powerflow,
            lambda_capacity=lambda_capacity,
            lambda_stability=lambda_stability
        )
    
    def forward(
        self,
        x_sequence: List[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_properties: Dict[str, torch.Tensor],
        temporal_states: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with physics-informed features.
        
        Args:
            x_sequence: List of node features at each timestep [B, N, F]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [B, E, F_edge]
            graph_properties: Physical properties of the grid
            temporal_states: Previous temporal states
            
        Returns:
            Dictionary of predictions
        """
        # Augment features with physics-based domain knowledge
        augmented_sequence = []
        for x_t in x_sequence:
            x_augmented = self.physics_extractor(
                node_features=x_t,
                edge_index=edge_index,
                graph_properties=graph_properties
            )
            augmented_sequence.append(x_augmented)
        
        # Forward pass through GNN
        predictions = self.gnn_model(
            x_sequence=augmented_sequence,
            edge_index=edge_index,
            edge_attr=edge_attr,
            temporal_states=temporal_states,
        )
        
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_properties: Dict[str, torch.Tensor],
        prev_predictions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            graph_properties: Physical properties
            prev_predictions: Previous predictions for temporal loss
            
        Returns:
            Tuple of (loss, loss_components)
        """
        return self.physics_loss(
            predictions=predictions,
            targets=targets,
            graph_properties=graph_properties,
            prev_predictions=prev_predictions
        )
