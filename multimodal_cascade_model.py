"""
Unified Cascade Failure Prediction Model
=========================================
Combines ALL features:
1. Graph Neural Networks (GNN) with graph attention
2. Physics-informed learning (power flow, stability constraints)
3. Multi-modal data fusion (environmental, infrastructure, robotic)
4. Temporal dynamics with LSTM
5. Seven-dimensional risk assessment


Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional, Tuple, Dict


# ============================================================================
# MULTI-MODAL EMBEDDING NETWORKS (Section 3.3.2)
# ============================================================================

class EnvironmentalEmbedding(nn.Module):
    """Embedding network for environmental data (φ_env)."""
    
    def __init__(self, satellite_channels: int = 12, weather_features: int = 8,
                 threat_features: int = 6, embedding_dim: int = 128):
        super(EnvironmentalEmbedding, self).__init__()
        
        # Satellite imagery CNN
        self.satellite_cnn = nn.Sequential(
            nn.Conv2d(satellite_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Weather temporal processing
        self.weather_lstm = nn.LSTM(weather_features, 32, num_layers=1, batch_first=True)
        
        # Threat encoder
        self.threat_encoder = nn.Sequential(
            nn.Linear(threat_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, satellite_data: torch.Tensor, weather_sequence: torch.Tensor,
                threat_indicators: torch.Tensor) -> torch.Tensor:
        B, N = satellite_data.size(0), satellite_data.size(1)
        
        # Process satellite imagery
        sat_flat = satellite_data.reshape(B * N, *satellite_data.shape[2:])
        sat_features = self.satellite_cnn(sat_flat).reshape(B, N, 32)
        
        # Process weather sequences
        weather_flat = weather_sequence.reshape(B * N, *weather_sequence.shape[2:])
        _, (weather_hidden, _) = self.weather_lstm(weather_flat)
        weather_features = weather_hidden[-1].reshape(B, N, 32)
        
        # Process threat indicators
        threat_features = self.threat_encoder(threat_indicators)
        
        # Fuse all environmental modalities
        combined = torch.cat([sat_features, weather_features, threat_features], dim=-1)
        return self.fusion(combined)


class InfrastructureEmbedding(nn.Module):
    """Embedding network for infrastructure data (φ_infra)."""
    
    def __init__(self, scada_features: int = 20, pmu_features: int = 15,
                 equipment_features: int = 10, embedding_dim: int = 128):
        super(InfrastructureEmbedding, self).__init__()
        
        self.scada_encoder = nn.Sequential(
            nn.Linear(scada_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.pmu_projection = nn.Sequential(
            nn.Linear(pmu_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.equipment_encoder = nn.Sequential(
            nn.Linear(equipment_features, 32),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, scada_data: torch.Tensor, pmu_sequence: torch.Tensor,
                equipment_status: torch.Tensor) -> torch.Tensor:
        scada_features = self.scada_encoder(scada_data)
        
        if pmu_sequence.dim() == 4:
            # Has time dimension: [B, N, T, features] -> average over time
            pmu_avg = pmu_sequence.mean(dim=2)  # [B, N, features]
        elif pmu_sequence.dim() == 3:
            # No time dimension (last_timestep mode): [B, N, features]
            pmu_avg = pmu_sequence
        else:
            raise ValueError(f"Unexpected pmu_sequence dimensions: {pmu_sequence.dim()}")
        
        pmu_features = self.pmu_projection(pmu_avg)
        
        equip_features = self.equipment_encoder(equipment_status)
        
        combined = torch.cat([scada_features, pmu_features, equip_features], dim=-1)
        return self.fusion(combined)


class RoboticEmbedding(nn.Module):
    """Embedding network for robotic sensor data (φ_robot)."""
    
    def __init__(self, visual_channels: int = 3, thermal_channels: int = 1,
                 sensor_features: int = 12, embedding_dim: int = 128):
        super(RoboticEmbedding, self).__init__()
        
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(visual_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.thermal_cnn = nn.Sequential(
            nn.Conv2d(thermal_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_features, 32),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 32, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, visual_data: torch.Tensor, thermal_data: torch.Tensor,
                sensor_data: torch.Tensor) -> torch.Tensor:
        B, N = visual_data.size(0), visual_data.size(1)
        
        vis_flat = visual_data.reshape(B * N, *visual_data.shape[2:])
        vis_features = self.visual_cnn(vis_flat).reshape(B, N, 32)
        
        therm_flat = thermal_data.reshape(B * N, *thermal_data.shape[2:])
        therm_features = self.thermal_cnn(therm_flat).reshape(B, N, 16)
        
        sensor_features = self.sensor_encoder(sensor_data)
        
        combined = torch.cat([vis_features, therm_features, sensor_features], dim=-1)
        return self.fusion(combined)


# ============================================================================
# GRAPH ATTENTION LAYER WITH PHYSICS (Section 3.4 + 4.1)
# ============================================================================

class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Network layer with physics-aware message passing.
    Implements Equations 2, 3, 4 from the paper.
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 concat: bool = True, dropout: float = 0.1, edge_dim: Optional[int] = None):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.register_parameter('lin_edge', None)
            self.register_parameter('att_edge', None)
        
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H, C_out = self.heads, self.out_channels
        
        x_flat = x.reshape(B * N, C)
        x_transformed = self.lin(x_flat).reshape(B * N, H, C_out)
        
        edge_attr_transformed = None
        if edge_attr is not None and self.lin_edge is not None:
            B_e, E, edge_dim = edge_attr.shape
            edge_attr_flat = edge_attr.reshape(B_e * E, edge_dim)
            edge_attr_transformed = self.lin_edge(edge_attr_flat).reshape(B_e * E, H, C_out)
        
        # Create batched edge_index
        edge_index_batched = []
        for b in range(B):
            edge_index_batched.append(edge_index + b * N)
        edge_index_batched = torch.cat(edge_index_batched, dim=1)
        
        if True:  # add_self_loops
            edge_index_batched, _ = add_self_loops(edge_index_batched, num_nodes=B * N)
            if edge_attr_transformed is not None:
                num_self_loops = B * N
                self_loop_attr = torch.zeros(num_self_loops, H, C_out, device=edge_attr_transformed.device)
                edge_attr_transformed = torch.cat([edge_attr_transformed, self_loop_attr], dim=0)
        
        out = self.propagate(
            edge_index_batched,
            x=x_transformed,
            edge_attr=edge_attr_transformed,
            size=(B * N, B * N)
        )
        
        out = out.reshape(B, N, H, C_out)
        
        if self.concat:
            out = out.reshape(B, N, H * C_out)
        else:
            out = out.mean(dim=2)
        
        out = out + self.bias
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_index_i: torch.Tensor, size_i: Optional[int],
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        
        if edge_attr is not None and self.att_edge is not None:
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)


# ============================================================================
# TEMPORAL GNN WITH LSTM (Section 4.1.4)
# ============================================================================

class TemporalGNNCell(nn.Module):
    """Temporal GNN Cell combining graph attention with LSTM."""
    
    def __init__(self, node_features: int, hidden_dim: int,
                 edge_dim: Optional[int] = None, num_heads: int = 4, dropout: float = 0.1):
        super(TemporalGNNCell, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gat_out_channels_per_head = hidden_dim // num_heads
        self.gat_out_dim = num_heads * self.gat_out_channels_per_head
        
        self.gat = GraphAttentionLayer(
            in_channels=node_features,
            out_channels=self.gat_out_channels_per_head,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        if self.gat_out_dim != hidden_dim:
            self.projection = nn.Linear(self.gat_out_dim, hidden_dim)
        else:
            self.projection = None
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,
            num_layers=3,  # Increased from 1 to 3 layers
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape
        
        spatial_features = self.gat(x, edge_index, edge_attr)
        
        if self.projection is not None:
            spatial_features = self.projection(spatial_features)
        
        if h_prev is None:
            h_prev = (
                torch.zeros(3, B * N, self.hidden_dim, device=x.device),  # 3 layers
                torch.zeros(3, B * N, self.hidden_dim, device=x.device)
            )
        
        spatial_flat = spatial_features.reshape(B * N, 1, self.hidden_dim)
        
        output, (h_new, c_new) = self.lstm(spatial_flat, h_prev)
        
        h_out = output.squeeze(1).reshape(B, N, self.hidden_dim)
        h_out = self.layer_norm(h_out)
        
        return h_out, (h_new, c_new)


class RelayTimingModel(nn.Module):
    """
    Models deterministic relay operations with inverse-time characteristics.
    Implements IEEE inverse-time overcurrent relay curves.
    """
    
    def __init__(self, hidden_dim: int):
        super(RelayTimingModel, self).__init__()
        
        # Predict relay parameters
        self.time_dial_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # Changed from hidden_dim to hidden_dim * 2
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Time dial: 0-1
        )
        
        self.pickup_current_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # Changed from hidden_dim to hidden_dim * 2
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Pickup current > 0
        )
        
        # Relay curve constants (IEEE standard)
        self.register_buffer('curve_A', torch.tensor(0.14))
        self.register_buffer('curve_B', torch.tensor(0.02))
        self.register_buffer('curve_p', torch.tensor(0.02))
    
    def compute_operating_time(self, current: torch.Tensor, time_dial: torch.Tensor,
                               pickup_current: torch.Tensor) -> torch.Tensor:
        """
        Compute relay operating time using inverse-time curve.
        
        Args:
            current: Line current [batch_size, num_edges, 1]
            time_dial: Time dial setting [batch_size, num_edges, 1]
            pickup_current: Pickup current setting [batch_size, num_edges, 1]
        
        Returns:
            Operating time in seconds [batch_size, num_edges, 1]
        """
        I_pu = current / (pickup_current + 1e-6)
        
        # IEEE inverse-time curve: t = TD * (A / (I^p - 1) + B)
        operating_time = time_dial * (
            self.curve_A / (torch.pow(I_pu, self.curve_p) - 1 + 1e-6) + self.curve_B
        )
        
        # Clamp to reasonable range (0.1 to 60 seconds)
        operating_time = torch.clamp(operating_time, 0.1, 60.0)
        
        return operating_time
    
    def forward(self, edge_embeddings: torch.Tensor, line_currents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict relay parameters and operating times.
        
        Args:
            edge_embeddings: Edge feature embeddings [batch_size, num_edges, hidden_dim * 2]
            line_currents: Computed line currents [batch_size, num_edges, 1]
        
        Returns:
            Dictionary with relay predictions
        """
        time_dial = self.time_dial_predictor(edge_embeddings)
        pickup_current = self.pickup_current_predictor(edge_embeddings)
        
        operating_time = self.compute_operating_time(line_currents, time_dial, pickup_current)
        
        return {
            'time_dial': time_dial,
            'pickup_current': pickup_current,
            'operating_time': operating_time,
            'will_operate': (line_currents > pickup_current).float()
        }


# ============================================================================
# PHYSICS-INFORMED LOSS (Section 4.2)
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function with power flow constraints."""
    
    def __init__(self, lambda_powerflow: float = 0.1, lambda_capacity: float = 0.05,
                 lambda_stability: float = 0.05, lambda_frequency: float = 0.08):
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_capacity = lambda_capacity
        self.lambda_stability = lambda_stability
        self.lambda_frequency = lambda_frequency  # Added frequency loss weight
    
    def power_flow_loss(self, voltages: torch.Tensor, angles: torch.Tensor,
                       edge_index: torch.Tensor, conductance: torch.Tensor,
                       susceptance: torch.Tensor, power_injection: torch.Tensor) -> torch.Tensor:
        """
        Compute power flow loss with proper dimension handling.
        
        Args:
            voltages: Node voltages [batch_size, num_nodes, 1]
            angles: Node angles [batch_size, num_nodes, 1]
            edge_index: Edge connectivity [2, num_edges]
            conductance: Edge conductance [batch_size, num_edges]
            susceptance: Edge susceptance [batch_size, num_edges]
            power_injection: Node power injection [batch_size, num_nodes, 1]
        """
        src, dst = edge_index
        batch_size, num_nodes, _ = voltages.shape
        
        # Ensure conductance and susceptance have shape [batch_size, num_edges, 1]
        if conductance.dim() == 1:
            # Shape: [num_edges] -> [1, num_edges, 1] -> [batch_size, num_edges, 1]
            conductance = conductance.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        elif conductance.dim() == 2:
            # Shape: [batch_size, num_edges] -> [batch_size, num_edges, 1]
            conductance = conductance.unsqueeze(-1)
        
        if susceptance.dim() == 1:
            susceptance = susceptance.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        elif susceptance.dim() == 2:
            susceptance = susceptance.unsqueeze(-1)
        
        # Use advanced indexing to get values for all edges in all batches
        # voltages: [batch_size, num_nodes, 1] -> V_i, V_j: [batch_size, num_edges, 1]
        V_i = voltages[:, src, :]  # [batch_size, num_edges, 1]
        V_j = voltages[:, dst, :]  # [batch_size, num_edges, 1]
        theta_i = angles[:, src, :]  # [batch_size, num_edges, 1]
        theta_j = angles[:, dst, :]  # [batch_size, num_edges, 1]
        
        # Compute angle differences
        theta_ij = theta_i - theta_j  # [batch_size, num_edges, 1]
        
        # All tensors now have shape [batch_size, num_edges, 1]
        P_ij = V_i * V_j * (conductance * torch.cos(theta_ij) + susceptance * torch.sin(theta_ij))
        # P_ij: [batch_size, num_edges, 1]
        
        P_ij_squeezed = P_ij.squeeze(-1)  # [batch_size, num_edges]
        P_calc = torch.zeros(batch_size, num_nodes, device=voltages.device)
        
        # Sum incoming and outgoing flows for each node
        for b in range(batch_size):
            P_calc[b].index_add_(0, src, P_ij_squeezed[b])
            P_calc[b].index_add_(0, dst, -P_ij_squeezed[b])
        
        P_calc = P_calc.unsqueeze(-1)  # [batch_size, num_nodes, 1]
        
        # Compute MSE between calculated and injected power
        return F.mse_loss(P_calc, power_injection)
    
    def capacity_loss(self, line_flows: torch.Tensor, thermal_limits: torch.Tensor) -> torch.Tensor:
        """
        Compute capacity constraint violations.
        
        Args:
            line_flows: Computed line flows [batch_size, num_edges, 1]
            thermal_limits: Thermal limits [batch_size, num_edges] or [num_edges]
        """
        if thermal_limits.dim() == 1:
            thermal_limits = thermal_limits.unsqueeze(0).unsqueeze(-1)
        elif thermal_limits.dim() == 2:
            thermal_limits = thermal_limits.unsqueeze(-1)
        
        violations = F.relu(line_flows - thermal_limits)
        return torch.mean(violations ** 2)
    
    def voltage_stability_loss(self, voltages: torch.Tensor,
                              voltage_min: float = 0.95, voltage_max: float = 1.05) -> torch.Tensor:
        """
        Compute voltage stability constraint violations.
        
        Args:
            voltages: Node voltages [batch_size, num_nodes, 1]
            voltage_min: Minimum allowed voltage (p.u.)
            voltage_max: Maximum allowed voltage (p.u.)
        """
        low_violations = F.relu(voltage_min - voltages)
        high_violations = F.relu(voltages - voltage_max)
        return torch.mean(low_violations ** 2 + high_violations ** 2)
    
    def frequency_loss(self, frequency: torch.Tensor, power_imbalance: torch.Tensor,
                      total_inertia: float = 5.0, nominal_freq: float = 60.0) -> torch.Tensor:
        """
        Compute frequency dynamics loss based on swing equation.
        
        Args:
            frequency: Predicted frequency [batch_size, 1]
            power_imbalance: Power generation - load [batch_size, 1]
            total_inertia: System inertia constant (seconds)
            nominal_freq: Nominal frequency (Hz)
        
        Returns:
            Frequency dynamics loss
        """
        # Swing equation: df/dt = (P_gen - P_load) / (2 * H * S_base) * f_nominal
        # For steady state: frequency deviation proportional to power imbalance
        expected_freq_deviation = power_imbalance / (2 * total_inertia) * nominal_freq
        expected_frequency = nominal_freq + expected_freq_deviation
        
        # Loss: predicted frequency should match swing equation
        return F.mse_loss(frequency, expected_frequency)
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            graph_properties: Graph properties including edge attributes
        
        Returns:
            Total loss and loss components dictionary
        """
        # Prediction loss
        L_prediction = F.binary_cross_entropy(
            predictions['failure_probability'],
            targets['failure_label']
        )
        
        edge_index = graph_properties['edge_index']
        num_edges = edge_index.shape[1]
        batch_size = predictions['voltages'].shape[0]
        
        # Physics losses with proper default handling
        L_powerflow = self.power_flow_loss(
            voltages=predictions['voltages'],
            angles=predictions['angles'],
            edge_index=edge_index,
            conductance=graph_properties.get('conductance', torch.ones(batch_size, num_edges, device=predictions['voltages'].device)),
            susceptance=graph_properties.get('susceptance', torch.ones(batch_size, num_edges, device=predictions['voltages'].device)),
            power_injection=graph_properties.get('power_injection', torch.zeros_like(predictions['voltages']))
        )
        
        L_capacity = self.capacity_loss(
            line_flows=predictions['line_flows'],
            thermal_limits=graph_properties.get('thermal_limits', torch.ones(batch_size, num_edges, device=predictions['line_flows'].device) * 1000)
        )
        
        L_stability = self.voltage_stability_loss(voltages=predictions['voltages'])
        
        L_frequency = torch.tensor(0.0, device=predictions['voltages'].device)
        if 'frequency' in predictions and 'power_imbalance' in graph_properties:
            L_frequency = self.frequency_loss(
                frequency=predictions['frequency'],
                power_imbalance=graph_properties['power_imbalance']
            )
        
        L_total = (L_prediction + 
                  self.lambda_powerflow * L_powerflow +
                  self.lambda_capacity * L_capacity + 
                  self.lambda_stability * L_stability +
                  self.lambda_frequency * L_frequency)
        
        return L_total, {
            'total': L_total.item(),
            'prediction': L_prediction.item(),
            'powerflow': L_powerflow.item(),
            'capacity': L_capacity.item(),
            'stability': L_stability.item(),
            'frequency': L_frequency.item()  # Added frequency loss tracking
        }


# ============================================================================
# UNIFIED CASCADE PREDICTION MODEL
# ============================================================================

class UnifiedCascadePredictionModel(nn.Module):
    """
    COMPLETE unified model combining:
    - Multi-modal fusion (environmental, infrastructure, robotic)
    - Physics-informed GNN with graph attention
    - Temporal dynamics with multi-layer LSTM
    - Frequency dynamics modeling
    - Deterministic relay timing
    - Multi-task prediction
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128,
                 num_gnn_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super(UnifiedCascadePredictionModel, self).__init__()
        
        # Multi-modal embeddings
        self.env_embedding = EnvironmentalEmbedding(embedding_dim=embedding_dim)
        self.infra_embedding = InfrastructureEmbedding(embedding_dim=embedding_dim)
        self.robot_embedding = RoboticEmbedding(embedding_dim=embedding_dim)
        
        # Multi-modal fusion with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=heads, dropout=dropout, batch_first=True
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
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.failure_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.failure_time_head = nn.Sequential(
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
        
        # Frequency prediction head
        self.frequency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output: 0-1, scaled to 57-63 Hz range
        )
        
        # Relay timing model
        self.relay_model = RelayTimingModel(hidden_dim=hidden_dim)
        
        # Seven-dimensional risk assessment with supervision
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7),  # Changed from 3 to 7 dimensions
            nn.Sigmoid()
        )
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all modalities.
        
        Args:
            batch: Dictionary containing all input modalities
            return_sequence: If True, process full temporal sequence
        
        Returns:
            Dictionary with predictions
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
        
        # Attention-based fusion
        multi_modal = torch.stack([env_emb, infra_emb, robot_emb], dim=2)
        B, N, M, D = multi_modal.shape
        multi_modal_flat = multi_modal.reshape(B * N, M, D)
        
        fused, _ = self.fusion_attention(
            multi_modal_flat, multi_modal_flat, multi_modal_flat
        )
        fused = fused.mean(dim=1).reshape(B, N, D)
        fused = self.fusion_norm(fused)
        
        # Edge embedding
        edge_embedded = self.edge_embedding(batch.get('edge_attr', torch.zeros(B, batch['edge_index'].shape[1], 10, device=fused.device)))
        
        if return_sequence and 'temporal_sequence' in batch:
            # Process full 60-timestep sequence
            sequence_length = batch['temporal_sequence'].shape[2]
            h_states = []
            lstm_state = None
            
            for t in range(sequence_length):
                x_t = batch['temporal_sequence'][:, :, t, :]
                h_t, lstm_state = self.temporal_gnn(x_t, batch['edge_index'], edge_embedded, lstm_state)
                h_states.append(h_t)
            
            h = torch.stack(h_states, dim=2)  # [B, N, T, D]
            h = h[:, :, -1, :]  # Use last timestep for predictions
        else:
            # Single timestep processing
            h, _ = self.temporal_gnn(fused, batch['edge_index'], edge_embedded)
        
        # Additional GNN layers
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn_layer(h, batch['edge_index'], edge_embedded)
            h = layer_norm(h + h_new)
        
        # Multi-task predictions
        failure_prob = self.failure_prob_head(h)
        voltages = self.voltage_head(h)
        angles = self.angle_head(h)
        
        h_global = h.mean(dim=1, keepdim=True)  # Global pooling
        frequency_normalized = self.frequency_head(h_global)
        frequency = 57.0 + frequency_normalized * 6.0  # Scale to 57-63 Hz
        
        # Line flow prediction
        src, dst = batch['edge_index']
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        line_flows = self.line_flow_head(edge_features)
        
        relay_predictions = self.relay_model(edge_features, line_flows)
        
        risk_scores = self.risk_head(h)  # [B, N, 7]: threat_severity, vulnerability, operational_impact, 
                                          # cascade_probability, response_complexity, public_safety, urgency
        
        return {
            'failure_probability': failure_prob,
            'failure_timing': relay_predictions['operating_time'],  # Use relay model
            'voltages': voltages,
            'angles': angles,
            'line_flows': line_flows,
            'frequency': frequency,  # Added frequency prediction
            'risk_scores': risk_scores,  # Now 7 dimensions
            'relay_time_dial': relay_predictions['time_dial'],
            'relay_pickup_current': relay_predictions['pickup_current'],
            'relay_will_operate': relay_predictions['will_operate'],
            'node_embeddings': h,
            'env_embedding': env_emb,
            'infra_embedding': infra_emb,
            'robot_embedding': robot_emb,
            'fused_embedding': fused
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with physics constraints."""
        return self.physics_loss(predictions, targets, graph_properties)
