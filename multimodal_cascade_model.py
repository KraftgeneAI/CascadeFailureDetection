"""
Unified Cascade Failure Prediction Model
=========================================
(MODIFIED to fix non-physical prediction heads)
at = sat_data.view(
Combines ALL features:
1. Graph Neural Networks (GNN) with graph attention
2. Physics-informed learning (power flow, stability constraints)
3. Multi-modal data fusion (environmental, infrastructure, robotic)
4. Temporal dynamics with LSTM
5. Seven-dimensional risk assessment

*** IMPROVEMENT: Replaced complex edge-based RelayTimingModel with a 
*** direct node-based failure_time_head for simpler and more
*** effective causal path prediction.
***
*** IMPROVEMENT 2 (CRITICAL): Fixed all physics prediction heads.
*** - Removed non-physical activations (Sigmoid, Softplus)
*** - Removed hard-coded scaling (voltage, angle, frequency)
*** - Added a dedicated head for reactive_flow.
*** This forces the model to learn the real physics.

*** IMPROVEMENT 3: DYNAMIC TOPOLOGY MASKING
*** - Added 'edge_mask' support to GAT layers to simulate line failures
*** - Allows "zeroing out" connections without rebuilding graph objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional, Tuple, Dict
import logging # Added for debug logging


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# MULTI-MODAL EMBEDDING NETWORKS (Section 3.3.2)
# ============================================================================

class EnvironmentalEmbedding(nn.Module):
    """Embedding network for environmental data (φ_env)."""
    
    def __init__(self, satellite_channels: int = 12, weather_features: int = 80,
                 threat_features: int = 6, embedding_dim: int = 128):
        super(EnvironmentalEmbedding, self).__init__()
        
        # Satellite imagery CNN
        self.satellite_cnn = nn.Sequential(
            nn.Conv2d(satellite_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Added dropout
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Added dropout
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Weather temporal processing
        self.weather_lstm = nn.LSTM(weather_features, 32, num_layers=2, batch_first=True, dropout=0.3)
        
        # Threat encoder
        self.threat_encoder = nn.Sequential(
            nn.Linear(threat_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, satellite_data: torch.Tensor, weather_sequence: torch.Tensor,
                threat_indicators: torch.Tensor) -> torch.Tensor:
        
        # --- 1. SATELLITE DATA PROCESSING ---
        # Expected input: [B, T, N, C, H, W] where C=12, H=16, W=16
        # Goal: Process each node's satellite image through CNN, aggregate over time
        
        if satellite_data.dim() != 6:
            raise ValueError(f"Expected satellite_data to have 6 dimensions [B, T, N, C, H, W], "
                           f"but got shape {satellite_data.shape}")
        
        B, T, N, C, H, W = satellite_data.shape
        
        # Reshape to [B*T*N, C, H, W] for CNN processing
        # This processes all images (across batch, time, and nodes) in parallel
        sat_input = satellite_data.reshape(B * T * N, C, H, W)
        
        # Pass through CNN: [B*T*N, C, H, W] -> [B*T*N, 32, 1, 1]
        cnn_out = self.satellite_cnn(sat_input)
        
        # Flatten spatial dimensions: [B*T*N, 32, 1, 1] -> [B*T*N, 32]
        cnn_out = cnn_out.squeeze(-1).squeeze(-1)
        
        # Reshape back to [B, T, N, 32]
        sat_features_temporal = cnn_out.reshape(B, T, N, 32)
        
        # Aggregate over time (mean pooling): [B, T, N, 32] -> [B, N, 32]
        sat_features = sat_features_temporal.mean(dim=1)

        # --- 2. WEATHER DATA PROCESSING ---
        # Expected input: [B, T, N, weather_features]
        # Goal: Process temporal weather sequences through LSTM
        
        # Handle 5D edge case [B, T, N, H, W] - flatten spatial dimensions
        if weather_sequence.dim() == 5:
            B_w, T_w, N_w, H_w, W_w = weather_sequence.shape
            weather_sequence = weather_sequence.reshape(B_w, T_w, N_w, H_w * W_w)
        
        # Reshape for LSTM: [B, T, N, F] -> [B*N, T, F]
        weather_reshaped = weather_sequence.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        
        # Pad or slice to match LSTM input size
        if hasattr(self, 'weather_lstm'):
            target_dim = self.weather_lstm.input_size
            current_dim = weather_reshaped.shape[-1]
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    # Pad with zeros
                    padding = torch.zeros(
                        weather_reshaped.shape[0], 
                        weather_reshaped.shape[1], 
                        target_dim - current_dim,
                        device=weather_reshaped.device,
                        dtype=weather_reshaped.dtype
                    )
                    weather_reshaped = torch.cat([weather_reshaped, padding], dim=-1)
                else:
                    # Slice if too large
                    weather_reshaped = weather_reshaped[..., :target_dim]

            # Process through LSTM: [B*N, T, F] -> [B*N, T, 32]
            weather_out, _ = self.weather_lstm(weather_reshaped)
            # Take last timestep: [B*N, T, 32] -> [B*N, 32]
            weather_features = weather_out[:, -1, :].reshape(B, N, 32)
        else:
            # Fallback for Linear layer
            weather_features = self.weather_fc(weather_reshaped.reshape(B*N, -1)).reshape(B, N, 32)

        # --- 3. THREAT INDICATORS PROCESSING ---
        # Expected input: [B, T, N, 6]
        # Goal: Encode threat indicators and aggregate over time
        
        threat_input = threat_indicators
        
        # Check if we need to adjust dimensions for the encoder
        if hasattr(self, 'threat_encoder'):
            # Get the expected input dimension from the first layer
            if isinstance(self.threat_encoder, nn.Sequential):
                first_layer = self.threat_encoder[0]
                if isinstance(first_layer, nn.Linear):
                    target_t_dim = first_layer.in_features
                    
                    if threat_input.shape[-1] != target_t_dim:
                        if threat_input.shape[-1] < target_t_dim:
                            # Pad with zeros
                            pad_t = torch.zeros(
                                *threat_input.shape[:-1], 
                                target_t_dim - threat_input.shape[-1], 
                                device=threat_input.device,
                                dtype=threat_input.dtype
                            )
                            threat_input = torch.cat([threat_input, pad_t], dim=-1)
                        else:
                            # Slice if too large
                            threat_input = threat_input[..., :target_t_dim]
        
        # Process through encoder: [B, T, N, 6] -> [B, T, N, 32]
        threat_features = self.threat_encoder(threat_input)
        
        # Aggregate over time: [B, T, N, 32] -> [B, N, 32]
        if threat_features.dim() == 4:
            threat_features = threat_features.mean(dim=1)

        # --- 4. FUSION ---
        # Concatenate all features: [B, N, 32] + [B, N, 32] + [B, N, 32] -> [B, N, 96]
        combined = torch.cat([sat_features, weather_features, threat_features], dim=-1)
        
        # Fuse into final embedding: [B, N, 96] -> [B, N, embedding_dim]
        fused = self.fusion(combined)
        
        # Expand back to [B, T, N, embedding_dim] for temporal consistency
        # This allows the model to process temporal sequences downstream
        fused = fused.unsqueeze(1).expand(-1, T, -1, -1)
             
        return fused
class InfrastructureEmbedding(nn.Module):
    """Embedding network for infrastructure data (φ_infra)."""
    
    def __init__(self, scada_features: int = 13, pmu_features: int = 8, 
                 equipment_features: int = 10, embedding_dim: int = 128):
        super(InfrastructureEmbedding, self).__init__()
        
        # Output: 64
        self.scada_encoder = nn.Sequential(
            nn.Linear(scada_features, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64)
        )
        
        # Output: 32
        self.pmu_projection = nn.Sequential(
            nn.Linear(pmu_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32)
        )
        
        # Output: 32
        self.equipment_encoder = nn.Sequential(
            nn.Linear(equipment_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Input: 64 + 32 + 32 = 128 -> Output: 128
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, scada_data, pmu_data, equipment_data=None):
        B, T, N, _ = scada_data.shape
        
        # --- 1. SCADA PROCESSING ---
        # Flatten: (Batch*Time*Nodes, Features)
        scada_flat = scada_data.reshape(B * T * N, -1)
        
        # Use the correct attribute name: scada_encoder
        if hasattr(self, 'scada_encoder'):
            # Dynamic input dimension handling
            target_dim = self.scada_encoder[0].in_features
            current_dim = scada_flat.shape[-1]
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    padding = torch.zeros(scada_flat.shape[0], target_dim - current_dim, device=scada_flat.device)
                    scada_flat = torch.cat([scada_flat, padding], dim=-1)
                else:
                    scada_flat = scada_flat[:, :target_dim]
            
            # Encoder outputs 64 features
            scada_features = self.scada_encoder(scada_flat).reshape(B, T, N, 64)
        else:
            # Fallback (should not happen if initialized correctly)
            scada_features = torch.zeros(B, T, N, 64, device=scada_data.device)

        # --- 2. PMU PROCESSING ---
        pmu_flat = pmu_data.reshape(B * T * N, -1)
        
        if hasattr(self, 'pmu_projection'):
            # Access first layer [0] for in_features
            target_dim = self.pmu_projection[0].in_features
            current_dim = pmu_flat.shape[-1]
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    padding = torch.zeros(pmu_flat.shape[0], target_dim - current_dim, device=pmu_flat.device)
                    pmu_flat = torch.cat([pmu_flat, padding], dim=-1)
                else:
                    pmu_flat = pmu_flat[:, :target_dim]

            # Projection outputs 32 features
            pmu_features = self.pmu_projection(pmu_flat).reshape(B, T, N, 32)
        else:
            pmu_features = torch.zeros(B, T, N, 32, device=scada_data.device)
            
        # --- 3. EQUIPMENT PROCESSING ---
        # Handle cases where equipment_data might be missing or inside *args in other implementations
        if equipment_data is not None:
            equip_flat = equipment_data.reshape(B * T * N, -1)
            
            if hasattr(self, 'equipment_encoder'):
                target_dim = self.equipment_encoder[0].in_features
                current_dim = equip_flat.shape[-1]
                
                if current_dim != target_dim:
                    if current_dim < target_dim:
                        padding = torch.zeros(equip_flat.shape[0], target_dim - current_dim, device=equip_flat.device)
                        equip_flat = torch.cat([equip_flat, padding], dim=-1)
                    else:
                        equip_flat = equip_flat[:, :target_dim]
                
                # Encoder outputs 32 features
                equip_features = self.equipment_encoder(equip_flat).reshape(B, T, N, 32)
            else:
                 equip_features = torch.zeros(B, T, N, 32, device=scada_data.device)
        else:
            # If no equipment data provided, create zeros
            equip_features = torch.zeros(B, T, N, 32, device=scada_data.device)

        # --- 4. FUSION ---
        # Combine: 64 + 32 + 32 = 128
        combined = torch.cat([scada_features, pmu_features, equip_features], dim=-1)
        
        # Flatten for the linear fusion layer
        combined_flat = combined.reshape(B * T * N, -1)
        fused = self.fusion(combined_flat)
        
        # Reshape back to [B, T, N, 128]
        return fused.reshape(B, T, N, -1)
    
class RoboticEmbedding(nn.Module):
    """Embedding network for robotic sensor data (φ_robot)."""
    
    def __init__(self, visual_channels: int = 3, thermal_channels: int = 1,
                 sensor_features: int = 12, embedding_dim: int = 128):
        super(RoboticEmbedding, self).__init__()
        
        # We will create the visual_cnn dynamically in forward if channels don't match,
        # but we initialize it here with defaults to satisfy PyTorch.
        self.visual_channels = visual_channels
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(visual_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.thermal_cnn = nn.Sequential(
            nn.Conv2d(thermal_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # FIX: Fusion layer should expect 32 + 16 + 32 = 80 input features
        # visual_cnn outputs 32, thermal_cnn outputs 16, sensor_encoder outputs 32
        self.fusion = nn.Sequential(
            nn.Linear(80, embedding_dim),  # Changed from 256 to 80
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, visual_data: torch.Tensor, thermal_data: torch.Tensor,
                sensor_data: torch.Tensor) -> torch.Tensor:
        
        # --- 1. VISUAL DATA FIX (Dynamic Channels & Shaping) ---
        # Detect strange input shape [Batch, Channels, Nodes, Features] e.g., [1, 12, 118, 64]
        if visual_data.dim() == 4:
            # Assume [B, C, N, F] -> Permute to [B, N, C, F]
            # Example: [1, 12, 118, 64] -> [1, 118, 12, 64]
            visual_data = visual_data.permute(0, 2, 1, 3)
            
            # Reshape features (64) into image (8x8)
            # Result: [B, N, C, 8, 8]
            B, N, C, F = visual_data.shape
            side = int(F**0.5) # Try to find square root (sqrt(64)=8)
            if side * side == F:
                visual_data = visual_data.reshape(B, N, C, side, side)
            else:
                # Fallback: keep as 1xFeature strip if not square
                visual_data = visual_data.unsqueeze(-2) # [B, N, C, 1, F]

        # Check temporal dimension
        has_temporal = visual_data.dim() == 6  # [B, T, N, C, H, W]
        
        # --- DYNAMIC LAYER ADAPTATION ---
        # Check actual input channels vs expected channels
        # Shape is either [B, N, C, H, W] or [B, T, N, C, H, W]
        current_channels = visual_data.shape[3] if has_temporal else visual_data.shape[2]
        
        if current_channels != self.visual_channels:
            # If the layer expects 3 but gets 12, we must replace the first layer
            device = visual_data.device
            # Retrieve the existing first layer's parameters to match standard
            out_channels = self.visual_cnn[0].out_channels
            kernel_size = self.visual_cnn[0].kernel_size
            padding = self.visual_cnn[0].padding
            
            # Replace the first layer on the fly
            new_layer = nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding).to(device)
            self.visual_cnn[0] = new_layer
            self.visual_channels = current_channels # Update state
        # --------------------------------

        if has_temporal:
            B, T, N = visual_data.size(0), visual_data.size(1), visual_data.size(2)
            
            vis_features_list = []
            therm_features_list = []
            
            for t in range(T):
                # VISUAL
                vis_t = visual_data[:, t, :, :, :, :]  # [B, N, C, H, W]
                vis_flat = vis_t.reshape(B * N, *vis_t.shape[2:]) # [B*N, C, H, W]
                vis_feat = self.visual_cnn(vis_flat).reshape(B, N, 32)
                vis_features_list.append(vis_feat)
                
                # THERMAL (Handle missing/wrong shape roughly similarly if needed)
                if thermal_data.dim() == 6:
                     therm_t = thermal_data[:, t, :, :, :, :]
                     therm_flat = therm_t.reshape(B * N, *therm_t.shape[2:])
                     therm_feat = self.thermal_cnn(therm_flat).reshape(B, N, 16)
                     therm_features_list.append(therm_feat)
            
            vis_features = torch.stack(vis_features_list, dim=1)  # [B, T, N, 32]
            
            if len(therm_features_list) > 0:
                therm_features = torch.stack(therm_features_list, dim=1)
            else:
                # Fallback if thermal data structure is weird
                therm_features = torch.zeros(B, T, N, 16, device=visual_data.device)
            
            # SENSOR
            sensor_flat = sensor_data.reshape(B * T * N, -1)
            sensor_features = self.sensor_encoder(sensor_flat).reshape(B, T, N, 32)
            
            combined = torch.cat([vis_features, therm_features, sensor_features], dim=-1)
            combined_flat = combined.reshape(B * T * N, -1)
            fused = self.fusion(combined_flat).reshape(B, T, N, -1)
            return fused

        else:
            # Original non-temporal processing
            B, N = visual_data.size(0), visual_data.size(1)
            
            # VISUAL
            # Flatten B and N: [B*N, C, H, W]
            vis_flat = visual_data.reshape(B * N, *visual_data.shape[2:])
            vis_features = self.visual_cnn(vis_flat).reshape(B, N, 32)
            
            # THERMAL
            # Check for same 4D issue on thermal
            if thermal_data.dim() == 4:
                # [B, C, N, F] -> [B, N, C, 8, 8]
                thermal_data = thermal_data.permute(0, 2, 1, 3)
                B_th, N_th, C_th, F_th = thermal_data.shape
                side_th = int(F_th**0.5)
                thermal_data = thermal_data.reshape(B_th, N_th, C_th, side_th, side_th)

            therm_flat = thermal_data.reshape(B * N, *thermal_data.shape[2:])
            # Uses .reshape() to handle non-contiguous memory automatically
            # Uses -1 in the last reshape to automatically match the model's output feature size
            therm_features = self.thermal_cnn(therm_flat.reshape(-1, 1, 8, 8)).reshape(B, N, -1)
            
            # SENSOR
            # Flatten: [B*N, F]
            sensor_flat = sensor_data.reshape(B*N, -1)
            
            # Dynamic check for sensor encoder
            if hasattr(self, 'sensor_encoder'):
                target_dim = self.sensor_encoder[0].in_features
                current_dim = sensor_flat.shape[-1]
                if current_dim != target_dim:
                    # Pad
                    if current_dim < target_dim:
                        pad = torch.zeros(sensor_flat.shape[0], target_dim - current_dim, device=sensor_flat.device)
                        sensor_flat = torch.cat([sensor_flat, pad], dim=-1)
                    else:
                        sensor_flat = sensor_flat[:, :target_dim]
                        
            sensor_features = self.sensor_encoder(sensor_flat).reshape(B, N, 32)
            
            combined = torch.cat([vis_features, therm_features, sensor_features], dim=-1)
            
            # Flatten for fusion
            combined_flat = combined.reshape(B * N, -1)
            return self.fusion(combined_flat).reshape(B, N, -1)

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
                edge_attr: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None) -> torch.Tensor: # <--- Added edge_mask
        B, N, C = x.shape
        H, C_out = self.heads, self.out_channels
        
        x_flat = x.reshape(B * N, C)
        x_transformed = self.lin(x_flat).reshape(B * N, H, C_out)
        
        edge_attr_transformed = None
        if edge_attr is not None and self.lin_edge is not None:
            # Handle both batched [B, E, D] and unbatched [E, D] edge_attr
            if edge_attr.dim() == 3:
                B_e, E, edge_dim = edge_attr.shape
                edge_attr_flat = edge_attr.reshape(B_e * E, edge_dim)
            else: # dim == 2
                E, edge_dim = edge_attr.shape
                edge_attr_flat = edge_attr
            
            edge_attr_transformed = self.lin_edge(edge_attr_flat).reshape(-1, H, C_out)
        
        # Create batched edge_index
        edge_index_batched = []
        for b in range(B):
            edge_index_batched.append(edge_index + b * N)
        edge_index_batched = torch.cat(edge_index_batched, dim=1)
        
        # Handle batched edge attributes
        if edge_attr is not None and edge_attr.dim() == 3 and edge_attr_transformed is not None:
             edge_attr_propagated = edge_attr_transformed.reshape(B*edge_attr.shape[1], H, C_out)
        else:
             edge_attr_propagated = edge_attr_transformed # Use as is (either [E,H,C] or None)

        # --- NEW: Process Edge Mask for Batching ---
        edge_mask_propagated = None
        if edge_mask is not None:
             # edge_mask comes in as [B, E]
             # Flatten to [B*E, 1] to match the batched graph
             edge_mask_propagated = edge_mask.reshape(-1, 1)
        # -------------------------------------------
        
        if True:  # add_self_loops
            edge_index_batched, _ = add_self_loops(edge_index_batched, num_nodes=B * N)
            
            # Handle Self Loop Attributes
            if edge_attr_propagated is not None:
                num_self_loops = B * N
                self_loop_attr = torch.zeros(num_self_loops, H, C_out, device=edge_attr_propagated.device)
                
                # If unbatched, expand to match batched self-loops
                if edge_attr_propagated.shape[0] == edge_attr.shape[0]: # [E, H, C]
                    edge_attr_propagated = edge_attr_propagated.repeat(B, 1, 1)

                edge_attr_propagated = torch.cat([edge_attr_propagated, self_loop_attr], dim=0)

            # --- NEW: Handle Mask for Self Loops ---
            if edge_mask_propagated is not None:
                # Self loops are always "active" (1.0)
                # We need to append B*N ones
                mask_self_loops = torch.ones(B * N, 1, device=edge_mask.device)
                edge_mask_propagated = torch.cat([edge_mask_propagated, mask_self_loops], dim=0)
            # ---------------------------------------
        
        out = self.propagate(
            edge_index_batched,
            x=x_transformed,
            edge_attr=edge_attr_propagated,
            size=(B * N, B * N),
            edge_mask=edge_mask_propagated # <--- Pass mask to propagate
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
                edge_attr: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None) -> torch.Tensor: # <--- Added edge_mask
        
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        
        if edge_attr is not None and self.att_edge is not None:
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # --- NEW: Apply Masking ---
        msg = x_j * alpha.unsqueeze(-1)
        
        if edge_mask is not None:
            # edge_mask is [Total_Edges, 1]
            # Broadcast to [Total_Edges, Heads, Channels]
            mask_broadcast = edge_mask.unsqueeze(1) 
            msg = msg * mask_broadcast # Zero out messages from failed edges
        
        return msg


# ============================================================================
# TEMPORAL GNN WITH LSTM (Section 4.1.4)
# ============================================================================

class TemporalGNNCell(nn.Module):
    """Temporal GNN Cell combining graph attention with LSTM."""
    
    def __init__(self, node_features: int, hidden_dim: int,
                 edge_dim: Optional[int] = None, num_heads: int = 4, dropout: float = 0.3):  # Increased default dropout
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
            num_layers=3,
            batch_first=True,
            dropout=0.3  # Increased from 0.1 to 0.3
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None, # <--- Added edge_mask
                h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape
        
        # Pass mask to GAT
        spatial_features = self.gat(x, edge_index, edge_attr, edge_mask=edge_mask)
        
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


# ============================================================================
# PHYSICS-INFORMED LOSS (Section 4.2)
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function with power flow constraints."""
    
    def __init__(self, lambda_powerflow: float = 0.1, lambda_capacity: float = 0.05,
                 lambda_stability: float = 0.05, lambda_frequency: float = 0.08,
                 lambda_reactive: float = 0.1, lambda_voltage: float = 1.0):
        
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_capacity = lambda_capacity
        self.lambda_stability = lambda_stability
        self.lambda_frequency = lambda_frequency  # Added frequency loss weight
        self.lambda_reactive = lambda_reactive  # New
        self.lambda_voltage = lambda_voltage    # New
    
    def power_flow_loss(self, voltages: torch.Tensor, angles: torch.Tensor,
                       edge_index: torch.Tensor, conductance: torch.Tensor,
                       susceptance: torch.Tensor, power_injection: torch.Tensor) -> torch.Tensor:
        """
        Compute power flow loss with proper dimension handling.
        
        Args:
            voltages: Node voltages [batch_size, num_nodes, 1]
            angles: Node angles [batch_size, num_nodes, 1]
            edge_index: Edge connectivity [2, num_edges]
            conductance: Edge conductance [batch_size, num_edges] or [num_edges]
            susceptance: Edge susceptance [batch_size, num_edges] or [num_edges]
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
        
        # Violations are when |line_flow| > limit
        violations = F.relu(torch.abs(line_flows) - thermal_limits)
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
    
    def reactive_power_loss(self, voltages, angles, edge_index, conductance, susceptance, reactive_injection):
        """
        Q_calc = V_i * sum( V_j * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij)) )
        """
        src, dst = edge_index
        batch_size = voltages.shape[0]

        # Ensure shapes [B, E, 1]
        if conductance.dim() == 2: conductance = conductance.unsqueeze(-1)
        if susceptance.dim() == 2: susceptance = susceptance.unsqueeze(-1)

        V_i = voltages[:, src, :]
        V_j = voltages[:, dst, :]
        theta_ij = angles[:, src, :] - angles[:, dst, :]

        # Reactive Power Flow Equation (AC Physics)
        # Note the minus sign on B_ij * cos(theta_ij)
        Q_ij = V_i * V_j * (conductance * torch.sin(theta_ij) - susceptance * torch.cos(theta_ij))
        Q_ij = Q_ij.squeeze(-1)

        Q_calc = torch.zeros(batch_size, voltages.shape[1], device=voltages.device)
        for b in range(batch_size):
            Q_calc[b].index_add_(0, src, Q_ij[b])
            Q_calc[b].index_add_(0, dst, -Q_ij[b]) # Conservation

        return F.mse_loss(Q_calc.unsqueeze(-1), reactive_injection)

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
        
        L_voltage = F.mse_loss(predictions['voltages'], targets['voltages'])

        batch_size = predictions['voltages'].shape[0]
        num_edges = graph_properties['edge_index'].shape[1]
        
        L_reactive = self.reactive_power_loss(
            voltages=predictions['voltages'],
            angles=predictions['angles'],
            edge_index=graph_properties['edge_index'],
            conductance=graph_properties.get('conductance', torch.ones(batch_size, num_edges, device=predictions['voltages'].device)),
            susceptance=graph_properties.get('susceptance', torch.ones(batch_size, num_edges, device=predictions['voltages'].device)),
            reactive_injection=graph_properties.get('reactive_injection', torch.zeros_like(predictions['voltages']))
        )

        L_total = (L_prediction + 
                   self.lambda_powerflow * L_powerflow + # Assuming you calculated L_powerflow above
                   self.lambda_capacity * L_capacity +   # Assuming L_capacity above
                   self.lambda_stability * L_stability + # Assuming L_stability above
                   self.lambda_frequency * L_frequency + # Assuming L_frequency above
                   self.lambda_voltage * L_voltage +     # <--- Add this
                   self.lambda_reactive * L_reactive)    # <--- Add this
        
        return L_total, {
            'total': L_total.item(),
            'prediction': L_prediction.item(),
            'powerflow': L_powerflow.item(),
            'capacity': L_capacity.item(),
            'stability': L_stability.item(),
            'frequency': L_frequency.item(),
            'voltage': L_voltage.item(),   # <--- Now non-zero
            'reactive': L_reactive.item()  # <--- Now non-zero
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
    - Multi-task prediction (including direct node timing)
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128,
                 num_gnn_layers: int = 3, heads: int = 4, dropout: float = 0.3):  # Increased default dropout
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
        
        # Edge embedding - Updated to accept 5 features:
        # [reactance, thermal_limits, resistance, susceptance, conductance]
        self.edge_embedding = nn.Sequential(
            nn.Linear(5, hidden_dim),  # Changed from 4 to 5
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.failure_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # ====================================================================
        # START: IMPROVEMENT - Direct Node-Timing Head
        # ====================================================================
        self.failure_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus() # Ensures time is always positive
        )
        # ====================================================================
        # END: IMPROVEMENT
        # ====================================================================

        # ====================================================================
        # START: PHYSICS HEAD FIXES
        # ====================================================================
        
        # Voltage head: Must be able to predict < 0.9 and > 1.1
        # Removed Sigmoid, replaced with ReLU (voltage is positive)
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU() # <-- FIX: Allows prediction of any positive voltage
        )
        
        # Angle head: Must predict small radians. Tanh [-1, 1] is a good
        # range for this. The scaling was the bug.
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # <-- Kept, as it's a good range for radians
        )
        
        # Line flow head: Must predict positive AND negative values.
        # Removed Softplus, now a linear output.
        self.line_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
            # <-- FIX: Removed Softplus
        )
        
        # Reactive flow head: NEW head, learns this task separately.
        # Also a linear output.
        self.reactive_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Frequency head: Must be able to predict "bad" frequencies.
        # Removed Sigmoid, replaced with ReLU.
        self.frequency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # <-- FIX: Allows prediction of any positive frequency
        )
        

        # ====================================================================
        # START: ADDITION
        # ====================================================================
        # New head for direct temperature prediction
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU() # Temperature must be positive
        )
        # ====================================================================
        # END: ADDITION
        # ====================================================================

        # ====================================================================
        # END: PHYSICS HEAD FIXES
        # ====================================================================
        
        
        # Seven-dimensional risk assessment with supervision
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_dim // 2, 7),
            nn.Sigmoid()
        )
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all modalities.
        """
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
        
        edge_attr_input = batch.get('edge_attr')
        if edge_attr_input is None:
            E = batch['edge_index'].shape[1]
            edge_attr_input = torch.zeros(env_emb.shape[0], E, 5, device=env_emb.device)  # Changed from 4 to 5
        
        if edge_attr_input.dim() == 2 and env_emb.dim() > 2:
             edge_attr_input = edge_attr_input.unsqueeze(0).expand(env_emb.shape[0], -1, -1)
        
        edge_mask_input = batch.get('edge_mask') 

        if has_temporal:
            B, T, N, D = env_emb.shape
            
            fused_list = []
            for t in range(T):
                # --- FIX 1: Broadcast Robot Embedding ---
                # robot_emb shape: [B, T, N, D]
                # Select timestep t: [B, N, D]
                robot_feat_t = robot_emb[:, t, :, :]  # Explicitly select all dimensions
                
                # robot_feat_t should now be [B, N, D], no need to unsqueeze/expand
                # It already has the correct shape to stack with env and infra

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
                
                # --- FIX 2: Handle Edge Mask Dimensions ---
                if edge_mask_input is not None and edge_mask_input.dim() == 3:
                    mask_t = edge_mask_input[:, t, :]
                else:
                    mask_t = edge_mask_input
                
                h_t, lstm_state = self.temporal_gnn(x_t, batch['edge_index'], edge_embedded, 
                                                  edge_mask=mask_t,
                                                  h_prev=lstm_state)
                h_states.append(h_t)
            
            h_stack = torch.stack(h_states, dim=2)
            
            if 'sequence_length' in batch:
                lengths = batch['sequence_length']
                h_final_list = []
                for b in range(B):
                    # --- FIX 3: Cast Tensor to Integer for Slicing ---
                    valid_idx = int(lengths[b]) - 1  # <--- CAST TO INT
                    
                    if valid_idx < 0: valid_idx = 0
                    if valid_idx >= T: valid_idx = T - 1
                    h_final_list.append(h_stack[b, :, valid_idx, :])
                h = torch.stack(h_final_list, dim=0)
            else:
                h = h_stack[:, :, -1, :]

        else:
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
            
            h, _ = self.temporal_gnn(fused, batch['edge_index'], edge_embedded, edge_mask=edge_mask_input)
        
        final_mask = edge_mask_input[:, -1, :] if (edge_mask_input is not None and edge_mask_input.dim() == 3) else edge_mask_input
        
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, batch['edge_index'], edge_embedded, edge_mask=final_mask)
            h = layer_norm(h + h_new)
        
        failure_prob = self.failure_prob_head(h)
        failure_timing = self.failure_time_head(h)
        voltages = self.voltage_head(h)
        angles = self.angle_head(h)
        
        h_global = h.mean(dim=1, keepdim=True)
        frequency = self.frequency_head(h_global)
        temperature = self.temperature_head(h)

        src, dst = batch['edge_index']
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        
        line_flows = self.line_flow_head(edge_features)
        reactive_flows = self.reactive_flow_head(edge_features)

        risk_scores = self.risk_head(h)
        
        if torch.isnan(failure_prob).any(): logging.error("[ERROR] NaN detected in failure_prob!")
        
        return {
            'failure_probability': failure_prob,
            'failure_timing': failure_timing,
            'cascade_timing': failure_timing,
            'voltages': voltages,
            'angles': angles,
            'line_flows': line_flows,
            'temperature': temperature,
            'reactive_flows': reactive_flows,
            'frequency': frequency,
            'risk_scores': risk_scores,
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
        logging.debug("Computing total loss with physics constraints")
        loss_details = self.physics_loss(predictions, targets, graph_properties)
        logging.debug(f"Loss computation complete. Total loss: {loss_details[0].item()}")
        return loss_details