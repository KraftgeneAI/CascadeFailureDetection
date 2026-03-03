"""
Environmental Embedding Network
================================
Processes satellite imagery, weather sequences, and threat indicators.
"""

import torch
import torch.nn as nn


class EnvironmentalEmbedding(nn.Module):
    """Embedding network for environmental data (φ_env)."""
    
    def __init__(self, satellite_channels: int = 12, weather_features: int = 80,
                 threat_features: int = 6, embedding_dim: int = 128):
        super(EnvironmentalEmbedding, self).__init__()
        
        # Satellite imagery CNN
        self.satellite_cnn = nn.Sequential(
            nn.Conv2d(satellite_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
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
        if satellite_data.dim() != 6:
            raise ValueError(f"Expected satellite_data to have 6 dimensions [B, T, N, C, H, W], "
                           f"but got shape {satellite_data.shape}")
        
        B, T, N, C, H, W = satellite_data.shape
        
        # Reshape to [B*T*N, C, H, W] for CNN processing
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
                    padding = torch.zeros(
                        weather_reshaped.shape[0], 
                        weather_reshaped.shape[1], 
                        target_dim - current_dim,
                        device=weather_reshaped.device,
                        dtype=weather_reshaped.dtype
                    )
                    weather_reshaped = torch.cat([weather_reshaped, padding], dim=-1)
                else:
                    weather_reshaped = weather_reshaped[..., :target_dim]

            # Process through LSTM: [B*N, T, F] -> [B*N, T, 32]
            weather_out, _ = self.weather_lstm(weather_reshaped)
            # Take last timestep: [B*N, T, 32] -> [B*N, 32]
            weather_features = weather_out[:, -1, :].reshape(B, N, 32)
        else:
            weather_features = self.weather_fc(weather_reshaped.reshape(B*N, -1)).reshape(B, N, 32)

        # --- 3. THREAT INDICATORS PROCESSING ---
        threat_input = threat_indicators
        
        # Check if we need to adjust dimensions for the encoder
        if hasattr(self, 'threat_encoder'):
            if isinstance(self.threat_encoder, nn.Sequential):
                first_layer = self.threat_encoder[0]
                if isinstance(first_layer, nn.Linear):
                    target_t_dim = first_layer.in_features
                    
                    if threat_input.shape[-1] != target_t_dim:
                        if threat_input.shape[-1] < target_t_dim:
                            pad_t = torch.zeros(
                                *threat_input.shape[:-1], 
                                target_t_dim - threat_input.shape[-1], 
                                device=threat_input.device,
                                dtype=threat_input.dtype
                            )
                            threat_input = torch.cat([threat_input, pad_t], dim=-1)
                        else:
                            threat_input = threat_input[..., :target_t_dim]
        
        # Process through encoder: [B, T, N, 6] -> [B, T, N, 32]
        threat_features = self.threat_encoder(threat_input)
        
        # Aggregate over time: [B, T, N, 32] -> [B, N, 32]
        if threat_features.dim() == 4:
            threat_features = threat_features.mean(dim=1)

        # --- 4. FUSION ---
        combined = torch.cat([sat_features, weather_features, threat_features], dim=-1)
        fused = self.fusion(combined)
        
        # Expand back to [B, T, N, embedding_dim] for temporal consistency
        fused = fused.unsqueeze(1).expand(-1, T, -1, -1)
             
        return fused
