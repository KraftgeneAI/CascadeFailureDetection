"""
Robotic Embedding Network
==========================
Processes visual data, thermal imagery, and sensor readings from robotic inspections.
"""

import torch
import torch.nn as nn

from cascade_prediction.data.generator.config import Settings


class RoboticEmbedding(nn.Module):
    """Embedding network for robotic sensor data (φ_robot)."""
    
    def __init__(
        self,
        visual_channels: int = Settings.Embedding.ROBOT_VISUAL_CHANNELS,
        thermal_channels: int = Settings.Embedding.ROBOT_THERMAL_CHANNELS,
        sensor_features: int = Settings.Embedding.ROBOT_SENSOR_FEATURES,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM
    ):
        super(RoboticEmbedding, self).__init__()

        self.visual_channels = visual_channels
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(visual_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(Settings.Embedding.DROPOUT_CNN),
            nn.MaxPool2d(2),
            nn.Conv2d(16, Settings.Embedding.ROBOT_VIS_HIDDEN, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(Settings.Embedding.DROPOUT_CNN),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.thermal_cnn = nn.Sequential(
            nn.Conv2d(thermal_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(Settings.Embedding.DROPOUT_CNN),
            nn.MaxPool2d(2),
            nn.Conv2d(8, Settings.Embedding.ROBOT_THERM_HIDDEN, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(Settings.Embedding.DROPOUT_CNN),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_features, Settings.Embedding.ROBOT_SENSOR_HIDDEN),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC)
        )

        self.fusion = nn.Sequential(
            nn.Linear(Settings.Embedding.ROBOT_FUSION_INPUT, embedding_dim),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, visual_data: torch.Tensor, thermal_data: torch.Tensor,
                sensor_data: torch.Tensor) -> torch.Tensor:
        
        # --- 1. VISUAL DATA FIX (Dynamic Channels & Shaping) ---
        if visual_data.dim() == 4:
            visual_data = visual_data.permute(0, 2, 1, 3)
            
            B, N, C, F = visual_data.shape
            side = int(F**0.5)
            if side * side == F:
                visual_data = visual_data.reshape(B, N, C, side, side)
            else:
                visual_data = visual_data.unsqueeze(-2)

        has_temporal = visual_data.dim() == 6
        
        # --- DYNAMIC LAYER ADAPTATION ---
        current_channels = visual_data.shape[3] if has_temporal else visual_data.shape[2]
        
        if current_channels != self.visual_channels:
            device = visual_data.device
            out_channels = self.visual_cnn[0].out_channels
            kernel_size = self.visual_cnn[0].kernel_size
            padding = self.visual_cnn[0].padding
            
            new_layer = nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding).to(device)
            self.visual_cnn[0] = new_layer
            self.visual_channels = current_channels

        if has_temporal:
            B, T, N = visual_data.size(0), visual_data.size(1), visual_data.size(2)
            
            vis_features_list = []
            therm_features_list = []
            
            for t in range(T):
                vis_t = visual_data[:, t, :, :, :, :]
                vis_flat = vis_t.reshape(B * N, *vis_t.shape[2:])
                vis_feat = self.visual_cnn(vis_flat).reshape(B, N, Settings.Embedding.ROBOT_VIS_HIDDEN)
                vis_features_list.append(vis_feat)

                if thermal_data.dim() == 6:
                    therm_t = thermal_data[:, t, :, :, :, :]
                    therm_flat = therm_t.reshape(B * N, *therm_t.shape[2:])
                    therm_feat = self.thermal_cnn(therm_flat).reshape(B, N, Settings.Embedding.ROBOT_THERM_HIDDEN)
                    therm_features_list.append(therm_feat)
            
            vis_features = torch.stack(vis_features_list, dim=1)
            
            if len(therm_features_list) > 0:
                therm_features = torch.stack(therm_features_list, dim=1)
            else:
                therm_features = torch.zeros(B, T, N, Settings.Embedding.ROBOT_THERM_HIDDEN, device=visual_data.device)
            
            sensor_flat = sensor_data.reshape(B * T * N, -1)
            sensor_features = self.sensor_encoder(sensor_flat).reshape(B, T, N, Settings.Embedding.ROBOT_SENSOR_HIDDEN)
            
            combined = torch.cat([vis_features, therm_features, sensor_features], dim=-1)
            combined_flat = combined.reshape(B * T * N, -1)
            fused = self.fusion(combined_flat).reshape(B, T, N, -1)
            return fused

        else:
            B, N = visual_data.size(0), visual_data.size(1)
            
            vis_flat = visual_data.reshape(B * N, *visual_data.shape[2:])
            vis_features = self.visual_cnn(vis_flat).reshape(B, N, Settings.Embedding.ROBOT_VIS_HIDDEN)
            
            if thermal_data.dim() == 4:
                thermal_data = thermal_data.permute(0, 2, 1, 3)
                B_th, N_th, C_th, F_th = thermal_data.shape
                side_th = int(F_th**0.5)
                thermal_data = thermal_data.reshape(B_th, N_th, C_th, side_th, side_th)

            therm_flat = thermal_data.reshape(B * N, *thermal_data.shape[2:])
            therm_features = self.thermal_cnn(therm_flat.reshape(-1, 1, 8, 8)).reshape(B, N, -1)
            
            sensor_flat = sensor_data.reshape(B*N, -1)
            
            if hasattr(self, 'sensor_encoder'):
                target_dim = self.sensor_encoder[0].in_features
                current_dim = sensor_flat.shape[-1]
                if current_dim != target_dim:
                    if current_dim < target_dim:
                        pad = torch.zeros(sensor_flat.shape[0], target_dim - current_dim, device=sensor_flat.device)
                        sensor_flat = torch.cat([sensor_flat, pad], dim=-1)
                    else:
                        sensor_flat = sensor_flat[:, :target_dim]
                        
            sensor_features = self.sensor_encoder(sensor_flat).reshape(B, N, Settings.Embedding.ROBOT_SENSOR_HIDDEN)
            
            combined = torch.cat([vis_features, therm_features, sensor_features], dim=-1)
            combined_flat = combined.reshape(B * N, -1)
            return self.fusion(combined_flat).reshape(B, N, -1)
