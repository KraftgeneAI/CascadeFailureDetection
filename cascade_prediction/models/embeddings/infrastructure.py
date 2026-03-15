"""
Infrastructure Embedding Network
=================================
Processes SCADA data, PMU measurements, and equipment status.
"""

import torch
import torch.nn as nn

from cascade_prediction.data.generator.config import Settings


class InfrastructureEmbedding(nn.Module):
    """Embedding network for infrastructure data (φ_infra)."""
    
    def __init__(
        self,
        scada_features: int = Settings.Embedding.INFRA_SCADA_FEATURES,
        pmu_features: int = Settings.Embedding.INFRA_PMU_FEATURES,
        equipment_features: int = Settings.Embedding.INFRA_EQUIPMENT_FEATURES,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM
    ):
        super(InfrastructureEmbedding, self).__init__()

        # SCADA encoder — output: INFRA_SCADA_HIDDEN
        self.scada_encoder = nn.Sequential(
            nn.Linear(scada_features, Settings.Embedding.INFRA_SCADA_HIDDEN),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC),
            nn.Linear(Settings.Embedding.INFRA_SCADA_HIDDEN, Settings.Embedding.INFRA_SCADA_HIDDEN)
        )

        # PMU projection — output: INFRA_PMU_HIDDEN
        self.pmu_projection = nn.Sequential(
            nn.Linear(pmu_features, Settings.Embedding.INFRA_PMU_HIDDEN),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC),
            nn.Linear(Settings.Embedding.INFRA_PMU_HIDDEN, Settings.Embedding.INFRA_PMU_HIDDEN)
        )

        # Equipment encoder — output: INFRA_EQUIP_HIDDEN
        self.equipment_encoder = nn.Sequential(
            nn.Linear(equipment_features, Settings.Embedding.INFRA_EQUIP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC)
        )

        # Fusion: SCADA + PMU + EQUIP -> embedding_dim
        fusion_input = (
            Settings.Embedding.INFRA_SCADA_HIDDEN
            + Settings.Embedding.INFRA_PMU_HIDDEN
            + Settings.Embedding.INFRA_EQUIP_HIDDEN
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, embedding_dim),
            nn.ReLU(),
            nn.Dropout(Settings.Embedding.DROPOUT_FC),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, scada_data, pmu_data, equipment_data=None):
        B, T, N, _ = scada_data.shape
        
        # --- 1. SCADA PROCESSING ---
        scada_flat = scada_data.reshape(B * T * N, -1)
        
        if hasattr(self, 'scada_encoder'):
            target_dim = self.scada_encoder[0].in_features
            current_dim = scada_flat.shape[-1]
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    padding = torch.zeros(scada_flat.shape[0], target_dim - current_dim, device=scada_flat.device)
                    scada_flat = torch.cat([scada_flat, padding], dim=-1)
                else:
                    scada_flat = scada_flat[:, :target_dim]
            
            scada_features = self.scada_encoder(scada_flat).reshape(B, T, N, Settings.Embedding.INFRA_SCADA_HIDDEN)
        else:
            scada_features = torch.zeros(B, T, N, Settings.Embedding.INFRA_SCADA_HIDDEN, device=scada_data.device)

        # --- 2. PMU PROCESSING ---
        pmu_flat = pmu_data.reshape(B * T * N, -1)
        
        if hasattr(self, 'pmu_projection'):
            target_dim = self.pmu_projection[0].in_features
            current_dim = pmu_flat.shape[-1]
            
            if current_dim != target_dim:
                if current_dim < target_dim:
                    padding = torch.zeros(pmu_flat.shape[0], target_dim - current_dim, device=pmu_flat.device)
                    pmu_flat = torch.cat([pmu_flat, padding], dim=-1)
                else:
                    pmu_flat = pmu_flat[:, :target_dim]

            pmu_features = self.pmu_projection(pmu_flat).reshape(B, T, N, Settings.Embedding.INFRA_PMU_HIDDEN)
        else:
            pmu_features = torch.zeros(B, T, N, Settings.Embedding.INFRA_PMU_HIDDEN, device=scada_data.device)
            
        # --- 3. EQUIPMENT PROCESSING ---
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
                
                equip_features = self.equipment_encoder(equip_flat).reshape(B, T, N, Settings.Embedding.INFRA_EQUIP_HIDDEN)
            else:
                equip_features = torch.zeros(B, T, N, Settings.Embedding.INFRA_EQUIP_HIDDEN, device=scada_data.device)
        else:
            equip_features = torch.zeros(B, T, N, Settings.Embedding.INFRA_EQUIP_HIDDEN, device=scada_data.device)

        # --- 4. FUSION ---
        combined = torch.cat([scada_features, pmu_features, equip_features], dim=-1)
        combined_flat = combined.reshape(B * T * N, -1)
        fused = self.fusion(combined_flat)
        
        return fused.reshape(B, T, N, -1)
