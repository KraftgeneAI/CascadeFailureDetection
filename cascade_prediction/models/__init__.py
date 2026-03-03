"""
Models Package
==============
Complete cascade prediction model architecture.

This package contains:
- Embeddings: Multi-modal feature extraction
- Layers: Graph attention and temporal GNN
- Heads: Multi-task prediction heads
- Loss: Physics-informed loss functions
- Unified Model: Complete integrated model
"""

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

# Import unified model
from .unified_model import UnifiedCascadePredictionModel

__all__ = [
    # Embeddings
    'EnvironmentalEmbedding',
    'InfrastructureEmbedding',
    'RoboticEmbedding',
    # Layers
    'GraphAttentionLayer',
    'TemporalGNNCell',
    # Heads
    'FailureProbabilityHead',
    'VoltageHead',
    'AngleHead',
    'FrequencyHead',
    'TemperatureHead',
    'LineFlowHead',
    'ReactiveFlowHead',
    'ActivePowerLineFlowHead',
    'RiskHead',
    'TimingHead',
    # Loss
    'PhysicsInformedLoss',
    # Unified Model
    'UnifiedCascadePredictionModel',
]
