"""
Unified Cascade Failure Prediction Model (Backward Compatibility Wrapper)
==========================================================================
DEPRECATED: This file is maintained for backward compatibility only.

Please use the new modular structure:
    from cascade_prediction.models import UnifiedCascadePredictionModel

The new structure provides:
- Better code organization
- Easier maintenance
- Modular components
- Improved testing capabilities
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "multimodal_cascade_model.py is deprecated. "
    "Please use: from cascade_prediction.models import UnifiedCascadePredictionModel",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular structure
from cascade_prediction.models.embeddings import (
    EnvironmentalEmbedding,
    InfrastructureEmbedding,
    RoboticEmbedding,
)

from cascade_prediction.models.layers import (
    GraphAttentionLayer,
    TemporalGNNCell,
)

from cascade_prediction.models.loss import PhysicsInformedLoss

from cascade_prediction.models import UnifiedCascadePredictionModel

# Export for backward compatibility
__all__ = [
    'EnvironmentalEmbedding',
    'InfrastructureEmbedding',
    'RoboticEmbedding',
    'GraphAttentionLayer',
    'TemporalGNNCell',
    'PhysicsInformedLoss',
    'UnifiedCascadePredictionModel',
]
