"""
Cascade Prediction System
==========================
Physics-informed deep learning for power grid cascade failure prediction.

This package provides a complete system for:
- Multi-modal data fusion (environmental, infrastructure, robotic sensors)
- Graph neural networks with physics-aware message passing
- Temporal dynamics modeling with LSTM
- Multi-task prediction (failure probability, timing, risk assessment)
- Physics-informed loss functions
- Training utilities with metrics and checkpointing
- Inference utilities with reporting

Main Components:
- models: Neural network architectures
- data: Dataset and preprocessing utilities
- training: Training loops, metrics, and checkpointing
- inference: Prediction, analysis, and reporting
- utils: Utility functions

Version: 1.0.0
Author: Kraftgene AI Inc.
Date: October 2025
"""

__version__ = '1.0.0'

# Import main components
from .models import UnifiedCascadePredictionModel, PhysicsInformedLoss
from .data import CascadeDataset, collate_cascade_batch

# Training and inference are available as subpackages
# from cascade_prediction.training import Trainer
# from cascade_prediction.inference import CascadePredictor

__all__ = [
    'UnifiedCascadePredictionModel',
    'PhysicsInformedLoss',
    'CascadeDataset',
    'collate_cascade_batch',
]
