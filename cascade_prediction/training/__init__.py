"""
Training Package
================

This package provides training utilities for the cascade prediction model.

Modules:
--------
- trainer: Main training loop and model management
- calibration: Dynamic loss weight calibration utilities

Classes:
--------
- Trainer: Training manager for cascade prediction model

Functions:
----------
- calibrate_loss_weights: Calibrate physics-informed loss weights dynamically
- apply_calibrated_weights: Apply calibrated weights to base lambda values
"""

from .trainer import Trainer
from .calibration import calibrate_loss_weights, apply_calibrated_weights

__all__ = ['Trainer', 'calibrate_loss_weights', 'apply_calibrated_weights']
