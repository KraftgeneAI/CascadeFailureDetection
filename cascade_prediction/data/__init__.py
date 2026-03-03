"""
Data Package
============
Data loading and preprocessing for cascade prediction.

This package contains:
- Dataset: Main CascadeDataset class
- Collation: Batch collation functions
- Preprocessing: Normalization, truncation, and edge masking utilities
"""

from .dataset import CascadeDataset
from .collation import collate_cascade_batch
from .preprocessing import (
    normalize_power,
    normalize_frequency,
    denormalize_power,
    denormalize_frequency,
    calculate_truncation_window,
    apply_truncation,
    create_edge_mask_from_failures,
    create_edge_mask_sequence,
    to_tensor,
)

__all__ = [
    # Dataset
    'CascadeDataset',
    # Collation
    'collate_cascade_batch',
    # Preprocessing
    'normalize_power',
    'normalize_frequency',
    'denormalize_power',
    'denormalize_frequency',
    'calculate_truncation_window',
    'apply_truncation',
    'create_edge_mask_from_failures',
    'create_edge_mask_sequence',
    'to_tensor',
]
