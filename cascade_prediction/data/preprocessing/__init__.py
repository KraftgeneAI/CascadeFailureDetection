"""
Preprocessing Package
=====================
Data preprocessing utilities for cascade prediction.

This package contains:
- Normalization: Physics-based power and frequency normalization
- Truncation: Sliding window truncation for temporal sequences
- Edge Masking: Dynamic topology masking for line failures
"""

from .normalization import (
    normalize_power,
    normalize_frequency,
    denormalize_power,
    denormalize_frequency,
)

from .truncation import (
    calculate_truncation_window,
    apply_truncation,
)

from .edge_masking import (
    create_edge_mask_from_failures,
    create_edge_mask_sequence,
    to_tensor,
)

__all__ = [
    # Normalization
    'normalize_power',
    'normalize_frequency',
    'denormalize_power',
    'denormalize_frequency',
    # Truncation
    'calculate_truncation_window',
    'apply_truncation',
    # Edge Masking
    'create_edge_mask_from_failures',
    'create_edge_mask_sequence',
    'to_tensor',
]
