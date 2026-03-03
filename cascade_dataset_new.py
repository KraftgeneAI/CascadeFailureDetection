"""
Cascade Dataset (Backward Compatibility Wrapper)
=================================================
DEPRECATED: This file is maintained for backward compatibility only.

Please use the new modular structure:
    from cascade_prediction.data import CascadeDataset, collate_cascade_batch

The new structure provides:
- Better code organization
- Modular preprocessing components
- Easier testing
- Improved maintainability
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "cascade_dataset.py is deprecated. "
    "Please use: from cascade_prediction.data import CascadeDataset, collate_cascade_batch",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular structure
from cascade_prediction.data import CascadeDataset, collate_cascade_batch

# Export for backward compatibility
__all__ = [
    'CascadeDataset',
    'collate_cascade_batch',
]
