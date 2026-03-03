"""
Inference Utilities
==================
Helper utilities for inference operations.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
