"""
JSON Encoder Module
==================
Provides custom JSON encoder for NumPy types.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types.
    
    Converts NumPy integers, floats, arrays, and booleans to native Python types.
    """
    
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
