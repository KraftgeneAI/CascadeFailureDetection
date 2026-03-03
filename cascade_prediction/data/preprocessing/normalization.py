"""
Normalization Module
====================
Physics-based normalization functions for power system data.

This module provides normalization utilities for:
- Power values (MW to per-unit using base MVA)
- Frequency values (Hz to per-unit using base frequency)
"""

import torch
import numpy as np
from typing import Union


def normalize_power(
    power_values: Union[torch.Tensor, np.ndarray],
    base_mva: float = 100.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize power values to per-unit using base MVA.
    
    Args:
        power_values: Power values in MW
        base_mva: Base MVA for normalization (default: 100.0)
    
    Returns:
        Normalized power values in per-unit
    """
    return power_values / base_mva


def normalize_frequency(
    frequency_values: Union[torch.Tensor, np.ndarray],
    base_frequency: float = 60.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize frequency values to per-unit using base frequency.
    
    Args:
        frequency_values: Frequency values in Hz
        base_frequency: Base frequency for normalization (default: 60.0 Hz)
    
    Returns:
        Normalized frequency values in per-unit
    """
    return frequency_values / base_frequency


def denormalize_power(
    power_pu: Union[torch.Tensor, np.ndarray],
    base_mva: float = 100.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize power values from per-unit to MW.
    
    Args:
        power_pu: Power values in per-unit
        base_mva: Base MVA for denormalization (default: 100.0)
    
    Returns:
        Power values in MW
    """
    return power_pu * base_mva


def denormalize_frequency(
    frequency_pu: Union[torch.Tensor, np.ndarray],
    base_frequency: float = 60.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize frequency values from per-unit to Hz.
    
    Args:
        frequency_pu: Frequency values in per-unit
        base_frequency: Base frequency for denormalization (default: 60.0 Hz)
    
    Returns:
        Frequency values in Hz
    """
    return frequency_pu * base_frequency
