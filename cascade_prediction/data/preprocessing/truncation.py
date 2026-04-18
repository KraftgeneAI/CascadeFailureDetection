"""
Truncation Module
=================
Sliding window truncation logic for temporal sequences.

This module implements random truncation strategies to force the model
to predict cascade failures from pre-cascade observations only, preserving
the full predictive challenge of the task.
"""

import numpy as np
from typing import Tuple


def calculate_truncation_window(
    sequence_length: int,
    cascade_start_time: int,
    is_cascade: bool,
    min_length_ratio: float = 0.3,
) -> Tuple[int, int]:
    """
    Calculate sliding window truncation indices for a sequence.

    This function implements random truncation to:
    1. Force the model to predict cascade failures from PRE-CASCADE observations
       only (sequence ends at cascade_start - 5), preserving the full predictive
       challenge.
    2. Overlap normal and cascade sequence length distributions.

    The truncation ends 5 timesteps BEFORE cascade onset, so the model never
    sees any cascade-phase data during training or validation. Predictions must
    rely on pre-cascade grid state features (loading ratios, topology-informed
    cascade susceptibility features, etc.).

    Args:
        sequence_length: Total length of the original sequence
        cascade_start_time: Time when cascade starts (for cascade scenarios)
        is_cascade: Whether this is a cascade scenario
        min_length_ratio: Minimum sequence length as ratio of original (default: 0.3)

    Returns:
        Tuple of (start_idx, end_idx) for slicing the sequence
    """
    # Define the valid range of sequence lengths
    minimum_model_length = 10
    min_len = max(minimum_model_length, int(sequence_length * min_length_ratio))

    # --- 1. Determine the END point (Truncation) ---
    if is_cascade:
        # Cascade case: end 5 steps BEFORE cascade onset — model must predict
        # cascade from pre-cascade state only (no cascade visibility).
        hard_limit = int(cascade_start_time) - 5

        # Safety check to ensure we have enough data
        if hard_limit < min_len:
            hard_limit = min_len

        # Random end point within pre-cascade window
        end_idx = np.random.randint(min_len, hard_limit + 1)
    else:
        # Normal case: End anywhere, but capped to prevent "Long=Safe" leak
        global_max_cascade_len = int(sequence_length * 0.85) - 5

        # Ensure bounds are valid
        if global_max_cascade_len < min_len:
            global_max_cascade_len = min_len + 1

        end_idx = np.random.randint(min_len, global_max_cascade_len + 1)

    # --- 2. Determine the START point (Sliding Window) ---
    max_start = end_idx - minimum_model_length

    if max_start > 0:
        start_idx = np.random.randint(0, max_start)
    else:
        start_idx = 0

    return start_idx, end_idx


def apply_truncation(
    sequence: list,
    start_idx: int,
    end_idx: int,
    min_fallback_length: int = 10
) -> list:
    """
    Apply truncation to a sequence with fallback for empty results.

    Args:
        sequence: Original sequence to truncate
        start_idx: Start index for slicing
        end_idx: End index for slicing
        min_fallback_length: Minimum length to use as fallback

    Returns:
        Truncated sequence
    """
    truncated = sequence[start_idx:end_idx]

    # Fallback for empty sequences
    if len(truncated) == 0:
        truncated = sequence[:min_fallback_length]

    return truncated
