"""
Threshold Optimization Utilities
=================================
Reusable helpers for finding optimal classification thresholds.
"""

import numpy as np
import torch


def find_best_f1(probs: torch.Tensor, targets: torch.Tensor):
    """
    Sweep thresholds [0.05, 0.95] and return the one with the highest F1.

    Returns:
        best_f1     : float
        best_thresh : float
    """
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs > t).float()
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        if f1 > best_f1:
            best_f1 = f1.item()
            best_thresh = t
    return best_f1, best_thresh


def find_best_fbeta(probs: torch.Tensor, targets: torch.Tensor, beta: float = 0.5):
    """
    Sweep thresholds [0.05, 0.95] and return the one with the highest F-beta score.

    beta < 1 favours precision; beta > 1 favours recall.

    Returns:
        best_score  : float
        best_thresh : float
    """
    best_score, best_thresh = 0.0, 0.5
    beta_sq = beta ** 2
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs > t).float()
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-7)
        if score > best_score:
            best_score = score.item()
            best_thresh = t
    return best_score, best_thresh
