"""
Trainer Module
==============
Training manager for cascade prediction model.
"""

# Standard library
import os
import json
from typing import Dict, Tuple

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Local
from .metrics import calculate_cascade_metrics, calculate_node_metrics, calculate_timing_metrics, calculate_risk_metrics
from .checkpointing import save_checkpoint, load_checkpoint


class Trainer:
    """Training manager for cascade prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        learning_rate: float = 0.0001,
        output_dir: str = "checkpoints",
        max_grad_norm: float = 5.0,
        use_amp: bool = False,
        model_outputs_logits: bool = False,
        base_mva: float = 100.0,
        base_freq: float = 60.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
