"""
Trainer Module
==============
Training manager for cascade prediction model.

This is a simplified, self-contained trainer that includes all necessary
functionality without requiring separate metrics and checkpointing modules.
"""

# Standard library
import os
import json
from typing import Dict, Optional

# Third-party
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Training manager for cascade prediction model.
    
    This trainer provides:
    - Training loop with progress tracking
    - Validation with metrics
    - Automatic checkpointing (best model, latest model)
    - Training history tracking
    """
    
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
        patience: int = 25,
        use_amp: bool = False,
        model_outputs_logits: bool = False,
        base_mva: float = 100.0,
        base_freq: float = 60.0
    ):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : nn.Module
            The model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function
        device : torch.device
            Device to train on (cuda/cpu)
        learning_rate : float
            Learning rate for optimizer
        output_dir : str
            Directory to save checkpoints
        max_grad_norm : float
            Maximum gradient norm for clipping
        patience : int
            Number of epochs to wait for improvement before early stopping
        use_amp : bool
            Whether to use Automatic Mixed Precision (AMP) for faster training
        model_outputs_logits : bool
            Whether the model outputs logits (True) or probabilities (False)
        base_mva : float
            Base MVA for physics normalization
        base_freq : float
            Base frequency (Hz) for physics normalization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.patience = patience
        self.use_amp = use_amp
        self.model_outputs_logits = model_outputs_logits
        self.base_mva = base_mva
        self.base_freq = base_freq
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_cascade_acc': [], 'val_cascade_acc': [],
            'train_cascade_f1': [], 'val_cascade_f1': [],
            'train_cascade_precision': [], 'val_cascade_precision': [],
            'train_cascade_recall': [], 'val_cascade_recall': [],
            'train_node_acc': [], 'val_node_acc': [],
            'train_node_f1': [], 'val_node_f1': [],
            'train_node_precision': [], 'val_node_precision': [],
            'train_node_recall': [], 'val_node_recall': [],
            'train_time_mae': [], 'val_time_mae': [],
            'train_risk_mse': [], 'val_risk_mse': [],
            'learning_rate': []
        }
        
        # Training state tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_timing_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Thresholds for metrics calculation
        self.cascade_threshold = 0.25
        self.node_threshold = 0.25
        
        # Model validation flag
        self._model_validated = False
