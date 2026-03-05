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
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Thresholds for metrics calculation
        self.cascade_threshold = 0.5
        self.node_threshold = 0.5
        
        # Model validation flag
        self._model_validated = False
    
    def _validate_model_outputs(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Validate that model outputs match expected format."""
        print("\n" + "="*80)
        print("MODEL OUTPUT VALIDATION (First Batch)")
        print("="*80)
        
        if 'node_failure_labels' not in batch:
            print("  ✗ Batch missing 'node_failure_labels'. Skipping validation.")
            return
        if 'edge_index' not in batch:
            print("  ✗ Batch missing 'edge_index'. Skipping validation.")
            return

        B = batch['node_failure_labels'].shape[0]
        N = batch['node_failure_labels'].shape[1]
        E = batch['edge_index'].shape[1]

        def check_shape(key, expected_dims):
            if key not in outputs:
                print(f"  ✗ {key}: Missing from model output!")
                return
            
            shape = tuple(outputs[key].shape)
            
            valid = True
            if len(shape) != len(expected_dims):
                valid = False
            else:
                for i, dim in enumerate(expected_dims):
                    if dim == 'B': valid = valid and (shape[i] == B)
                    elif dim == 'N': valid = valid and (shape[i] == N)
                    elif dim == 'E': valid = valid and (shape[i] == E)
                    elif isinstance(dim, int): valid = valid and (shape[i] == dim)

            if valid:
                print(f"  ✓ {key}: shape {shape} (Matches expected)")
            else:
                print(f"  ✗ {key}: SHAPE MISMATCH! Got {shape}, expected {expected_dims}")

        print("Checking required outputs for loss calculation...")
        check_shape('failure_probability', ('B', 'N', 1))
        check_shape('voltages', ('B', 'N', 1))
        check_shape('angles', ('B', 'N', 1))
        check_shape('line_flows', ('B', 'E', 1))
        check_shape('frequency', ('B', 1, 1))
        check_shape('risk_scores', ('B', 'N', 7))
        check_shape('cascade_timing', ('B', 'N', 1))
        check_shape('temperature', ('B', 'N', 1))
        check_shape('reactive_nodes', ('B', 'N', 1))

        if 'temporal_sequence' in batch:
            if batch['temporal_sequence'].dim() > 0:
                T = batch['temporal_sequence'].shape[1]
                print(f"\nTemporal sequence detected: B={B}, T={T}, N={N}")
                print(f"  ✓ 3-layer LSTM IS BEING UTILIZED.")
            else:
                print(f"\n  ✗ Temporal sequence is empty! Check data generator.")
        else:
            print(f"\n  ✗ No temporal sequence found! Model is in single-step mode.")
            print(f"  ✗ 3-layer LSTM is NOT being utilized effectively.")

        print("="*80 + "\n")
    
    
    def _calculate_metrics(self, outputs, batch):
        """Helper to calculate all metrics for a batch."""
        
        node_probs = outputs['failure_probability'].squeeze(-1)  # [B, N]
        node_pred = (node_probs > self.node_threshold).float()  # [B, N]
        node_labels = batch['node_failure_labels']  # [B, N]
        
        cascade_prob = node_probs.max(dim=1)[0]  # [B]
        cascade_pred = (cascade_prob > self.cascade_threshold).float()
        cascade_labels = (node_labels.max(dim=1)[0] > 0.5).float()
        
        cascade_tp = ((cascade_pred == 1) & (cascade_labels == 1)).sum().item()
        cascade_fp = ((cascade_pred == 1) & (cascade_labels == 0)).sum().item()
        cascade_tn = ((cascade_pred == 0) & (cascade_labels == 0)).sum().item()
        cascade_fn = ((cascade_pred == 0) & (cascade_labels == 1)).sum().item()
        
        node_tp = ((node_pred == 1) & (node_labels == 1)).sum().item()
        node_fp = ((node_pred == 1) & (node_labels == 0)).sum().item()
        node_tn = ((node_pred == 0) & (node_labels == 0)).sum().item()
        node_fn = ((node_pred == 0) & (node_labels == 1)).sum().item()
        
        risk_mse = 0.0
        if 'risk_scores' in outputs and 'ground_truth_risk' in batch and batch['ground_truth_risk'] is not None:
            pred_risk_agg = torch.mean(outputs['risk_scores'], dim=1)  # [B, N, 7] -> [B, 7]
            target_risk = batch['ground_truth_risk']  # [B, 7]
            risk_mse = nn.functional.mse_loss(pred_risk_agg, target_risk).item()
        
        pairwise_acc = 0.0
        valid_timing_nodes = 0
        
        if 'cascade_timing' in outputs and 'cascade_timing' in batch:
            pred_times = outputs['cascade_timing'].squeeze(-1)
            target_times = batch['cascade_timing']
            
            correct_pairs = 0
            total_pairs = 0
            
            for b in range(pred_times.shape[0]):
                p = pred_times[b]
                t = target_times[b]
                mask = t >= 0
                
                if mask.sum() < 2:
                    continue
                
                # Get indices of failed nodes
                idx = torch.where(mask)[0]
                
                # Compare every pair
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        u, v = idx[i], idx[j]
                        
                        # Skip if ground truth times are identical
                        if t[u] == t[v]:
                            continue
                        
                        total_pairs += 1
                        
                        # Check if order matches
                        if (t[u] < t[v] and p[u] < p[v]) or (t[u] > t[v] and p[u] > p[v]):
                            correct_pairs += 1
            
            if total_pairs > 0:
                pairwise_acc = correct_pairs / total_pairs
                valid_timing_nodes = 1
        
        return {
            'cascade_tp': cascade_tp, 'cascade_fp': cascade_fp,
            'cascade_tn': cascade_tn, 'cascade_fn': cascade_fn,
            'node_tp': node_tp, 'node_fp': node_fp,
            'node_tn': node_tn, 'node_fn': node_fn,
            'risk_mse': risk_mse,
            'time_mae': pairwise_acc,
            'valid_timing_nodes': valid_timing_nodes
        }
    
    def _aggregate_epoch_metrics(self, metric_sums, total_batches, total_timing_batches):
        """Helper to compute final epoch metrics from sums."""
        
        cascade_tp = metric_sums.get('cascade_tp', 0)
        cascade_fp = metric_sums.get('cascade_fp', 0)
        cascade_tn = metric_sums.get('cascade_tn', 0)
        cascade_fn = metric_sums.get('cascade_fn', 0)
        
        node_tp = metric_sums.get('node_tp', 0)
        node_fp = metric_sums.get('node_fp', 0)
        node_tn = metric_sums.get('node_tn', 0)
        node_fn = metric_sums.get('node_fn', 0)
        
        cascade_precision = cascade_tp / (cascade_tp + cascade_fp + 1e-7)
        cascade_recall = cascade_tp / (cascade_tp + cascade_fn + 1e-7)
        cascade_f1 = 2 * cascade_precision * cascade_recall / (cascade_precision + cascade_recall + 1e-7)
        cascade_acc = (cascade_tp + cascade_tn) / (cascade_tp + cascade_tn + cascade_fp + cascade_fn + 1e-7)
        
        node_precision = node_tp / (node_tp + node_fp + 1e-7)
        node_recall = node_tp / (node_tp + node_fn + 1e-7)
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-7)
        node_acc = (node_tp + node_tn) / (node_tp + node_tn + node_fp + node_fn + 1e-7)
        
        risk_mse = metric_sums.get('risk_mse', 0) / (total_batches + 1e-7)
        
        if total_timing_batches == 0:
            time_mae = 0.0
        else:
            time_mae = metric_sums.get('time_mae', 0) / total_timing_batches
        
        return {
            'cascade_acc': cascade_acc, 'cascade_f1': cascade_f1,
            'cascade_precision': cascade_precision, 'cascade_recall': cascade_recall,
            'node_acc': node_acc, 'node_f1': node_f1,
            'node_precision': node_precision, 'node_recall': node_recall,
            'risk_mse': risk_mse,
            'time_mae': time_mae
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing loss and all metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        loss_component_sums = {}
        grad_norms = []
        
        metric_sums = {}
        total_timing_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", mininterval=240.0)
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (handle graph_properties specially)
            batch_device = {}
            for k, v in batch.items():
                if k == 'graph_properties':
                    batch_device[k] = {
                        prop_k: prop_v.to(self.device) if isinstance(prop_v, torch.Tensor) else prop_v
                        for prop_k, prop_v in v.items()
                    }
                elif isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(self.device)
                else:
                    batch_device[k] = v
            
            if 'node_failure_labels' not in batch_device:
                continue
            
            self.optimizer.zero_grad()
            
            # Prepare graph_properties and targets for loss function
            graph_properties = batch_device.get('graph_properties', {})
            if 'edge_index' not in graph_properties:
                graph_properties['edge_index'] = batch_device['edge_index']
            
            targets = {
                'failure_label': batch_device['node_failure_labels'],
                'ground_truth_risk': batch_device.get('ground_truth_risk'),
                'cascade_timing': batch_device.get('cascade_timing'),
                'voltages': batch_device['scada_data'][:, -1, :, 0:1] if 'scada_data' in batch_device else None,
                'node_reactive_power': batch_device['scada_data'][:, -1, :, 3:4] if 'scada_data' in batch_device else None,
                'line_reactive_power': batch_device['edge_attr'][:, :, 6:7] if 'edge_attr' in batch_device else None,
                'active_power_line_flows': batch_device['edge_attr'][:, :, 5:6] if 'edge_attr' in batch_device else None,
            }
            
            # Forward pass with optional AMP
            if self.use_amp and self.scaler is not None:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_device)
                    
                    # Validate model outputs on first batch
                    if not self._model_validated:
                        self._validate_model_outputs(outputs, batch_device)
                        self._model_validated = True
                    
                    # Extract edge mask
                    edge_mask = batch_device.get('edge_mask')
                    if edge_mask is not None and edge_mask.dim() == 3:
                        edge_mask = edge_mask[:, -1, :]
                    
                    loss, loss_components = self.criterion(
                        outputs,
                        targets,
                        graph_properties,
                        edge_mask=edge_mask
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first)
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                grad_norms.append(grad_norm.item())
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(batch_device)
                
                # Validate model outputs on first batch
                if not self._model_validated:
                    self._validate_model_outputs(outputs, batch_device)
                    self._model_validated = True
                
                # Extract edge mask
                edge_mask = batch_device.get('edge_mask')
                if edge_mask is not None and edge_mask.dim() == 3:
                    edge_mask = edge_mask[:, -1, :]
                
                loss, loss_components = self.criterion(
                    outputs,
                    targets,
                    graph_properties,
                    edge_mask=edge_mask
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                grad_norms.append(grad_norm.item())
                
                # Optimizer step
                self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Track loss components
            for comp_name, comp_value in loss_components.items():
                loss_component_sums[comp_name] = loss_component_sums.get(comp_name, 0.0) + comp_value
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = self._calculate_metrics(outputs, batch_device)
            
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
            
            if batch_metrics['valid_timing_nodes'] > 0:
                total_timing_batches += 1
            
            # Update progress bar with running metrics
            running_metrics = self._aggregate_epoch_metrics(metric_sums, batch_idx + 1, total_timing_batches)
            pbar.set_postfix({
                'cF1': f"{running_metrics['cascade_f1']:.3f}",
                'nF1': f"{running_metrics['node_f1']:.3f}",
                'rMSE': f"{running_metrics['risk_mse']:.3f}",
                'pL': f"{loss_components.get('prediction', 0):.3f}",
                'tL': f"{loss_components.get('timing', 0):.3f}",
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_metrics = self._aggregate_epoch_metrics(metric_sums, num_batches, total_timing_batches)
        epoch_metrics['loss'] = avg_loss
        
        # Print gradient norm statistics
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        print(f"\n  Average gradient norm: {avg_grad_norm:.4f}")
        
        # Print loss components
        if loss_component_sums:
            print(f"  Average loss components:")
            for comp_name, comp_sum in loss_component_sums.items():
                avg_comp = comp_sum / num_batches
                print(f"    {comp_name}: {avg_comp:.6f}")
        
        return epoch_metrics
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing loss and all metrics for validation
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        metric_sums = {}
        total_timing_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", mininterval=240.0)
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device (handle graph_properties specially)
                batch_device = {}
                for k, v in batch.items():
                    if k == 'graph_properties':
                        batch_device[k] = {
                            prop_k: prop_v.to(self.device) if isinstance(prop_v, torch.Tensor) else prop_v
                            for prop_k, prop_v in v.items()
                        }
                    elif isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(self.device)
                    else:
                        batch_device[k] = v
                
                if 'node_failure_labels' not in batch_device:
                    continue
                
                # Prepare graph_properties and targets for loss function
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'ground_truth_risk': batch_device.get('ground_truth_risk'),
                    'cascade_timing': batch_device.get('cascade_timing'),
                    'voltages': batch_device['scada_data'][:, -1, :, 0:1] if 'scada_data' in batch_device else None,
                    'node_reactive_power': batch_device['scada_data'][:, -1, :, 3:4] if 'scada_data' in batch_device else None,
                    'line_reactive_power': batch_device['edge_attr'][:, :, 6:7] if 'edge_attr' in batch_device else None,
                    'active_power_line_flows': batch_device['edge_attr'][:, :, 5:6] if 'edge_attr' in batch_device else None,
                }
                
                # Forward pass
                outputs = self.model(batch_device)
                
                # Extract edge mask
                edge_mask = batch_device.get('edge_mask')
                if edge_mask is not None and edge_mask.dim() == 3:
                    edge_mask = edge_mask[:, -1, :]
                
                loss, loss_components = self.criterion(
                    outputs,
                    targets,
                    graph_properties,
                    edge_mask=edge_mask
                )
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(outputs, batch_device)
                
                for key, value in batch_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0) + value
                
                if batch_metrics['valid_timing_nodes'] > 0:
                    total_timing_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_metrics = self._aggregate_epoch_metrics(metric_sums, num_batches, total_timing_batches)
        epoch_metrics['loss'] = avg_loss
        
        return epoch_metrics
    
    def save_checkpoint(self, filename: str, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Parameters:
        -----------
        filename : str
            Checkpoint filename
        epoch : int
            Current epoch number
        is_best : bool
            Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filename: str) -> int:
        """
        Load model checkpoint.
        
        Parameters:
        -----------
        filename : str
            Checkpoint filename
        
        Returns:
        --------
        epoch : int
            Epoch number from checkpoint
        """
        filepath = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, save_every: int = 5) -> Dict:
        """
        Train the model for multiple epochs.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
        save_every : int
            Save checkpoint every N epochs
        
        Returns:
        --------
        history : Dict
            Training history
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Early stopping patience: {self.patience} epochs")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history - all metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_cascade_acc'].append(train_metrics['cascade_acc'])
            self.history['val_cascade_acc'].append(val_metrics['cascade_acc'])
            self.history['train_cascade_f1'].append(train_metrics['cascade_f1'])
            self.history['val_cascade_f1'].append(val_metrics['cascade_f1'])
            self.history['train_cascade_precision'].append(train_metrics['cascade_precision'])
            self.history['val_cascade_precision'].append(val_metrics['cascade_precision'])
            self.history['train_cascade_recall'].append(train_metrics['cascade_recall'])
            self.history['val_cascade_recall'].append(val_metrics['cascade_recall'])
            self.history['train_node_acc'].append(train_metrics['node_acc'])
            self.history['val_node_acc'].append(val_metrics['node_acc'])
            self.history['train_node_f1'].append(train_metrics['node_f1'])
            self.history['val_node_f1'].append(val_metrics['node_f1'])
            self.history['train_node_precision'].append(train_metrics['node_precision'])
            self.history['val_node_precision'].append(val_metrics['node_precision'])
            self.history['train_node_recall'].append(train_metrics['node_recall'])
            self.history['val_node_recall'].append(val_metrics['node_recall'])
            self.history['train_time_mae'].append(train_metrics['time_mae'])
            self.history['val_time_mae'].append(val_metrics['time_mae'])
            self.history['train_risk_mse'].append(train_metrics['risk_mse'])
            self.history['val_risk_mse'].append(val_metrics['risk_mse'])
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Cascade F1: Train={train_metrics['cascade_f1']:.4f}, Val={val_metrics['cascade_f1']:.4f}")
            print(f"  Node F1:    Train={train_metrics['node_f1']:.4f}, Val={val_metrics['node_f1']:.4f}")
            print(f"  Time MAE:   Train={train_metrics['time_mae']:.4f}, Val={val_metrics['time_mae']:.4f}")
            print(f"  Risk MSE:   Train={train_metrics['risk_mse']:.4f}, Val={val_metrics['risk_mse']:.4f}")
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                print(f"  New best model! (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(
                    f'checkpoint_epoch_{epoch}.pth',
                    epoch,
                    is_best=is_best
                )
            
            # Always save latest
            self.save_checkpoint('latest_checkpoint.pth', epoch)
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} epochs.")
                print(f"Stopping training at epoch {epoch}.")
                break
        
        # Save final history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        
        return self.history
