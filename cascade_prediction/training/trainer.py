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

from cascade_prediction.data.generator.config import Settings


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
        learning_rate: float = Settings.Training.LEARNING_RATE,
        output_dir: str = "checkpoints",
        max_grad_norm: float = Settings.Training.TRAINER_MAX_GRAD_NORM,
        patience: int = Settings.Training.PATIENCE,
        use_amp: bool = False,
        model_outputs_logits: bool = False,
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_freq: float = Settings.Dataset.BASE_FREQUENCY
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
            weight_decay=Settings.Training.WEIGHT_DECAY
        )

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=Settings.Training.SCHEDULER_PATIENCE
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
        
        # Training state tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_timing_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Thresholds for metrics calculation
        self.cascade_threshold = Settings.Training.CASCADE_THRESHOLD
        self.node_threshold = Settings.Training.NODE_THRESHOLD
        
        # Model validation flag
        self._model_validated = False
    
    def _prepare_targets(self, batch_device: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare targets dictionary from batch data.
        
        Args:
            batch_device: Batch data dictionary on device
        
        Returns:
            Dictionary of target tensors for loss calculation
        """
        # Extract features from the last valid timestep of each sequence
        # scada_data shape: [B, T, N, F]
        # sequence_length shape: [B]
        
        if 'scada_data' in batch_device and 'sequence_length' in batch_device:
            B = batch_device['scada_data'].shape[0]
            N = batch_device['scada_data'].shape[2]
            
            # Get the indices of the last valid timesteps (0-based indexing).
            # Clamp to actual T after truncation — sequence_length may exceed the
            # collated T when the batch was truncated to global_min_len.
            T = batch_device['scada_data'].shape[1]
            last_step_indices = (batch_device['sequence_length'] - 1).clamp(0, T - 1)  # [B]
            
            # Create batch indices for advanced indexing
            batch_indices = torch.arange(B, device=last_step_indices.device)  # [B]
            
            # Extract voltages and reactive power from last valid timestep
            # Use advanced indexing: [batch_indices, last_step_indices, :, feature_idx]
            voltages = batch_device['scada_data'][batch_indices, last_step_indices, :, 0:1]  # [B, N, 1]
            node_reactive_power = batch_device['scada_data'][batch_indices, last_step_indices, :, 3:4]  # [B, N, 1]
        else:
            voltages = None
            node_reactive_power = None
        
        targets = {
            'failure_label': batch_device['node_failure_labels'],
            'ground_truth_risk': batch_device.get('ground_truth_risk'),
            'cascade_timing': batch_device.get('cascade_timing'),
            'voltages': voltages,
            'node_reactive_power': node_reactive_power,
            'line_reactive_power': batch_device['edge_attr'][:, :, 6:7] if 'edge_attr' in batch_device else None,
            'active_power_line_flows': batch_device['edge_attr'][:, :, 5:6] if 'edge_attr' in batch_device else None,
        }
        return targets
    
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
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing loss and all metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        grad_norms = []
        loss_component_sums = {}
        
        metric_sums = {}
        total_timing_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", mininterval=240.0)
        for batch_idx, batch in enumerate(pbar):
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
            
            graph_properties = batch_device.get('graph_properties', {})
            if 'edge_index' not in graph_properties:
                graph_properties['edge_index'] = batch_device['edge_index']
            
            # Prepare targets using helper method
            targets = self._prepare_targets(batch_device)

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_device, return_sequence=True)
                    
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
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_device, return_sequence=True)
                
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
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.optimizer.step()
            
            total_loss += loss.item()
            
            for comp_name, comp_value in loss_components.items():
                loss_component_sums[comp_name] = loss_component_sums.get(comp_name, 0.0) + comp_value
            
            
            pbar.set_description(f"Training (Loss: {loss.item():.4f}, Grad: {grad_norm:.2f})")
            
            with torch.no_grad():
                batch_metrics = self._calculate_metrics(outputs, batch_device)
                
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
            if batch_metrics['valid_timing_nodes'] > 0:
                total_timing_batches += 1

            running_metrics = self._aggregate_epoch_metrics(metric_sums, batch_idx + 1, total_timing_batches)
            
            pbar.set_postfix({
                'cF1': f"{running_metrics['cascade_f1']:.3f}",
                'nF1': f"{running_metrics['node_f1']:.3f}",
                'rMSE': f"{running_metrics['risk_mse']:.3f}",
                'pL': f"{loss_components.get('prediction', 0):.3f}",
                'tL': f"{loss_components.get('timing', 0):.3f}",
            })
            

        avg_loss = total_loss / (len(self.train_loader) + 1e-7)
        epoch_metrics = self._aggregate_epoch_metrics(metric_sums, len(self.train_loader), total_timing_batches)
        epoch_metrics['loss'] = avg_loss
        epoch_metrics['timing_loss'] = loss_component_sums.get('timing', 0.0) / (len(self.train_loader) + 1e-7)

        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        print(f"\n  Average gradient norm: {avg_grad_norm:.4f}")
        if loss_component_sums:
            print(f"  Average loss components:")
            for comp_name, comp_sum in loss_component_sums.items():
                avg_comp = comp_sum / (len(self.train_loader) + 1e-7)
                print(f"    {comp_name}: {avg_comp:.6f}")
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate using dynamic threshold optimization (find_best_f1 and find_best_fbeta).
        """
        self.model.eval()
        
        total_loss = 0.0
        total_timing_loss_sum = 0.0
        
        metric_sums = {}
        total_timing_batches = 0
        
        # Collect all probabilities and labels for threshold optimization
        all_node_probs = []
        all_node_labels = []
        all_cascade_probs = []
        all_cascade_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", mininterval=240.0)
            for batch_idx, batch in enumerate(pbar):
                # 1. Move to device
                batch_device = {}
                for k, v in batch.items():
                    if k == 'graph_properties':
                        batch_device[k] = {
                            pk: pv.to(self.device) if isinstance(pv, torch.Tensor) else pv
                            for pk, pv in v.items()
                        }
                    elif isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(self.device)
                    else:
                        batch_device[k] = v
                
                if 'node_failure_labels' not in batch_device: 
                    continue

                # 2. Forward
                outputs = self.model(batch_device, return_sequence=True)
                
                # 3. Loss
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                # Prepare targets using helper method
                targets = self._prepare_targets(batch_device)
                
                # Extract edge mask
                edge_mask = batch_device.get('edge_mask')
                if edge_mask is not None and edge_mask.dim() == 3:
                    edge_mask = edge_mask[:, -1, :]

                loss, loss_components = self.criterion(outputs, targets, graph_properties, edge_mask=edge_mask)
                total_loss += loss.item()
                total_timing_loss_sum += loss_components.get('timing', 0.0)
                
                # 4. Collect probabilities and labels for threshold optimization
                node_probs = outputs['failure_probability'].squeeze(-1)  # [B, N]
                node_labels = batch_device['node_failure_labels']  # [B, N]
                
                # Flatten and collect
                all_node_probs.append(node_probs.flatten())
                all_node_labels.append(node_labels.flatten())
                
                # Cascade level
                cascade_prob = node_probs.max(dim=1)[0]  # [B]
                cascade_labels = (node_labels.max(dim=1)[0] > 0.5).float()
                
                all_cascade_probs.append(cascade_prob)
                all_cascade_labels.append(cascade_labels)
                
                # 5. Other metrics (risk, timing)
                batch_metrics = self._calculate_metrics(outputs, batch_device)
                for key, value in batch_metrics.items():
                    if key not in ['cascade_tp', 'cascade_fp', 'cascade_tn', 'cascade_fn',
                                   'node_tp', 'node_fp', 'node_tn', 'node_fn']:
                        metric_sums[key] = metric_sums.get(key, 0) + value
                
                if batch_metrics['valid_timing_nodes'] > 0:
                    total_timing_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Concatenate all probabilities and labels
        global_node_probs = torch.cat(all_node_probs)
        global_node_labels = torch.cat(all_node_labels)
        global_cascade_probs = torch.cat(all_cascade_probs)
        global_cascade_labels = torch.cat(all_cascade_labels)
        
        # Helper 1: Best F1 (Standard)
        def find_best_f1(probs, targets):
            best_f1, best_thresh = 0.0, 0.5
            for t in np.arange(0.05, 0.96, 0.05):
                preds = (probs > t).float()
                tp = (preds * targets).sum()
                fp = (preds * (1-targets)).sum()
                fn = ((1-preds) * targets).sum()
                f1 = 2*tp / (2*tp + fp + fn + 1e-7)
                if f1 > best_f1:
                    best_f1 = f1.item()
                    best_thresh = t
            return best_f1, best_thresh

        # Helper 2: Best F-beta Score (Favors Precision)
        def find_best_fbeta(probs, targets, beta=0.5):
            best_score, best_thresh = 0.0, 0.5
            beta_sq = beta**2
            
            for t in np.arange(0.05, 0.96, 0.05):
                preds = (probs > t).float()
                tp = (preds * targets).sum()
                fp = (preds * (1-targets)).sum()
                fn = ((1-preds) * targets).sum()
                
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                
                # Calculate F-beta Score
                score = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-7)
                
                if score > best_score:
                    best_score = score.item()
                    best_thresh = t
            return best_score, best_thresh

        # --- 1. Find Thresholds ---
        best_c_f1, best_c_thresh = find_best_f1(global_cascade_probs, global_cascade_labels)
        
        # Use the F-beta finder for Nodes, with beta (Precision focus)
        best_n_score, best_n_thresh = find_best_fbeta(
            global_node_probs, global_node_labels, beta=Settings.Training.FBETA
        )
        
        # --- 2. Recalculate Metrics ---
        
        # Node Metrics (using F-beta Threshold)
        final_n_preds = (global_node_probs > best_n_thresh).float()
        node_tp = (final_n_preds * global_node_labels).sum().item()
        node_fp = (final_n_preds * (1-global_node_labels)).sum().item()
        node_tn = ((1-final_n_preds) * (1-global_node_labels)).sum().item()
        node_fn = ((1-final_n_preds) * global_node_labels).sum().item()
        
        node_precision = node_tp / (node_tp + node_fp + 1e-7)
        node_recall = node_tp / (node_tp + node_fn + 1e-7)
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-7)
        node_acc = (node_tp + node_tn) / (node_tp + node_tn + node_fp + node_fn + 1e-7)
        
        # Cascade Metrics (using F1 Threshold)
        final_c_preds = (global_cascade_probs > best_c_thresh).float()
        cascade_tp = (final_c_preds * global_cascade_labels).sum().item()
        cascade_fp = (final_c_preds * (1-global_cascade_labels)).sum().item()
        cascade_tn = ((1-final_c_preds) * (1-global_cascade_labels)).sum().item()
        cascade_fn = ((1-final_c_preds) * global_cascade_labels).sum().item()
        
        cascade_precision = cascade_tp / (cascade_tp + cascade_fp + 1e-7)
        cascade_recall = cascade_tp / (cascade_tp + cascade_fn + 1e-7)
        cascade_f1 = 2 * cascade_precision * cascade_recall / (cascade_precision + cascade_recall + 1e-7)
        cascade_acc = (cascade_tp + cascade_tn) / (cascade_tp + cascade_tn + cascade_fp + cascade_fn + 1e-7)
        
        avg_loss = total_loss / (len(self.val_loader) + 1e-7)
        avg_timing_loss = total_timing_loss_sum / (len(self.val_loader) + 1e-7)

        return {
            'loss': avg_loss,
            'timing_loss': avg_timing_loss,
            'time_mae': metric_sums.get('time_mae', 0) / (total_timing_batches + 1e-7),
            'risk_mse': metric_sums.get('risk_mse', 0) / (len(self.val_loader) + 1e-7),
            
            # Metrics
            'node_f1': node_f1,
            'node_precision': node_precision,
            'node_recall': node_recall,
            'node_acc': node_acc,
            
            'cascade_f1': cascade_f1,
            'cascade_precision': cascade_precision,
            'cascade_recall': cascade_recall,
            'cascade_acc': cascade_acc,
            
            # Thresholds
            'best_cascade_threshold': best_c_thresh,
            'best_node_threshold': best_n_thresh,
        }
    
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'cascade_threshold': self.cascade_threshold,
            'node_threshold': self.node_threshold
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
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        
        # Load thresholds if available
        self.cascade_threshold = checkpoint.get('cascade_threshold', Settings.Training.CASCADE_THRESHOLD)
        self.node_threshold = checkpoint.get('node_threshold', Settings.Training.NODE_THRESHOLD)
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int):
        """Train the model and save history/plots."""
        patience_counter = 0
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Note: Scheduler still steps on TOTAL validation loss
            self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Append history (all keys)
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

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            
            # LOG VALIDATION METRICS (using dynamic thresholds)
            print(f"\n  VALIDATION METRICS (Dynamic Thresholds):")
            print(f"    Cascade F1: {val_metrics['cascade_f1']:.4f} (Thresh: {val_metrics['best_cascade_threshold']:.2f})")
            print(f"    Node F1:    {val_metrics['node_f1']:.4f} (Thresh: {val_metrics['best_node_threshold']:.2f})")
            print(f"    Node Prec:  {val_metrics['node_precision']:.4f} | Node Rec: {val_metrics['node_recall']:.4f}")
            
            # Update thresholds with dynamically found values
            self.cascade_threshold = val_metrics['best_cascade_threshold']
            self.node_threshold = val_metrics['best_node_threshold']
            
            #current_f1 = (val_metrics['cascade_f1'] + val_metrics['node_f1']) / 2.0
            # At later stage of fine-tuning node f1
            current_f1 = val_metrics['node_f1']
            
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'cascade_f1': val_metrics['cascade_f1'],
                    'node_f1': val_metrics['node_f1'],
                    'cascade_threshold': self.cascade_threshold,
                    'node_threshold': self.node_threshold,
                    'history': self.history
                }, f"{self.output_dir}/best_f1_model.pth")
                
                print(f"  ★ SAVED BEST F1 MODEL (Avg F1: {current_f1:.4f} | cF1: {val_metrics['cascade_f1']:.3f}, nF1: {val_metrics['node_f1']:.3f})")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_timing_loss': val_metrics['timing_loss'], 
                    
                    'cascade_threshold': self.cascade_threshold,
                    'node_threshold': self.node_threshold,
                    
                    'history': self.history
                }, f"{self.output_dir}/best_model.pth")
                print(f"  ✓ Saved best model (New best Val Loss: {val_metrics['loss']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # Save latest checkpoint (always includes the current fixed thresholds)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_time_mae': val_metrics['time_mae'],
                'val_timing_loss': val_metrics['timing_loss'],
                
                'cascade_threshold': self.cascade_threshold,
                'node_threshold': self.node_threshold,
                
                'history': self.history
            }, f"{self.output_dir}/latest_checkpoint.pth")
        
        self.save_history()
        self.plot_training_curves()
        
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"  Best validation Timing Loss: {self.best_val_timing_loss:.4f}")
        print(f"  Training history saved to: {self.output_dir}/training_history.json")
        print(f"  Training curves saved to: {self.output_dir}/training_curves.png")
        print(f"  Best model saved to: {self.output_dir}/best_model.pth")
        print(f"{'='*80}")
        
        return self.history

    def save_history(self):
        history_path = f"{self.output_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")
    
    def plot_training_curves(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        if not self.history['train_loss']:
            print("No history to plot. Skipping plot generation.")
            plt.close()
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['train_cascade_f1'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_cascade_f1'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Cascade Detection F1')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs, self.history['train_node_f1'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_node_f1'], 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Node Failure F1')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['train_cascade_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_cascade_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['train_cascade_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_cascade_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Cascade Precision/Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.history['train_node_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['train_node_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Node Precision/Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(epochs, self.history.get('train_time_mae', [0]*len(epochs)), 'b-', label='Train', linewidth=2)
        axes[1, 2].plot(epochs, self.history.get('val_time_mae', [0]*len(epochs)), 'r-', label='Validation', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('MAE (minutes)')
        axes[1, 2].set_title('Cascade Timing MAE')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        axes[2, 0].plot(epochs, self.history.get('train_risk_mse', [0]*len(epochs)), 'b-', label='Train', linewidth=2)
        axes[2, 0].plot(epochs, self.history.get('val_risk_mse', [0]*len(epochs)), 'r-', label='Validation', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('MSE')
        axes[2, 0].set_title('7-D Risk Score MSE')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(epochs, self.history['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('LR')
        axes[2, 1].set_title('Learning Rate Schedule')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        axes[2, 2].plot(epochs, self.history['train_cascade_acc'], 'b-', label='Train Cascade', linewidth=2)
        axes[2, 2].plot(epochs, self.history['val_cascade_acc'], 'r-', label='Val Cascade', linewidth=2)
        axes[2, 2].plot(epochs, self.history['train_node_acc'], 'b--', label='Train Node', linewidth=2, alpha=0.6)
        axes[2, 2].plot(epochs, self.history['val_node_acc'], 'r--', label='Val Node', linewidth=2, alpha=0.6)
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Accuracy')
        axes[2, 2].set_title('Accuracy Comparison')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.output_dir}/training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {plot_path}")
