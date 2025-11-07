"""
Training Script for Cascade Prediction Model
======================================================
This script trains the UnifiedCascadePredictionModel using data generated
by multimodal_data_generator.py. It incorporates:
- Command-line arguments for hyperparameters.
- Automatic GPU (CUDA) detection.
- Weighted sampling to handle class imbalance.
- A merged physics-informed focal loss function.
- Dynamic loss weight calibration.
- Checkpointing and metrics logging.

Usage:
    # Basic training
    python train_model.py --data_dir ./data --output_dir ./checkpoints --epochs 100 --batch_size 8

    # Adjust learning rate and other parameters
    python train_model.py --data_dir ./data --epochs 50 --batch_size 16 --lr 0.001 --grad_clip 5.0

    # Resume training from a checkpoint
    python train_model.py --resume
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from typing import Dict, Tuple
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import argparse # Import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from multimodal_cascade_model import UnifiedCascadePredictionModel
    from cascade_dataset import CascadeDataset, collate_cascade_batch
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Please ensure multimodal_cascade_model.py and cascade_dataset.py are in the same directory.")
    sys.exit(1)


# ============================================================================
# MERGED & CORRECTED PHYSICS-INFORMED FOCAL LOSS
# ============================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function with FOCAL LOSS for severe class imbalance.
    
    Combines:
    - Focal Loss (from train_model.py) for class imbalance.
    - REAL Physics Constraints (from multimodal_cascade_model.py) for power flow,
      capacity, and voltage stability.
    - Physics-based frequency loss using the swing equation.***
    """
    
    def __init__(self, lambda_powerflow: float = 0.1, lambda_capacity: float = 0.1,
                 lambda_stability: float = 0.001, lambda_frequency: float = 0.1,
                 lambda_reactive: float = 0.1, 
                 pos_weight: float = 10.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.15, use_logits: bool = False,
                 base_mva: float = 100.0, base_freq: float = 60.0):
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_capacity = lambda_capacity
        self.lambda_stability = lambda_stability
        self.lambda_frequency = lambda_frequency
        self.lambda_reactive = lambda_reactive 
        
        # Focal Loss & Imbalance parameters
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_logits = use_logits
        
        self._warned_missing_outputs = set()
        
        # Constants for physics (can be overridden by graph_properties if available)
        self.power_base = base_mva
        self.freq_nominal = base_freq

    # --- Focal Loss (from original train_model.py) ---
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss with label smoothing and POS_WEIGHT for handling severe class imbalance.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) * pos_weight
        """
        # Smooth labels
        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets_smooth, 
            pos_weight=pos_weight_tensor,
            reduction='none'
        )
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

    # --- CORRECT Physics Methods (from multimodal_cascade_model.py) ---
    def power_flow_loss(self, voltages: torch.Tensor, angles: torch.Tensor,
                       edge_index: torch.Tensor, conductance: torch.Tensor,
                       susceptance: torch.Tensor, power_injection: torch.Tensor) -> torch.Tensor:
        """
        Compute REAL AC power flow loss.
        """
        src, dst = edge_index
        batch_size, num_nodes, _ = voltages.shape
        num_edges = edge_index.shape[1]

        # Ensure properties are correctly broadcastable
        def prep_prop(prop, shape):
            if prop.dim() == 1: # [E]
                return prop.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
            if prop.dim() == 2: # [B, E]
                return prop.unsqueeze(-1)
            if prop.shape == shape:
                return prop
            # Fallback for shape mismatch (e.g., during calibration with B=1)
            if prop.numel() == shape[1]:
                 return prop.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
            return prop.expand(batch_size, num_edges, 1)

        conductance = prep_prop(conductance, (batch_size, num_edges, 1))
        susceptance = prep_prop(susceptance, (batch_size, num_edges, 1))
        
        V_i = voltages[:, src, :]  # [B, E, 1]
        V_j = voltages[:, dst, :]  # [B, E, 1]
        theta_i = angles[:, src, :]  # [B, E, 1]
        theta_j = angles[:, dst, :]  # [B, E, 1]
        
        theta_ij = theta_i - theta_j  # [B, E, 1]
        
        P_ij = V_i * V_j * (conductance * torch.cos(theta_ij) + susceptance * torch.sin(theta_ij))
        P_ij_squeezed = P_ij.squeeze(-1)  # [B, E]
        
        P_calc_flat = torch.zeros(batch_size * num_nodes, device=voltages.device)
        
        # Create batched indices
        batch_offset = torch.arange(0, batch_size, device=voltages.device) * num_nodes
        src_flat = src.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        dst_flat = dst.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        
        P_ij_flat = P_ij_squeezed.flatten() # [B*E]
        
        P_calc_flat.index_add_(0, src_flat, P_ij_flat)
        P_calc_flat.index_add_(0, dst_flat, -P_ij_flat)
        
        P_calc = P_calc_flat.reshape(batch_size, num_nodes, 1)
        
        # power_injection is [B, N] from data loader, needs to be [B, N, 1]
        if power_injection.dim() == 2:
            power_injection = power_injection.unsqueeze(-1)
            
        return F.mse_loss(P_calc, power_injection)

    def capacity_loss(self, line_flows: torch.Tensor, thermal_limits: torch.Tensor) -> torch.Tensor:
        """
        Compute capacity constraint violations.
        """
        batch_size = line_flows.shape[0]
        num_edges = line_flows.shape[1]

        if thermal_limits.dim() == 1: # [E]
            thermal_limits = thermal_limits.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        elif thermal_limits.dim() == 2: # [B, E]
            thermal_limits = thermal_limits.unsqueeze(-1)
        
        # line_flows is [B, E, 1], thermal_limits is now [B, E, 1]
        violations = F.relu(torch.abs(line_flows) - thermal_limits)
        return torch.mean(violations ** 2)

    def voltage_stability_loss(self, voltages: torch.Tensor,
                              voltage_min: float = 0.9, voltage_max: float = 1.1) -> torch.Tensor:
        """
        Compute voltage stability constraint violations.
        (Using wider, more realistic 0.9-1.1 p.u. limits)
        """
        low_violations = F.relu(voltage_min - voltages)
        high_violations = F.relu(voltages - voltage_max)
        return torch.mean(low_violations ** 2 + high_violations ** 2)

    # ====================================================================
    # START: PHYSICS-BASED FREQUENCY LOSS
    # ====================================================================
    def frequency_loss(self, predicted_freq_Hz: torch.Tensor, 
                       power_injection_pu: torch.Tensor,
                       nominal_freq_Hz: float = 60.0,
                       total_inertia_H: float = 5.0) -> torch.Tensor:
        """
        Compute REAL physics-based frequency loss using the swing equation.
        Converts predicted Hz and p.u. power imbalance to a common p.u. space.
        
        Args:
            predicted_freq_Hz: Model's predicted frequency [B, 1, 1] (in Hz)
            power_injection_pu: Net power injection at each node [B, N] (in p.u.)
            nominal_freq_Hz: System nominal frequency (e.g., 60.0)
            total_inertia_H: Heuristic system inertia constant (H) in seconds.
        """
        
        # 1. Convert predicted Hz to p.u.
        # predicted_freq_Hz is [B, 1, 1]
        predicted_freq_pu = predicted_freq_Hz / nominal_freq_Hz
        
        # 2. Calculate total system power imbalance (p.u.)
        # power_injection_pu is [B, N], sum over nodes (dim=1)
        # Result is [B]
        system_power_imbalance_pu = torch.sum(power_injection_pu, dim=1)
        
        # 3. Simplified Swing Equation (steady-state deviation)
        # d_f_pu = P_imbalance_pu / (2 * H_total)
        # P_imbalance is [B], H is scalar. Result is [B]
        expected_freq_deviation_pu = system_power_imbalance_pu / (2 * total_inertia_H)
        
        # 4. Calculate expected p.u. frequency
        # expected_freq = 1.0 p.u. + deviation
        # Reshape deviation to match predicted_freq_pu [B, 1, 1]
        expected_freq_pu = 1.0 + expected_freq_deviation_pu.view(-1, 1, 1)
        
        # 5. Compute MSE loss in p.u. space
        return F.mse_loss(predicted_freq_pu, expected_freq_pu)
    # ====================================================================
    # END: PHYSICS-BASED FREQUENCY LOSS
    # ====================================================================

    # --- Merged Forward Pass ---
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        failure_prob = predictions['failure_probability']  # [B, N, 1]
        B, N, _ = failure_prob.shape
        
        failure_prob_flat = failure_prob.reshape(-1)  # [B*N]
        targets_flat = targets['failure_label'].reshape(-1)  # [B*N]
        
        if self.use_logits:
            logits = failure_prob_flat
        else:
            probs = failure_prob_flat.clamp(1e-7, 1 - 1e-7)
            logits = torch.log(probs / (1 - probs))
        
        # --- Main Prediction Loss ---
        L_prediction = self.focal_loss(logits, targets_flat)
        
        loss_components = {'prediction': L_prediction.item()}
        total_loss = L_prediction
        
        # --- Physics-Informed Losses ---
        
        # Helper to get graph properties
        def get_prop(key):
            if graph_properties and key in graph_properties:
                return graph_properties[key]
            if key not in self._warned_missing_outputs:
                # print(f"Warning: {key} not in graph_properties for loss calc.")
                self._warned_missing_outputs.add(key)
            return None

        # 1. Power Flow Loss
        if 'voltages' in predictions and 'angles' in predictions and \
           get_prop('conductance') is not None and get_prop('susceptance') is not None and \
           get_prop('power_injection') is not None:
            
            L_powerflow = self.power_flow_loss(
                voltages=predictions['voltages'],
                angles=predictions['angles'],
                edge_index=graph_properties['edge_index'], # edge_index must exist
                conductance=get_prop('conductance'),
                susceptance=get_prop('susceptance'),
                power_injection=get_prop('power_injection')
            )
            L_powerflow = torch.clamp(L_powerflow, 0.0, 10.0) # Prevent explosion
            total_loss += self.lambda_powerflow * L_powerflow
            loss_components['powerflow'] = L_powerflow.item()
        
        # 2. Capacity Loss
        if 'line_flows' in predictions and get_prop('thermal_limits') is not None:
            L_capacity = self.capacity_loss(
                line_flows=predictions['line_flows'],
                thermal_limits=get_prop('thermal_limits')
            )
            L_capacity = torch.clamp(L_capacity, 0.0, 10.0) # Prevent explosion
            total_loss += self.lambda_capacity * L_capacity
            loss_components['capacity'] = L_capacity.item()

        # 3. Voltage Stability Loss
        if 'voltages' in predictions:
            L_stability = self.voltage_stability_loss(
                voltages=predictions['voltages']
            )
            total_loss += self.lambda_stability * L_stability
            loss_components['voltage'] = L_stability.item()

        # ====================================================================
        # START: REPLACED HEURISTIC WITH PHYSICS-BASED LOSS
        # ====================================================================
        # 4. Frequency Loss (PHYSICS-BASED)
        if 'frequency' in predictions and get_prop('power_injection') is not None:
            
            predicted_freq_Hz = predictions['frequency']  # [B, 1, 1] (in Hz)
            
            # power_injection is [B, N] from data loader (already p.u.)
            power_injection_pu = get_prop('power_injection')
            
            L_frequency = self.frequency_loss(
                predicted_freq_Hz,
                power_injection_pu,
                nominal_freq_Hz=self.freq_nominal,
                total_inertia_H=5.0 # Heuristic system inertia
            )
            
            L_frequency = torch.clamp(L_frequency, 0.0, 10.0)
            total_loss += self.lambda_frequency * L_frequency
            loss_components['frequency'] = L_frequency.item()
        # ====================================================================
        # END: REPLACEMENT
        # ====================================================================
        
        return total_loss, loss_components


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """Training manager for cascade prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.003,
        output_dir: str = "checkpoints",
        max_grad_norm: float = 5.0,
        use_amp: bool = False,
        model_outputs_logits: bool = False,
        base_mva: float = 100.0,    # Added for loss
        base_freq: float = 60.0     # Added for loss
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.model_outputs_logits = model_outputs_logits
        
        # Pass base mva/freq to loss
        self.base_mva = base_mva
        self.base_freq = base_freq
        
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and self.device.type == 'cuda' else None
        
        os.makedirs(output_dir, exist_ok=True)
        
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
            'learning_rate': []
        }
        
        print("\n" + "="*80)
        print("STARTING DYNAMIC LOSS WEIGHT CALIBRATION")
        print("="*80)
        calibrated_lambdas = self._calibrate_loss_weights()
        print("="*80)
        print("CALIBRATION COMPLETE")
        print("="*80 + "\n")
        
        # --- MODIFIED: Use calibrated weights with the NEW merged loss class ---
        self.criterion = PhysicsInformedLoss(
            lambda_powerflow=calibrated_lambdas.get('lambda_powerflow', 0.1),
            lambda_capacity=calibrated_lambdas.get('lambda_capacity', 0.1),
            lambda_stability=calibrated_lambdas.get('voltage', 0.001), # Calibrator uses 'voltage'
            lambda_frequency=calibrated_lambdas.get('lambda_frequency', 0.1),
            
            pos_weight=20.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            label_smoothing=0.1,
            use_logits=model_outputs_logits,
            base_mva=self.base_mva,
            base_freq=self.base_freq
        )
        print("\n✓ PhysicsInformedLoss (Merged) initialized with dynamically calibrated weights.")
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        self.cascade_threshold = 0.5 # Start at 0.5, let it adjust
        self.node_threshold = 0.5 # Start at 0.5, let it adjust
        self.best_val_f1 = 0.0
        
        self._model_validated = False

    def _calibrate_loss_weights(self, num_batches=20) -> Dict[str, float]:
        """
        Run a few batches to find the average raw loss for each component
        and compute balancing weights.
        """
        print(f"Running loss calibration for {num_batches} batches...")
        self.model.eval()
        
        # Use a "dummy" criterion with all weights at 1.0 to get raw loss magnitudes
        dummy_criterion = PhysicsInformedLoss(
            lambda_powerflow=1.0, lambda_capacity=1.0,
            lambda_stability=1.0, lambda_frequency=1.0,
            lambda_reactive=1.0, 
            use_logits=self.model_outputs_logits,
            base_mva=self.base_mva,
            base_freq=self.base_freq
        )
        
        loss_sums = {}
        total_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= num_batches:
                    break
                
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
                
                # Stop if batch is empty
                if 'node_failure_labels' not in batch_device:
                    print("  Warning: Empty batch encountered during calibration.")
                    continue

                outputs = self.model(batch_device, return_sequence=True)
                
                # --- This is the key part ---
                # We need the edge_index in graph_properties for the new loss function
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                _, loss_components = dummy_criterion(
                    outputs, 
                    {'failure_label': batch_device['node_failure_labels'].reshape(-1)},
                    graph_properties
                )
                
                for key, val in loss_components.items():
                    if not np.isnan(val) and not np.isinf(val):
                        loss_sums[key] = loss_sums.get(key, 0.0) + val
                total_batches += 1

        if total_batches == 0:
            print("[ERROR] Calibration failed: No data loaded.")
            return {
                'lambda_powerflow': 0.1, 'lambda_capacity': 0.1,
                'voltage': 0.001, 'lambda_frequency': 0.1
            }
        
        avg_losses = {key: val / total_batches for key, val in loss_sums.items()}
        
        print("  Average raw loss components (unweighted):")
        for key, val in avg_losses.items():
            print(f"    {key}: {val:.6f}")
            
        # ====================================================================
        # START: BALANCING LOGIC
        # ====================================================================
        
        target_magnitude = avg_losses.get('prediction', 0.1)
        # Ensure target_magnitude is not zero to prevent division by zero
        if target_magnitude < 1e-9: target_magnitude = 1e-9 
        print(f"\n  Target magnitude (from prediction loss): {target_magnitude:.6f}")

        physics_loss_keys = ['powerflow', 'capacity', 'voltage', 'frequency']
        calibrated_lambdas = {}
        
        print("\n  Calibrating lambda weights (Target-Value Strategy):")
        
        for key in physics_loss_keys:
            raw_loss = avg_losses.get(key, 0.0)
            
            # This is the logic:
            # If the raw loss is near zero, we calculate its lambda
            # as if its value was the target_magnitude.
            if raw_loss < 1e-6:
                denominator = target_magnitude 
                # This makes lambda_val = target_magnitude / target_magnitude = 1.0
            else:
                denominator = raw_loss
            
            lambda_val = target_magnitude / denominator
            
            # Handle the 'lambda_' prefix for the criterion's keys
            lambda_key = f"lambda_{key}" if key != 'voltage' else 'voltage'
            calibrated_lambdas[lambda_key] = lambda_val
            
            print(f"    {lambda_key}: {lambda_val:10.4f}  (Raw Loss: {raw_loss:.6f}, Denom: {denominator:.6f}, Initial Weighted Loss: {lambda_val * raw_loss:.4f})")
        
        # ====================================================================
        # END: BALANCING LOGIC
        # ====================================================================
            
        self.model.train()
        return calibrated_lambdas
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.5)
        self.node_threshold = checkpoint.get('node_threshold', 0.5)
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ Resumed from epoch {self.start_epoch} (best val_loss: {self.best_val_loss:.4f})")
        print(f"✓ Loaded thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
        return True
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        cascade_tp = cascade_fp = cascade_tn = cascade_fn = 0
        node_tp = node_fp = node_tn = node_fn = 0
        grad_norms = []
        loss_component_sums = {}
        
        pbar = tqdm(self.train_loader, desc="Training")
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
            
            # Check for empty batch (can happen with small datasets)
            if 'node_failure_labels' not in batch_device:
                continue

            self.optimizer.zero_grad()
            
            # --- Prepare graph_properties for the new loss function ---
            graph_properties = batch_device.get('graph_properties', {})
            if 'edge_index' not in graph_properties:
                graph_properties['edge_index'] = batch_device['edge_index']
            # --- End modification ---
            
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_device, return_sequence=True)
                    
                    if not self._model_validated:
                        self._validate_model_outputs(outputs, batch_device)
                        self._model_validated = True
                    
                    loss, loss_components = self.criterion(
                        outputs, 
                        {'failure_label': batch_device['node_failure_labels']}, # Pass [B,N] labels
                        graph_properties
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
                
                loss, loss_components = self.criterion(
                    outputs, 
                    {'failure_label': batch_device['node_failure_labels']}, # Pass [B,N] labels
                    graph_properties
                )
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.optimizer.step()
            
            total_loss += loss.item()
            
            for comp_name, comp_value in loss_components.items():
                loss_component_sums[comp_name] = loss_component_sums.get(comp_name, 0.0) + comp_value
            
            # ====================================================================
            # START: TQDM DISPLAY LOGIC
            # ====================================================================
            
            # 1. Set a simple description that won't overflow
            pbar.set_description(f"Training (Loss: {loss.item():.4f}, Grad: {grad_norm:.2f})")
            
            # --- Metrics Calculation (moved up for postfix) ---
            node_probs_current = outputs['failure_probability'].squeeze(-1) # [B, N]
            node_pred_current = (node_probs_current > self.node_threshold).float() # [B, N]
            node_labels_current = batch_device['node_failure_labels']  # [B, N]
            
            cascade_prob_current = node_probs_current.max(dim=1)[0] # [B]
            cascade_pred_current = (cascade_prob_current > self.cascade_threshold).float()
            cascade_labels_current = (node_labels_current.max(dim=1)[0] > 0.5).float()
            
            # Update totals
            cascade_tp += ((cascade_pred_current == 1) & (cascade_labels_current == 1)).sum().item()
            cascade_fp += ((cascade_pred_current == 1) & (cascade_labels_current == 0)).sum().item()
            cascade_tn += ((cascade_pred_current == 0) & (cascade_labels_current == 0)).sum().item()
            cascade_fn += ((cascade_pred_current == 0) & (cascade_labels_current == 1)).sum().item()
            
            node_tp += ((node_pred_current == 1) & (node_labels_current == 1)).sum().item()
            node_fp += ((node_pred_current == 1) & (node_labels_current == 0)).sum().item()
            node_tn += ((node_pred_current == 0) & (node_labels_current == 0)).sum().item()
            node_fn += ((node_pred_current == 0) & (node_labels_current == 1)).sum().item()

            # Calculate running metrics for display
            cascade_precision = cascade_tp / (cascade_tp + cascade_fp + 1e-7)
            cascade_recall = cascade_tp / (cascade_tp + cascade_fn + 1e-7)
            cascade_f1 = 2 * cascade_precision * cascade_recall / (cascade_precision + cascade_recall + 1e-7)
            
            node_precision = node_tp / (node_tp + node_fp + 1e-7)
            node_recall = node_tp / (node_tp + node_fn + 1e-7)
            node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-7)
            
            # 2. Set a clean, ultra-compact postfix that will fit on one line
            pbar.set_postfix({
                'cF1': f"{cascade_f1:.3f}",
                'cR': f"{cascade_recall:.3f}",
                'nF1': f"{node_f1:.3f}",
                'pL': f"{loss_components.get('prediction', 0):.3f}",
                'pwL': f"{loss_components.get('powerflow', 0):.2f}",
                'cpL': f"{loss_components.get('capacity', 0):.2f}",
                'vL': f"{loss_components.get('voltage', 0):.2f}",
                'fL': f"{loss_components.get('frequency', 0):.2f}",
            })
            
            # ====================================================================
            # END: TQDM DISPLAY LOGIC
            # ====================================================================

        # Compute final epoch metrics
        avg_loss = total_loss / (len(self.train_loader) + 1e-7)
        cascade_precision = cascade_tp / (cascade_tp + cascade_fp + 1e-7)
        cascade_recall = cascade_tp / (cascade_tp + cascade_fn + 1e-7)
        cascade_f1 = 2 * cascade_precision * cascade_recall / (cascade_precision + cascade_recall + 1e-7)
        cascade_acc = (cascade_tp + cascade_tn) / (cascade_tp + cascade_tn + cascade_fp + cascade_fn + 1e-7)
        
        node_precision = node_tp / (node_tp + node_fp + 1e-7)
        node_recall = node_tp / (node_tp + node_fn + 1e-7)
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-7)
        node_acc = (node_tp + node_tn) / (node_tp + node_tn + node_fp + node_fn + 1e-7)
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        print(f"\n  Average gradient norm: {avg_grad_norm:.4f}")
        if loss_component_sums:
            print(f"  Average loss components:")
            for comp_name, comp_sum in loss_component_sums.items():
                avg_comp = comp_sum / (len(self.train_loader) + 1e-7)
                print(f"    {comp_name}: {avg_comp:.6f}")
        
        return {
            'loss': avg_loss,
            'cascade_acc': cascade_acc, 'cascade_f1': cascade_f1,
            'cascade_precision': cascade_precision, 'cascade_recall': cascade_recall,
            'node_acc': node_acc, 'node_f1': node_f1,
            'node_precision': node_precision, 'node_recall': node_recall
        }
    
    def _validate_model_outputs(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        """Validate that model outputs match expected format."""
        print("\n" + "="*80)
        print("MODEL OUTPUT VALIDATION (First Batch)")
        print("="*80)
        
        B = batch['node_failure_labels'].shape[0]
        N = batch['node_failure_labels'].shape[1]
        E = batch['edge_index'].shape[1]

        def check_shape(key, expected_dims):
            if key not in outputs:
                print(f"  ✗ {key}: Missing from model output!")
                return
            
            shape = tuple(outputs[key].shape)
            
            # Helper to check dimensions
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

        print("\nChecking other model outputs...")
        check_shape('risk_scores', ('B', 'N', 7))
        check_shape('reactive_flows', ('B', 'E', 1))

        if 'temporal_sequence' in batch:
            T = batch['temporal_sequence'].shape[1]
            print(f"\nTemporal sequence detected: B={B}, T={T}, N={N}")
            print(f"  ✓ 3-layer LSTM IS BEING UTILIZED.")
        else:
            print(f"\n  ✗ No temporal sequence found! Model is in single-step mode.")
            print(f"  ✗ 3-layer LSTM is NOT being utilized effectively.")

        print("="*80 + "\n")
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with PROPER METRICS."""
        self.model.eval()
        
        total_loss = 0.0
        cascade_tp = cascade_fp = cascade_tn = cascade_fn = 0
        node_tp = node_fp = node_tn = node_fn = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
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
                
                # Check for empty batch
                if 'node_failure_labels' not in batch_device:
                    continue

                # --- Prepare graph_properties for the new loss function ---
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                # --- End modification ---
                
                outputs = self.model(batch_device, return_sequence=True)
                
                loss, _ = self.criterion(
                    outputs,
                    {'failure_label': batch_device['node_failure_labels']}, # Pass [B,N] labels
                    graph_properties
                )
                
                total_loss += loss.item()
                
                # --- Metrics Calculation (unchanged) ---
                node_probs = outputs['failure_probability'].squeeze(-1) # [B, N]
                cascade_prob = node_probs.max(dim=1)[0] # [B]
                
                cascade_pred = (cascade_prob > self.cascade_threshold).float()
                cascade_labels = (batch_device['node_failure_labels'].max(dim=1)[0] > 0.5).float()
                
                cascade_tp += ((cascade_pred == 1) & (cascade_labels == 1)).sum().item()
                cascade_fp += ((cascade_pred == 1) & (cascade_labels == 0)).sum().item()
                cascade_tn += ((cascade_pred == 0) & (cascade_labels == 0)).sum().item()
                cascade_fn += ((cascade_pred == 0) & (cascade_labels == 1)).sum().item()
                
                node_pred = (node_probs > self.node_threshold).float() # [B, N]
                node_labels = batch_device['node_failure_labels']  # [B, N]
                
                node_tp += ((node_pred == 1) & (node_labels == 1)).sum().item()
                node_fp += ((node_pred == 1) & (node_labels == 0)).sum().item()
                node_tn += ((node_pred == 0) & (node_labels == 0)).sum().item()
                node_fn += ((node_pred == 0) & (node_labels == 1)).sum().item()
                
                cascade_f1 = 2 * cascade_tp / (2 * cascade_tp + cascade_fp + cascade_fn + 1e-7)
                node_f1 = 2 * node_tp / (2 * node_tp + node_fp + node_fn + 1e-7)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'casc_f1': f"{cascade_f1:.4f}",
                    'node_f1': f"{node_f1:.4f}"
                })
        
        # Compute final validation metrics
        avg_loss = total_loss / (len(self.val_loader) + 1e-7)
        cascade_precision = cascade_tp / (cascade_tp + cascade_fp + 1e-7)
        cascade_recall = cascade_tp / (cascade_tp + cascade_fn + 1e-7)
        cascade_f1 = 2 * cascade_precision * cascade_recall / (cascade_precision + cascade_recall + 1e-7)
        cascade_acc = (cascade_tp + cascade_tn) / (cascade_tp + cascade_tn + cascade_fp + cascade_fn + 1e-7)
        
        node_precision = node_tp / (node_tp + node_fp + 1e-7)
        node_recall = node_tp / (node_tp + node_fn + 1e-7)
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall + 1e-7)
        node_acc = (node_tp + node_tn) / (node_tp + node_tn + node_fp + node_fn + 1e-7)
        
        return {
            'loss': avg_loss,
            'cascade_acc': cascade_acc, 'cascade_f1': cascade_f1,
            'cascade_precision': cascade_precision, 'cascade_recall': cascade_recall,
            'node_acc': node_acc, 'node_f1': node_f1,
            'node_precision': node_precision, 'node_recall': node_recall
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Train the model and save history/plots."""
        patience_counter = 0
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # --- Dynamic Threshold Adjustment ---
            # Use combined F1 as the main driver for threshold tuning
            combined_f1 = (val_metrics['cascade_f1'] + val_metrics['node_f1']) / 2
            
            if combined_f1 > self.best_val_f1:
                self.best_val_f1 = combined_f1
                print(f"  ✓ Improved F1 score: {combined_f1:.4f} (thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f})")
            
            # If recall is suffering, lower thresholds to find more positives
            elif val_metrics['cascade_recall'] < 0.5 or val_metrics['node_recall'] < 0.3:
                self.cascade_threshold = max(0.1, self.cascade_threshold - 0.05) # Be more aggressive
                self.node_threshold = max(0.1, self.node_threshold - 0.05)
                print(f"  ⚠ Low recall detected - lowering thresholds to cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
            # If precision is suffering, raise thresholds to be more confident
            elif val_metrics['cascade_precision'] < 0.4 or val_metrics['node_precision'] < 0.3:
                self.cascade_threshold = min(0.8, self.cascade_threshold + 0.05) # Be more aggressive
                self.node_threshold = min(0.8, self.node_threshold + 0.05)
                print(f"  ⚠ Low precision detected - raising thresholds to cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
            self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Update history (keys are already initialized)
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
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"\n  CASCADE DETECTION (Thresh: {self.cascade_threshold:.3f}):")
            print(f"    F1 Score:  Train {train_metrics['cascade_f1']:.4f} | Val {val_metrics['cascade_f1']:.4f}")
            print(f"    Precision: Train {train_metrics['cascade_precision']:.4f} | Val {val_metrics['cascade_precision']:.4f}")
            print(f"    Recall:    Train {train_metrics['cascade_recall']:.4f} | Val {val_metrics['cascade_recall']:.4f}")
            print(f"\n  NODE FAILURE (Thresh: {self.node_threshold:.3f}):")
            print(f"    F1 Score:  Train {train_metrics['node_f1']:.4f} | Val {val_metrics['node_f1']:.4f}")
            print(f"    Precision: Train {train_metrics['node_precision']:.4f} | Val {val_metrics['node_precision']:.4f}")
            print(f"    Recall:    Train {train_metrics['node_recall']:.4f} | Val {val_metrics['node_recall']:.4f}")
            
            # Save best model based on validation loss
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_cascade_f1': val_metrics['cascade_f1'],
                    'val_node_f1': val_metrics['node_f1'],
                    'cascade_threshold': self.cascade_threshold,
                    'node_threshold': self.node_threshold,
                    'history': self.history
                }, f"{self.output_dir}/best_model.pth")
                print(f"  ✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'cascade_threshold': self.cascade_threshold,
                'node_threshold': self.node_threshold,
                'history': self.history
            }, f"{self.output_dir}/latest_checkpoint.pth")
        
        self.save_history()
        self.plot_training_curves()
        
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Training history saved to: {self.output_dir}/training_history.json")
        print(f"  Training curves saved to: {self.output_dir}/training_curves.png")
        print(f"  Best model saved to: {self.output_dir}/best_model.pth")
        print(f"{'='*80}")
    
    def save_history(self):
        history_path = f"{self.output_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        if not self.history['train_loss']:
            print("No history to plot. Skipping plot generation.")
            plt.close()
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cascade F1
        axes[0, 1].plot(epochs, self.history['train_cascade_f1'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_cascade_f1'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Cascade Detection F1')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cascade Precision/Recall
        axes[0, 2].plot(epochs, self.history['train_cascade_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_cascade_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[0, 2].plot(epochs, self.history['train_cascade_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_cascade_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Cascade Precision/Recall')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Node F1
        axes[1, 0].plot(epochs, self.history['train_node_f1'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_node_f1'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Node Failure F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Node Precision/Recall
        axes[1, 1].plot(epochs, self.history['train_node_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['train_node_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Node Precision/Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Accuracy comparison
        axes[1, 2].plot(epochs, self.history['train_cascade_acc'], 'b-', label='Train Cascade', linewidth=2)
        axes[1, 2].plot(epochs, self.history['val_cascade_acc'], 'r-', label='Val Cascade', linewidth=2)
        axes[1, 2].plot(epochs, self.history['train_node_acc'], 'b--', label='Train Node', linewidth=2, alpha=0.6)
        axes[1, 2].plot(epochs, self.history['val_node_acc'], 'r--', label='Val Node', linewidth=2, alpha=0.6)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Accuracy Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.output_dir}/training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {plot_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # --- ADDED: Argument Parser ---
    parser = argparse.ArgumentParser(description="Train Cascade Prediction Model")
    parser.add_argument('--data_dir', type=str, default="data", 
                        help="Root directory containing train/val/test data folders")
    parser.add_argument('--output_dir', type=str, default="checkpoints", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=50, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=4, 
                        help="Training and validation batch size")
    parser.add_argument('--lr', type=float, default=0.0005, 
                        help="Initial learning rate")
    parser.add_argument('--grad_clip', type=float, default=10.0, 
                        help="Max gradient norm for clipping")
    parser.add_argument('--patience', type=int, default=10, 
                        help="Epochs for early stopping patience")
    parser.add_argument('--resume', action='store_true', 
                        help="Resume training from latest_checkpoint.pth")
    parser.add_argument('--base_mva', type=float, default=100.0,
                        help="Base MVA for physics normalization")
    parser.add_argument('--base_freq', type=float, default=60.0,
                        help="Base frequency (Hz) for physics normalization")

    args = parser.parse_args()
    # --- END: Argument Parser ---
    
    
    print("="*80)
    print("CASCADE FAILURE PREDICTION - TRAINING SCRIPT (IMPROVED)")
    print("="*80)
    
    # --- CHANGED: Use args for configuration ---
    # Configuration
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MAX_GRAD_NORM = args.grad_clip
    EARLY_STOPPING_PATIENCE = args.patience
    
    # --- CHANGED: Auto-detect device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = (DEVICE.type == 'cuda')
    
    # Model's 'failure_prob_head' ends in Sigmoid, so it outputs probabilities.
    MODEL_OUTPUTS_LOGITS = False 
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    if not torch.cuda.is_available():
        print("  (WARNING: CUDA (GPU) not available, training will be slow on CPU.)")
    print(f"  Gradient clipping: {MAX_GRAD_NORM}")
    print(f"  Mixed precision: {USE_AMP}")
    print(f"  Resume training: {args.resume}")
    
    print(f"\nLoading datasets...")
    # --- Pass normalization constants to Dataset ---
    try:
        train_dataset = CascadeDataset(
            f"{DATA_DIR}/train", 
            mode='full_sequence', 
            base_mva=args.base_mva,
            base_frequency=args.base_freq
        )
        val_dataset = CascadeDataset(
            f"{DATA_DIR}/val", 
            mode='full_sequence', 
            base_mva=args.base_mva,
            base_frequency=args.base_freq
        )
    except ValueError as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        print("Please ensure you have run multimodal_data_generator.py to create the data")
        print(f"in '{DATA_DIR}/train' and '{DATA_DIR}/val' directories.")
        sys.exit(1)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("\n[ERROR] No training data found. Please generate data first.")
        sys.exit(1)
        
    print(f"  Mode: full_sequence (utilizing 3-layer LSTM for temporal modeling)")
    
    print(f"\nComputing sample weights for balanced sampling...")
    sample_weights = []
    positive_count = 0
    negative_count = 0
    
    for idx in range(len(train_dataset)):
        has_cascade = train_dataset.get_cascade_label(idx)
        
        if has_cascade:
            sample_weights.append(20.0) # Oversample positive class
            positive_count += 1
        else:
            sample_weights.append(1.0)
            negative_count += 1
    
    print(f"  Positive samples: {positive_count} ({positive_count/len(train_dataset)*100:.1f}%)")
    print(f"  Negative samples: {negative_count} ({negative_count/len(train_dataset)*100:.1f}%)")
    print(f"  Oversampling ratio: 20:1 (positive:negative)")
    
    if positive_count < 10 and len(train_dataset) > 0: # Added check for > 0
        print(f"\n{'='*80}")
        print(f"[CRITICAL WARNING] Only {positive_count} cascade scenarios found!")
        print("  This is not enough data to train the model.")
        print("  Please REGENERATE your data with more cascade scenarios (e.g., --cascade 4000).")
        print(f"{'='*80}\n")
        
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_cascade_batch,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_cascade_batch,
        persistent_workers=False
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = UnifiedCascadePredictionModel(
        embedding_dim=128,
        hidden_dim=128,
        num_gnn_layers=3,
        heads=4,
        dropout=0.3
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    # This will now run the calibration automatically using the NEW merged loss
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        max_grad_norm=MAX_GRAD_NORM,
        use_amp=USE_AMP,
        model_outputs_logits=MODEL_OUTPUTS_LOGITS,
        base_mva=args.base_mva,
        base_freq=args.base_freq
    )
    
    checkpoint_path = f"{OUTPUT_DIR}/latest_checkpoint.pth"
    if args.resume and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    
    print("\nTraining completed successfully!")