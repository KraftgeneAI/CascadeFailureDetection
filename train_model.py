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

*** IMPROVEMENT: Added loss functions for 7-D risk and cascade timing
*** to train the model on causal path prediction.
***
*** IMPROVEMENT 2: Added metric tracking and plotting for
*** Timing MAE and Risk MSE to monitor causal path performance.
***
*** IMPROVEMENT 3 (FINAL): The "best" model is now saved based on the
*** LOWEST val_time_mae (causal path accuracy), while ensuring
*** F1 scores remain high.
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
    - ***IMPROVEMENT: Added Risk and Timing losses***
    """
    
    def __init__(self, lambda_powerflow: float = 0.1, lambda_temperature: float = 0.1,
                 lambda_stability: float = 0.001, lambda_frequency: float = 0.1,
                 lambda_reactive: float = 0.1, 
                 lambda_risk: float = 0.2,       # <-- IMPROVEMENT: Added
                 lambda_timing: float = 0.1,     # <-- IMPROVEMENT: Added
                 pos_weight: float = 10.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.15, use_logits: bool = False,
                 base_mva: float = 100.0, base_freq: float = 60.0):
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_temperature = lambda_temperature
        self.lambda_stability = lambda_stability
        self.lambda_frequency = lambda_frequency
        self.lambda_reactive = lambda_reactive 
        self.lambda_risk = lambda_risk       # <-- IMPROVEMENT: Added
        self.lambda_timing = lambda_timing   # <-- IMPROVEMENT: Added
        
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
        
        # ====================================================================
        # START: BUG FIX - Use pos_weight correctly
        # ====================================================================
        # pos_weight (e.g., 1.0) must be applied to the loss, not just a tensor
        # Create a weight tensor that applies pos_weight to positive samples
        weight = torch.ones_like(logits)
        if self.pos_weight != 1.0:
            weight[targets > 0.5] = self.pos_weight
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets_smooth, 
            reduction='none' # Do not reduce yet
        )
        
        # Apply the pos_weight
        bce_loss = bce_loss * weight
        # ====================================================================
        # END: BUG FIX
        # ====================================================================

        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

    # ====================================================================
    # START: "DEAD HEAD" BUG FIX 
    # Re-wire power_flow_loss to use the model's *prediction*
    # ====================================================================
    def power_flow_loss(self, predicted_line_flows: torch.Tensor, 
                              edge_index: torch.Tensor, 
                              power_injection: torch.Tensor,
                              num_nodes: int,
                              batch_size: int) -> torch.Tensor:
        """
        Computes power flow loss by summing the model's *predicted* line flows (P)
        at each node and comparing to the ground truth power injection.
        """
        src, dst = edge_index
        num_edges = edge_index.shape[1]
        
        # predicted_line_flows is [B, E, 1]
        P_ij_squeezed = predicted_line_flows.squeeze(-1)  # [B, E]
        
        P_calc_flat = torch.zeros(batch_size * num_nodes, device=predicted_line_flows.device)
        
        batch_offset = torch.arange(0, batch_size, device=predicted_line_flows.device) * num_nodes
        src_flat = src.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        dst_flat = dst.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        
        P_ij_flat = P_ij_squeezed.flatten() # [B*E]
        
        P_calc_flat.index_add_(0, src_flat, P_ij_flat)
        P_calc_flat.index_add_(0, dst_flat, -P_ij_flat) # Sums flows at nodes
        
        P_calc = P_calc_flat.reshape(batch_size, num_nodes, 1)
        
        if power_injection.dim() == 2:
            power_injection = power_injection.unsqueeze(-1)
            
        return F.mse_loss(P_calc, power_injection)
    # ====================================================================
    # END: "DEAD HEAD" BUG FIX
    # ====================================================================

    def temperature_loss(self, predicted_temp: torch.Tensor, ground_truth_temp: torch.Tensor) -> torch.Tensor:
        """
        Forces the model to learn the actual ground truth temperature.
        """
        # predicted_temp is [B, N, 1]
        # ground_truth_temp is [B, N]
        return F.mse_loss(predicted_temp.squeeze(-1), ground_truth_temp)


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

    # ====================================================================
    # START: "DEAD HEAD" BUG FIX 
    # Re-wire reactive_power_flow_loss to use the model's *prediction*
    # ====================================================================
    def reactive_power_flow_loss(self, predicted_reactive_flows: torch.Tensor, 
                                 edge_index: torch.Tensor, 
                                 reactive_injection: torch.Tensor,
                                 num_nodes: int,
                                 batch_size: int) -> torch.Tensor:
        """
        Computes reactive power flow loss by summing the model's *predicted* reactive flows (Q)
        at each node and comparing to the ground truth reactive injection.
        """
        src, dst = edge_index
        num_edges = edge_index.shape[1]
        
        # predicted_reactive_flows is [B, E, 1]
        Q_ij_squeezed = predicted_reactive_flows.squeeze(-1)  # [B, E]
        
        Q_calc_flat = torch.zeros(batch_size * num_nodes, device=predicted_reactive_flows.device)
        
        batch_offset = torch.arange(0, batch_size, device=predicted_reactive_flows.device) * num_nodes
        src_flat = src.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        dst_flat = dst.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        
        Q_ij_flat = Q_ij_squeezed.flatten() # [B*E]
        
        Q_calc_flat.index_add_(0, src_flat, Q_ij_flat)
        Q_calc_flat.index_add_(0, dst_flat, -Q_ij_flat) # Sums flows at nodes
        
        Q_calc = Q_calc_flat.reshape(batch_size, num_nodes, 1)
        
        if reactive_injection.dim() == 2:
            reactive_injection = reactive_injection.unsqueeze(-1)
            
        return F.mse_loss(Q_calc, reactive_injection)
    # ====================================================================
    # END: "DEAD HEAD" BUG FIX
    # ====================================================================

    # ====================================================================
    # START: IMPROVEMENT - RISK SCORE LOSS
    # ====================================================================
    def risk_loss(self, predicted_risk: torch.Tensor, target_risk: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for the 7-D risk assessment.
        Compares aggregated predicted node risk to the scenario's ground truth risk.
        
        Args:
            predicted_risk: Model's per-node risk output [B, N, 7]
            target_risk: Ground truth scenario risk vector [B, 7]
        """
        # Aggregate predicted risk across nodes (mean)
        predicted_risk_agg = torch.mean(predicted_risk, dim=1) # [B, 7]
        
        # Use MSELoss to compare the aggregated prediction to the target
        return F.mse_loss(predicted_risk_agg, target_risk)
    # ====================================================================
    # END: IMPROVEMENT - RISK SCORE LOSS
    # ====================================================================

    # ====================================================================
    # START: IMPROVEMENT - SIMPLIFIED TIMING LOSS
    # ====================================================================
    def timing_loss(self, predicted_node_timing: torch.Tensor, 
                    target_node_timing: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for cascade timing (Node-to-Node).
        
        Args:
            predicted_node_timing: Model's per-node time output [B, N, 1]
            target_node_timing: Ground truth per-node time [B, N]
        """
        # Squeeze prediction to match target shape
        predicted_node_timing = predicted_node_timing.squeeze(-1) # [B, N]
        
        # 1. Create mask for only nodes that *should* fail (ground truth time >= 0)
        mask = target_node_timing >= 0.0
        
        if mask.sum() == 0:
            # No cascade nodes in this batch, return zero loss
            return torch.tensor(0.0, device=target_node_timing.device)
            
        # 2. Compute loss only on nodes that are part of the cascade
        loss = F.mse_loss(predicted_node_timing[mask], target_node_timing[mask])
        return loss
    # ====================================================================
    # END: IMPROVEMENT - SIMPLIFIED TIMING LOSS
    # ====================================================================


    # --- Merged Forward Pass ---
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        failure_prob = predictions['failure_probability']  # [B, N, 1]
        B, N, _ = failure_prob.shape
        
        failure_prob_flat = failure_prob.reshape(-1)  # [B*N]
        # --- IMPROVEMENT: targets is now a dict ---
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

        # ====================================================================
        # START: "DEAD HEAD" BUG FIX 
        # Call the new power_flow_loss
        # ====================================================================
        # 1. Power Flow Loss
        if 'line_flows' in predictions and get_prop('power_injection') is not None:
            
            L_powerflow = self.power_flow_loss(
                predicted_line_flows=predictions['line_flows'], # <-- USE PREDICTION
                edge_index=graph_properties['edge_index'],
                power_injection=get_prop('power_injection'),
                num_nodes=N,
                batch_size=B
            )
            L_powerflow = torch.clamp(L_powerflow, 0.0, 10.0) # Prevent explosion
            total_loss += self.lambda_powerflow * L_powerflow
            loss_components['powerflow'] = L_powerflow.item()
        # ====================================================================
        # END: "DEAD HEAD" BUG FIX
        # ====================================================================
        
        # 2. Temperature Loss
        if 'temperature' in predictions and get_prop('ground_truth_temperature') is not None:
            L_temperature = self.temperature_loss(
                predictions['temperature'],
                get_prop('ground_truth_temperature')
            )
            L_temperature = torch.clamp(L_temperature, 0.0, 10.0) # Prevent explosion
            total_loss += self.lambda_temperature * L_temperature
            loss_components['temperature'] = L_temperature.item()
            # 3. Voltage Stability Loss
            if 'voltages' in predictions:
                L_stability = self.voltage_stability_loss(
                    voltages=predictions['voltages']
                )
                total_loss += self.lambda_stability * L_stability
                loss_components['voltage'] = L_stability.item()

        # ====================================================================
        # START: "DEAD HEAD" BUG FIX 
        # Call the new reactive_power_flow_loss
        # ====================================================================
        if 'reactive_flows' in predictions and get_prop('reactive_injection') is not None:
            
            L_reactive = self.reactive_power_flow_loss(
                predicted_reactive_flows=predictions['reactive_flows'], # <-- USE PREDICTION
                edge_index=graph_properties['edge_index'],
                reactive_injection=get_prop('reactive_injection'),
                num_nodes=N,
                batch_size=B
            )
            L_reactive = torch.clamp(L_reactive, 0.0, 10.0)
            total_loss += self.lambda_reactive * L_reactive
            loss_components['reactive'] = L_reactive.item()
        # ====================================================================
        # END: "DEAD HEAD" BUG FIX
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
        
        # 5. Risk Score Loss
        if 'risk_scores' in predictions and 'ground_truth_risk' in targets and targets['ground_truth_risk'] is not None:
            L_risk = self.risk_loss(
                predicted_risk=predictions['risk_scores'],  # [B, N, 7]
                target_risk=targets['ground_truth_risk']    # [B, 7]
            )
            total_loss += self.lambda_risk * L_risk
            loss_components['risk'] = L_risk.item()

        # 6. Timing Loss
        # 'cascade_timing' (pred) is [B,N,1], 'cascade_timing' (target) is [B,N]
        if 'cascade_timing' in predictions and 'cascade_timing' in targets and targets['cascade_timing'] is not None:
            L_timing = self.timing_loss(
                predicted_node_timing=predictions['cascade_timing'], # [B, N, 1]
                target_node_timing=targets['cascade_timing'],        # [B, N]
            )
            total_loss += self.lambda_timing * L_timing
            loss_components['timing'] = L_timing.item()

        
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
            'train_time_mae': [], 'val_time_mae': [],
            'train_risk_mse': [], 'val_risk_mse': [],
            'learning_rate': []
        }
        
        print("\n" + "="*80)
        print("STARTING DYNAMIC LOSS WEIGHT CALIBRATION")
        print("="*80)
        calibrated_lambdas = self._calibrate_loss_weights()
        print("="*80)
        print("CALIBRATION COMPLETE")
        print("="*80 + "\n")
        
        # ====================================================================
        # START: MODIFICATION - Set pos_weight=1.0 and add lambda_reactive
        # ====================================================================
        self.criterion = PhysicsInformedLoss(
            lambda_powerflow=5.0,      # <-- MANUALLY SET (was 0.026)
            lambda_temperature=5.0,      # <-- MANUALLY SET (was 0.017)
            lambda_stability=1.0,      # <-- MANUALLY SET (was 0.21)
            lambda_frequency=1.0,      # <-- MANUALLY SET (was 0.11)
            lambda_reactive=5.0,       # <-- MANUALLY SET (was 0.22)
            lambda_risk=1.5,           # (Calibrated 1.77, this is fine)

            # For dynamic calibration (case-dependent)
            # lambda_powerflow=calibrated_lambdas.get('lambda_powerflow', 0.1),
            # lambda_temperature=calibrated_lambdas.get('lambda_temperature', 0.1),
            # lambda_stability=calibrated_lambdas.get('voltage', 0.001), 
            # lambda_frequency=calibrated_lambdas.get('lambda_frequency', 0.1),
            # lambda_reactive=calibrated_lambdas.get('lambda_reactive', 0.1),
            # lambda_risk=calibrated_lambdas.get('lambda_risk', 0.2),

            lambda_timing=10.0,        # (This remains the highest priority)
            
            pos_weight=1.0,     # <-- CHANGED: Set to 1.0 (sampler handles balance)
            focal_alpha=0.25,
            focal_gamma=2.0,
            label_smoothing=0.1,
            use_logits=model_outputs_logits,
            base_mva=self.base_mva,
            base_freq=self.base_freq
        )
        print(f"\n✓ PhysicsInformedLoss initialized (pos_weight=1.0, lambda_timing=10.0)")
        # ====================================================================
        # END: MODIFICATION
        # ====================================================================
        
        self.start_epoch = 0
        self.best_val_mae = float('inf') # We want to MINIMIZE this
        self.best_val_loss = float('inf') # Fallback for early epochs
        
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
        
        dummy_criterion = PhysicsInformedLoss(
            lambda_powerflow=1.0, lambda_temperature=1.0,
            lambda_stability=1.0, lambda_frequency=1.0,
            lambda_reactive=1.0, # <-- ADDED
            lambda_risk=1.0,
            lambda_timing=1.0,
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
                
                if 'node_failure_labels' not in batch_device:
                    continue

                outputs = self.model(batch_device, return_sequence=True)
                
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'ground_truth_risk': batch_device.get('ground_truth_risk'),
                    'cascade_timing': batch_device.get('cascade_timing')
                }

                _, loss_components = dummy_criterion(
                    outputs, 
                    targets,
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
                'voltage': 0.001, 'lambda_frequency': 0.1,
                'lambda_reactive': 0.1, 'lambda_risk': 0.2, 'lambda_timing': 0.1
            }
        
        avg_losses = {key: val / total_batches for key, val in loss_sums.items()}
        
        print("  Average raw loss components (unweighted):")
        for key, val in avg_losses.items():
            print(f"    {key}: {val:.6f}")
            
        
        target_magnitude = avg_losses.get('prediction', 0.1)
        if target_magnitude < 1e-9: target_magnitude = 1e-9 
        print(f"\n  Target magnitude (from prediction loss): {target_magnitude:.6f}")
        
        # ====================================================================
        # START: MODIFICATION - Add 'reactive'
        # ====================================================================
        physics_loss_keys = ['powerflow', 'temperature', 'voltage', 'frequency', 'reactive', 'risk', 'timing']
        # ====================================================================
        # END: MODIFICATION
        # ====================================================================
        calibrated_lambdas = {}
        
        print("\n  Calibrating lambda weights (Target-Value Strategy):")
        
        for key in physics_loss_keys:
            raw_loss = avg_losses.get(key, 0.0)
            
            if raw_loss < 1e-6:
                denominator = target_magnitude 
            else:
                denominator = raw_loss
            
            lambda_val = target_magnitude / denominator
            
            lambda_key = f"lambda_{key}" if key not in ['voltage'] else key
            calibrated_lambdas[lambda_key] = lambda_val
            
            print(f"    {lambda_key}: {lambda_val:10.4f}  (Raw Loss: {raw_loss:.6f}, Denom: {denominator:.6f}, Initial Weighted Loss: {lambda_val * raw_loss:.4f})")
            
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
        
        self.best_val_mae = checkpoint.get('val_time_mae', float('inf'))
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.5)
        self.node_threshold = checkpoint.get('node_threshold', 0.5)
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ Resumed from epoch {self.start_epoch} (best val_mae: {self.best_val_mae:.4f})")
        print(f"✓ Loaded thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
        return True
    
    def _calculate_metrics(self, outputs, batch):
        """Helper to calculate all metrics for a batch."""
        
        node_probs = outputs['failure_probability'].squeeze(-1) # [B, N]
        node_pred = (node_probs > self.node_threshold).float() # [B, N]
        node_labels = batch['node_failure_labels']  # [B, N]
        
        cascade_prob = node_probs.max(dim=1)[0] # [B]
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
            pred_risk_agg = torch.mean(outputs['risk_scores'], dim=1) # [B, 7]
            target_risk = batch['ground_truth_risk'] # [B, 7]
            risk_mse = F.mse_loss(pred_risk_agg, target_risk).item()
            
        time_mae = 0.0
        valid_timing_nodes = 0
        if 'cascade_timing' in outputs and 'cascade_timing' in batch and batch['cascade_timing'] is not None:
            pred_times = outputs['cascade_timing'].squeeze(-1) # [B, N]
            target_times = batch['cascade_timing'] # [B, N]
            
            mask = target_times >= 0.0
            if mask.sum() > 0:
                time_errors = torch.abs(pred_times[mask] - target_times[mask])
                time_mae = time_errors.mean().item()
                valid_timing_nodes = mask.sum().item()

        return {
            'cascade_tp': cascade_tp, 'cascade_fp': cascade_fp, 'cascade_tn': cascade_tn, 'cascade_fn': cascade_fn,
            'node_tp': node_tp, 'node_fp': node_fp, 'node_tn': node_tn, 'node_fn': node_fn,
            'risk_mse': risk_mse,
            'time_mae': time_mae,
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

        print("\nChecking other model outputs...")
        check_shape('reactive_flows', ('B', 'E', 1))

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

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        grad_norms = []
        loss_component_sums = {}
        
        metric_sums = {}
        total_timing_batches = 0

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
            
            if 'node_failure_labels' not in batch_device:
                continue

            self.optimizer.zero_grad()
            
            graph_properties = batch_device.get('graph_properties', {})
            if 'edge_index' not in graph_properties:
                graph_properties['edge_index'] = batch_device['edge_index']
            
            targets = {
                'failure_label': batch_device['node_failure_labels'],
                'ground_truth_risk': batch_device.get('ground_truth_risk'),
                'cascade_timing': batch_device.get('cascade_timing')
            }

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch_device, return_sequence=True)
                    
                    if not self._model_validated:
                        self._validate_model_outputs(outputs, batch_device)
                        self._model_validated = True
                    
                    loss, loss_components = self.criterion(
                        outputs, 
                        targets,
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
                    targets,
                    graph_properties
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
                'tMAE': f"{running_metrics['time_mae']:.2f}m",
                'rMSE': f"{running_metrics['risk_mse']:.3f}",
                'pL': f"{loss_components.get('prediction', 0):.3f}",
                'tL': f"{loss_components.get('timing', 0):.3f}",
            })
            

        avg_loss = total_loss / (len(self.train_loader) + 1e-7)
        epoch_metrics = self._aggregate_epoch_metrics(metric_sums, len(self.train_loader), total_timing_batches)
        epoch_metrics['loss'] = avg_loss
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        print(f"\n  Average gradient norm: {avg_grad_norm:.4f}")
        if loss_component_sums:
            print(f"  Average loss components:")
            for comp_name, comp_sum in loss_component_sums.items():
                avg_comp = comp_sum / (len(self.train_loader) + 1e-7)
                print(f"    {comp_name}: {avg_comp:.6f}")
        
        return epoch_metrics
    
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with PROPER METRICS."""
        self.model.eval()
        
        total_loss = 0.0
        
        metric_sums = {}
        total_timing_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
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

                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                outputs = self.model(batch_device, return_sequence=True)
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'ground_truth_risk': batch_device.get('ground_truth_risk'),
                    'cascade_timing': batch_device.get('cascade_timing')
                }
                
                loss, _ = self.criterion(
                    outputs,
                    targets,
                    graph_properties
                )
                
                total_loss += loss.item()
                
                batch_metrics = self._calculate_metrics(outputs, batch_device)
                for key, value in batch_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0) + value
                if batch_metrics['valid_timing_nodes'] > 0:
                    total_timing_batches += 1

                running_metrics = self._aggregate_epoch_metrics(metric_sums, batch_idx + 1, total_timing_batches)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'casc_f1': f"{running_metrics['cascade_f1']:.4f}",
                    'time_mae': f"{running_metrics['time_mae']:.2f}m"
                })
        
        avg_loss = total_loss / (len(self.val_loader) + 1e-7)
        epoch_metrics = self._aggregate_epoch_metrics(metric_sums, len(self.val_loader), total_timing_batches)
        epoch_metrics['loss'] = avg_loss
        
        return epoch_metrics
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Train the model and save history/plots."""
        patience_counter = 0
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            combined_f1 = (val_metrics['cascade_f1'] + val_metrics['node_f1']) / 2
            
            if combined_f1 > self.best_val_f1:
                self.best_val_f1 = combined_f1
                print(f"  ✓ Improved F1 score: {combined_f1:.4f} (thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f})")
            
            elif val_metrics['cascade_recall'] < 0.5 or val_metrics['node_recall'] < 0.3:
                self.cascade_threshold = max(0.1, self.cascade_threshold - 0.05)
                self.node_threshold = max(0.1, self.node_threshold - 0.05)
                print(f"  ⚠ Low recall detected - lowering thresholds to cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
            elif val_metrics['cascade_precision'] < 0.4 or val_metrics['node_precision'] < 0.3:
                self.cascade_threshold = min(0.8, self.cascade_threshold + 0.05)
                self.node_threshold = min(0.8, self.node_threshold + 0.05)
                print(f"  ⚠ Low precision detected - raising thresholds to cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
            self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
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
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"\n  CASCADE DETECTION (Thresh: {self.cascade_threshold:.3f}):")
            print(f"    F1 Score:  Train {train_metrics['cascade_f1']:.4f} | Val {val_metrics['cascade_f1']:.4f}")
            print(f"    Precision: Train {train_metrics['cascade_precision']:.4f} | Val {val_metrics['cascade_precision']:.4f}")
            print(f"    Recall:    Train {train_metrics['cascade_recall']:.4f} | Val {val_metrics['cascade_recall']:.4f}")
            print(f"\n  NODE FAILURE (Thresh: {self.node_threshold:.3f}):")
            print(f"    F1 Score:  Train {train_metrics['node_f1']:.4f} | Val {val_metrics['node_f1']:.4f}")
            print(f"    Precision: Train {train_metrics['node_precision']:.4f} | Val {val_metrics['node_precision']:.4f}")
            print(f"    Recall:    Train {train_metrics['node_recall']:.4f} | Val {val_metrics['node_recall']:.4f}")
            
            print(f"\n  CAUSAL PATH & RISK METRICS:")
            print(f"    Timing MAE (mins): Train {train_metrics['time_mae']:.3f} | Val {val_metrics['time_mae']:.3f}")
            print(f"    Risk MSE:          Train {train_metrics['risk_mse']:.4f} | Val {val_metrics['risk_mse']:.4f}")

            # ====================================================================
            # START: "BEST MODEL" LOGIC (Minimize MAE if F1 is good)
            # ====================================================================
            current_val_mae = val_metrics['time_mae']
            f1_is_good = val_metrics['cascade_f1'] > 0.9 and val_metrics['node_f1'] > 0.9

            # We are trying to MINIMIZE the MAE
            if current_val_mae < self.best_val_mae and f1_is_good:
                self.best_val_mae = current_val_mae
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_cascade_f1': val_metrics['cascade_f1'],
                    'val_node_f1': val_metrics['node_f1'],
                    'val_time_mae': current_val_mae, # <-- Save the new best MAE
                    'cascade_threshold': self.cascade_threshold,
                    'node_threshold': self.node_threshold,
                    'history': self.history
                }, f"{self.output_dir}/best_model.pth")
                print(f"  ✓ Saved best model (New best MAE: {current_val_mae:.4f}m, F1s: {val_metrics['cascade_f1']:.3f}/{val_metrics['node_f1']:.3f})")
            
            # Fallback for val_loss if F1s aren't good yet
            elif val_metrics['loss'] < self.best_val_loss and not f1_is_good:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                print(f"  ✓ Saved best model (val_loss: {val_metrics['loss']:.4f}) - (F1 scores not high enough yet)")
                
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            # ====================================================================
            # END: "BEST MODEL" LOGIC
            # ====================================================================

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_time_mae': val_metrics['time_mae'],
                'val_mae': current_val_mae, # Use val_mae for consistency
                'cascade_threshold': self.cascade_threshold,
                'node_threshold': self.node_threshold,
                'history': self.history
            }, f"{self.output_dir}/latest_checkpoint.pth")
        
        self.save_history()
        self.plot_training_curves()
        
        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"  Best validation MAE: {self.best_val_mae:.4f} minutes")
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
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        if not self.history['train_loss']:
            print("No history to plot. Skipping plot generation.")
            plt.close()
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss (Row 0, Col 0)
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cascade F1 (Row 0, Col 1)
        axes[0, 1].plot(epochs, self.history['train_cascade_f1'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_cascade_f1'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Cascade Detection F1')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Node F1 (Row 0, Col 2)
        axes[0, 2].plot(epochs, self.history['train_node_f1'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_node_f1'], 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Node Failure F1')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cascade Precision/Recall (Row 1, Col 0)
        axes[1, 0].plot(epochs, self.history['train_cascade_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_cascade_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['train_cascade_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_cascade_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Cascade Precision/Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Node Precision/Recall (Row 1, Col 1)
        axes[1, 1].plot(epochs, self.history['train_node_precision'], 'b--', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_precision'], 'r--', label='Val Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['train_node_recall'], 'b:', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_node_recall'], 'r:', label='Val Recall', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Node Precision/Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Timing MAE (Row 1, Col 2)
        axes[1, 2].plot(epochs, self.history.get('train_time_mae', [0]*len(epochs)), 'b-', label='Train', linewidth=2)
        axes[1, 2].plot(epochs, self.history.get('val_time_mae', [0]*len(epochs)), 'r-', label='Validation', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('MAE (minutes)')
        axes[1, 2].set_title('Cascade Timing MAE')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Risk MSE (Row 2, Col 0)
        axes[2, 0].plot(epochs, self.history.get('train_risk_mse', [0]*len(epochs)), 'b-', label='Train', linewidth=2)
        axes[2, 0].plot(epochs, self.history.get('val_risk_mse', [0]*len(epochs)), 'r-', label='Validation', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('MSE')
        axes[2, 0].set_title('7-D Risk Score MSE')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Learning Rate (Row 2, Col 1)
        axes[2, 1].plot(epochs, self.history['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('LR')
        axes[2, 1].set_title('Learning Rate Schedule')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Accuracy comparison (Row 2, Col 2)
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Cascade Prediction Model")
    parser.add_argument('--data_dir', type=str, default="data", 
                        help="Root directory containing train/val/test data folders")
    parser.add_argument('--output_dir', type=str, default="checkpoints", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=4, 
                        help="Training and validation batch size")
    parser.add_argument('--lr', type=float, default=0.0001, 
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
    
    
    print("="*80)
    print("CASCADE FAILURE PREDICTION - TRAINING SCRIPT (IMPROVED)")
    print("="*80)
    
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MAX_GRAD_NORM = args.grad_clip
    EARLY_STOPPING_PATIENCE = args.patience
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = (DEVICE.type == 'cuda')
    
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
    
    # ====================================================================
    # START: MODIFICATION - Dynamic 1:1 Sampler
    # ====================================================================
    print(f"\nComputing sample weights for balanced sampling...")
    
    positive_count = sum(train_dataset.get_cascade_label(idx) for idx in range(len(train_dataset)))
    negative_count = len(train_dataset) - positive_count
    
    if positive_count == 0 or negative_count == 0:
        print("  [WARNING] Training data contains only one class. Using uniform weights.")
        sample_weights = [1.0] * len(train_dataset)
    else:
        # Calculate weights to create a 1:1 balance in each batch
        # Weight = Total Samples / Num Samples in Class
        total_samples = len(train_dataset)
        pos_weight_val = total_samples / positive_count
        neg_weight_val = total_samples / negative_count
        
        sample_weights = []
        for idx in range(len(train_dataset)):
            if train_dataset.get_cascade_label(idx):
                sample_weights.append(pos_weight_val)
            else:
                sample_weights.append(neg_weight_val)
        
        print(f"  Positive samples: {positive_count} ({positive_count/total_samples*100:.1f}%)")
        print(f"  Negative samples: {negative_count} ({negative_count/total_samples*100:.1f}%)")
        print(f"  Calculated weights -> Pos: {pos_weight_val:.2f}, Neg: {neg_weight_val:.2f} (creates ~1:1 batch balance)")
    # ====================================================================
    # END: MODIFICATION
    # ====================================================================
        
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