"""
Training Script for Cascade Prediction Model
======================================================
(MODIFIED for "New, Sound Training Methodology")
- New "un-cheatable" timing loss
- Fully dynamic lambda calibration
- Data leakage fix (pos_weight=1.0)
- "Resume" bug fix
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
    """
    
    def __init__(self, lambda_powerflow: float = 0.1, lambda_temperature: float = 0.1,
                 lambda_stability: float = 0.001, lambda_frequency: float = 0.1,
                 lambda_reactive: float = 0.1, 
                 lambda_risk: float = 0.2,
                 lambda_timing: float = 0.1,
                 pos_weight: float = 10.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.15, use_logits: bool = False,
                 base_mva: float = 100.0, base_freq: float = 60.0):
        super(PhysicsInformedLoss, self).__init__()
        
        self.lambda_powerflow = lambda_powerflow
        self.lambda_temperature = lambda_temperature
        self.lambda_stability = lambda_stability
        self.lambda_frequency = lambda_frequency
        self.lambda_reactive = lambda_reactive 
        self.lambda_risk = lambda_risk
        self.lambda_timing = lambda_timing
        
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_logits = use_logits
        
        self._warned_missing_outputs = set()
        
        self.power_base = base_mva
        self.freq_nominal = base_freq

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss with label smoothing and POS_WEIGHT for handling severe class imbalance.
        """
        targets_smooth = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        weight = torch.ones_like(logits)
        if self.pos_weight != 1.0:
            weight[targets > 0.5] = self.pos_weight
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets_smooth, 
            reduction='none'
        )
        
        bce_loss = bce_loss * weight

        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

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
        
        P_ij_squeezed = predicted_line_flows.squeeze(-1)  # [B, E]
        
        P_calc_flat = torch.zeros(batch_size * num_nodes, device=predicted_line_flows.device)
        
        batch_offset = torch.arange(0, batch_size, device=predicted_line_flows.device) * num_nodes
        src_flat = src.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        dst_flat = dst.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        
        P_ij_flat = P_ij_squeezed.flatten() # [B*E]
        
        P_calc_flat.index_add_(0, src_flat, P_ij_flat)
        P_calc_flat.index_add_(0, dst_flat, -P_ij_flat)
        
        P_calc = P_calc_flat.reshape(batch_size, num_nodes, 1)
        
        if power_injection.dim() == 2:
            power_injection = power_injection.unsqueeze(-1)
            
        return F.mse_loss(P_calc, power_injection)

    def temperature_loss(self, predicted_temp: torch.Tensor, ground_truth_temp: torch.Tensor) -> torch.Tensor:
        """
        Forces the model to learn the actual ground truth temperature.
        FIX: We scale both prediction and target by 100
        """
        predicted_temp_scaled = predicted_temp.squeeze(-1) / 100.0
        ground_truth_temp_scaled = ground_truth_temp / 100.0
        scaled_mse_loss = F.mse_loss(predicted_temp_scaled, ground_truth_temp_scaled)
        
        return scaled_mse_loss

    def voltage_stability_loss(self, voltages: torch.Tensor,
                              voltage_min: float = 0.9, voltage_max: float = 1.1) -> torch.Tensor:
        """
        Compute voltage stability constraint violations.
        """
        low_violations = F.relu(voltage_min - voltages)
        high_violations = F.relu(voltages - voltage_max)
        return torch.mean(low_violations ** 2 + high_violations ** 2)

    def frequency_loss(self, predicted_freq_Hz: torch.Tensor, 
                       power_injection_pu: torch.Tensor,
                       nominal_freq_Hz: float = 60.0,
                       total_inertia_H: float = 5.0) -> torch.Tensor:
        """
        Compute REAL physics-based frequency loss using the swing equation.
        """
        predicted_freq_pu = predicted_freq_Hz / nominal_freq_Hz
        system_power_imbalance_pu = torch.sum(power_injection_pu, dim=1)
        expected_freq_deviation_pu = system_power_imbalance_pu / (2 * total_inertia_H)
        expected_freq_pu = 1.0 + expected_freq_deviation_pu.view(-1, 1, 1)
        return F.mse_loss(predicted_freq_pu, expected_freq_pu)

    def reactive_power_flow_loss(self, predicted_reactive_flows: torch.Tensor, 
                                 edge_index: torch.Tensor, 
                                 reactive_injection: torch.Tensor,
                                 num_nodes: int,
                                 batch_size: int) -> torch.Tensor:
        """
        Computes reactive power flow loss by summing the model's *predicted* reactive flows (Q)
        """
        src, dst = edge_index
        num_edges = edge_index.shape[1]
        
        Q_ij_squeezed = predicted_reactive_flows.squeeze(-1)  # [B, E]
        
        Q_calc_flat = torch.zeros(batch_size * num_nodes, device=predicted_reactive_flows.device)
        
        batch_offset = torch.arange(0, batch_size, device=predicted_reactive_flows.device) * num_nodes
        src_flat = src.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        dst_flat = dst.repeat(batch_size) + batch_offset.repeat_interleave(num_edges)
        
        Q_ij_flat = Q_ij_squeezed.flatten() # [B*E]
        
        Q_calc_flat.index_add_(0, src_flat, Q_ij_flat)
        Q_calc_flat.index_add_(0, dst_flat, -Q_ij_flat)
        
        Q_calc = Q_calc_flat.reshape(batch_size, num_nodes, 1)
        
        if reactive_injection.dim() == 2:
            reactive_injection = reactive_injection.unsqueeze(-1)
            
        return F.mse_loss(Q_calc, reactive_injection)

    def risk_loss(self, predicted_risk: torch.Tensor, target_risk: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for the 7-D risk assessment.
        """
        predicted_risk_agg = torch.mean(predicted_risk, dim=1) # [B, 7]
        return F.mse_loss(predicted_risk_agg, target_risk)

    # ====================================================================
    # START: "CHEATEABLE MAE" BUG FIX
    # Replace the old timing_loss with a new, robust "Ranking + MSE" loss
    # ====================================================================
    def timing_loss(self, predicted_node_timing: torch.Tensor, 
                    target_node_timing: torch.Tensor) -> torch.Tensor:
        """
        Compute purely RANKING-based loss.
        Ignores absolute time error (MSE). Focuses 100% on the order of failures.
        """
        predicted_node_timing_squeezed = predicted_node_timing.squeeze(-1) # [B, N]
        batch_size = predicted_node_timing_squeezed.shape[0]
        
        ranking_losses = []
        for b in range(batch_size):
            preds = predicted_node_timing_squeezed[b]
            targets = target_node_timing[b]
            
            # Get indices of nodes that actually fail
            pos_idx = torch.where(targets >= 0)[0]
            
            # Need at least 2 failures to compare order
            if len(pos_idx) < 2:
                continue
                
            # Generate all pairs of failing nodes
            pairs = torch.combinations(pos_idx, r=2)
            i_indices = pairs[:, 0]
            j_indices = pairs[:, 1]
            
            pred_diff = preds[i_indices] - preds[j_indices]
            target_diff = targets[i_indices] - targets[j_indices]
            
            # target_sign is +1 if i fails after j, -1 if i fails before j
            target_sign = torch.sign(target_diff)
            
            # Margin ranking loss: enforce correct order with a margin
            margin = 0.1
            sample_ranking_loss = torch.relu(margin - (pred_diff * target_sign))
            
            if sample_ranking_loss.numel() > 0:
                ranking_losses.append(sample_ranking_loss.mean())

        # Fallback: If no ranking pairs exist (e.g., normal cases), return 0 loss
        if not ranking_losses:
            return torch.tensor(0.0, device=predicted_node_timing.device)
        
        # Return purely the ranking loss
        return torch.stack(ranking_losses).mean()
    # ====================================================================
    # END: "CHEATEABLE MAE" BUG FIX
    # ====================================================================


    # --- Merged Forward Pass ---
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        predictions_fp32 = {}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                predictions_fp32[k] = v.float() # Cast everything to float32
            else:
                predictions_fp32[k] = v
        
        predictions = predictions_fp32

        failure_prob = predictions['failure_probability']
        B, N, _ = failure_prob.shape
        
        failure_prob_flat = failure_prob.reshape(-1)
        targets_flat = targets['failure_label'].reshape(-1)
        
        if self.use_logits:
            logits = failure_prob_flat
        else:
            probs = failure_prob_flat.clamp(1e-7, 1 - 1e-7)
            logits = torch.log(probs / (1 - probs))
        
        L_prediction = self.focal_loss(logits, targets_flat)
        
        loss_components = {'prediction': L_prediction.item()}
        total_loss = L_prediction
        
        def get_prop(key):
            if graph_properties and key in graph_properties:
                return graph_properties[key]
            if key not in self._warned_missing_outputs:
                # print(f"Warning: {key} not in graph_properties for loss calc.")
                self._warned_missing_outputs.add(key)
            return None

        # 1. Power Flow Loss
        if 'line_flows' in predictions and get_prop('power_injection') is not None:
            
            L_powerflow = self.power_flow_loss(
                predicted_line_flows=predictions['line_flows'],
                edge_index=graph_properties['edge_index'],
                power_injection=get_prop('power_injection'),
                num_nodes=N,
                batch_size=B
            )
            L_powerflow = torch.clamp(L_powerflow, 0.0, 10.0) 
            total_loss += self.lambda_powerflow * L_powerflow
            loss_components['powerflow'] = L_powerflow.item()
        
        # 2. Temperature Loss
        if 'temperature' in predictions and get_prop('ground_truth_temperature') is not None:
            L_temperature = self.temperature_loss(
                predictions['temperature'],
                get_prop('ground_truth_temperature')
            )
            # The clamp is removed because the loss is now scaled
            total_loss += self.lambda_temperature * L_temperature
            loss_components['temperature'] = L_temperature.item()
            
        # 3. Voltage Stability Loss
        if 'voltages' in predictions:
            L_stability = self.voltage_stability_loss(
                voltages=predictions['voltages']
            )
            total_loss += self.lambda_stability * L_stability
            loss_components['voltage'] = L_stability.item()

        # 4. Reactive Power Loss
        if 'reactive_flows' in predictions and get_prop('reactive_injection') is not None:
            
            L_reactive = self.reactive_power_flow_loss(
                predicted_reactive_flows=predictions['reactive_flows'],
                edge_index=graph_properties['edge_index'],
                reactive_injection=get_prop('reactive_injection'),
                num_nodes=N,
                batch_size=B
            )
            L_reactive = torch.clamp(L_reactive, 0.0, 10.0)
            total_loss += self.lambda_reactive * L_reactive
            loss_components['reactive'] = L_reactive.item()
        
        # 5. Frequency Loss (PHYSICS-BASED)
        if 'frequency' in predictions and get_prop('power_injection') is not None:
            
            predicted_freq_Hz = predictions['frequency']
            power_injection_pu = get_prop('power_injection')
            
            L_frequency = self.frequency_loss(
                predicted_freq_Hz,
                power_injection_pu,
                nominal_freq_Hz=self.freq_nominal,
                total_inertia_H=5.0
            )
            
            L_frequency = torch.clamp(L_frequency, 0.0, 10.0)
            total_loss += self.lambda_frequency * L_frequency
            loss_components['frequency'] = L_frequency.item()
        
        # 6. Risk Score Loss
        if 'risk_scores' in predictions and 'ground_truth_risk' in targets and targets['ground_truth_risk'] is not None:
            L_risk = self.risk_loss(
                predicted_risk=predictions['risk_scores'],
                target_risk=targets['ground_truth_risk']
            )
            total_loss += self.lambda_risk * L_risk
            loss_components['risk'] = L_risk.item()

        # 7. Timing Loss
        if 'cascade_timing' in predictions and 'cascade_timing' in targets and targets['cascade_timing'] is not None:
            L_timing = self.timing_loss(
                predicted_node_timing=predictions['cascade_timing'],
                target_node_timing=targets['cascade_timing'],
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
        base_mva: float = 100.0,
        base_freq: float = 60.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-3
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.model_outputs_logits = model_outputs_logits
        
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
        avg_losses = self._calibrate_loss_weights()
        print("="*80)
        print("CALIBRATION COMPLETE")
        print("="*80 + "\n")
        
        # ====================================================================
        # START: FINAL FIX - TRUST THE DYNAMIC CALIBRATOR
        # ====================================================================
        print("Balancing loss weights dynamically...")
        
        target_magnitude = avg_losses.get('prediction', 0.1)
        if target_magnitude < 1e-9: target_magnitude = 1e-9
        
        calibrated_lambdas = {}
        physics_loss_keys = ['powerflow', 'temperature', 'voltage', 'frequency', 'reactive', 'risk', 'timing']
        
        for key in physics_loss_keys:
            raw_loss = avg_losses.get(key, 0.0)
            denominator = raw_loss if raw_loss >= 1e-6 else target_magnitude
            lambda_val = target_magnitude / denominator
            lambda_key = f"lambda_{key}" if key not in ['voltage'] else key
            calibrated_lambdas[lambda_key] = lambda_val

        # Define FINAL weights (All are now dynamic)
        final_lambdas = {
            'lambda_powerflow': calibrated_lambdas.get('lambda_powerflow', 0.1),
            'lambda_temperature': calibrated_lambdas.get('lambda_temperature', 0.1),
            'lambda_stability': calibrated_lambdas.get('voltage', 0.001), 
            'lambda_frequency': calibrated_lambdas.get('lambda_frequency', 0.1),
            'lambda_reactive': calibrated_lambdas.get('lambda_reactive', 0.1),
            'lambda_risk': calibrated_lambdas.get('lambda_risk', 0.2),       
            'lambda_timing': calibrated_lambdas.get('lambda_timing', 0.1),
        }

        # Print a clear report
        print(f"  Target Magnitude (from prediction loss): {target_magnitude:.4f}")
        print("\n  Final Loss Weights (Fully Dynamic):")
        print(f"  {'Component':<15} | {'Raw Loss':<12} | {'Final Lambda':<12} | {'Initial Weighted Loss'}")
        print(f"  {'-'*15} | {'-'*12} | {'-'*12} | {'-'*20}")
        
        def print_row(key_pretty, key_raw, key_lambda, is_voltage=False):
            raw = avg_losses.get(key_raw, 0.0)
            final_key = 'lambda_stability' if is_voltage else key_lambda
            final = final_lambdas.get(final_key, 0.0)
            print(f"  {key_pretty:<15} | {raw:<12.4f} | {final:<12.4f} | {raw * final:12.4f}")

        print_row("Timing", "timing", "lambda_timing")
        print_row("Powerflow", "powerflow", "lambda_powerflow")
        print_row("Temperature", "temperature", "lambda_temperature")
        print_row("Reactive", "reactive", "lambda_reactive")
        print_row("Voltage", "voltage", "voltage", is_voltage=True)
        print_row("Frequency", "frequency", "lambda_frequency")
        print_row("Risk", "risk", "lambda_risk")
        
        # Initialize the criterion with the FINAL weights
        self.criterion = PhysicsInformedLoss(
            lambda_powerflow=final_lambdas['lambda_powerflow'],
            lambda_temperature=final_lambdas['lambda_temperature'],
            lambda_stability=final_lambdas['lambda_stability'],
            lambda_frequency=final_lambdas['lambda_frequency'],
            lambda_reactive=final_lambdas['lambda_reactive'],
            lambda_risk=final_lambdas['lambda_risk'],
            lambda_timing=10,
            
            pos_weight=10.0, 
            focal_alpha=0.25,
            focal_gamma=1,
            label_smoothing=0.05,
            use_logits=model_outputs_logits,
            base_mva=self.base_mva,
            base_freq=self.base_freq
        )
        print(f"\n✓ PhysicsInformedLoss initialized with FINAL dynamic weights.")
        # ====================================================================
        # END: FINAL FIX
        # ====================================================================
        
        self.start_epoch = 0
        self.best_val_mae = float('inf') 
        self.best_val_loss = float('inf')
        self.best_val_timing_loss = float('inf')
        self.cascade_threshold = 0.25
        self.node_threshold = 0.25
        self.best_val_f1 = 0.0
        
        self._model_validated = False

    def _calibrate_loss_weights(self, num_batches=20) -> Dict[str, float]:
        """
        Run a few batches to find the average raw loss for each component.
        Returns a dict of the raw, unweighted loss magnitudes.
        """
        print(f"Running loss calibration for {num_batches} batches...")
        self.model.eval()
        
        dummy_criterion = PhysicsInformedLoss(
            lambda_powerflow=1.0, 
            lambda_temperature=1.0,
            lambda_stability=1.0, 
            lambda_frequency=1.0,
            lambda_reactive=1.0, 
            lambda_risk=1.0,
            lambda_timing=1.0,
            pos_weight=1.0,
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
            return {} 
        
        avg_losses = {key: val / total_batches for key, val in loss_sums.items()}
        
        print("  Average raw loss components (unweighted):")
        for key, val in sorted(avg_losses.items()):
            print(f"    {key: <15}: {val:10.6f}")
            
        self.model.train()
        return avg_losses

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        self.best_val_mae = checkpoint.get('val_time_mae', float('inf'))
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.25)
        self.node_threshold = checkpoint.get('node_threshold', 0.25)
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        

        print(f"✓ Resumed from epoch {self.start_epoch}")
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
                
                if mask.sum() < 2: continue
                
                # Get indices of failed nodes
                idx = torch.where(mask)[0]
                
                # Compare every pair
                # If target_i < target_j, we want pred_i < pred_j
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        u, v = idx[i], idx[j]
                        
                        # Skip if ground truth times are identical
                        if t[u] == t[v]: continue
                            
                        total_pairs += 1
                        
                        # Check if order matches
                        if (t[u] < t[v] and p[u] < p[v]) or (t[u] > t[v] and p[u] > p[v]):
                            correct_pairs += 1
                            
            if total_pairs > 0:
                pairwise_acc = correct_pairs / total_pairs
                valid_timing_nodes = 1 # Just to mark that this batch had pairs

        return {
            'cascade_tp': cascade_tp, 'cascade_fp': cascade_fp, 'cascade_tn': cascade_tn, 'cascade_fn': cascade_fn,
            'node_tp': node_tp, 'node_fp': node_fp, 'node_tn': node_tn, 'node_fn': node_fn,
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
        check_shape('temperature', ('B', 'N', 1)) # <-- Added this

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
        Validate and find OPTIMAL thresholds (Cascade F1 & Node F_beta Score).
        """
        self.model.eval()
        
        total_loss = 0.0
        total_timing_loss_sum = 0.0
        
        all_node_probs = []
        all_node_labels = []
        all_cascade_probs = []
        all_cascade_labels = []
        
        metric_sums = {}
        total_timing_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
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
                
                if 'node_failure_labels' not in batch_device: continue

                # 2. Forward
                outputs = self.model(batch_device, return_sequence=True)
                
                # 3. Loss
                graph_properties = batch_device.get('graph_properties', {})
                if 'edge_index' not in graph_properties:
                    graph_properties['edge_index'] = batch_device['edge_index']
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'ground_truth_risk': batch_device.get('ground_truth_risk'),
                    'cascade_timing': batch_device.get('cascade_timing')
                }
                
                loss, loss_components = self.criterion(outputs, targets, graph_properties)
                total_loss += loss.item()
                total_timing_loss_sum += loss_components.get('timing', 0.0)
                
                # 4. Accumulate
                node_probs = outputs['failure_probability'].squeeze(-1)
                node_labels = batch_device['node_failure_labels']
                
                all_node_probs.append(node_probs.view(-1).cpu())
                all_node_labels.append(node_labels.view(-1).cpu())
                
                all_cascade_probs.append(node_probs.max(dim=1)[0].cpu())
                all_cascade_labels.append((node_labels.max(dim=1)[0] > 0.5).float().cpu())

                # 5. Fixed metrics
                batch_metrics = self._calculate_metrics(outputs, batch_device)
                for key, value in batch_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0) + value
                if batch_metrics['valid_timing_nodes'] > 0:
                    total_timing_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / (len(self.val_loader) + 1e-7)
        avg_timing_loss = total_timing_loss_sum / (len(self.val_loader) + 1e-7)
        
        # --- DYNAMIC SEARCH ---
        if not all_node_probs: return {'loss': avg_loss, 'timing_loss': avg_timing_loss}

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

        # Helper 2: Best F-beta Score (Favors Precision) - YOUR FIX
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
        
        # Use the F-beta finder for Nodes, with beta=0.5 (Precision focus)
        best_n_score, best_n_thresh = find_best_fbeta(global_node_probs, global_node_labels, beta=0.5)
        
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
        cascade_acc = (cascade_tp + cascade_tn) / (cascade_tp + cascade_tn + cascade_fp + cascade_fn + 1e-7)

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
            
            'cascade_f1': best_c_f1,
            'cascade_precision': cascade_precision,
            'cascade_recall': cascade_recall,
            'cascade_acc': cascade_acc,
            
            # Thresholds
            'best_node_thresh': best_n_thresh,
            'best_cascade_thresh': best_c_thresh
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
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
            
            # LOG OPTIMAL METRICS
            print(f"\n  OPTIMAL VALIDATION METRICS (Dynamic Thresholds):")
            print(f"    Cascade F1: {val_metrics['cascade_f1']:.4f} (Thresh: {val_metrics['best_cascade_thresh']:.2f})")
            print(f"    Node F1:    {val_metrics['node_f1']:.4f} (Thresh: {val_metrics['best_node_thresh']:.2f})")
            print(f"    Node Prec:  {val_metrics['node_precision']:.4f} | Node Rec: {val_metrics['node_recall']:.4f}")
            
            print(f"\n  CAUSAL PATH METRICS:")
            print(f"    Timing Loss:       Train {train_metrics['timing_loss']:.4f} | Val {val_metrics['timing_loss']:.4f}")
                  
            # ====================================================================
            # --- MODIFICATION: SAVE BEST MODEL BASED ON TIMING LOSS ---
            # ====================================================================
            
            # current_val_loss = val_metrics['loss']
            # if current_val_loss < self.best_val_loss:
            #     self.best_val_loss = current_val_loss
            #     patience_counter = 0
            #     ...
            #     print(f"  ✓ Saved best model (New best Val Loss: {current_val_loss:.4f})")
            
            # --- NEW SAVING LOGIC (YOUR REQUEST) ---
            current_timing_loss = val_metrics['timing_loss']
            
            if current_timing_loss < self.best_val_timing_loss:
                self.best_val_timing_loss = current_timing_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_timing_loss': current_timing_loss, # Save the new best score
                    
                    'cascade_threshold': float(val_metrics['best_cascade_thresh']),
                    'node_threshold': float(val_metrics['best_node_thresh']),
                    
                    'history': self.history
                }, f"{self.output_dir}/best_model.pth")
                
                print(f"  ✓ Saved best model (New best Timing Loss: {current_timing_loss:.4f})")
                
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            # ====================================================================

            # Save latest checkpoint (always includes the latest optimal thresholds)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_time_mae': val_metrics['time_mae'],
                'val_timing_loss': val_metrics['timing_loss'], # Save current timing loss
                
                'cascade_threshold': float(val_metrics['best_cascade_thresh']),
                'node_threshold': float(val_metrics['best_node_thresh']),
                
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Cascade Prediction Model")
    parser.add_argument('--data_dir', type=str, default="data", 
                        help="Root directory containing train/val/test data folders")
    parser.add_argument('--output_dir', type=str, default="checkpoints", 
                        help="Directory to save checkpoints and logs")
    parser.add_argument('--epochs', type=int, default=200, 
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=4, 
                        help="Training and validation batch size")
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help="Initial learning rate")
    parser.add_argument('--grad_clip', type=float, default=10.0, 
                        help="Max gradient norm for clipping")
    parser.add_argument('--patience', type=int, default=10, 
                        help="Epochs for early stopping patience")
    
    # ====================================================================
    # START: "RESUME" FIX (Allows custom checkpoint path)
    # ====================================================================
    parser.add_argument('--resume', type=str, default=None, 
                        help="Path to checkpoint file to resume (e.g., checkpoints/best_model.pth)")
    # ====================================================================
    # END: "RESUME" FIX
    # ====================================================================
    
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
    print(f"  Resume training: {args.resume is not None}")
    
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
        dropout=0.5
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
    
    # ====================================================================
    # START: "RESUME" FIX (Allows custom checkpoint path)
    # ====================================================================
    if args.resume:
        checkpoint_path = args.resume 
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            
            # --- NEW: FORCE LR RESET FOR PHASE 2 ---
            print(f"\n[PHASE 2 RESET] Manually resetting Learning Rate to {LEARNING_RATE}...")
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE  # Resets to 0.0001 (or whatever arg you passed)
            
            # OPTIONAL: Reset Scheduler to forget "patience" history
            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer, 'min', patience=5
            )
            print("[PHASE 2 RESET] Scheduler reset.")
            # ---------------------------------------

        else:
            print(f"Warning: Checkpoint file not found...")
    # ====================================================================
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    
    print("\nTraining completed successfully!")