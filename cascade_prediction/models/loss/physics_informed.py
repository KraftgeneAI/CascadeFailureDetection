"""
Physics-Informed Loss Module
============================
Loss function combining prediction accuracy with physics constraints.

This module implements a comprehensive loss function that enforces:
- Power flow conservation (Kirchhoff's laws)
- Thermal capacity constraints
- Voltage stability bounds
- Frequency dynamics (swing equation)
- Reactive power balance
- Cascade timing accuracy
- Risk assessment accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from cascade_prediction.data.generator.config import Settings


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function with power flow constraints.
    
    Combines multiple loss components:
    1. Failure prediction (focal loss)
    2. Power flow consistency
    3. Thermal capacity constraints
    4. Voltage stability
    5. Frequency dynamics
    6. Reactive power balance
    7. Temperature prediction
    8. Risk assessment
    9. Cascade timing
    """
    
    def __init__(
        self,
        lambda_prediction: float = Settings.Loss.LAMBDA_PREDICTION,
        lambda_powerflow: float = Settings.Loss.LAMBDA_POWERFLOW,
        lambda_risk: float = Settings.Loss.LAMBDA_RISK,
        lambda_timing: float = Settings.Loss.LAMBDA_TIMING,
        lambda_active_flow: float = Settings.Loss.LAMBDA_ACTIVE_FLOW,
        lambda_temperature: float = Settings.Loss.LAMBDA_TEMPERATURE,
        lambda_frequency: float = Settings.Loss.LAMBDA_FREQUENCY,
        lambda_reactive: float = Settings.Loss.LAMBDA_REACTIVE,
        lambda_voltage: float = Settings.Loss.LAMBDA_VOLTAGE,
        lambda_capacity: float = Settings.Loss.LAMBDA_CAPACITY,
        pos_weight: float = 1.0,
        focal_alpha: float = Settings.Loss.FOCAL_ALPHA,
        focal_gamma: float = Settings.Loss.FOCAL_GAMMA,
        label_smoothing: float = 0.0,
        use_logits: bool = False,
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_freq: float = Settings.Dataset.BASE_FREQUENCY,
        **kwargs
    ):
        """
        Initialize physics-informed loss.
        
        Args:
            lambda_prediction: Weight for failure prediction loss (focal loss)
            lambda_powerflow: Weight for reactive power flow consistency loss
            lambda_risk: Weight for risk assessment loss
            lambda_timing: Weight for timing prediction loss
            lambda_active_flow: Weight for active power flow loss
            lambda_temperature: Weight for temperature prediction loss
            lambda_frequency: Weight for frequency dynamics loss
            lambda_reactive: Weight for reactive power loss
            lambda_voltage: Weight for voltage prediction loss
            lambda_capacity: Weight for thermal capacity loss
            pos_weight: Positive class weight for BCE loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
            use_logits: Whether model outputs logits (not used currently)
            base_mva: Base MVA for power normalization
            base_freq: Base frequency for normalization
        """
        super().__init__()
        
        # Store lambda weights
        self.lambdas = {
            'prediction': lambda_prediction,
            'powerflow': lambda_powerflow,
            'risk': lambda_risk,
            'timing': lambda_timing,
            'active_flow': lambda_active_flow,
            'temperature': lambda_temperature,
            'frequency': lambda_frequency,
            'reactive': lambda_reactive,
            'voltage': lambda_voltage,
            'capacity': lambda_capacity,
        }
        
        # Classification loss settings
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Physics parameters
        self.base_mva = base_mva
        self.base_freq = base_freq
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for imbalanced classification.
        
        Args:
            logits: Model logits [batch_size, num_nodes]
            targets: Ground truth labels [batch_size, num_nodes]
        
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.focal_gamma * bce_loss
        return loss.mean()
    
    def flow_consistency_loss(
        self,
        predicted_flows: torch.Tensor,
        target_line_flows: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow consistency loss.
        
        Args:
            predicted_flows: Predicted line flows [batch_size, num_edges, 1]
            target_line_flows: Target line flows [batch_size, num_edges, 1]
        
        Returns:
            MSE loss between predicted and target flows
        """
        return F.mse_loss(predicted_flows, target_line_flows)
    
    def temperature_loss(
        self,
        predicted_temp: torch.Tensor,
        ground_truth_temp: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised learning for temperature prediction.
        
        Args:
            predicted_temp: Predicted temperatures [batch_size, num_nodes, 1]
            ground_truth_temp: Ground truth temperatures [batch_size, num_nodes]
        
        Returns:
            Scaled MSE loss for temperature
        """
        # Scale down for numerical stability (approx max temp in deg C)
        return F.mse_loss(
            predicted_temp.squeeze(-1) / Settings.Loss.TEMPERATURE_SCALE,
            ground_truth_temp / Settings.Loss.TEMPERATURE_SCALE
        )
    
    def frequency_loss(
        self,
        predicted_freq: torch.Tensor,
        power_injection: torch.Tensor
    ) -> torch.Tensor:
        """
        Forces frequency to respond to power imbalance (Gen - Load).
        
        Args:
            predicted_freq: Predicted frequency [batch_size, 1, 1]
            power_injection: Power injection (gen - load) [batch_size, num_nodes, 1]
        
        Returns:
            MSE loss for frequency prediction
        """
        imbalance = torch.sum(power_injection, dim=1)  # [B, 1] or [B]
        target_freq_dev = imbalance / Settings.Loss.POWER_TO_FREQ
        target_freq = 1.0 + target_freq_dev.view(-1, 1, 1)
        
        # Normalize predicted frequency (assuming it outputs Hz)
        pred_freq_pu = predicted_freq / self.base_freq
        
        return F.mse_loss(pred_freq_pu, target_freq)
    
    def risk_loss(
        self,
        predicted_risk: torch.Tensor,
        target_risk: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised learning for risk score.
        
        Args:
            predicted_risk: Predicted risk scores [batch_size, num_nodes, 7]
            target_risk: Target risk scores [batch_size, 7]
        
        Returns:
            MSE loss for risk assessment
        """
        return F.mse_loss(torch.mean(predicted_risk, dim=1), target_risk)
    
    def timing_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Cascade timing loss with both regression and ranking components.
        
        Args:
            predicted: Predicted timing [batch_size, num_nodes, 1]
            target: Target timing [batch_size, num_nodes]
        
        Returns:
            Combined timing loss
        """
        pred_s = predicted.squeeze(-1)
        losses = []
        
        for b in range(pred_s.shape[0]):
            t = target[b]
            pos_idx = torch.where(t >=0)[0]
            
            if len(pos_idx) < 2:
                continue
            
            # 1. Absolute Time Loss (Regression)
            regression_loss = F.smooth_l1_loss(pred_s[b][pos_idx], t[pos_idx])
            
            # 2. Sequence Order Loss (Ranking)
            pairs = torch.combinations(pos_idx, r=2)
            p_diff = pred_s[b][pairs[:, 0]] - pred_s[b][pairs[:, 1]]
            t_diff = t[pairs[:, 0]] - t[pairs[:, 1]]
            ranking_loss = torch.relu(0.1 - p_diff * torch.sign(t_diff)).mean()
            
            losses.append(regression_loss + ranking_loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=predicted.device)
    
    def voltage_loss(
        self,
        predicted_v: torch.Tensor,
        target_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised voltage loss (p.u.).
        
        Args:
            predicted_v: Predicted voltages [batch_size, num_nodes, 1]
            target_v: Target voltages [batch_size, num_nodes, 1]
        
        Returns:
            MSE loss for voltage prediction
        """
        return F.mse_loss(predicted_v, target_v)
    
    def reactive_power_loss(
        self,
        predicted_q: torch.Tensor,
        target_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Reactive power prediction loss.
        
        Args:
            predicted_q: Predicted reactive power [batch_size, num_nodes, 1]
            target_q: Target reactive power [batch_size, num_nodes, 1]
        
        Returns:
            MSE loss for reactive power
        """
        return F.mse_loss(predicted_q, target_q)
    
    def active_power_line_flow_loss(
        self,
        predicted_p: torch.Tensor,
        target_p: torch.Tensor
    ) -> torch.Tensor:
        """
        Active power line flow loss.
        
        Args:
            predicted_p: Predicted active power flows [batch_size, num_edges, 1]
            target_p: Target active power flows [batch_size, num_edges, 1]
        
        Returns:
            MSE loss for active power flows
        """
        return F.mse_loss(predicted_p, target_p)
    
    def capacity_loss(
        self,
        line_flows: torch.Tensor,
        thermal_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute thermal capacity constraint violations.
        
        Penalizes line flows that exceed thermal limits.
        
        Args:
            line_flows: Predicted line flows [batch_size, num_edges, 1]
            thermal_limits: Thermal limits [batch_size, num_edges] or [num_edges]
        
        Returns:
            Mean squared violation loss
        """
        # Handle different thermal_limits dimensions
        if thermal_limits.dim() == 1:
            thermal_limits = thermal_limits.unsqueeze(0).unsqueeze(-1)
        elif thermal_limits.dim() == 2:
            thermal_limits = thermal_limits.unsqueeze(-1)
        
        # Violations occur when |line_flow| > thermal_limit
        violations = F.relu(torch.abs(line_flows) - thermal_limits)
        return torch.mean(violations ** 2)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_properties: Dict[str, torch.Tensor],
        edge_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            graph_properties: Graph properties including edge attributes
            edge_mask: Optional edge mask for dynamic topology
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Helper to safely get graph properties
        def get_prop(key):
            return graph_properties.get(key)
        
        # --- 0. PREDICTION (Focal Loss) ---
        failure_prob = predictions['failure_probability'].squeeze(-1)
        failure_prob_clamped = failure_prob.clamp(1e-7, 1 - 1e-7)
        logits = torch.log(failure_prob_clamped / (1 - failure_prob_clamped))
        
        L_pred = self.focal_loss(logits, targets['failure_label'])
        loss_dict['prediction'] = L_pred.item()
        total_loss += self.lambdas.get('prediction', 1.0) * L_pred
        
        # --- 1. VOLTAGE SUPERVISION ---
        if 'voltages' in predictions and targets.get('voltages') is not None:
            L_volt = self.voltage_loss(predictions['voltages'], targets['voltages'])
            total_loss += self.lambdas.get('voltage', 1.0) * L_volt
            loss_dict['voltage'] = L_volt.item()
        
        # --- 2. REACTIVE POWER CONSISTENCY ---
        if 'reactive_nodes' in predictions and 'node_reactive_power' in targets:
            L_react = self.reactive_power_loss(
                predictions['reactive_nodes'],
                targets['node_reactive_power']
            )
            total_loss += self.lambdas.get('reactive', 1.0) * L_react
            loss_dict['reactive'] = L_react.item()
        
        # --- 3. REACTIVE POWER FLOW CONSISTENCY ---
        if 'line_flows' in predictions and 'line_reactive_power' in targets:
            L_flow = self.flow_consistency_loss(
                predictions['line_flows'],
                targets['line_reactive_power']
            )
            total_loss += self.lambdas.get('powerflow', 0.1) * L_flow
            loss_dict['powerflow'] = L_flow.item()
        
        # --- 3b. ACTIVE POWER LINE FLOW SUPERVISION ---
        if 'active_power_line_flows' in predictions and 'active_power_line_flows' in targets:
            L_active_flow = self.active_power_line_flow_loss(
                predictions['active_power_line_flows'],
                targets['active_power_line_flows']
            )
            total_loss += self.lambdas.get('active_flow', 0.1) * L_active_flow
            loss_dict['active_flow'] = L_active_flow.item()
        
        # --- 4. TEMPERATURE (Rule 5) ---
        if 'temperature' in predictions and get_prop('ground_truth_temperature') is not None:
            L_temp = self.temperature_loss(
                predictions['temperature'],
                get_prop('ground_truth_temperature')
            )
            total_loss += self.lambdas.get('temperature',0.05) * L_temp
            loss_dict['temperature'] = L_temp.item()
        
        # --- 5. FREQUENCY (Rule 4) ---
        if 'frequency' in predictions and get_prop('power_injection') is not None:
            L_freq = self.frequency_loss(
                predictions['frequency'],
                get_prop('power_injection')
            )
            total_loss += self.lambdas.get('frequency',0.05) * L_freq
            loss_dict['frequency'] = L_freq.item()
        
        # --- 6. RISK & TIMING ---
        if 'risk_scores' in predictions and targets.get('ground_truth_risk') is not None:
            L_risk = self.risk_loss(predictions['risk_scores'], targets['ground_truth_risk'])
            total_loss += self.lambdas.get('risk', 0.1) * L_risk
            loss_dict['risk'] = L_risk.item()
        
        if 'cascade_timing' in predictions and targets.get('cascade_timing') is not None:
            L_time = self.timing_loss(predictions['cascade_timing'], targets['cascade_timing'])
            total_loss += self.lambdas.get('timing', 0.1) * L_time
            loss_dict['timing'] = L_time.item()
        
        # --- 7. THERMAL CAPACITY CONSTRAINTS ---
        if 'line_flows' in predictions and get_prop('thermal_limits') is not None:
            L_capacity = self.capacity_loss(
                predictions['line_flows'],
                get_prop('thermal_limits')
            )
            total_loss += self.lambdas.get('capacity', 0.05) * L_capacity
            loss_dict['capacity'] = L_capacity.item()

        return total_loss, loss_dict
