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
        focal_alpha: float = Settings.Loss.FOCAL_ALPHA,
        focal_gamma: float = Settings.Loss.FOCAL_GAMMA,
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
            lambda_frequency: Weight for frequency supervision loss
            lambda_reactive: Weight for reactive power loss
            lambda_voltage: Weight for voltage prediction loss
            lambda_capacity: Weight for thermal capacity constraint loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            base_mva: Base MVA for power normalization
            base_freq: Base frequency (Hz) for normalization
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

        # Focal loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

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
        ground_truth_freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct supervised frequency loss against ground-truth SCADA measurements.

        The previous physics-derived target (1.0 + sum(P_inj)/POWER_TO_FREQ) was
        incorrect: the POWER_TO_FREQ=10 constant produced targets clustered near
        1.0 p.u. (≈60 Hz) even during cascades where the observed system frequency
        drops to ~38 Hz (0.64 p.u.).  This caused the frequency head to learn the
        wrong operating point and injected a misleading gradient signal into the
        shared encoder.

        We now supervise directly against the ground-truth frequency recorded in
        SCADA column 6 (in Hz), normalised to per-unit by dividing by base_freq.

        Args:
            predicted_freq:    [batch_size, 1, 1] — FrequencyHead output (Hz)
            ground_truth_freq: [batch_size, 1, 1] — SCADA ground truth (Hz)

        Returns:
            MSE loss in per-unit space
        """
        pred_pu   = predicted_freq   / self.base_freq
        target_pu = ground_truth_freq / self.base_freq
        return F.mse_loss(pred_pu, target_pu)
    
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
        IMPROVED cascade timing loss (v3) — absolute-normalised targets in [0, 1].

        Combines four components:
          1. MSE + Smooth-L1 regression on failing nodes (primary regression).
          2. Bias-correction term: penalises the mean prediction deviating from
             the mean target within each sample.  This directly counteracts the
             systematic early-prediction bias that was the observed failure mode
             (model predicted ~41 min while actual was ~48 min).
          3. Temporal-spread term: enforces that the predicted time range
             (max - min) matches the true range of the cascade progression,
             preventing the model from collapsing all timing to a single value.
          4. Pairwise ranking loss with margin 0.04 (≈ 1.2 timesteps in a
             30-step scenario), enforcing the correct failure sequence order.

        All targets are now ABSOLUTE (failure_time / DEFAULT_SEQUENCE_LENGTH),
        making them consistent across different sliding-window crops and with
        the inference decode formula:
            decoded_minutes = pred_normed × DEFAULT_SEQ_LEN × DT_MINUTES

        Args:
            predicted: Predicted timing [batch_size, num_nodes, 1],
                       Sigmoid output in (0, 1).
            target:    Target timing [batch_size, num_nodes],
                       absolute-normalised to [0, 1] (−1 = non-failing node,
                       masked out of all loss terms).

        Returns:
            Combined timing loss scalar.
        """
        pred_s = predicted.squeeze(-1)  # [B, N]
        losses = []

        for b in range(pred_s.shape[0]):
            t = target[b]
            pos_idx = torch.where(t >= 0)[0]

            if len(pos_idx) == 0:
                continue

            p_valid = pred_s[b][pos_idx]   # model predictions for failing nodes
            t_valid = t[pos_idx]            # absolute-normalised ground-truth times

            # ── 1. Regression loss ────────────────────────────────────────────
            mse_loss = F.mse_loss(p_valid, t_valid)
            sl1_loss = F.smooth_l1_loss(p_valid, t_valid, beta=0.05)
            regression_loss = 0.6 * mse_loss + 0.3 * sl1_loss

            # ── 2. Bias-correction loss ───────────────────────────────────────
            # Penalise the mean predicted time deviating from the mean actual
            # time.  This is the primary counter-measure to the systematic
            # early-prediction bias: by explicitly penalising the *mean* offset
            # the model is pushed to centre its predictions correctly even when
            # individual node rankings are uncertain.
            mean_pred_bias   = p_valid.mean()
            mean_target_bias = t_valid.mean()
            bias_loss = (mean_pred_bias - mean_target_bias).pow(2)
            regression_loss = regression_loss + 0.1 * bias_loss

            # ── 3. Temporal spread loss ───────────────────────────────────────
            # Encourage the model to reproduce the correct time RANGE of the
            # cascade (last_failure - first_failure).  Without this, the model
            # tends to cluster all timing predictions around a single value.
            if len(pos_idx) >= 2:
                pred_spread   = p_valid.max() - p_valid.min()
                target_spread = t_valid.max() - t_valid.min()
                spread_loss   = F.mse_loss(pred_spread.unsqueeze(0),
                                           target_spread.unsqueeze(0))
                regression_loss = regression_loss + 0.1 * spread_loss

            # ── 4. Pairwise ranking loss ──────────────────────────────────────
            # Enforce correct cascade sequence order with a margin of 0.04
            # (≈ 1.2 timesteps at DEFAULT_SEQUENCE_LENGTH = 30, previously 0.02
            # which was too lenient — ~0.6 steps — letting mis-ordered pairs
            # slip through the penalty).
            if len(pos_idx) >= 2:
                pairs = torch.combinations(pos_idx, r=2)
                p_diff = pred_s[b][pairs[:, 0]] - pred_s[b][pairs[:, 1]]
                t_diff = t[pairs[:, 0]] - t[pairs[:, 1]]
                # Only penalise pairs with meaningfully different actual times
                # (gap > 1 timestep ≈ 0.033 in normalised space)
                sig_mask = t_diff.abs() > 0.033
                if sig_mask.sum() > 0:
                    ranking_loss = torch.relu(
                        0.04 - p_diff[sig_mask] * torch.sign(t_diff[sig_mask])
                    ).mean()
                else:
                    ranking_loss = torch.tensor(0.0, device=predicted.device)

                losses.append(regression_loss + 0.4 * ranking_loss)
            else:
                losses.append(regression_loss)

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
        active_flows: torch.Tensor,
        reactive_flows: torch.Tensor,
        thermal_limits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute thermal capacity constraint violations using apparent power.

        Previous implementation compared only reactive (Q) line flows against MVA
        thermal limits, which is physically incorrect: thermal ratings bound apparent
        power S = sqrt(P² + Q²), not Q alone.  Using Q-only understated violations on
        lines carrying large active power flows and overstated them on lines with high
        reactive flow but low active flow.

        This version computes S = sqrt(P² + Q²) and penalises violations of the form
        max(0, S - S_limit)², giving a smooth, physically correct constraint signal.

        Args:
            active_flows:   Predicted active power flows P  [batch_size, num_edges, 1]
            reactive_flows: Predicted reactive power flows Q [batch_size, num_edges, 1]
            thermal_limits: MVA thermal limits  [batch_size, num_edges] or [num_edges]

        Returns:
            Mean squared violation loss
        """
        # Broadcast thermal_limits to [B, E, 1]
        if thermal_limits.dim() == 1:
            thermal_limits = thermal_limits.unsqueeze(0).unsqueeze(-1)
        elif thermal_limits.dim() == 2:
            thermal_limits = thermal_limits.unsqueeze(-1)

        # Apparent power S = sqrt(P² + Q²) in per-unit
        apparent_power = torch.sqrt(active_flows.pow(2) + reactive_flows.pow(2) + 1e-8)

        # Violations only when apparent power exceeds thermal limit
        violations = F.relu(apparent_power - thermal_limits)
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
        # Head outputs probabilities via Sigmoid — convert to logits for numerically
        # stable focal loss via binary_cross_entropy_with_logits.
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
        
        # --- 5. FREQUENCY SUPERVISION ---
        # Uses ground-truth SCADA frequency (col 6) rather than physics-derived
        # target.  The old formula used POWER_TO_FREQ=10 which produced targets near
        # 60 Hz even during cascades where the true frequency drops to ~38 Hz.
        if 'frequency' in predictions and targets.get('ground_truth_frequency') is not None:
            L_freq = self.frequency_loss(
                predictions['frequency'],
                targets['ground_truth_frequency'],
            )
            total_loss += self.lambdas.get('frequency', 0.1) * L_freq
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
        # Uses apparent power S = sqrt(P² + Q²) compared against MVA thermal limits,
        # replacing the previous Q-only comparison which was physically incorrect.
        if ('line_flows' in predictions
                and 'active_power_line_flows' in predictions
                and get_prop('thermal_limits') is not None):
            L_capacity = self.capacity_loss(
                predictions['active_power_line_flows'],
                predictions['line_flows'],
                get_prop('thermal_limits'),
            )
            total_loss += self.lambdas.get('capacity', 0.05) * L_capacity
            loss_dict['capacity'] = L_capacity.item()
        elif 'line_flows' in predictions and get_prop('thermal_limits') is not None:
            # Fallback: only reactive flows available — use Q as proxy
            apparent_proxy = torch.abs(predictions['line_flows'])
            L_capacity = self.capacity_loss(
                apparent_proxy,
                torch.zeros_like(apparent_proxy),
                get_prop('thermal_limits'),
            )
            total_loss += self.lambdas.get('capacity', 0.05) * L_capacity
            loss_dict['capacity'] = L_capacity.item()

        return total_loss, loss_dict
