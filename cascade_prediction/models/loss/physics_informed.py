"""
Physics-Informed Loss Module
============================
Loss function for UnifiedCascadePredictionModel.

Active heads and their loss terms:
  failure_probability  [B, N, 1]  → focal loss          (prediction)
  cascade_timing       [B, N, 1]  → timing loss          (timing)
  risk_scores          [B, N, 7]  → MSE loss             (risk)
  parent_logits        [B, N, N+1]→ masked cross-entropy (parent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from cascade_prediction.data.generator.config import Settings


class PhysicsInformedLoss(nn.Module):
    """
    Loss function combining:
    1. Failure prediction (focal loss)           — long-range, from c_T
    2. Risk assessment (MSE)                     — long-range, from c_T
    3. Cascade timing (regression + ranking)     — long-range, from c_T
    4. Causal parent prediction (cross-entropy)  — long-range, from c_T
    5. Voltage prediction (MSE)                  — short-range physics, from h_T
    6. Temperature prediction (MSE)              — short-range physics, from h_T
    7. Line flow prediction (MSE)                — short-range physics, from h_T
    """

    def __init__(
        self,
        lambda_prediction: float = Settings.Loss.LAMBDA_PREDICTION,
        lambda_risk: float = Settings.Loss.LAMBDA_RISK,
        lambda_timing: float = Settings.Loss.LAMBDA_TIMING,
        lambda_parent: float = Settings.Loss.LAMBDA_PARENT,
        focal_alpha: float = Settings.Loss.FOCAL_ALPHA,
        focal_gamma: float = Settings.Loss.FOCAL_GAMMA,
        parent_non_trigger_weight: float = Settings.Loss.PARENT_NON_TRIGGER_WEIGHT,
        lambda_voltage: float = Settings.PhysicsPred.LAMBDA_VOLTAGE,
        lambda_temp: float = Settings.PhysicsPred.LAMBDA_TEMP,
        lambda_flow: float = Settings.PhysicsPred.LAMBDA_FLOW,
        **kwargs,  # absorbs removed auxiliary-head params from legacy call sites
    ):
        super().__init__()

        self.lambdas = {
            'prediction': lambda_prediction,
            'risk':       lambda_risk,
            'timing':     lambda_timing,
            'parent':     lambda_parent,
            'voltage':    lambda_voltage,
            'temp':       lambda_temp,
            'flow':       lambda_flow,
        }
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.parent_non_trigger_weight = parent_non_trigger_weight

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for imbalanced node-failure classification.

        Args:
            logits:  [B, N] raw logits
            targets: [B, N] binary labels
        """
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.focal_gamma * bce).mean()

    def risk_loss(
        self,
        predicted_risk: torch.Tensor,
        target_risk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-node supervised risk loss.

        Args:
            predicted_risk: [B, N, 7]
            target_risk:    [B, N, 7]
        """
        return F.mse_loss(predicted_risk, target_risk)

    def timing_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cascade timing loss — absolute-normalised targets in [0, 1].

        Combines:
          1. MSE + Smooth-L1 regression on failing nodes.
          2. Bias-correction: penalises mean-prediction offset.
          3. Temporal-spread: matches predicted vs actual failure time range.
          4. Pairwise ranking with margin 0.04 (~1.2 timesteps).

        Args:
            predicted: [B, N, 1] Sigmoid output in (0, 1).
            target:    [B, N]    Absolute-normalised times; -1 = non-failing (masked).
        """
        pred_s = predicted.squeeze(-1)  # [B, N]
        losses = []

        for b in range(pred_s.shape[0]):
            t       = target[b]
            pos_idx = torch.where(t >= 0)[0]
            if len(pos_idx) == 0:
                continue

            p_valid = pred_s[b][pos_idx]
            t_valid = t[pos_idx]

            # 1. Regression
            reg = 0.6 * F.mse_loss(p_valid, t_valid) + 0.3 * F.smooth_l1_loss(p_valid, t_valid, beta=0.05)

            # 2. Bias correction
            reg = reg + 0.1 * (p_valid.mean() - t_valid.mean()).pow(2)

            # 3. Temporal spread
            if len(pos_idx) >= 2:
                spread_loss = F.mse_loss(
                    (p_valid.max() - p_valid.min()).unsqueeze(0),
                    (t_valid.max() - t_valid.min()).unsqueeze(0),
                )
                reg = reg + 0.1 * spread_loss

            # 4. Pairwise ranking
            if len(pos_idx) >= 2:
                pairs  = torch.combinations(pos_idx, r=2)
                p_diff = pred_s[b][pairs[:, 0]] - pred_s[b][pairs[:, 1]]
                t_diff = t[pairs[:, 0]] - t[pairs[:, 1]]
                sig    = t_diff.abs() > 0.033
                ranking_loss = (
                    torch.relu(0.04 - p_diff[sig] * torch.sign(t_diff[sig])).mean()
                    if sig.sum() > 0
                    else torch.tensor(0.0, device=predicted.device)
                )
                losses.append(reg + 0.4 * ranking_loss)
            else:
                losses.append(reg)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=predicted.device)

    def physics_supervision_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        MSE losses comparing short-range physics predictions (from h_T)
        against the ground-truth physics values at the held-out timestep T.

        Targets are extracted from the raw SCADA / edge_attr tensors by the
        trainer and stored under the keys below.  If a target is absent the
        corresponding term is silently skipped.

        Keys expected in targets:
          'physics_voltage_target' : [B, N]  — scada[:, -1, :, 0]  (p.u. voltage)
          'physics_temp_target'    : [B, N]  — scada[:, -1, :, 5]  (°C temperature)
          'physics_flow_target'    : [B, E]  — edge_attr[:, -1, :, 5] (MW line flow)
        """
        total = torch.tensor(0.0, device=next(iter(predictions.values()),
                                              torch.zeros(1)).device)
        loss_dict: Dict[str, float] = {}

        def _mse(pred: Optional[torch.Tensor],
                 target: Optional[torch.Tensor],
                 key: str, lam: float):
            nonlocal total
            if pred is None or target is None:
                return
            p = pred.squeeze(-1)   # [B, N] or [B, E]
            if p.shape != target.shape:
                return
            L = F.mse_loss(p, target.float())
            total = total + lam * L
            loss_dict[key] = L.item()

        _mse(predictions.get('voltage_pred'),
             targets.get('physics_voltage_target'),
             'voltage_pred', self.lambdas['voltage'])

        raw_temp = targets.get('physics_temp_target')
        temp_target_normed = (raw_temp / 100.0) if raw_temp is not None else None
        _mse(predictions.get('temp_pred'),
             temp_target_normed,
             'temp_pred', self.lambdas['temp'])

        _mse(predictions.get('flow_pred'),
             targets.get('physics_flow_target'),
             'flow_pred', self.lambdas['flow'])

        return total, loss_dict

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_properties: Dict[str, torch.Tensor],
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions:      Output of UnifiedCascadePredictionModel.forward()
            targets:          Output of Trainer._prepare_targets()
            graph_properties: Graph-level properties (passed through; not used here)
            edge_mask:        Unused; kept for API compatibility

        Returns:
            (total_loss, loss_components_dict)
        """
        loss_dict  = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # --- 1. FAILURE PREDICTION (Focal Loss) ---
        failure_prob = predictions['failure_probability'].squeeze(-1)  # [B, N]
        L_pred = self.focal_loss(failure_prob, targets['failure_label'])
        total_loss = total_loss + self.lambdas['prediction'] * L_pred
        loss_dict['prediction'] = L_pred.item()

        # --- 2. RISK ---
        if 'risk_scores' in predictions and targets.get('ground_truth_risk') is not None:
            L_risk = self.risk_loss(predictions['risk_scores'], targets['ground_truth_risk'])
            total_loss = total_loss + self.lambdas['risk'] * L_risk
            loss_dict['risk'] = L_risk.item()

        # --- 3. CASCADE TIMING ---
        if 'cascade_timing' in predictions and targets.get('cascade_timing') is not None:
            L_time = self.timing_loss(predictions['cascade_timing'], targets['cascade_timing'])
            total_loss = total_loss + self.lambdas['timing'] * L_time
            loss_dict['timing'] = L_time.item()

        # --- 4. CAUSAL PARENT PREDICTION ---
        # parent_labels[i] = -1  → did not fail (ignored)
        # parent_labels[i] = N   → trigger node (no causal parent)
        # parent_labels[i] = j   → node j caused node i to fail
        if 'parent_logits' in predictions and targets.get('parent_labels') is not None:
            parent_logits = predictions['parent_logits']          # [B, N, N+1]
            parent_labels = targets['parent_labels'].long()       # [B, N]
            B, N, _ = parent_logits.shape
            logits_flat = parent_logits.reshape(B * N, N + 1)
            labels_flat = parent_labels.reshape(B * N)
            mask        = labels_flat != -1
            if mask.sum() > 0:
                # Up-weight actual-parent classes (0..N-1) vs the trigger class (N).
                # Trigger dominates labels, so without this the model always predicts N.
                class_weight = torch.full(
                    (N + 1,), self.parent_non_trigger_weight,
                    device=logits_flat.device, dtype=torch.float32,
                )
                class_weight[N] = 1.0  # trigger class retains base weight
                L_parent = F.cross_entropy(
                    logits_flat[mask].float(), labels_flat[mask],
                    weight=class_weight,
                )
                total_loss = total_loss + self.lambdas['parent'] * L_parent
                loss_dict['parent'] = L_parent.item()

        # --- 5. PHYSICS SUPERVISION (short-range, from h_T) ---
        phys_loss, phys_dict = self.physics_supervision_loss(predictions, targets)
        total_loss = total_loss + phys_loss
        loss_dict.update(phys_dict)

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
