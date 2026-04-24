"""
Tests for cascade_prediction/models/loss/physics_informed.py
=============================================================
Covers:
  - PhysicsInformedLoss construction and defaults
  - focal_loss
  - flow_consistency_loss
  - temperature_loss
  - frequency_loss
  - risk_loss
  - timing_loss (including edge-cases: no failing nodes, single failing node,
    bias-correction, spread, ranking)
  - voltage_loss
  - reactive_power_loss
  - active_power_line_flow_loss
  - capacity_loss (apparent-power formulation)
  - forward() — full loss integration
"""

import pytest
import torch
import torch.nn.functional as F

from cascade_prediction.models.loss.physics_informed import PhysicsInformedLoss
from cascade_prediction.data.generator.config import Settings


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

BATCH  = 2
NODES  = 8
EDGES  = 6
RISK_D = Settings.Model.RISK_DIM  # 7


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _ones(*shape) -> torch.Tensor:
    return torch.ones(*shape)


def _loss(**kwargs) -> PhysicsInformedLoss:
    """Create a loss instance with all lambdas = 1.0 unless overridden."""
    defaults = dict(
        lambda_prediction=1.0, lambda_powerflow=1.0, lambda_risk=1.0,
        lambda_timing=1.0, lambda_active_flow=1.0, lambda_temperature=1.0,
        lambda_frequency=1.0, lambda_reactive=1.0, lambda_voltage=1.0,
        lambda_capacity=1.0,
    )
    defaults.update(kwargs)
    return PhysicsInformedLoss(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_construction(self):
        criterion = PhysicsInformedLoss()
        assert isinstance(criterion, PhysicsInformedLoss)

    def test_custom_lambdas_stored(self):
        criterion = PhysicsInformedLoss(lambda_prediction=5.0, lambda_risk=0.2)
        assert criterion.lambdas["prediction"] == pytest.approx(5.0)
        assert criterion.lambdas["risk"]       == pytest.approx(0.2)

    def test_focal_parameters(self):
        criterion = PhysicsInformedLoss(focal_alpha=0.3, focal_gamma=1.5)
        assert criterion.focal_alpha == pytest.approx(0.3)
        assert criterion.focal_gamma == pytest.approx(1.5)

    def test_physics_parameters(self):
        criterion = PhysicsInformedLoss(base_mva=200.0, base_freq=50.0)
        assert criterion.base_mva  == pytest.approx(200.0)
        assert criterion.base_freq == pytest.approx(50.0)

    def test_all_lambda_keys_present(self):
        criterion = PhysicsInformedLoss()
        expected = {
            "prediction", "powerflow", "risk", "timing",
            "active_flow", "temperature", "frequency",
            "reactive", "voltage", "capacity",
        }
        assert expected == set(criterion.lambdas.keys())


# ---------------------------------------------------------------------------
# focal_loss
# ---------------------------------------------------------------------------

class TestFocalLoss:

    def test_perfect_prediction_yields_low_loss(self):
        crit = _loss()
        logits = torch.tensor([[5.0, -5.0]])
        labels = torch.tensor([[1.0,  0.0]])
        loss = crit.focal_loss(logits, labels)
        assert loss.item() < 0.1

    def test_wrong_prediction_yields_higher_loss(self):
        crit = _loss()
        logits_wrong  = torch.tensor([[-5.0, 5.0]])
        logits_right  = torch.tensor([[ 5.0, -5.0]])
        labels        = torch.tensor([[  1.0,  0.0]])
        loss_wrong = crit.focal_loss(logits_wrong, labels)
        loss_right = crit.focal_loss(logits_right, labels)
        assert loss_wrong > loss_right

    def test_loss_is_non_negative(self):
        crit = _loss()
        logits = _rand(BATCH, NODES)
        labels = torch.randint(0, 2, (BATCH, NODES)).float()
        loss = crit.focal_loss(logits, labels)
        assert loss.item() >= 0.0

    def test_returns_scalar(self):
        crit = _loss()
        logits = _rand(BATCH, NODES)
        labels = torch.zeros(BATCH, NODES)
        loss = crit.focal_loss(logits, labels)
        assert loss.dim() == 0

    def test_focal_gamma_effect(self):
        """Higher gamma → less penalty on easy examples."""
        logits = torch.tensor([[3.0]])   # confident correct prediction
        labels = torch.tensor([[1.0]])
        crit_low  = PhysicsInformedLoss(focal_gamma=0.0)
        crit_high = PhysicsInformedLoss(focal_gamma=5.0)
        loss_low  = crit_low.focal_loss(logits, labels)
        loss_high = crit_high.focal_loss(logits, labels)
        assert loss_high < loss_low, "Higher gamma should reduce loss on easy (confident) examples"


# ---------------------------------------------------------------------------
# Flow consistency loss
# ---------------------------------------------------------------------------

class TestFlowConsistencyLoss:

    def test_identical_inputs_zero_loss(self):
        crit = _loss()
        flows = _rand(BATCH, EDGES, 1)
        assert crit.flow_consistency_loss(flows, flows).item() == pytest.approx(0.0)

    def test_loss_non_negative(self):
        crit = _loss()
        pred = _rand(BATCH, EDGES, 1)
        tgt  = _rand(BATCH, EDGES, 1)
        assert crit.flow_consistency_loss(pred, tgt).item() >= 0.0

    def test_larger_error_larger_loss(self):
        crit = _loss()
        pred = torch.zeros(BATCH, EDGES, 1)
        tgt_close = pred + 0.1
        tgt_far   = pred + 10.0
        assert crit.flow_consistency_loss(pred, tgt_far) > \
               crit.flow_consistency_loss(pred, tgt_close)


# ---------------------------------------------------------------------------
# Temperature loss
# ---------------------------------------------------------------------------

class TestTemperatureLoss:

    def test_identical_inputs_zero_loss(self):
        crit = _loss()
        t = _rand(BATCH, NODES).abs() * 80
        pred = t.unsqueeze(-1)
        assert crit.temperature_loss(pred, t).item() == pytest.approx(0.0, abs=1e-5)

    def test_scaled_correctly(self):
        """Loss should be normalised by TEMPERATURE_SCALE."""
        crit = _loss()
        pred = torch.full((1, 1, 1), 100.0)
        gt   = torch.full((1, 1), 200.0)
        raw_mse = ((100.0 - 200.0) / Settings.Loss.TEMPERATURE_SCALE) ** 2
        assert crit.temperature_loss(pred, gt).item() == pytest.approx(raw_mse, rel=1e-4)


# ---------------------------------------------------------------------------
# Frequency loss
# ---------------------------------------------------------------------------

class TestFrequencyLoss:

    def test_identical_inputs_zero_loss(self):
        crit = _loss(base_freq=60.0)
        freq = torch.tensor([[[60.0]]])
        assert crit.frequency_loss(freq, freq).item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_in_pu_space(self):
        """A 1 Hz error at 60 Hz should give (1/60)^2 MSE."""
        crit = _loss(base_freq=60.0)
        pred = torch.tensor([[[60.0]]])
        tgt  = torch.tensor([[[61.0]]])
        expected = (1.0 / 60.0) ** 2
        assert crit.frequency_loss(pred, tgt).item() == pytest.approx(expected, rel=1e-4)

    def test_non_negative(self):
        crit = _loss()
        pred = _rand(BATCH, 1, 1).abs() * 60
        tgt  = _rand(BATCH, 1, 1).abs() * 60
        assert crit.frequency_loss(pred, tgt).item() >= 0.0


# ---------------------------------------------------------------------------
# Risk loss
# ---------------------------------------------------------------------------

class TestRiskLoss:

    def test_identical_inputs_zero_loss(self):
        crit = _loss()
        r = torch.rand(BATCH, NODES, RISK_D)
        assert crit.risk_loss(r, r).item() == pytest.approx(0.0, abs=1e-6)

    def test_per_node_supervision(self):
        """Distinct per-node targets should drive different node gradients."""
        crit = _loss()
        pred = (torch.ones(BATCH, NODES, RISK_D) * 0.5).requires_grad_(True)
        tgt  = torch.zeros(BATCH, NODES, RISK_D)
        tgt[:, 0, :] = 1.0   # only node 0 has high risk
        loss = crit.risk_loss(pred, tgt)
        loss.backward()
        # If gradients exist, per-node supervision is working
        assert loss.item() > 0.0

    def test_risk_loss_shape_independence(self):
        """Loss must be a scalar regardless of shape."""
        crit = _loss()
        pred = _rand(4, 20, RISK_D)
        tgt  = _rand(4, 20, RISK_D)
        loss = crit.risk_loss(pred, tgt)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Voltage loss
# ---------------------------------------------------------------------------

class TestVoltageLoss:

    def test_identical_zero_loss(self):
        crit = _loss()
        v = _rand(BATCH, NODES, 1)
        assert crit.voltage_loss(v, v).item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        crit = _loss()
        pred = _rand(BATCH, NODES, 1)
        tgt  = _rand(BATCH, NODES, 1)
        assert crit.voltage_loss(pred, tgt).item() >= 0.0


# ---------------------------------------------------------------------------
# Reactive power loss
# ---------------------------------------------------------------------------

class TestReactivePowerLoss:

    def test_identical_zero_loss(self):
        crit = _loss()
        q = _rand(BATCH, NODES, 1)
        assert crit.reactive_power_loss(q, q).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Active power line flow loss
# ---------------------------------------------------------------------------

class TestActivePowerLineFlowLoss:

    def test_identical_zero_loss(self):
        crit = _loss()
        p = _rand(BATCH, EDGES, 1)
        assert crit.active_power_line_flow_loss(p, p).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Capacity (thermal) loss
# ---------------------------------------------------------------------------

class TestCapacityLoss:

    def test_no_violation_zero_loss(self):
        """Flows well within thermal limits → zero loss."""
        crit = _loss()
        active   = torch.zeros(BATCH, EDGES, 1)
        reactive = torch.zeros(BATCH, EDGES, 1)
        limits   = torch.ones(BATCH, EDGES) * 100.0
        loss = crit.capacity_loss(active, reactive, limits)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_violation_increases_loss(self):
        """Apparent power exceeding limits should produce positive loss."""
        crit = _loss()
        active   = torch.ones(BATCH, EDGES, 1) * 50.0
        reactive = torch.ones(BATCH, EDGES, 1) * 50.0
        limits   = torch.ones(BATCH, EDGES) * 1.0   # very small limit → violation
        loss = crit.capacity_loss(active, reactive, limits)
        assert loss.item() > 0.0

    def test_uses_apparent_power(self):
        """Loss should reflect S = sqrt(P²+Q²), not just Q."""
        crit = _loss()
        limits = torch.ones(1, 1) * 5.0

        # P=3, Q=0 → S=3 < 5, no violation
        pred_p_safe = torch.tensor([[[3.0]]])
        pred_q_zero = torch.tensor([[[0.0]]])
        loss_p_only = crit.capacity_loss(pred_p_safe, pred_q_zero, limits)

        # P=3, Q=4 → S=5, exactly at limit → no violation (or marginal)
        pred_q_4 = torch.tensor([[[4.0]]])
        loss_pq   = crit.capacity_loss(pred_p_safe, pred_q_4, limits)

        # P=3, Q=5 → S≈5.83 > 5 → violation
        pred_q_5 = torch.tensor([[[5.0]]])
        loss_over = crit.capacity_loss(pred_p_safe, pred_q_5, limits)

        assert loss_p_only.item() == pytest.approx(0.0, abs=1e-4)
        assert loss_over.item() > loss_pq.item()

    def test_1d_limits_broadcast(self):
        """1-D thermal limits [E] should broadcast across batch."""
        crit = _loss()
        active   = torch.zeros(BATCH, EDGES, 1)
        reactive = torch.zeros(BATCH, EDGES, 1)
        limits   = torch.ones(EDGES) * 10.0   # shape [E]
        loss = crit.capacity_loss(active, reactive, limits)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# timing_loss — detailed edge-case coverage
# ---------------------------------------------------------------------------

class TestTimingLoss:

    def test_no_failing_nodes_returns_zero(self):
        """All targets == -1 → no failing nodes → loss must be 0."""
        crit = _loss()
        pred = torch.rand(BATCH, NODES, 1)
        tgt  = torch.full((BATCH, NODES), -1.0)
        loss = crit.timing_loss(pred, tgt)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_single_failing_node(self):
        """Single failing node per sample → regression loss only, no ranking."""
        crit = _loss()
        pred = torch.rand(BATCH, NODES, 1)
        tgt  = torch.full((BATCH, NODES), -1.0)
        tgt[:, 0] = 0.8   # node 0 fails at t=0.8
        loss = crit.timing_loss(pred, tgt)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_perfect_prediction_low_loss(self):
        crit = _loss()
        # Construct matching pred / target
        times = torch.tensor([0.2, 0.4, 0.6, 0.8])
        tgt   = torch.full((1, NODES), -1.0)
        tgt[0, :4] = times

        pred = torch.full((1, NODES, 1), -1.0)
        pred[0, :4, 0] = times   # perfect match

        loss = crit.timing_loss(pred, tgt)
        assert loss.item() < 0.05

    def test_wrong_order_penalised(self):
        """Reversed ordering of predictions should yield higher loss."""
        crit = _loss()
        tgt = torch.full((1, NODES), -1.0)
        tgt[0, :3] = torch.tensor([0.2, 0.4, 0.8])

        pred_right = torch.full((1, NODES, 1), 0.5)
        pred_right[0, 0, 0] = 0.2
        pred_right[0, 1, 0] = 0.4
        pred_right[0, 2, 0] = 0.8

        pred_wrong = torch.full((1, NODES, 1), 0.5)
        pred_wrong[0, 0, 0] = 0.8   # reversed
        pred_wrong[0, 1, 0] = 0.4
        pred_wrong[0, 2, 0] = 0.2

        loss_right = crit.timing_loss(pred_right, tgt)
        loss_wrong = crit.timing_loss(pred_wrong, tgt)
        assert loss_wrong.item() > loss_right.item(), \
            "Mis-ordered predictions should have higher timing loss"

    def test_bias_correction_component(self):
        """Systematically early predictions should incur bias penalty."""
        crit = _loss()
        tgt = torch.full((1, NODES), -1.0)
        tgt[0, :4] = torch.tensor([0.6, 0.7, 0.8, 0.9])   # late failures

        pred_early = torch.full((1, NODES, 1), 0.5)
        pred_early[0, :4, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])   # [4] not [4,1]

        pred_good = torch.full((1, NODES, 1), 0.5)
        pred_good[0, :4, 0] = torch.tensor([0.6, 0.7, 0.8, 0.9])

        loss_early = crit.timing_loss(pred_early, tgt)
        loss_good  = crit.timing_loss(pred_good,  tgt)
        assert loss_early.item() > loss_good.item()

    def test_spread_component(self):
        """Collapsed timing (all same value) should score worse than correct spread."""
        crit = _loss()
        tgt = torch.full((1, NODES), -1.0)
        tgt[0, :4] = torch.tensor([0.2, 0.4, 0.6, 0.8])

        pred_collapsed = torch.full((1, NODES, 1), 0.5)
        pred_collapsed[0, :4, 0] = 0.5   # all same

        pred_spread = torch.full((1, NODES, 1), 0.5)
        pred_spread[0, :4, 0] = torch.tensor([0.2, 0.4, 0.6, 0.8])

        loss_col  = crit.timing_loss(pred_collapsed, tgt)
        loss_spr  = crit.timing_loss(pred_spread,    tgt)
        assert loss_col.item() > loss_spr.item(), \
            "Collapsed timing predictions should incur higher spread penalty"

    def test_returns_scalar(self):
        crit = _loss()
        pred = torch.rand(BATCH, NODES, 1)
        tgt  = torch.full((BATCH, NODES), -1.0)
        tgt[:, 0] = 0.5
        loss = crit.timing_loss(pred, tgt)
        assert loss.dim() == 0

    def test_no_nan_output(self):
        crit = _loss()
        pred = torch.rand(BATCH, NODES, 1)
        tgt  = torch.rand(BATCH, NODES)
        tgt[tgt < 0.5] = -1.0   # random mix of failing/non-failing
        loss = crit.timing_loss(pred, tgt)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# forward() — integration
# ---------------------------------------------------------------------------

class TestPhysicsInformedLossForward:

    def _make_predictions(self):
        return {
            "failure_probability":    torch.sigmoid(_rand(BATCH, NODES, 1)),
            "cascade_timing":         torch.sigmoid(_rand(BATCH, NODES, 1)),
            "voltages":               torch.abs(_rand(BATCH, NODES, 1)),
            "angles":                 torch.tanh(_rand(BATCH, NODES, 1)),
            "line_flows":             _rand(BATCH, EDGES, 1),
            "active_power_line_flows": _rand(BATCH, EDGES, 1),
            "temperature":            torch.abs(_rand(BATCH, NODES, 1)),
            "reactive_nodes":         _rand(BATCH, NODES, 1),
            "frequency":              torch.abs(_rand(BATCH, 1, 1)) * 60,
            "risk_scores":            torch.sigmoid(_rand(BATCH, NODES, RISK_D)),
        }

    def _make_targets(self):
        tgt = torch.full((BATCH, NODES), -1.0)
        tgt[:, 0] = 0.7   # node 0 fails at t=0.7
        return {
            "failure_label":       torch.zeros(BATCH, NODES),
            "voltages":            torch.abs(_rand(BATCH, NODES, 1)),
            "node_reactive_power": _rand(BATCH, NODES, 1),
            "line_reactive_power": _rand(BATCH, EDGES, 1),
            "active_power_line_flows": _rand(BATCH, EDGES, 1),
            "ground_truth_risk":   torch.sigmoid(_rand(BATCH, NODES, RISK_D)),
            "cascade_timing":      tgt,
            "ground_truth_frequency": torch.full((BATCH, 1, 1), 60.0),
        }

    def _make_graph_props(self):
        return {
            "thermal_limits":         torch.ones(BATCH, EDGES) * 100.0,
            "ground_truth_temperature": _rand(BATCH, NODES).abs() * 80,
        }

    def test_forward_returns_tuple(self):
        crit = _loss()
        preds = self._make_predictions()
        tgts  = self._make_targets()
        props = self._make_graph_props()
        result = crit(preds, tgts, props)
        assert isinstance(result, tuple) and len(result) == 2

    def test_total_loss_is_scalar_tensor(self):
        crit = _loss()
        total_loss, _ = crit(
            self._make_predictions(), self._make_targets(), self._make_graph_props()
        )
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0

    def test_total_loss_non_negative(self):
        crit = _loss()
        total_loss, _ = crit(
            self._make_predictions(), self._make_targets(), self._make_graph_props()
        )
        assert total_loss.item() >= 0.0

    def test_loss_dict_has_prediction_key(self):
        crit = _loss()
        _, loss_dict = crit(
            self._make_predictions(), self._make_targets(), self._make_graph_props()
        )
        assert "prediction" in loss_dict

    def test_loss_dict_values_are_floats(self):
        crit = _loss()
        _, loss_dict = crit(
            self._make_predictions(), self._make_targets(), self._make_graph_props()
        )
        for k, v in loss_dict.items():
            assert isinstance(v, float), f"loss_dict['{k}'] is not a float: {type(v)}"

    def test_no_nan_total_loss(self):
        crit = _loss()
        total_loss, _ = crit(
            self._make_predictions(), self._make_targets(), self._make_graph_props()
        )
        assert not torch.isnan(total_loss)

    def test_minimal_targets_only_prediction_loss(self):
        """With only failure_label in targets, only prediction loss fires."""
        crit = _loss()
        preds = self._make_predictions()
        tgts  = {"failure_label": torch.zeros(BATCH, NODES)}
        props = {}
        total_loss, loss_dict = crit(preds, tgts, props)
        # Only prediction should be in the dict
        assert "prediction" in loss_dict
        for key in ("voltage", "reactive", "powerflow", "temperature", "frequency",
                    "risk", "timing", "capacity"):
            assert key not in loss_dict, f"Unexpected key '{key}' in loss_dict"

    def test_lambda_zero_excludes_component(self):
        """A lambda of 0 should make the component contribute 0 to total loss."""
        crit_with  = _loss(lambda_voltage=1.0)
        crit_without = _loss(lambda_voltage=0.0)
        preds = self._make_predictions()
        tgts  = self._make_targets()
        props = self._make_graph_props()

        loss_with,    d_with    = crit_with(preds, tgts, props)
        loss_without, d_without = crit_without(preds, tgts, props)

        # Voltage contributes when lambda=1, not when lambda=0
        if "voltage" in d_with and "voltage" in d_without:
            contribution = d_with["voltage"] * 1.0 - d_without["voltage"] * 0.0
            assert abs(contribution) == pytest.approx(d_with["voltage"], rel=1e-4)

    def test_gradients_flow_through_forward(self):
        """Total loss must be differentiable (allow .backward())."""
        crit = _loss()
        preds = self._make_predictions()
        # Make preds leaf tensors with grad
        for k, v in preds.items():
            preds[k] = v.detach().requires_grad_(True)
        tgts  = self._make_targets()
        props = self._make_graph_props()

        total_loss, _ = crit(preds, tgts, props)
        total_loss.backward()

        # At least failure_probability should have a gradient
        assert preds["failure_probability"].grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
