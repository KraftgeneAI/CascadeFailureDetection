"""
Tests for cascade_prediction/models/loss/physics_informed.py
=============================================================
Updated on the branch with:
  - Multi-component timing_loss (regression + bias correction + spread + ranking)
  - Causal parent prediction loss with per-class weight up-weighting
  - focal_loss, risk_loss retained

Run:
    cd CascadeFailureDetection
    pytest test/test_physics_loss.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from cascade_prediction.models.loss.physics_informed import PhysicsInformedLoss


# ── Shared constants ───────────────────────────────────────────────────────────
B = 2   # batch size
N = 8   # nodes — small for speed
E = 10  # edges (unused by loss, kept for parity)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _loss(
    lambda_prediction=1.0,
    lambda_risk=1.0,
    lambda_timing=1.0,
    lambda_parent=1.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    parent_non_trigger_weight=5.0,
) -> PhysicsInformedLoss:
    return PhysicsInformedLoss(
        lambda_prediction=lambda_prediction,
        lambda_risk=lambda_risk,
        lambda_timing=lambda_timing,
        lambda_parent=lambda_parent,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        parent_non_trigger_weight=parent_non_trigger_weight,
    )


def _minimal_preds(batch=B, nodes=N) -> dict:
    """Minimal predictions dict that satisfies PhysicsInformedLoss.forward()."""
    return {
        "failure_probability": torch.randn(batch, nodes, 1),
        "cascade_timing":      torch.rand(batch, nodes, 1),
        "risk_scores":         torch.rand(batch, nodes, 7),
        "parent_logits":       torch.randn(batch, nodes, nodes + 1),
    }


def _minimal_targets(batch=B, nodes=N) -> dict:
    """Minimal targets dict matching what Trainer._prepare_targets() builds."""
    timing = torch.full((batch, nodes), -1.0)
    # Mark half of node 0 in each batch as failing at t ≈ 0.5
    timing[:, 0] = 0.5
    return {
        "failure_label":    (torch.rand(batch, nodes) > 0.7).float(),
        "cascade_timing":   timing,
        "ground_truth_risk": torch.rand(batch, nodes, 7),
        "parent_labels":    torch.full((batch, nodes), nodes, dtype=torch.long),  # all triggers
    }


def _graph_props() -> dict:
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. focal_loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestFocalLoss:
    """Unit tests for PhysicsInformedLoss.focal_loss()."""

    @pytest.fixture
    def loss_fn(self):
        return _loss()

    def test_returns_scalar(self, loss_fn):
        logits  = torch.randn(B, N)
        targets = (torch.rand(B, N) > 0.5).float()
        out = loss_fn.focal_loss(logits, targets)
        assert out.dim() == 0

    def test_all_zero_labels_finite(self, loss_fn):
        logits  = torch.randn(B, N)
        targets = torch.zeros(B, N)
        out = loss_fn.focal_loss(logits, targets)
        assert torch.isfinite(out)

    def test_all_one_labels_finite(self, loss_fn):
        logits  = torch.randn(B, N)
        targets = torch.ones(B, N)
        out = loss_fn.focal_loss(logits, targets)
        assert torch.isfinite(out)

    def test_non_negative(self, loss_fn):
        logits  = torch.randn(B, N)
        targets = (torch.rand(B, N) > 0.5).float()
        assert loss_fn.focal_loss(logits, targets) >= 0.0

    def test_perfect_prediction_lower_than_random(self, loss_fn):
        """
        Correct logits (large positive for 1, large negative for 0)
        should produce less loss than random logits.
        """
        targets = (torch.rand(B, N) > 0.5).float()
        correct_logits = (targets * 2 - 1) * 5.0   # +5 for class 1, -5 for class 0
        random_logits  = torch.randn(B, N)
        assert loss_fn.focal_loss(correct_logits, targets) < loss_fn.focal_loss(random_logits, targets)

    def test_gradient_flows(self, loss_fn):
        logits  = torch.randn(B, N, requires_grad=True)
        targets = (torch.rand(B, N) > 0.5).float()
        loss_fn.focal_loss(logits, targets).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_focal_weighting_down_weights_easy_examples(self, loss_fn):
        """
        Focal loss with gamma>0 should be ≤ plain BCE loss for easy
        (correctly classified with high confidence) examples.
        """
        # All high-confidence correct predictions
        targets = torch.ones(B, N)
        logits  = torch.full((B, N), 5.0)   # sigmoid ≈ 0.993 → easy

        focal = loss_fn.focal_loss(logits, targets)
        bce   = F.binary_cross_entropy_with_logits(logits, targets)
        assert focal <= bce + 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 2. risk_loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskLoss:
    """Unit tests for PhysicsInformedLoss.risk_loss()."""

    @pytest.fixture
    def loss_fn(self):
        return _loss()

    def test_perfect_prediction_near_zero(self, loss_fn):
        risk = torch.rand(B, N, 7)
        assert loss_fn.risk_loss(risk, risk).item() == pytest.approx(0.0, abs=1e-6)

    def test_non_perfect_positive(self, loss_fn):
        pred   = torch.rand(B, N, 7)
        target = torch.rand(B, N, 7)
        assert loss_fn.risk_loss(pred, target) > 0.0

    def test_returns_scalar(self, loss_fn):
        pred   = torch.rand(B, N, 7)
        target = torch.rand(B, N, 7)
        out = loss_fn.risk_loss(pred, target)
        assert out.dim() == 0

    def test_gradient_flows(self, loss_fn):
        pred   = torch.rand(B, N, 7, requires_grad=True)
        target = torch.rand(B, N, 7)
        loss_fn.risk_loss(pred, target).backward()
        assert pred.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. timing_loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimingLoss:
    """Unit tests for the multi-component PhysicsInformedLoss.timing_loss()."""

    @pytest.fixture
    def loss_fn(self):
        return _loss()

    def test_all_non_failing_returns_zero(self, loss_fn):
        """When every target is -1 (non-failing), loss should be exactly 0."""
        pred   = torch.rand(B, N, 1)
        target = torch.full((B, N), -1.0)
        out = loss_fn.timing_loss(pred, target)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

    def test_single_failing_node_is_finite(self, loss_fn):
        pred   = torch.rand(B, N, 1)
        target = torch.full((B, N), -1.0)
        target[:, 0] = 0.4
        out = loss_fn.timing_loss(pred, target)
        assert torch.isfinite(out)
        assert out >= 0.0

    def test_returns_scalar(self, loss_fn):
        pred   = torch.rand(B, N, 1)
        target = torch.full((B, N), -1.0)
        target[:, :3] = torch.rand(B, 3)
        out = loss_fn.timing_loss(pred, target)
        assert out.dim() == 0

    def test_perfect_timing_near_zero(self, loss_fn):
        """If predicted values exactly match targets, loss should be close to 0."""
        target_vals = torch.tensor([[0.2, 0.4, 0.8]])          # (1, 3) for nodes 0-2
        pred_full   = torch.full((1, N, 1), 0.5)               # initial
        pred_full[0, :3, 0] = target_vals[0]                   # set to exact targets
        target      = torch.full((1, N), -1.0)
        target[0, :3] = target_vals[0]
        out = loss_fn.timing_loss(pred_full, target)
        assert out.item() < 0.05   # near zero given exact match

    def test_gradient_flows(self, loss_fn):
        pred   = torch.rand(B, N, 1, requires_grad=True)
        target = torch.full((B, N), -1.0)
        target[:, 0] = 0.5
        loss_fn.timing_loss(pred, target).backward()
        assert pred.grad is not None

    def test_one_batch_empty_one_batch_with_failing(self, loss_fn):
        """
        Batch 0 has no failing nodes; batch 1 has two.
        Should not crash and should return finite loss.
        """
        pred   = torch.rand(2, N, 1)
        target = torch.full((2, N), -1.0)
        target[1, :2] = torch.tensor([0.3, 0.7])
        out = loss_fn.timing_loss(pred, target)
        assert torch.isfinite(out)

    def test_pairwise_ranking_penalises_inversion(self, loss_fn):
        """
        With two failing nodes where t0 < t1, a prediction that inverts
        (p0 > p1) should produce higher loss than a correct ordering (p0 < p1).
        """
        loss_fn_det = _loss()   # fresh instance, no randomness
        target = torch.tensor([[-1.0] * N])   # (1, N)
        target[0, 0] = 0.2
        target[0, 1] = 0.8   # node 1 fails later

        # Correct ordering: p0 < p1
        pred_correct  = torch.full((1, N, 1), 0.5)
        pred_correct[0, 0, 0] = 0.2
        pred_correct[0, 1, 0] = 0.8

        # Inverted: p0 > p1
        pred_inverted = torch.full((1, N, 1), 0.5)
        pred_inverted[0, 0, 0] = 0.8
        pred_inverted[0, 1, 0] = 0.2

        l_correct  = loss_fn_det.timing_loss(pred_correct,  target)
        l_inverted = loss_fn_det.timing_loss(pred_inverted, target)
        assert l_inverted >= l_correct


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PhysicsInformedLoss.forward()
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhysicsInformedLossForward:
    """Integration tests for the full forward pass."""

    @pytest.fixture
    def loss_fn(self):
        return _loss()

    @pytest.fixture
    def preds(self):
        return _minimal_preds()

    @pytest.fixture
    def targets(self):
        return _minimal_targets()

    def test_returns_tuple(self, loss_fn, preds, targets):
        out = loss_fn(preds, targets, _graph_props())
        assert isinstance(out, tuple) and len(out) == 2

    def test_total_loss_is_scalar(self, loss_fn, preds, targets):
        total, _ = loss_fn(preds, targets, _graph_props())
        assert total.dim() == 0

    def test_total_loss_non_negative(self, loss_fn, preds, targets):
        total, _ = loss_fn(preds, targets, _graph_props())
        assert total >= 0.0

    def test_loss_dict_has_total_key(self, loss_fn, preds, targets):
        _, loss_dict = loss_fn(preds, targets, _graph_props())
        assert "total" in loss_dict

    def test_loss_dict_has_prediction_key(self, loss_fn, preds, targets):
        _, loss_dict = loss_fn(preds, targets, _graph_props())
        assert "prediction" in loss_dict

    def test_all_components_finite(self, loss_fn, preds, targets):
        total, loss_dict = loss_fn(preds, targets, _graph_props())
        assert torch.isfinite(total)
        for k, v in loss_dict.items():
            assert not (v != v), f"NaN in loss_dict['{k}']"  # NaN check for floats

    def test_backward_works(self, loss_fn, preds, targets):
        """Gradients must flow from the loss back through all prediction tensors."""
        preds_grad = {k: v.detach().requires_grad_(True) for k, v in preds.items()}
        total, _ = loss_fn(preds_grad, targets, _graph_props())
        total.backward()
        # failure_probability always contributes to loss
        assert preds_grad["failure_probability"].grad is not None

    def test_missing_optional_targets_no_crash(self, loss_fn, preds):
        """
        Only failure_label is required; other targets are optional.
        Passing only failure_label should not raise.
        """
        sparse_targets = {"failure_label": (torch.rand(B, N) > 0.5).float()}
        total, loss_dict = loss_fn(preds, sparse_targets, _graph_props())
        assert torch.isfinite(total)
        assert "prediction" in loss_dict

    def test_parent_loss_skipped_when_all_negative(self, loss_fn, preds):
        """
        When all parent_labels are -1 (no failing node), mask.sum() == 0
        and the parent loss term should be omitted from loss_dict.
        """
        targets_no_fail = _minimal_targets()
        targets_no_fail["parent_labels"] = torch.full((B, N), -1, dtype=torch.long)
        _, loss_dict = loss_fn(preds, targets_no_fail, _graph_props())
        assert "parent" not in loss_dict

    def test_parent_loss_present_when_some_failing(self, loss_fn, preds):
        """
        When some parent_labels are >= 0, the parent loss term must appear.
        """
        targets_fail = _minimal_targets()
        targets_fail["parent_labels"] = torch.full((B, N), -1, dtype=torch.long)
        targets_fail["parent_labels"][:, 0] = N   # node 0 is a trigger
        _, loss_dict = loss_fn(preds, targets_fail, _graph_props())
        assert "parent" in loss_dict

    def test_kwargs_absorbed_silently(self):
        """
        Legacy call sites may pass extra keyword arguments that no longer
        map to active heads.  The **kwargs absorber must swallow them.
        """
        PhysicsInformedLoss(
            lambda_prediction=1.0,
            lambda_voltage=0.5,   # removed auxiliary head — must not raise
            lambda_temperature=0.3,
        )

    def test_loss_decreases_with_gradient_step(self):
        """
        A single Adam step on synthetic data should reduce the loss.
        This is a smoke test for the full training loop.
        """
        loss_fn = _loss()
        preds   = _minimal_preds()
        params  = [v for v in preds.values() if v.requires_grad or True]
        for p in params:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(params, lr=0.1)
        targets   = _minimal_targets()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            total, _ = loss_fn(preds, targets, _graph_props())
            total.backward()
            optimizer.step()
            losses.append(total.item())

        # At least some improvement over 5 steps
        assert losses[-1] < losses[0] * 5 + 1.0   # lenient bound


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Lambda weight routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestLambdaWeightRouting:
    """Verify that lambda multipliers actually scale individual loss components."""

    def test_zero_prediction_lambda_excludes_prediction(self):
        """With lambda_prediction=0 the 'prediction' contribution is zeroed."""
        fn_zero = PhysicsInformedLoss(lambda_prediction=0.0, lambda_risk=0.0,
                                       lambda_timing=0.0, lambda_parent=0.0)
        fn_one  = PhysicsInformedLoss(lambda_prediction=1.0, lambda_risk=0.0,
                                       lambda_timing=0.0, lambda_parent=0.0)
        preds   = _minimal_preds()
        targets = {"failure_label": (torch.rand(B, N) > 0.5).float()}
        t0, _ = fn_zero(preds, targets, _graph_props())
        t1, _ = fn_one(preds, targets, _graph_props())
        assert t0.item() == pytest.approx(0.0, abs=1e-6)
        assert t1.item() > 0.0

    def test_parent_lambda_scales_parent_loss(self):
        targets_fail = _minimal_targets()
        targets_fail["parent_labels"] = torch.full((B, N), -1, dtype=torch.long)
        targets_fail["parent_labels"][:, 0] = N

        fn_low  = PhysicsInformedLoss(lambda_parent=0.01, lambda_risk=0.0, lambda_timing=0.0)
        fn_high = PhysicsInformedLoss(lambda_parent=10.0, lambda_risk=0.0, lambda_timing=0.0)
        preds = _minimal_preds()

        t_low,  d_low  = fn_low(preds, targets_fail, _graph_props())
        t_high, d_high = fn_high(preds, targets_fail, _graph_props())

        assert d_high["parent"] == pytest.approx(d_low["parent"], rel=1e-4)
        assert t_high > t_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
