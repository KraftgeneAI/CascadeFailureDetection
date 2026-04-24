"""
Tests for cascade_prediction/models/grid_state_forecaster.py
=============================================================
Covers:
  - Module-level constants (index lists, counts)
  - GridStateForecaster forward pass
      - output keys, shapes, no-NaN
      - optional edge_attr / edge_mask paths
      - static vs per-timestep edge_attr
      - gradient flow
      - eval-mode determinism
  - GridStateForecaster.compute_loss
      - perfect prediction → near-zero loss
      - loss dict keys and float values
      - total == sum of components
      - differentiable
  - extract_next_step_targets
      - extracts last timestep variable indices
      - SCADA var / equip var subset correctness
  - assemble_full_scada
      - output shape [B, N, 18]
      - var indices written correctly
      - const indices written correctly
      - non-index positions stay zero
  - assemble_full_equip
      - output shape [B, N, 10]
      - var / const placement correctness
  - Round-trip: extract → assemble reconstructs the last timestep
"""

import pytest
import torch

from cascade_prediction.models.grid_state_forecaster import (
    GridStateForecaster,
    extract_next_step_targets,
    assemble_full_scada,
    assemble_full_equip,
    SCADA_VAR_IDX,
    SCADA_CONST_IDX,
    EQUIP_VAR_IDX,
    EQUIP_CONST_IDX,
    N_SCADA_VAR,
    N_PMU,
    N_EQUIP_VAR,
)
from cascade_prediction.data.generator.config import Settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH  = 2
NODES  = 8
EDGES  = 10
T      = 4   # sequence length

SCADA_F = Settings.Embedding.INFRA_SCADA_FEATURES    # 18
PMU_F   = Settings.Embedding.INFRA_PMU_FEATURES      # 8
EQUIP_F = Settings.Embedding.INFRA_EQUIPMENT_FEATURES # 10
NODE_F  = Settings.Embedding.NODE_FEATURE_DIM         # 119
EDGE_F  = Settings.Model.EDGE_FEATURES                # 7
EMB     = Settings.Model.EMBEDDING_DIM                # 128
HEADS   = Settings.Model.HEADS                        # 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _edge_index(n=NODES, e=EDGES) -> torch.Tensor:
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    return torch.stack([src, dst], dim=0)


def _make_batch(
    batch_size=BATCH,
    num_nodes=NODES,
    num_edges=EDGES,
    seq_len=T,
    include_edge_attr=True,
    include_edge_mask=True,
    static_edge_attr=False,
):
    """Build a minimal batch dict that GridStateForecaster.forward() expects."""
    B, N, E = batch_size, num_nodes, num_edges
    ei = _edge_index(N, E)

    batch = {
        'scada_data':       _rand(B, seq_len, N, SCADA_F),
        'pmu_sequence':     _rand(B, seq_len, N, PMU_F),
        'equipment_status': _rand(B, seq_len, N, EQUIP_F),
        'node_features':    _rand(B, seq_len, N, NODE_F),
        'edge_index':       ei,
    }

    if include_edge_attr:
        if static_edge_attr:
            batch['edge_attr'] = _rand(B, E, EDGE_F)          # [B, E, F]
        else:
            batch['edge_attr'] = _rand(B, seq_len, E, EDGE_F) # [B, T, E, F]

    if include_edge_mask:
        batch['edge_mask'] = torch.ones(B, seq_len, E)

    return batch


def _make_model(num_gnn_layers=1) -> GridStateForecaster:
    """Create a small, fast model for testing."""
    return GridStateForecaster(
        embedding_dim=EMB,
        num_gnn_layers=num_gnn_layers,
        heads=HEADS,
        dropout=0.0,
    ).eval()


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Verify the index split constants are internally consistent."""

    def test_scada_var_count(self):
        assert N_SCADA_VAR == len(SCADA_VAR_IDX) == 13

    def test_pmu_count(self):
        assert N_PMU == PMU_F == 8

    def test_equip_var_count(self):
        assert N_EQUIP_VAR == len(EQUIP_VAR_IDX) == 3

    def test_scada_const_count(self):
        assert len(SCADA_CONST_IDX) == 5

    def test_equip_const_count(self):
        assert len(EQUIP_CONST_IDX) == 7

    def test_scada_indices_partition_18(self):
        """Variable + constant indices must cover exactly {0..17} with no overlap."""
        all_idx = sorted(SCADA_VAR_IDX + SCADA_CONST_IDX)
        assert all_idx == list(range(18)), \
            f"SCADA indices do not partition [0,17]: {all_idx}"

    def test_equip_indices_partition_10(self):
        """Variable + constant indices must cover exactly {0..9} with no overlap."""
        all_idx = sorted(EQUIP_VAR_IDX + EQUIP_CONST_IDX)
        assert all_idx == list(range(10)), \
            f"Equipment indices do not partition [0,9]: {all_idx}"

    def test_no_overlap_scada(self):
        assert set(SCADA_VAR_IDX) & set(SCADA_CONST_IDX) == set()

    def test_no_overlap_equip(self):
        assert set(EQUIP_VAR_IDX) & set(EQUIP_CONST_IDX) == set()


# ---------------------------------------------------------------------------
# GridStateForecaster — forward()
# ---------------------------------------------------------------------------

class TestGridStateForecasterForward:

    # ── Output keys and shapes ───────────────────────────────────────────────

    def test_output_keys(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert set(out.keys()) == {'next_scada_vars', 'next_pmu', 'next_equip_vars'}

    def test_next_scada_vars_shape(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR), \
            f"Expected ({BATCH}, {NODES}, {N_SCADA_VAR}), got {out['next_scada_vars'].shape}"

    def test_next_pmu_shape(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out['next_pmu'].shape == (BATCH, NODES, N_PMU)

    def test_next_equip_vars_shape(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out['next_equip_vars'].shape == (BATCH, NODES, N_EQUIP_VAR)

    # ── No NaN ──────────────────────────────────────────────────────────────

    def test_no_nan_in_outputs(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        for key, val in out.items():
            assert not torch.isnan(val).any(), f"NaN detected in '{key}'"

    # ── Edge-attr / mask optional paths ──────────────────────────────────────

    def test_forward_without_edge_attr(self):
        """Model must fall back to zero edge attributes when key is absent."""
        model = _make_model()
        batch = _make_batch(include_edge_attr=False, include_edge_mask=False)
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)
        assert not torch.isnan(out['next_scada_vars']).any()

    def test_forward_without_edge_mask(self):
        """Model must work with edge_attr but no edge_mask."""
        model = _make_model()
        batch = _make_batch(include_edge_mask=False)
        with torch.no_grad():
            out = model(batch)
        assert out['next_pmu'].shape == (BATCH, NODES, N_PMU)

    def test_forward_with_static_edge_attr(self):
        """Static [B, E, F] edge attributes (no time dimension) should be accepted."""
        model = _make_model()
        batch = _make_batch(static_edge_attr=True)
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)

    def test_forward_2d_edge_attr(self):
        """Bare [E, F] edge attributes (legacy format) should be accepted."""
        model = _make_model()
        batch = _make_batch(include_edge_attr=False, include_edge_mask=False)
        batch['edge_attr'] = _rand(EDGES, EDGE_F)   # 2-D
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)

    # ── Temporal consistency ─────────────────────────────────────────────────

    def test_different_sequence_lengths(self):
        """Output shape must be independent of sequence length T."""
        model = _make_model()
        for seq_len in [1, 3, 8]:
            batch = _make_batch(seq_len=seq_len)
            with torch.no_grad():
                out = model(batch)
            assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR), \
                f"Shape mismatch for seq_len={seq_len}"

    def test_longer_sequence_changes_output(self):
        """A longer history should produce different predictions than a shorter one."""
        model = _make_model()
        torch.manual_seed(0)
        b_short = _make_batch(seq_len=1)
        torch.manual_seed(0)
        b_long  = _make_batch(seq_len=5)
        with torch.no_grad():
            out_s = model(b_short)
            out_l = model(b_long)
        # Different input histories → different outputs
        assert not torch.allclose(out_s['next_scada_vars'], out_l['next_scada_vars']), \
            "Predictions should differ for different sequence lengths"

    # ── Batch size independence ───────────────────────────────────────────────

    def test_single_sample_batch(self):
        model = _make_model()
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (1, NODES, N_SCADA_VAR)

    def test_large_batch(self):
        model = _make_model()
        batch = _make_batch(batch_size=8)
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (8, NODES, N_SCADA_VAR)

    # ── Eval-mode determinism ────────────────────────────────────────────────

    def test_eval_mode_is_deterministic(self):
        model = _make_model()
        batch = _make_batch()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        assert torch.allclose(out1['next_scada_vars'], out2['next_scada_vars']), \
            "Eval-mode forward must be deterministic"

    # ── Gradient flow ─────────────────────────────────────────────────────────

    def test_gradients_flow_through_forward(self):
        """All decoders must be reachable by backprop."""
        model = GridStateForecaster(
            embedding_dim=EMB, num_gnn_layers=1, heads=HEADS, dropout=0.0
        ).train()
        batch = _make_batch()
        out = model(batch)
        # Sum all outputs to get a scalar loss
        loss = (
            out['next_scada_vars'].mean()
            + out['next_pmu'].mean()
            + out['next_equip_vars'].mean()
        )
        loss.backward()
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No gradients flowed through the model"

    def test_each_decoder_receives_gradient(self):
        """Each decoder head must have its own gradient signal."""
        model = GridStateForecaster(
            embedding_dim=EMB, num_gnn_layers=1, heads=HEADS, dropout=0.0
        ).train()
        batch = _make_batch()

        decoders = {
            'scada_decoder': model.scada_decoder,
            'pmu_decoder':   model.pmu_decoder,
            'equip_decoder': model.equip_decoder,
        }

        for name, decoder in decoders.items():
            # Zero all grads
            model.zero_grad()
            out = model(batch)
            # Only backprop through this decoder's output
            key = name.replace('_decoder', '').replace('scada', 'next_scada_vars')
            key_map = {
                'scada_decoder': 'next_scada_vars',
                'pmu_decoder':   'next_pmu',
                'equip_decoder': 'next_equip_vars',
            }
            out[key_map[name]].mean().backward()
            # Last linear layer of this decoder should have a gradient
            last_linear = list(decoder.children())[-1]
            assert last_linear.weight.grad is not None, \
                f"{name} last layer has no gradient"

    # ── Multi-GNN-layer depth ─────────────────────────────────────────────────

    def test_deeper_model_runs(self):
        model = _make_model(num_gnn_layers=3)
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)
        assert not torch.isnan(out['next_scada_vars']).any()

    def test_zero_gnn_layers(self):
        """With num_gnn_layers=0, model has no extra GNN refinement — should still work."""
        model = _make_model(num_gnn_layers=0)
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)


# ---------------------------------------------------------------------------
# GridStateForecaster — compute_loss()
# ---------------------------------------------------------------------------

class TestGridStateForecasterComputeLoss:

    @pytest.fixture
    def predictions_and_targets(self):
        preds = {
            'next_scada_vars': _rand(BATCH, NODES, N_SCADA_VAR),
            'next_pmu':        _rand(BATCH, NODES, N_PMU),
            'next_equip_vars': _rand(BATCH, NODES, N_EQUIP_VAR),
        }
        tgts = {
            'next_scada_vars': _rand(BATCH, NODES, N_SCADA_VAR),
            'next_pmu':        _rand(BATCH, NODES, N_PMU),
            'next_equip_vars': _rand(BATCH, NODES, N_EQUIP_VAR),
        }
        return preds, tgts

    def test_returns_tuple_of_two(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        result = model.compute_loss(preds, tgts)
        assert isinstance(result, tuple) and len(result) == 2

    def test_total_loss_is_scalar_tensor(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        total, _ = model.compute_loss(preds, tgts)
        assert isinstance(total, torch.Tensor)
        assert total.dim() == 0

    def test_total_loss_non_negative(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        total, _ = model.compute_loss(preds, tgts)
        assert total.item() >= 0.0

    def test_loss_dict_keys(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        _, loss_dict = model.compute_loss(preds, tgts)
        assert set(loss_dict.keys()) == {'loss_scada', 'loss_pmu', 'loss_equip', 'total'}

    def test_loss_dict_values_are_floats(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        _, loss_dict = model.compute_loss(preds, tgts)
        for k, v in loss_dict.items():
            assert isinstance(v, float), f"loss_dict['{k}'] is not a float"

    def test_total_equals_sum_of_components(self, predictions_and_targets):
        model = _make_model()
        preds, tgts = predictions_and_targets
        total, loss_dict = model.compute_loss(preds, tgts)
        expected = loss_dict['loss_scada'] + loss_dict['loss_pmu'] + loss_dict['loss_equip']
        assert total.item() == pytest.approx(expected, rel=1e-5)

    def test_perfect_prediction_zero_loss(self):
        """When predictions == targets, every component loss must be 0."""
        model = _make_model()
        x = _rand(BATCH, NODES, N_SCADA_VAR)
        preds = {
            'next_scada_vars': x.clone(),
            'next_pmu':        _rand(BATCH, NODES, N_PMU),
            'next_equip_vars': _rand(BATCH, NODES, N_EQUIP_VAR),
        }
        tgts = {
            'next_scada_vars': x.clone(),
            'next_pmu':        preds['next_pmu'].clone(),
            'next_equip_vars': preds['next_equip_vars'].clone(),
        }
        total, loss_dict = model.compute_loss(preds, tgts)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_dict['loss_scada'] == pytest.approx(0.0, abs=1e-6)
        assert loss_dict['loss_pmu']   == pytest.approx(0.0, abs=1e-6)
        assert loss_dict['loss_equip'] == pytest.approx(0.0, abs=1e-6)

    def test_larger_error_yields_larger_loss(self):
        model = _make_model()
        zero_pred = {
            'next_scada_vars': torch.zeros(BATCH, NODES, N_SCADA_VAR),
            'next_pmu':        torch.zeros(BATCH, NODES, N_PMU),
            'next_equip_vars': torch.zeros(BATCH, NODES, N_EQUIP_VAR),
        }
        tgt_close = {k: v + 0.1 for k, v in zero_pred.items()}
        tgt_far   = {k: v + 10.0 for k, v in zero_pred.items()}
        loss_close, _ = model.compute_loss(zero_pred, tgt_close)
        loss_far,   _ = model.compute_loss(zero_pred, tgt_far)
        assert loss_far.item() > loss_close.item()

    def test_loss_is_differentiable(self):
        """Total loss must support .backward() for training."""
        model = GridStateForecaster(
            embedding_dim=EMB, num_gnn_layers=1, heads=HEADS, dropout=0.0
        ).train()
        batch = _make_batch()
        out = model(batch)
        tgts = {
            'next_scada_vars': _rand(BATCH, NODES, N_SCADA_VAR),
            'next_pmu':        _rand(BATCH, NODES, N_PMU),
            'next_equip_vars': _rand(BATCH, NODES, N_EQUIP_VAR),
        }
        total, _ = model.compute_loss(out, tgts)
        total.backward()
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0


# ---------------------------------------------------------------------------
# extract_next_step_targets()
# ---------------------------------------------------------------------------

class TestExtractNextStepTargets:
    """Tests for the training-time target extraction utility."""

    @pytest.fixture
    def full_tensors(self):
        scada = _rand(BATCH, T, NODES, SCADA_F)
        pmu   = _rand(BATCH, T, NODES, PMU_F)
        equip = _rand(BATCH, T, NODES, EQUIP_F)
        return scada, pmu, equip

    def test_output_keys(self, full_tensors):
        tgts = extract_next_step_targets(*full_tensors)
        assert set(tgts.keys()) == {'next_scada_vars', 'next_pmu', 'next_equip_vars'}

    def test_scada_var_shape(self, full_tensors):
        tgts = extract_next_step_targets(*full_tensors)
        assert tgts['next_scada_vars'].shape == (BATCH, NODES, N_SCADA_VAR)

    def test_pmu_shape(self, full_tensors):
        tgts = extract_next_step_targets(*full_tensors)
        assert tgts['next_pmu'].shape == (BATCH, NODES, N_PMU)

    def test_equip_var_shape(self, full_tensors):
        tgts = extract_next_step_targets(*full_tensors)
        assert tgts['next_equip_vars'].shape == (BATCH, NODES, N_EQUIP_VAR)

    def test_extracts_last_timestep(self, full_tensors):
        """Returned values must come from timestep T-1, not any earlier step."""
        scada, pmu, equip = full_tensors
        tgts = extract_next_step_targets(scada, pmu, equip)

        expected_scada = scada[:, -1, :, :][:, :, SCADA_VAR_IDX]
        assert torch.allclose(tgts['next_scada_vars'], expected_scada)

    def test_pmu_is_full_last_timestep(self, full_tensors):
        """PMU output should be the complete last-timestep PMU slice."""
        scada, pmu, equip = full_tensors
        tgts = extract_next_step_targets(scada, pmu, equip)
        assert torch.allclose(tgts['next_pmu'], pmu[:, -1, :, :])

    def test_equip_var_indices_correct(self, full_tensors):
        """Equipment variable indices must match EQUIP_VAR_IDX exactly."""
        scada, pmu, equip = full_tensors
        tgts = extract_next_step_targets(scada, pmu, equip)
        expected = equip[:, -1, :, :][:, :, EQUIP_VAR_IDX]
        assert torch.allclose(tgts['next_equip_vars'], expected)

    def test_single_timestep_sequence(self):
        """T=1: the only timestep IS the last timestep."""
        scada = _rand(BATCH, 1, NODES, SCADA_F)
        pmu   = _rand(BATCH, 1, NODES, PMU_F)
        equip = _rand(BATCH, 1, NODES, EQUIP_F)
        tgts  = extract_next_step_targets(scada, pmu, equip)
        expected_scada = scada[:, 0, :, :][:, :, SCADA_VAR_IDX]
        assert torch.allclose(tgts['next_scada_vars'], expected_scada)

    def test_does_not_modify_input(self, full_tensors):
        """Utility must be read-only — input tensors must be unchanged."""
        scada, pmu, equip = full_tensors
        scada_copy = scada.clone()
        extract_next_step_targets(scada, pmu, equip)
        assert torch.allclose(scada, scada_copy)


# ---------------------------------------------------------------------------
# assemble_full_scada()
# ---------------------------------------------------------------------------

class TestAssembleFullScada:

    @pytest.fixture
    def scada_parts(self):
        pred_vars = _rand(BATCH, NODES, N_SCADA_VAR)           # [B, N, 13]
        constants = _rand(BATCH, NODES, len(SCADA_CONST_IDX))  # [B, N,  5]
        return pred_vars, constants

    def test_output_shape(self, scada_parts):
        pred_vars, constants = scada_parts
        full = assemble_full_scada(pred_vars, constants)
        assert full.shape == (BATCH, NODES, 18)

    def test_var_indices_written(self, scada_parts):
        pred_vars, constants = scada_parts
        full = assemble_full_scada(pred_vars, constants)
        for out_i, src_i in enumerate(SCADA_VAR_IDX):
            assert torch.allclose(full[:, :, src_i], pred_vars[:, :, out_i]), \
                f"Variable index {src_i} (output col {out_i}) not written correctly"

    def test_const_indices_written(self, scada_parts):
        pred_vars, constants = scada_parts
        full = assemble_full_scada(pred_vars, constants)
        for out_i, src_i in enumerate(SCADA_CONST_IDX):
            assert torch.allclose(full[:, :, src_i], constants[:, :, out_i]), \
                f"Constant index {src_i} (input col {out_i}) not written correctly"

    def test_all_18_columns_covered(self, scada_parts):
        """No column in the assembled tensor should remain exactly zero
        if pred_vars and constants are non-zero random values."""
        pred_vars = torch.ones(BATCH, NODES, N_SCADA_VAR) * 3.0
        constants = torch.ones(BATCH, NODES, len(SCADA_CONST_IDX)) * 7.0
        full = assemble_full_scada(pred_vars, constants)
        # Every position should be either 3 (var) or 7 (const), not 0
        assert torch.all((full == 3.0) | (full == 7.0)), \
            "Some SCADA columns are unexpectedly zero"

    def test_output_dtype_matches_input(self):
        pred_vars = torch.ones(BATCH, NODES, N_SCADA_VAR, dtype=torch.float64)
        constants = torch.ones(BATCH, NODES, len(SCADA_CONST_IDX), dtype=torch.float64)
        full = assemble_full_scada(pred_vars, constants)
        assert full.dtype == torch.float64

    def test_single_sample(self):
        pred_vars = _rand(1, NODES, N_SCADA_VAR)
        constants = _rand(1, NODES, len(SCADA_CONST_IDX))
        full = assemble_full_scada(pred_vars, constants)
        assert full.shape == (1, NODES, 18)


# ---------------------------------------------------------------------------
# assemble_full_equip()
# ---------------------------------------------------------------------------

class TestAssembleFullEquip:

    @pytest.fixture
    def equip_parts(self):
        pred_vars = _rand(BATCH, NODES, N_EQUIP_VAR)            # [B, N, 3]
        constants = _rand(BATCH, NODES, len(EQUIP_CONST_IDX))   # [B, N, 7]
        return pred_vars, constants

    def test_output_shape(self, equip_parts):
        pred_vars, constants = equip_parts
        full = assemble_full_equip(pred_vars, constants)
        assert full.shape == (BATCH, NODES, 10)

    def test_var_indices_written(self, equip_parts):
        pred_vars, constants = equip_parts
        full = assemble_full_equip(pred_vars, constants)
        for out_i, src_i in enumerate(EQUIP_VAR_IDX):
            assert torch.allclose(full[:, :, src_i], pred_vars[:, :, out_i]), \
                f"Variable index {src_i} (output col {out_i}) not written correctly"

    def test_const_indices_written(self, equip_parts):
        pred_vars, constants = equip_parts
        full = assemble_full_equip(pred_vars, constants)
        for out_i, src_i in enumerate(EQUIP_CONST_IDX):
            assert torch.allclose(full[:, :, src_i], constants[:, :, out_i]), \
                f"Constant index {src_i} (input col {out_i}) not written correctly"

    def test_all_10_columns_covered(self):
        pred_vars = torch.ones(BATCH, NODES, N_EQUIP_VAR) * 2.0
        constants = torch.ones(BATCH, NODES, len(EQUIP_CONST_IDX)) * 5.0
        full = assemble_full_equip(pred_vars, constants)
        assert torch.all((full == 2.0) | (full == 5.0)), \
            "Some equipment columns are unexpectedly zero"

    def test_output_dtype_matches_input(self):
        pred_vars = torch.ones(BATCH, NODES, N_EQUIP_VAR, dtype=torch.float16)
        constants = torch.ones(BATCH, NODES, len(EQUIP_CONST_IDX), dtype=torch.float16)
        full = assemble_full_equip(pred_vars, constants)
        assert full.dtype == torch.float16


# ---------------------------------------------------------------------------
# Round-trip: extract → assemble reconstructs the original last timestep
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """
    Verify that extract_next_step_targets + assemble_full_scada/equip is
    a lossless round-trip for the variable portions of the last timestep.
    """

    def test_scada_roundtrip(self):
        scada = _rand(BATCH, T, NODES, SCADA_F)
        pmu   = _rand(BATCH, T, NODES, PMU_F)
        equip = _rand(BATCH, T, NODES, EQUIP_F)

        tgts = extract_next_step_targets(scada, pmu, equip)

        # Carry-forward constant columns from last timestep
        scada_consts = scada[:, -1, :, :][:, :, SCADA_CONST_IDX]  # [B, N, 5]
        full = assemble_full_scada(tgts['next_scada_vars'], scada_consts)

        # Variable columns must match the last-timestep original
        for out_i, src_i in enumerate(SCADA_VAR_IDX):
            orig = scada[:, -1, :, src_i]
            recon = full[:, :, src_i]
            assert torch.allclose(orig, recon), \
                f"Round-trip failed for SCADA variable index {src_i}"

    def test_equip_roundtrip(self):
        scada = _rand(BATCH, T, NODES, SCADA_F)
        pmu   = _rand(BATCH, T, NODES, PMU_F)
        equip = _rand(BATCH, T, NODES, EQUIP_F)

        tgts = extract_next_step_targets(scada, pmu, equip)

        equip_consts = equip[:, -1, :, :][:, :, EQUIP_CONST_IDX]  # [B, N, 7]
        full = assemble_full_equip(tgts['next_equip_vars'], equip_consts)

        for out_i, src_i in enumerate(EQUIP_VAR_IDX):
            orig = equip[:, -1, :, src_i]
            recon = full[:, :, src_i]
            assert torch.allclose(orig, recon), \
                f"Round-trip failed for equipment variable index {src_i}"

    def test_pmu_roundtrip(self):
        scada = _rand(BATCH, T, NODES, SCADA_F)
        pmu   = _rand(BATCH, T, NODES, PMU_F)
        equip = _rand(BATCH, T, NODES, EQUIP_F)

        tgts = extract_next_step_targets(scada, pmu, equip)

        # PMU has no constant split — entire last-timestep slice should match
        assert torch.allclose(tgts['next_pmu'], pmu[:, -1, :, :])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
