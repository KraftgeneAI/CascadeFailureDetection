"""
Tests for the code introduced in the current commit
====================================================
New / modified files:
  - cascade_prediction/data/sliding_window_dataset.py
      SlidingWindowDataset, _build_node_features
  - cascade_prediction/data/preprocessing/edge_masking.py
      create_edge_mask_sequence  (new function)
  - cascade_prediction/data/preprocessing/normalization.py
      denormalize_power, denormalize_frequency  (new functions)
  - cascade_prediction/data/__init__.py  (re-exports, smoke-tested)

Test classes
------------
  TestSlidingWindowDataset   — __init__, __len__, __getitem__,
                               get_cascade_label, window slicing,
                               training-mode noise, edge-mask range
  TestBuildNodeFeatures      — shape, delta correctness, t_pos monotone,
                               TTF clamp, no-NaN, single-timestep
  TestCreateEdgeMaskSequence — length, all-active baseline, t-1 lag masking,
                               first-step always unmasked, use_previous=False
  TestDenormalize            — denormalize_power, denormalize_frequency
  TestPackageExports         — import smoke-tests for __init__ re-exports
"""

import pickle
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from cascade_prediction.data.sliding_window_dataset import (
    SlidingWindowDataset,
    _build_node_features,
    WINDOW_SIZE,
    TEMPORAL_KEYS,
)
from cascade_prediction.data.preprocessing.edge_masking import (
    create_edge_mask_sequence,
    create_edge_mask_from_failures,
)
from cascade_prediction.data.preprocessing.normalization import (
    denormalize_power,
    denormalize_frequency,
)
from cascade_prediction.data.generator.config import Settings


# ---------------------------------------------------------------------------
# Scenario-building helpers
# ---------------------------------------------------------------------------

NUM_NODES = 10
NUM_EDGES = 8


def _edge_index(n=NUM_NODES, e=NUM_EDGES) -> np.ndarray:
    rng = np.random.default_rng(0)
    src = rng.integers(0, n, e)
    dst = rng.integers(0, n, e)
    return np.stack([src, dst], axis=0)


def _timestep(n=NUM_NODES, e=NUM_EDGES, failed_nodes=()) -> dict:
    ts = {
        'scada_data':       np.random.randn(n, 18).astype(np.float32),
        'pmu_sequence':     np.random.randn(n, 8).astype(np.float32),
        'equipment_status': np.random.randn(n, 10).astype(np.float32),
        'edge_attr':        np.random.randn(e, 7).astype(np.float32),
        'power_injection':  np.random.randn(n, 1).astype(np.float32),
        'reactive_injection': np.random.randn(n, 1).astype(np.float32),
        'node_labels':      np.zeros(n, dtype=np.float32),
    }
    for ni in failed_nodes:
        ts['node_labels'][ni] = 1.0
    return ts


def _scenario(seq_len=20, n=NUM_NODES, e=NUM_EDGES,
              is_cascade=False, failed_nodes=()) -> dict:
    """Build a minimal scenario dict that SlidingWindowDataset._load() accepts."""
    return {
        'sequence':   [_timestep(n, e, failed_nodes if is_cascade and t >= seq_len // 2 else ())
                       for t in range(seq_len)],
        'edge_index': _edge_index(n, e),
        'metadata':   {'is_cascade': is_cascade},
    }


def _save_scenario(path: Path, scenario: dict) -> None:
    with open(path, 'wb') as f:
        pickle.dump(scenario, f)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def data_dir_one_cascade(temp_dir):
    """Directory with a single cascade scenario (seq_len=20)."""
    _save_scenario(temp_dir / 'scenario_0.pkl',
                   _scenario(seq_len=20, is_cascade=True, failed_nodes=[0]))
    return temp_dir


@pytest.fixture
def data_dir_mixed(temp_dir):
    """Directory with one cascade (seq_len=20) and one normal (seq_len=15) scenario."""
    _save_scenario(temp_dir / 'scenario_0.pkl',
                   _scenario(seq_len=20, is_cascade=True, failed_nodes=[0]))
    _save_scenario(temp_dir / 'scenario_1.pkl',
                   _scenario(seq_len=15, is_cascade=False))
    return temp_dir


# ---------------------------------------------------------------------------
# SlidingWindowDataset
# ---------------------------------------------------------------------------

class TestSlidingWindowDataset:

    # ── Construction / length ───────────────────────────────────────────────

    def test_empty_dir_has_zero_length(self, temp_dir):
        ds = SlidingWindowDataset(str(temp_dir))
        assert len(ds) == 0

    def test_length_equals_windows_per_scenario(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        # seq_len=20, windows = 20 - WINDOW_SIZE
        expected = 20 - WINDOW_SIZE
        assert len(ds) == expected

    def test_length_sums_across_scenarios(self, data_dir_mixed):
        ds = SlidingWindowDataset(str(data_dir_mixed), window_size=WINDOW_SIZE)
        # cascade: 20 - W, normal: 15 - W
        expected = (20 - WINDOW_SIZE) + (15 - WINDOW_SIZE)
        assert len(ds) == expected

    def test_short_scenario_skipped(self, temp_dir):
        """A scenario shorter than window_size + 1 must be silently skipped."""
        _save_scenario(temp_dir / 'scenario_0.pkl',
                       _scenario(seq_len=WINDOW_SIZE))   # exactly window_size → 0 windows
        ds = SlidingWindowDataset(str(temp_dir), window_size=WINDOW_SIZE)
        assert len(ds) == 0

    def test_corrupted_file_skipped(self, temp_dir):
        """Unpickleable files must not crash __init__."""
        bad = temp_dir / 'scenario_0.pkl'
        bad.write_text("not a pickle")
        _save_scenario(temp_dir / 'scenario_1.pkl', _scenario(seq_len=20))
        ds = SlidingWindowDataset(str(temp_dir), window_size=WINDOW_SIZE)
        assert len(ds) == 20 - WINDOW_SIZE  # only valid scenario contributes

    # ── __getitem__ output keys and shapes ───────────────────────────────────

    def test_getitem_required_keys(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        required = {
            'scada_data', 'pmu_sequence', 'equipment_status',
            'node_features', 'edge_index', 'edge_attr', 'edge_mask',
        }
        missing = required - set(item.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_scada_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        W = WINDOW_SIZE
        assert item['scada_data'].shape == (W + 1, NUM_NODES, 18), \
            f"Expected ({W+1}, {NUM_NODES}, 18), got {item['scada_data'].shape}"

    def test_pmu_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['pmu_sequence'].shape == (WINDOW_SIZE + 1, NUM_NODES, 8)

    def test_equipment_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['equipment_status'].shape == (WINDOW_SIZE + 1, NUM_NODES, 10)

    def test_node_features_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['node_features'].shape == (WINDOW_SIZE + 1, NUM_NODES, 119)

    def test_edge_index_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['edge_index'].shape == (2, NUM_EDGES)
        assert item['edge_index'].dtype == torch.long

    def test_edge_attr_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['edge_attr'].shape == (WINDOW_SIZE + 1, NUM_EDGES, 7)

    def test_edge_mask_shape(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        assert item['edge_mask'].shape == (WINDOW_SIZE + 1, NUM_EDGES)

    def test_all_outputs_are_float_tensors(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        for k, v in item.items():
            if k == 'edge_index':
                continue
            assert isinstance(v, torch.Tensor), f"'{k}' is not a Tensor"
            assert v.dtype == torch.float32,     f"'{k}' dtype is {v.dtype}, not float32"

    def test_no_nan_in_outputs(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in '{k}'"

    # ── Window slicing correctness ────────────────────────────────────────────

    def test_consecutive_windows_shift_by_one(self, data_dir_one_cascade):
        """Window i+1 should be identical to window i shifted one step forward."""
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        if len(ds) < 2:
            pytest.skip("Need at least 2 windows")
        item0 = ds[0]
        item1 = ds[1]
        # Steps [1:W+1] of window 0 == steps [0:W] of window 1
        assert torch.allclose(
            item0['scada_data'][1:],
            item1['scada_data'][:-1],
        ), "Consecutive windows don't form a coherent sliding sequence"

    def test_last_window_accessible(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[len(ds) - 1]
        assert item['scada_data'].shape[0] == WINDOW_SIZE + 1

    def test_custom_window_size(self, temp_dir):
        _save_scenario(temp_dir / 'scenario_0.pkl', _scenario(seq_len=30))
        ds = SlidingWindowDataset(str(temp_dir), window_size=5)
        assert len(ds) == 30 - 5

    # ── get_cascade_label ─────────────────────────────────────────────────────

    def test_cascade_label_true(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        # All windows come from the same cascade scenario
        for idx in range(len(ds)):
            assert ds.get_cascade_label(idx) is True

    def test_cascade_label_false(self, temp_dir):
        _save_scenario(temp_dir / 'scenario_0.pkl',
                       _scenario(seq_len=20, is_cascade=False))
        ds = SlidingWindowDataset(str(temp_dir), window_size=WINDOW_SIZE)
        for idx in range(len(ds)):
            assert ds.get_cascade_label(idx) is False

    def test_cascade_label_per_scenario(self, data_dir_mixed):
        ds = SlidingWindowDataset(str(data_dir_mixed), window_size=WINDOW_SIZE)
        labels = [ds.get_cascade_label(i) for i in range(len(ds))]
        # Must contain both True and False across the two scenarios
        assert True  in labels
        assert False in labels

    # ── Edge-mask range ───────────────────────────────────────────────────────

    def test_edge_mask_in_unit_range(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        item = ds[0]
        mask = item['edge_mask']
        assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0), \
            f"Edge mask out of [0,1]: min={mask.min()}, max={mask.max()}"

    # ── Training-mode noise ────────────────────────────────────────────────────

    def test_training_noise_makes_samples_differ(self, temp_dir):
        """With is_training=True successive calls on the same window must differ."""
        _save_scenario(temp_dir / 'scenario_0.pkl', _scenario(seq_len=20))
        ds = SlidingWindowDataset(str(temp_dir) + '/', window_size=WINDOW_SIZE)
        ds.is_training = True   # force training mode
        s1 = ds[0]['scada_data'].clone()
        s2 = ds[0]['scada_data'].clone()
        # With Gaussian noise added, two independent draws should differ
        # (with probability 1 - 2^{-mantissa bits})
        assert not torch.allclose(s1, s2), \
            "Training-mode noise did not produce different samples"

    def test_eval_mode_is_deterministic(self, data_dir_one_cascade):
        ds = SlidingWindowDataset(str(data_dir_one_cascade), window_size=WINDOW_SIZE)
        ds.is_training = False
        s1 = ds[0]['scada_data'].clone()
        s2 = ds[0]['scada_data'].clone()
        assert torch.allclose(s1, s2), "Eval-mode should be deterministic"


# ---------------------------------------------------------------------------
# _build_node_features
# ---------------------------------------------------------------------------

class TestBuildNodeFeatures:

    T_LEN = 6
    N = 5

    @pytest.fixture
    def tensors(self):
        T, N = self.T_LEN, self.N
        scada = torch.randn(T, N, 18)
        pmu   = torch.randn(T, N, 8)
        equip = torch.randn(T, N, 10)
        p_inj = torch.randn(T, N, 1)
        q_inj = torch.randn(T, N, 1)
        return scada, pmu, equip, p_inj, q_inj

    def test_output_shape(self, tensors):
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        assert out.shape == (self.T_LEN, self.N, 119)

    def test_output_dtype_is_float32(self, tensors):
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        assert out.dtype == torch.float32

    def test_no_nan(self, tensors):
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        assert not torch.isnan(out).any(), "NaN in node features"

    def test_delta1_zero_at_t0(self, tensors):
        """1-step delta must be zero at t=0 (no previous step)."""
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        # delta1 occupies columns [38:76]
        delta1_t0 = out[0, :, 38:76]
        assert torch.all(delta1_t0 == 0.0), "delta1 at t=0 must be zero"

    def test_delta1_non_zero_after_t0(self, tensors):
        """1-step delta must be non-zero at t≥1 for non-constant input."""
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        delta1_t1 = out[1, :, 38:76]
        assert not torch.all(delta1_t1 == 0.0), "delta1 at t=1 should be non-zero"

    def test_delta2_zero_at_t0_and_t1(self, tensors):
        """2-step delta must be zero at t=0 and t=1."""
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        delta2_t0 = out[0, :, 76:114]
        delta2_t1 = out[1, :, 76:114]
        assert torch.all(delta2_t0 == 0.0), "delta2 at t=0 must be zero"
        assert torch.all(delta2_t1 == 0.0), "delta2 at t=1 must be zero"

    def test_delta2_non_zero_at_t2(self, tensors):
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        delta2_t2 = out[2, :, 76:114]
        assert not torch.all(delta2_t2 == 0.0), "delta2 at t=2 should be non-zero"

    def test_t_pos_monotone_increasing(self, tensors):
        """Absolute timestep position (col 114) must be strictly increasing in t."""
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=2, seq_len=20)
        t_pos = out[:, 0, 114]   # shape [T]
        diffs = t_pos[1:] - t_pos[:-1]
        assert torch.all(diffs > 0), f"t_pos not strictly increasing: {t_pos}"

    def test_t_pos_start_offset(self):
        """start_t should shift all t_pos values upward."""
        T, N = 4, 3
        scada = torch.zeros(T, N, 18)
        pmu   = torch.zeros(T, N, 8)
        equip = torch.zeros(T, N, 10)
        p_inj = torch.zeros(T, N, 1)
        q_inj = torch.zeros(T, N, 1)

        out0 = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        out5 = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=5, seq_len=20)

        t_pos0 = out0[:, 0, 114]
        t_pos5 = out5[:, 0, 114]
        assert torch.all(t_pos5 > t_pos0), \
            "Higher start_t should produce higher t_pos values"

    def test_ttf_features_clamped_to_unit_interval(self, tensors):
        """TTF features (cols 115:119) must lie in [0, 1]."""
        scada, pmu, equip, p_inj, q_inj = tensors
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=20)
        ttf = out[:, :, 115:119]
        assert torch.all(ttf >= 0.0) and torch.all(ttf <= 1.0), \
            f"TTF features out of [0,1]: min={ttf.min()}, max={ttf.max()}"

    def test_single_timestep(self):
        """T=1 is a valid edge case — delta1 and delta2 should both be zero."""
        T, N = 1, 5
        scada = torch.randn(T, N, 18)
        pmu   = torch.randn(T, N, 8)
        equip = torch.randn(T, N, 10)
        p_inj = torch.randn(T, N, 1)
        q_inj = torch.randn(T, N, 1)
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=10)
        assert out.shape == (1, N, 119)
        assert torch.all(out[0, :, 38:114] == 0.0), "All deltas must be zero for T=1"

    def test_base_features_match_concat(self):
        """First 38 columns must equal cat([scada, pmu, equip, p_inj, q_inj], dim=2)."""
        T, N = 3, 4
        scada = torch.randn(T, N, 18)
        pmu   = torch.randn(T, N, 8)
        equip = torch.randn(T, N, 10)
        p_inj = torch.randn(T, N, 1)
        q_inj = torch.randn(T, N, 1)
        out = _build_node_features(scada, pmu, equip, p_inj, q_inj, start_t=0, seq_len=10)
        expected_base = torch.cat([scada, pmu, equip, p_inj, q_inj], dim=2)
        assert torch.allclose(out[:, :, :38], expected_base), \
            "Base features (cols 0:38) should match concatenated inputs"


# ---------------------------------------------------------------------------
# create_edge_mask_sequence  (new function in edge_masking.py)
# ---------------------------------------------------------------------------

class TestCreateEdgeMaskSequence:

    @pytest.fixture
    def simple_edge_index(self):
        # 0→1, 1→2, 2→3, 3→0  (ring)
        return np.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    @pytest.fixture
    def no_failure_sequence(self):
        return [
            {'node_labels': np.zeros(4, dtype=np.float32)} for _ in range(5)
        ]

    # ── Basic contract ───────────────────────────────────────────────────────

    def test_output_length_matches_sequence(self, simple_edge_index, no_failure_sequence):
        masks = create_edge_mask_sequence(
            no_failure_sequence, simple_edge_index, num_edges=4
        )
        assert len(masks) == len(no_failure_sequence)

    def test_no_failures_all_active(self, simple_edge_index, no_failure_sequence):
        masks = create_edge_mask_sequence(
            no_failure_sequence, simple_edge_index, num_edges=4
        )
        for t, m in enumerate(masks):
            arr = m.numpy() if isinstance(m, torch.Tensor) else m
            assert np.all(arr == 1.0), f"All edges should be active at t={t}"

    def test_first_step_always_unmasked(self, simple_edge_index):
        """At t=0 there is no previous timestep — first mask must be all 1."""
        sequence = [
            {'node_labels': np.array([1.0, 0.0, 0.0, 0.0])},  # node 0 fails at t=0
            {'node_labels': np.zeros(4, dtype=np.float32)},
        ]
        masks = create_edge_mask_sequence(sequence, simple_edge_index, num_edges=4)
        first = masks[0]
        arr = first.numpy() if isinstance(first, torch.Tensor) else first
        assert np.all(arr == 1.0), "First timestep must not be masked (no prior failures)"

    def test_failure_at_t0_masks_edges_at_t1(self, simple_edge_index):
        """Node 0 fails at t=0 → edges 0→1 and 3→0 should be masked at t=1."""
        sequence = [
            {'node_labels': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)},
            {'node_labels': np.zeros(4, dtype=np.float32)},
        ]
        masks = create_edge_mask_sequence(sequence, simple_edge_index, num_edges=4)
        t1 = masks[1]
        arr = t1.numpy() if isinstance(t1, torch.Tensor) else t1
        # Edge 0: 0→1  (node 0 failed → masked)
        assert arr[0] == 0.0, "Edge 0→1 must be masked (node 0 failed at t=0)"
        # Edge 3: 3→0  (node 0 failed → masked)
        assert arr[3] == 0.0, "Edge 3→0 must be masked (node 0 failed at t=0)"
        # Edge 1: 1→2  (neither endpoint failed → active)
        assert arr[1] == 1.0, "Edge 1→2 must remain active"
        # Edge 2: 2→3  (neither endpoint failed → active)
        assert arr[2] == 1.0, "Edge 2→3 must remain active"

    def test_lag_is_one_step(self, simple_edge_index):
        """Failure at t=1 should only affect masks at t=2, not t=1."""
        sequence = [
            {'node_labels': np.zeros(4, dtype=np.float32)},
            {'node_labels': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)},
            {'node_labels': np.zeros(4, dtype=np.float32)},
        ]
        masks = create_edge_mask_sequence(sequence, simple_edge_index, num_edges=4)

        # t=1: node 1 just failed, but mask uses t=0's labels → all active
        arr_t1 = masks[1]
        arr_t1 = arr_t1.numpy() if isinstance(arr_t1, torch.Tensor) else arr_t1
        assert np.all(arr_t1 == 1.0), "Mask at t=1 must not reflect failures that happen at t=1"

        # t=2: t=1 failures propagate → edges touching node 1 masked
        arr_t2 = masks[2]
        arr_t2 = arr_t2.numpy() if isinstance(arr_t2, torch.Tensor) else arr_t2
        assert arr_t2[0] == 0.0, "Edge 0→1 must be masked at t=2 (node 1 failed at t=1)"
        assert arr_t2[1] == 0.0, "Edge 1→2 must be masked at t=2 (node 1 failed at t=1)"

    def test_use_previous_timestep_false_all_active(self, simple_edge_index):
        """With use_previous_timestep=False, all masks must be 1 regardless of labels."""
        sequence = [
            {'node_labels': np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)}
            for _ in range(4)
        ]
        masks = create_edge_mask_sequence(
            sequence, simple_edge_index, num_edges=4,
            use_previous_timestep=False
        )
        for t, m in enumerate(masks):
            arr = m.numpy() if isinstance(m, torch.Tensor) else m
            assert np.all(arr == 1.0), \
                f"With use_previous_timestep=False all masks must be 1 (t={t})"

    def test_torch_tensor_edge_index(self):
        """Function must accept a torch.Tensor edge index."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        sequence = [
            {'node_labels': np.zeros(2, dtype=np.float32)} for _ in range(3)
        ]
        masks = create_edge_mask_sequence(sequence, edge_index, num_edges=2)
        assert len(masks) == 3

    def test_single_timestep_sequence(self, simple_edge_index):
        sequence = [{'node_labels': np.ones(4, dtype=np.float32)}]
        masks = create_edge_mask_sequence(sequence, simple_edge_index, num_edges=4)
        assert len(masks) == 1
        arr = masks[0].numpy() if isinstance(masks[0], torch.Tensor) else masks[0]
        assert np.all(arr == 1.0), "Single-step sequence: mask must be all 1"

    def test_masks_are_binary_in_no_failure_case(self, simple_edge_index, no_failure_sequence):
        masks = create_edge_mask_sequence(
            no_failure_sequence, simple_edge_index, num_edges=4
        )
        for m in masks:
            arr = m.numpy() if isinstance(m, torch.Tensor) else m
            assert np.all((arr == 0.0) | (arr == 1.0)), \
                "Masks from create_edge_mask_sequence must be binary"


# ---------------------------------------------------------------------------
# denormalize_power / denormalize_frequency
# ---------------------------------------------------------------------------

class TestDenormalize:
    """Tests for the new inverse normalisation helpers."""

    # ── denormalize_power ────────────────────────────────────────────────────

    def test_power_roundtrip_tensor(self):
        """normalize then denormalize must recover original values."""
        from cascade_prediction.data.preprocessing.normalization import normalize_power
        original = torch.tensor([100.0, 200.0, 50.0])
        base_mva = 100.0
        pu       = normalize_power(original, base_mva)
        recovered = denormalize_power(pu, base_mva)
        assert torch.allclose(recovered, original), \
            f"Power round-trip failed: {recovered} vs {original}"

    def test_power_roundtrip_numpy(self):
        from cascade_prediction.data.preprocessing.normalization import normalize_power
        original = np.array([150.0, 300.0])
        base_mva = 100.0
        pu        = normalize_power(original, base_mva)
        recovered = denormalize_power(pu, base_mva)
        assert np.allclose(recovered, original)

    def test_power_scalar(self):
        assert denormalize_power(1.0, 100.0) == pytest.approx(100.0)
        assert denormalize_power(2.5, 200.0) == pytest.approx(500.0)

    def test_power_zero(self):
        result = denormalize_power(torch.zeros(5), 100.0)
        assert torch.all(result == 0.0)

    def test_power_preserves_shape(self):
        x = torch.randn(3, 4, 5)
        result = denormalize_power(x, 100.0)
        assert result.shape == x.shape

    def test_power_negative_values(self):
        """Reactive / two-directional flows can be negative."""
        pu = torch.tensor([-1.0, 0.0, 2.0])
        result = denormalize_power(pu, 100.0)
        assert torch.allclose(result, torch.tensor([-100.0, 0.0, 200.0]))

    # ── denormalize_frequency ────────────────────────────────────────────────

    def test_frequency_roundtrip_tensor(self):
        from cascade_prediction.data.preprocessing.normalization import normalize_frequency
        original  = torch.tensor([60.0, 59.5, 60.5])
        base_freq = 60.0
        pu        = normalize_frequency(original, base_freq)
        recovered = denormalize_frequency(pu, base_freq)
        assert torch.allclose(recovered, original)

    def test_frequency_roundtrip_numpy(self):
        from cascade_prediction.data.preprocessing.normalization import normalize_frequency
        original  = np.array([60.0, 59.0])
        base_freq = 60.0
        pu        = normalize_frequency(original, base_freq)
        recovered = denormalize_frequency(pu, base_freq)
        assert np.allclose(recovered, original)

    def test_frequency_scalar(self):
        assert denormalize_frequency(1.0, 60.0) == pytest.approx(60.0)
        assert denormalize_frequency(0.5, 60.0) == pytest.approx(30.0)

    def test_frequency_preserves_shape(self):
        x = torch.randn(2, 3)
        result = denormalize_frequency(x, 60.0)
        assert result.shape == x.shape

    def test_frequency_custom_base(self):
        pu = torch.tensor([1.0])
        assert denormalize_frequency(pu, 50.0).item() == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Package-level export smoke-tests
# ---------------------------------------------------------------------------

class TestPackageExports:
    """Verify that __init__.py re-exports are importable and correct."""

    def test_sliding_window_dataset_importable(self):
        from cascade_prediction.data import SlidingWindowDataset
        assert SlidingWindowDataset is not None

    def test_window_size_constant_importable(self):
        from cascade_prediction.data import WINDOW_SIZE
        assert isinstance(WINDOW_SIZE, int)
        assert WINDOW_SIZE > 0

    def test_temporal_keys_importable(self):
        from cascade_prediction.data import TEMPORAL_KEYS
        assert isinstance(TEMPORAL_KEYS, (set, frozenset))
        assert 'scada_data' in TEMPORAL_KEYS

    def test_create_edge_mask_sequence_importable(self):
        from cascade_prediction.data import create_edge_mask_sequence
        assert callable(create_edge_mask_sequence)

    def test_denormalize_power_importable(self):
        from cascade_prediction.data import denormalize_power
        assert callable(denormalize_power)

    def test_denormalize_frequency_importable(self):
        from cascade_prediction.data import denormalize_frequency
        assert callable(denormalize_frequency)

    def test_grid_state_forecaster_importable_from_models(self):
        from cascade_prediction.models import GridStateForecaster
        assert GridStateForecaster is not None

    def test_index_constants_importable_from_models(self):
        from cascade_prediction.models import (
            SCADA_VAR_IDX, SCADA_CONST_IDX,
            EQUIP_VAR_IDX, EQUIP_CONST_IDX,
        )
        assert len(SCADA_VAR_IDX) == 13
        assert len(SCADA_CONST_IDX) == 5
        assert len(EQUIP_VAR_IDX) == 3
        assert len(EQUIP_CONST_IDX) == 7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
