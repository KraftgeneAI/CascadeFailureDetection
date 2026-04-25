"""
Tests for cascade_prediction/inference/
========================================
New package introduced in the final-video-stress branch.

Modules under test
------------------
  dataset.py   — ScenarioInferenceDataset
  reporting.py — format_risk_assessment, print_prediction_report, helpers
  utils.py     — NumpyEncoder
  predictor.py — pure methods: _generate_cascade_path, _extract_ground_truth,
                 _analyze_predictions  (no model file needed)
  __init__.py  — re-export smoke tests

Test classes
------------
  TestScenarioInferenceDataset   — __len__, __getitem__ window slicing,
                                   edge-mask teacher-forcing, preprocessing
  TestFormatRiskAssessment       — level labels, insufficient data, formatting
  TestPrintPredictionReport      — TP/TN/FP/FN verdicts, node analysis,
                                   risk text present in output
  TestInferenceNumpyEncoder      — int/float/ndarray/bool serialisation
  TestCascadePredictorPureMethods — path generation, ground-truth extraction,
                                   result assembly without a real model
  TestInferencePackageExports    — importable names from __init__.py
"""

import io
import json
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cascade_prediction.inference.dataset import ScenarioInferenceDataset
from cascade_prediction.inference.reporting import (
    format_risk_assessment,
    print_prediction_report,
    _print_overall_verdict,
    _print_node_analysis,
    _print_cascade_path,
)
from cascade_prediction.inference.utils import NumpyEncoder
from cascade_prediction.inference.predictor import CascadePredictor


# ---------------------------------------------------------------------------
# Scenario-building helpers
# ---------------------------------------------------------------------------

NUM_NODES = 6
NUM_EDGES = 4


def _edge_index_np() -> np.ndarray:
    """Simple ring: 0→1, 1→2, 2→3, 3→0."""
    return np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)


def _timestep(n=NUM_NODES, e=NUM_EDGES, failed_nodes=()):
    ts = {
        "scada_data":       np.random.randn(n, 18).astype(np.float32),
        "pmu_sequence":     np.random.randn(n, 8).astype(np.float32),
        "satellite_data":   np.random.randn(n, 12, 4, 4).astype(np.float32),
        "weather_sequence": np.random.randn(n, 10, 8).astype(np.float32),
        "threat_indicators": np.random.randn(n, 6).astype(np.float32),
        "equipment_status": np.random.randn(n, 10).astype(np.float32),
        "visual_data":      np.random.randn(n, 3, 8, 8).astype(np.float32),
        "thermal_data":     np.random.randn(n, 1, 8, 8).astype(np.float32),
        "sensor_data":      np.random.randn(n, 12).astype(np.float32),
        "edge_attr":        np.random.randn(e, 7).astype(np.float32),
        "node_labels":      np.zeros(n, dtype=np.float32),
    }
    for ni in failed_nodes:
        ts["node_labels"][ni] = 1.0
    return ts


def _scenario(seq_len=8, failed_at_t0=False):
    sequence = []
    for t in range(seq_len):
        failed = (0,) if (failed_at_t0 and t == 0) else ()
        sequence.append(_timestep(failed_nodes=failed))
    return {
        "sequence":   sequence,
        "edge_index": _edge_index_np(),
        "metadata":   {
            "is_cascade": failed_at_t0,
            "failed_nodes": [0] if failed_at_t0 else [],
            "failure_times": [0.0] if failed_at_t0 else [],
            "ground_truth_risk": np.random.rand(7).astype(np.float32),
        },
    }


# ---------------------------------------------------------------------------
# ScenarioInferenceDataset
# ---------------------------------------------------------------------------


class TestScenarioInferenceDataset:

    # ── Length ───────────────────────────────────────────────────────────────

    def test_len_equals_sequence_length(self):
        ds = ScenarioInferenceDataset(_scenario(seq_len=7), window_size=5)
        assert len(ds) == 7

    def test_len_zero_for_empty_sequence(self):
        scenario = {"sequence": [], "edge_index": _edge_index_np()}
        ds = ScenarioInferenceDataset(scenario, window_size=5)
        assert len(ds) == 0

    # ── Window slicing ────────────────────────────────────────────────────────

    def test_window_at_index_0_has_length_1(self):
        """At idx=0 the window is [0:1] — a single timestep."""
        ds = ScenarioInferenceDataset(_scenario(seq_len=8), window_size=5)
        item = ds[0]
        assert item["sequence_length"] == 1

    def test_window_at_index_window_size_minus_1_has_length_window_size(self):
        """At idx=W-1, window = [0:W] — exactly window_size steps."""
        W = 5
        ds = ScenarioInferenceDataset(_scenario(seq_len=8), window_size=W)
        item = ds[W - 1]
        assert item["sequence_length"] == W

    def test_window_capped_at_window_size(self):
        """For idx > W, the window stays at window_size."""
        W = 3
        ds = ScenarioInferenceDataset(_scenario(seq_len=8), window_size=W)
        item = ds[W + 2]
        assert item["sequence_length"] == W

    def test_scada_data_window_shape(self):
        W = 4
        seq = 8
        ds = ScenarioInferenceDataset(_scenario(seq_len=seq), window_size=W)
        # At last index the window is [seq-W : seq]
        item = ds[seq - 1]
        assert item["scada_data"].shape[0] == W

    # ── Required keys ─────────────────────────────────────────────────────────

    def test_getitem_required_keys_present(self):
        ds = ScenarioInferenceDataset(_scenario(seq_len=6), window_size=4)
        item = ds[3]
        required = {
            "scada_data", "pmu_sequence", "equipment_status",
            "edge_index", "edge_attr", "edge_mask",
            "temporal_sequence", "sequence_length",
        }
        missing = required - set(item.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_edge_index_dtype_is_long(self):
        ds = ScenarioInferenceDataset(_scenario(), window_size=4)
        item = ds[0]
        assert item["edge_index"].dtype == torch.long

    def test_edge_index_numpy_input_converted_to_tensor(self):
        """Numpy edge_index in the scenario must be converted to torch.long."""
        scenario = _scenario()
        assert isinstance(scenario["edge_index"], np.ndarray)
        ds = ScenarioInferenceDataset(scenario, window_size=3)
        item = ds[0]
        assert isinstance(item["edge_index"], torch.Tensor)
        assert item["edge_index"].dtype == torch.long

    def test_edge_index_tensor_input_stays_tensor(self):
        """Torch tensor edge_index must remain a tensor."""
        scenario = _scenario()
        scenario["edge_index"] = torch.from_numpy(scenario["edge_index"]).long()
        ds = ScenarioInferenceDataset(scenario, window_size=3)
        item = ds[0]
        assert isinstance(item["edge_index"], torch.Tensor)

    def test_temporal_sequence_equals_scada_data(self):
        """temporal_sequence is just an alias for scada_data."""
        ds = ScenarioInferenceDataset(_scenario(seq_len=6), window_size=3)
        item = ds[2]
        assert torch.equal(item["temporal_sequence"], item["scada_data"])

    def test_graph_properties_is_empty_dict(self):
        ds = ScenarioInferenceDataset(_scenario(seq_len=5), window_size=3)
        item = ds[0]
        assert item["graph_properties"] == {}

    # ── Edge mask teacher-forcing ──────────────────────────────────────────────

    def test_edge_mask_at_t0_all_ones(self):
        """At idx=0 there is no previous step → mask must be all 1."""
        ds = ScenarioInferenceDataset(_scenario(failed_at_t0=True), window_size=5)
        # preprocessed_sequence['edge_mask'][0] == all ones
        mask_t0 = ds.preprocessed_sequence["edge_mask"][0]
        assert torch.all(mask_t0 == 1.0), "Edge mask at t=0 must be all 1"

    def test_edge_mask_tlag_failure_at_t0_propagates_to_t1(self):
        """Node 0 fails at t=0 → edges touching node 0 masked at t=1.

        Ring edge_index: [[0,1,2,3],[1,2,3,0]]
          Edge 0: 0→1  (src=0 failed → masked)
          Edge 3: 3→0  (dst=0 failed → masked)
          Edge 1: 1→2  (no failure → active)
          Edge 2: 2→3  (no failure → active)
        """
        ds = ScenarioInferenceDataset(
            _scenario(seq_len=4, failed_at_t0=True), window_size=5
        )
        mask_t1 = ds.preprocessed_sequence["edge_mask"][1]
        arr = mask_t1.numpy()

        assert arr[0] == 0.0, "Edge 0→1 must be masked (node 0 failed at t=0)"
        assert arr[3] == 0.0, "Edge 3→0 must be masked (node 0 failed at t=0)"
        assert arr[1] == 1.0, "Edge 1→2 must remain active"
        assert arr[2] == 1.0, "Edge 2→3 must remain active"

    def test_edge_mask_shape(self):
        seq_len = 6
        ds = ScenarioInferenceDataset(_scenario(seq_len=seq_len), window_size=4)
        mask_seq = ds.preprocessed_sequence["edge_mask"]
        assert mask_seq.shape == (seq_len, NUM_EDGES)

    def test_edge_mask_in_unit_range(self):
        ds = ScenarioInferenceDataset(_scenario(seq_len=6), window_size=4)
        mask = ds.preprocessed_sequence["edge_mask"]
        assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0)

    # ── Power normalisation ───────────────────────────────────────────────────

    def test_power_normalisation_applied_to_scada(self):
        """Columns 2-5 of SCADA are divided by base_mva during preprocessing."""
        base_mva = 100.0
        # Known values: scada col 2 = 200 MW → should become 2.0 p.u.
        scenario = _scenario(seq_len=2)
        scenario["sequence"][0]["scada_data"][:, 2] = 200.0

        ds = ScenarioInferenceDataset(scenario, window_size=5, base_mva=base_mva)
        # Check first timestep, all nodes, column 2 of stacked SCADA
        s0 = ds.preprocessed_sequence["scada_data"][0]  # [N, <=13]
        assert torch.allclose(s0[:, 2], torch.full((NUM_NODES,), 2.0)), \
            "SCADA power column must be normalised by base_mva"


# ---------------------------------------------------------------------------
# format_risk_assessment
# ---------------------------------------------------------------------------


class TestFormatRiskAssessment:

    def test_returns_string(self):
        scores = [0.5] * 7
        result = format_risk_assessment(scores)
        assert isinstance(result, str)

    def test_insufficient_data_fewer_than_7(self):
        result = format_risk_assessment([0.5, 0.3])
        assert "Insufficient" in result

    def test_exactly_7_scores_no_error(self):
        result = format_risk_assessment([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        assert isinstance(result, str)

    def test_all_labels_present_in_output(self):
        scores = [0.5] * 7
        result = format_risk_assessment(scores)
        for label in ("Threat", "Vulnerability", "Impact",
                      "Cascade Prob", "Response", "Safety", "Urgency"):
            assert label in result, f"Label '{label}' missing from risk assessment"

    def test_critical_level_for_score_above_0_8(self):
        scores = [0.9] * 7
        result = format_risk_assessment(scores)
        assert "(Critical)" in result

    def test_severe_level_for_score_between_0_6_and_0_8(self):
        scores = [0.7] * 7
        result = format_risk_assessment(scores)
        assert "(Severe)" in result

    def test_medium_level_for_score_between_0_3_and_0_6(self):
        scores = [0.4] * 7
        result = format_risk_assessment(scores)
        assert "(Medium)" in result

    def test_low_level_for_score_below_0_3(self):
        scores = [0.1] * 7
        result = format_risk_assessment(scores)
        assert "(Low)" in result

    def test_scores_formatted_to_three_decimal_places(self):
        scores = [0.123456] * 7
        result = format_risk_assessment(scores)
        assert "0.123" in result

    def test_all_thresholds_across_one_output(self):
        """Single call with mixed scores hits all four level labels."""
        scores = [0.9, 0.7, 0.4, 0.1, 0.5, 0.85, 0.2]
        result = format_risk_assessment(scores)
        assert "(Critical)" in result
        assert "(Severe)"   in result
        assert "(Medium)"   in result
        assert "(Low)"      in result


# ---------------------------------------------------------------------------
# print_prediction_report  (via redirect of stdout)
# ---------------------------------------------------------------------------

def _make_results_dict(
    cascade_detected=True,
    is_cascade=True,
    high_risk_nodes=None,
    failed_nodes=None,
    risk_scores=None,
):
    """Build a minimal results dict for print_prediction_report."""
    high_risk_nodes = high_risk_nodes or [0, 1]
    failed_nodes    = failed_nodes    or [0, 2]
    risk_scores     = risk_scores     or [0.5] * 7

    return {
        "inference_time":      0.123,
        "cascade_detected":    cascade_detected,
        "cascade_probability": 0.75 if cascade_detected else 0.05,
        "ground_truth": {
            "is_cascade":       is_cascade,
            "failed_nodes":     failed_nodes,
            "cascade_path":     [{"node_id": n, "time_minutes": float(i * 2)}
                                  for i, n in enumerate(failed_nodes)],
            "ground_truth_risk": risk_scores,
        },
        "high_risk_nodes": high_risk_nodes,
        "risk_assessment": risk_scores,
        "top_nodes": [
            {"node_id": n, "score": 0.8 - i * 0.1, "peak_time": i + 1}
            for i, n in enumerate(high_risk_nodes)
        ],
        "cascade_path": [
            {"order": i + 1, "node_id": n, "ranking_score": 0.8 - i * 0.1}
            for i, n in enumerate(high_risk_nodes)
        ],
        "system_state": {
            "frequency": 59.8,
            "voltages":  [0.95, 0.97, 1.01, 0.99, 0.98, 1.00],
        },
    }


class TestPrintOverallVerdict:
    """Test the TP/TN/FP/FN verdict strings via _print_overall_verdict."""

    def _capture(self, pred, actual) -> str:
        results = _make_results_dict(cascade_detected=pred, is_cascade=actual)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_overall_verdict(results, cascade_threshold=0.5)
        return buf.getvalue()

    def test_true_positive_verdict(self):
        out = self._capture(pred=True, actual=True)
        assert "Correctly detected" in out

    def test_true_negative_verdict(self):
        out = self._capture(pred=False, actual=False)
        assert "Correctly identified" in out

    def test_false_positive_verdict(self):
        out = self._capture(pred=True, actual=False)
        assert "FALSE POSITIVE" in out

    def test_false_negative_verdict(self):
        out = self._capture(pred=False, actual=True)
        assert "FALSE NEGATIVE" in out

    def test_cascade_probability_shown(self):
        out = self._capture(pred=True, actual=True)
        assert "Prob" in out

    def test_threshold_shown(self):
        out = self._capture(pred=True, actual=True)
        assert "Thresh" in out or "0.50" in out


class TestPrintNodeAnalysis:
    """Test the TP/FP/FN counts via _print_node_analysis."""

    def _capture(self, pred_nodes, actual_nodes) -> str:
        results = _make_results_dict(
            high_risk_nodes=list(pred_nodes),
            failed_nodes=list(actual_nodes),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_node_analysis(results, node_threshold=0.35)
        return buf.getvalue()

    def test_correct_tp_count(self):
        out = self._capture(pred_nodes={0, 1}, actual_nodes={0, 2})
        # TP = {0}
        assert "1" in out  # at least one "1" for the TP count

    def test_all_matched_zero_fn(self):
        out = self._capture(pred_nodes={0, 1}, actual_nodes={0, 1})
        # FN = 0
        assert "FN" in out or "Missed" in out

    def test_no_predictions_all_fn(self):
        out = self._capture(pred_nodes=set(), actual_nodes={0, 1, 2})
        assert "3" in out  # 3 missed nodes

    def test_extra_predictions_all_fp(self):
        out = self._capture(pred_nodes={5, 6, 7}, actual_nodes=set())
        # FP = 3, FN = 0
        assert "3" in out


class TestPrintCascadePath:
    """Test cascade path table output via _print_cascade_path."""

    def _capture(self, pred_path, actual_path) -> str:
        results = _make_results_dict()
        results["cascade_path"]                    = pred_path
        results["ground_truth"]["cascade_path"]    = actual_path
        results["ground_truth"]["failed_nodes"]    = [
            p["node_id"] for p in actual_path
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_cascade_path(results)
        return buf.getvalue()

    def test_empty_paths_no_crash(self):
        out = self._capture([], [])
        assert isinstance(out, str)

    def test_predicted_node_appears_in_output(self):
        pred  = [{"order": 1, "node_id": 7, "ranking_score": 0.85}]
        actual = [{"node_id": 7, "time_minutes": 3.5}]
        out = self._capture(pred, actual)
        assert "7" in out

    def test_actual_time_appears_in_output(self):
        pred  = [{"order": 1, "node_id": 0, "ranking_score": 0.9}]
        actual = [{"node_id": 0, "time_minutes": 12.5}]
        out = self._capture(pred, actual)
        assert "12.50" in out


class TestPrintPredictionReport:
    """Integration test for the full report."""

    def _run(self, pred=True, actual=True) -> str:
        results = _make_results_dict(cascade_detected=pred, is_cascade=actual)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_prediction_report(
                results,
                cascade_threshold=0.5,
                node_threshold=0.35,
            )
        return buf.getvalue()

    def test_report_contains_header(self):
        assert "PREDICTION RESULTS" in self._run()

    def test_report_contains_risk_labels(self):
        out = self._run()
        assert "Threat" in out

    def test_report_contains_inference_time(self):
        out = self._run()
        assert "Inference Time" in out

    def test_report_no_crash_for_tn(self):
        out = self._run(pred=False, actual=False)
        assert "Correctly identified" in out

    def test_report_no_crash_for_fp(self):
        out = self._run(pred=True, actual=False)
        assert "FALSE POSITIVE" in out


# ---------------------------------------------------------------------------
# NumpyEncoder (inference.utils)
# ---------------------------------------------------------------------------


class TestInferenceNumpyEncoder:

    def _encode(self, obj):
        return json.loads(json.dumps(obj, cls=NumpyEncoder))

    def test_int32_serialises_as_python_int(self):
        assert self._encode(np.int32(42)) == 42
        assert isinstance(self._encode(np.int32(42)), int)

    def test_int64_serialises_as_python_int(self):
        assert self._encode(np.int64(-7)) == -7

    def test_float32_serialises_as_python_float(self):
        result = self._encode(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-4

    def test_float64_serialises_as_python_float(self):
        assert abs(self._encode(np.float64(2.718)) - 2.718) < 1e-10

    def test_ndarray_serialises_as_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = self._encode(arr)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_2d_ndarray_serialises_nested_list(self):
        arr = np.array([[1, 2], [3, 4]])
        result = self._encode(arr)
        assert result == [[1, 2], [3, 4]]

    def test_bool_serialises_as_python_bool(self):
        assert self._encode(np.bool_(True))  is True
        assert self._encode(np.bool_(False)) is False

    def test_nested_dict_with_numpy_values(self):
        obj = {"count": np.int32(5), "score": np.float32(0.9)}
        result = self._encode(obj)
        assert result["count"] == 5
        assert abs(result["score"] - 0.9) < 1e-4

    def test_non_serialisable_raises_type_error(self):
        with pytest.raises(TypeError):
            json.dumps({"x": object()}, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# CascadePredictor — pure methods (no model/topology files required)
# ---------------------------------------------------------------------------


def _make_predictor() -> CascadePredictor:
    """Instantiate CascadePredictor bypassing __init__ (no file I/O)."""
    p = CascadePredictor.__new__(CascadePredictor)
    p.node_threshold    = 0.35
    p.cascade_threshold = 0.1
    return p


class TestCascadePredictorPureMethods:

    # ── _generate_cascade_path ───────────────────────────────────────────────

    def test_empty_ranked_nodes_returns_empty_path(self):
        p    = _make_predictor()
        path = p._generate_cascade_path([])
        assert path == []

    def test_single_node_path(self):
        p   = _make_predictor()
        inp = [{"node_id": 3, "score": 0.8, "peak_time": 1}]
        out = p._generate_cascade_path(inp)
        assert len(out) == 1
        assert out[0]["node_id"]      == 3
        assert out[0]["order"]        == 1
        assert out[0]["ranking_score"] == pytest.approx(0.8)

    def test_nodes_with_same_score_get_same_rank(self):
        p    = _make_predictor()
        inp  = [
            {"node_id": 0, "score": 0.9,  "peak_time": 1},
            {"node_id": 1, "score": 0.9,  "peak_time": 2},
        ]
        out = p._generate_cascade_path(inp)
        assert out[0]["order"] == out[1]["order"] == 1

    def test_nodes_with_large_score_gap_get_different_ranks(self):
        p   = _make_predictor()
        inp = [
            {"node_id": 0, "score": 0.9,  "peak_time": 1},
            {"node_id": 1, "score": 0.5,  "peak_time": 2},  # gap = 0.4 > 0.002
        ]
        out = p._generate_cascade_path(inp)
        assert out[0]["order"] < out[1]["order"]

    def test_cascade_path_preserves_all_nodes(self):
        p    = _make_predictor()
        inp  = [
            {"node_id": i, "score": 0.9 - i * 0.01, "peak_time": i}
            for i in range(5)
        ]
        out = p._generate_cascade_path(inp)
        assert len(out) == 5
        for i, item in enumerate(out):
            assert item["node_id"] == i

    # ── _extract_ground_truth ────────────────────────────────────────────────

    def test_extract_no_metadata(self):
        p   = _make_predictor()
        gt  = p._extract_ground_truth({"metadata": {}})
        assert gt["is_cascade"]      is None
        assert gt["failed_nodes"]    == []
        assert gt["cascade_path"]    == []
        assert gt["ground_truth_risk"] == []

    def test_extract_cascade_scenario(self):
        p        = _make_predictor()
        scenario = {
            "metadata": {
                "is_cascade":    True,
                "failed_nodes":  [0, 1, 2],
                "failure_times": [0.0, 1.5, 3.0],
                "ground_truth_risk": [0.9] * 7,
            }
        }
        gt = p._extract_ground_truth(scenario)
        assert gt["is_cascade"] is True
        assert len(gt["failed_nodes"]) == 3
        # Cascade path should be sorted by time
        times = [x["time_minutes"] for x in gt["cascade_path"]]
        assert times == sorted(times)

    def test_extract_normal_scenario(self):
        p        = _make_predictor()
        scenario = {
            "metadata": {
                "is_cascade":    False,
                "ground_truth_risk": [0.1] * 7,
            }
        }
        gt = p._extract_ground_truth(scenario)
        assert gt["is_cascade"] is False
        assert gt["failed_nodes"] == []

    def test_cascade_path_sorted_by_time(self):
        p        = _make_predictor()
        scenario = {
            "metadata": {
                "failed_nodes":  [2, 0, 1],
                "failure_times": [3.0, 0.0, 1.5],
            }
        }
        gt    = p._extract_ground_truth(scenario)
        times = [x["time_minutes"] for x in gt["cascade_path"]]
        assert times == sorted(times), "Cascade path must be sorted by failure time"

    # ── _analyze_predictions ─────────────────────────────────────────────────

    def test_analyze_no_risky_nodes(self):
        p    = _make_predictor()
        preds = {
            "max_probs":    {0: 0.1, 1: 0.2},   # all below node_threshold=0.35
            "first_time":   {0: 1,   1: 2},
            "risk_scores":  [0.4] * 7,
            "system_state": {"frequency": 60.0, "voltages": [1.0] * 4},
        }
        scenario = {"metadata": {"is_cascade": False}}
        results  = p._analyze_predictions(preds, scenario)

        assert results["cascade_detected"] is False
        assert results["high_risk_nodes"]  == []

    def test_analyze_risky_nodes_above_threshold(self):
        p    = _make_predictor()
        preds = {
            "max_probs":    {0: 0.8, 1: 0.2, 2: 0.7},
            "first_time":   {0: 1,   1: 2,   2: 3},
            "risk_scores":  [0.6] * 7,
            "system_state": {"frequency": 59.5, "voltages": [0.9, 1.0]},
        }
        scenario = {"metadata": {"is_cascade": True, "failed_nodes": [0],
                                  "failure_times": [0.0]}}
        results  = p._analyze_predictions(preds, scenario)

        assert results["cascade_detected"] is True
        assert 0 in results["high_risk_nodes"]
        assert 2 in results["high_risk_nodes"]
        assert 1 not in results["high_risk_nodes"]

    def test_analyze_top_nodes_sorted_by_score_descending(self):
        p    = _make_predictor()
        preds = {
            "max_probs":  {0: 0.9, 1: 0.6, 2: 0.75},
            "first_time": {0: 1,   1: 3,   2: 2},
            "risk_scores": None,
            "system_state": None,
        }
        scenario = {"metadata": {}}
        results  = p._analyze_predictions(preds, scenario)

        scores = [n["score"] for n in results["top_nodes"]]
        assert scores == sorted(scores, reverse=True)

    def test_analyze_none_risk_scores_replaced_with_zeros(self):
        p    = _make_predictor()
        preds = {
            "max_probs":    {0: 0.8},
            "first_time":   {0: 1},
            "risk_scores":  None,
            "system_state": None,
        }
        results = p._analyze_predictions(preds, {"metadata": {}})
        assert len(results["risk_assessment"]) == 7
        assert all(v == 0.0 for v in results["risk_assessment"])

    def test_analyze_none_system_state_replaced_with_defaults(self):
        p    = _make_predictor()
        preds = {
            "max_probs":    {0: 0.8},
            "first_time":   {0: 1},
            "risk_scores":  None,
            "system_state": None,
        }
        results = p._analyze_predictions(preds, {"metadata": {}})
        assert results["system_state"]["frequency"]  == 0.0
        assert results["system_state"]["voltages"]   == []

    def test_analyze_cascade_probability_is_top_node_score(self):
        p    = _make_predictor()
        preds = {
            "max_probs":    {0: 0.8, 1: 0.5},
            "first_time":   {0: 1,   1: 2},
            "risk_scores":  [0.5] * 7,
            "system_state": {"frequency": 60.0, "voltages": []},
        }
        results = p._analyze_predictions(preds, {"metadata": {}})
        assert results["cascade_probability"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Package-level export smoke tests
# ---------------------------------------------------------------------------


class TestInferencePackageExports:

    def test_cascade_predictor_importable(self):
        from cascade_prediction.inference import CascadePredictor
        assert CascadePredictor is not None

    def test_scenario_inference_dataset_importable(self):
        from cascade_prediction.inference import ScenarioInferenceDataset
        assert ScenarioInferenceDataset is not None

    def test_print_prediction_report_importable(self):
        from cascade_prediction.inference import print_prediction_report
        assert callable(print_prediction_report)

    def test_format_risk_assessment_importable(self):
        from cascade_prediction.inference import format_risk_assessment
        assert callable(format_risk_assessment)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
