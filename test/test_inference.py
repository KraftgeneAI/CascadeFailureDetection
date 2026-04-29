"""
Tests for cascade_prediction/inference/predictor.py and reporting.py
=====================================================================
Covers the two newest files on the branch (Apr 29 19:07):

  predictor.py
    _build_cascade_path()      – groups ranked nodes into ordered cascade steps
    _build_causal_sequence()   – builds parent-annotated causal chain
    CascadePredictor package exports

  reporting.py
    print_report()             – structured console report

All tests avoid loading model checkpoints or real data files by either:
  • calling pure static/instance methods directly on a bare instance
    built via CascadePredictor.__new__ (bypasses __init__'s file I/O), or
  • calling module-level functions with synthetic dicts.
"""

import io
import sys
import pytest
import numpy as np

# ── Imports ────────────────────────────────────────────────────────────────────

from cascade_prediction.inference.predictor import CascadePredictor
from cascade_prediction.inference.reporting import print_report
import cascade_prediction.inference as inference_pkg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_predictor(node_threshold=0.35, cascade_threshold=0.1) -> CascadePredictor:
    """Create a CascadePredictor without loading any checkpoint."""
    p = CascadePredictor.__new__(CascadePredictor)
    p.node_threshold = node_threshold
    p.cascade_threshold = cascade_threshold
    return p


def _capture_print(fn, *args, **kwargs) -> str:
    """Call fn(*args, **kwargs) and return everything it printed."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def _make_result(
    cascade_detected: bool = True,
    is_cascade: bool = True,
    failed_nodes: list = None,
    high_risk_nodes: list = None,
    cascade_path: list = None,
    cascade_sequence: list = None,
    risk_assessment: list = None,
) -> dict:
    """Build a minimal predict_scenario()-style result dict."""
    failed_nodes = failed_nodes or []
    high_risk_nodes = high_risk_nodes or []
    cascade_path = cascade_path or []
    cascade_sequence = cascade_sequence or []
    risk_assessment = risk_assessment or [0.5] * 7

    return {
        "inference_time": 0.042,
        "cascade_detected": cascade_detected,
        "cascade_probability": 0.75 if cascade_detected else 0.1,
        "high_risk_nodes": high_risk_nodes,
        "risk_assessment": risk_assessment,
        "top_nodes": [
            {"node_id": n, "score": 0.9 - i * 0.1, "pred_time_minutes": float(i * 2)}
            for i, n in enumerate(high_risk_nodes)
        ],
        "cascade_path": cascade_path,
        "cascade_sequence": cascade_sequence,
        "ground_truth": {
            "is_cascade": is_cascade,
            "failed_nodes": failed_nodes,
            "cascade_path": [
                {"node_id": n, "time_minutes": float(i * 2)}
                for i, n in enumerate(failed_nodes)
            ],
            "ground_truth_risk": [],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _build_cascade_path
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildCascadePath:
    """Tests for CascadePredictor._build_cascade_path()."""

    @pytest.fixture
    def predictor(self):
        return _make_predictor()

    def test_empty_input_returns_empty(self, predictor):
        assert predictor._build_cascade_path([]) == []

    def test_single_node_gets_order_one(self, predictor):
        ranked = [{"node_id": 5, "score": 0.9, "pred_time_minutes": 3.0}]
        path = predictor._build_cascade_path(ranked)
        assert len(path) == 1
        assert path[0]["order"] == 1
        assert path[0]["node_id"] == 5

    def test_two_nodes_within_half_minute_same_order(self, predictor):
        ranked = [
            {"node_id": 0, "score": 0.9, "pred_time_minutes": 2.0},
            {"node_id": 1, "score": 0.8, "pred_time_minutes": 2.4},   # Δ = 0.4 < 0.5
        ]
        path = predictor._build_cascade_path(ranked)
        assert path[0]["order"] == path[1]["order"] == 1

    def test_two_nodes_over_half_minute_different_orders(self, predictor):
        ranked = [
            {"node_id": 0, "score": 0.9, "pred_time_minutes": 2.0},
            {"node_id": 1, "score": 0.8, "pred_time_minutes": 2.6},   # Δ = 0.6 > 0.5
        ]
        path = predictor._build_cascade_path(ranked)
        assert path[0]["order"] == 1
        assert path[1]["order"] == 2

    def test_three_nodes_two_groups(self, predictor):
        ranked = [
            {"node_id": 0, "score": 0.9, "pred_time_minutes": 1.0},
            {"node_id": 1, "score": 0.85, "pred_time_minutes": 1.3},  # group 1
            {"node_id": 2, "score": 0.7, "pred_time_minutes": 5.0},   # group 2
        ]
        path = predictor._build_cascade_path(ranked)
        assert path[0]["order"] == 1
        assert path[1]["order"] == 1
        assert path[2]["order"] == 2

    def test_output_preserves_node_id_score_time(self, predictor):
        ranked = [{"node_id": 7, "score": 0.77, "pred_time_minutes": 4.5}]
        path = predictor._build_cascade_path(ranked)
        assert path[0]["node_id"] == 7
        assert path[0]["ranking_score"] == pytest.approx(0.77)
        assert path[0]["pred_time_minutes"] == pytest.approx(4.5)

    def test_all_nodes_exact_same_time_same_order(self, predictor):
        ranked = [
            {"node_id": i, "score": 0.9, "pred_time_minutes": 3.0}
            for i in range(5)
        ]
        path = predictor._build_cascade_path(ranked)
        orders = {p["order"] for p in path}
        assert orders == {1}

    def test_monotone_distinct_times_sequential_orders(self, predictor):
        ranked = [
            {"node_id": i, "score": 0.9, "pred_time_minutes": float(i * 2)}
            for i in range(4)
        ]
        path = predictor._build_cascade_path(ranked)
        assert [p["order"] for p in path] == [1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _build_causal_sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildCausalSequence:
    """Tests for CascadePredictor._build_causal_sequence()."""

    @pytest.fixture
    def predictor(self):
        return _make_predictor()

    def _call(self, predictor, risky_nodes, predicted_parents,
              timing, num_nodes=10, actual_labels=None):
        timing_arr = np.array(timing, dtype=np.float64)
        return predictor._build_causal_sequence(
            risky_nodes, predicted_parents, timing_arr, num_nodes, actual_labels
        )

    def test_empty_risky_nodes_returns_empty(self, predictor):
        result = self._call(predictor, [], [0] * 10, [0.0] * 10)
        assert result == []

    def test_single_trigger_node(self, predictor):
        """Parent index == num_nodes → trigger → pred_parent_id is None."""
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes  # all trigger
        timing = [1.0] * num_nodes
        result = self._call(predictor, [0], predicted_parents, timing, num_nodes)
        assert len(result) == 1
        assert result[0]["pred_parent_id"] is None

    def test_parent_in_risky_set_is_returned(self, predictor):
        num_nodes = 5
        # Node 2 predicts node 1 as parent; node 1 is risky
        predicted_parents = [num_nodes, num_nodes, 1, num_nodes, num_nodes]
        timing = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = self._call(predictor, [1, 2], predicted_parents, timing, num_nodes)
        node2_entry = next(e for e in result if e["node_id"] == 2)
        assert node2_entry["pred_parent_id"] == 1

    def test_parent_not_in_risky_set_is_none(self, predictor):
        num_nodes = 5
        # Node 2 predicts node 0 as parent, but node 0 is NOT in risky_nodes
        predicted_parents = [num_nodes, num_nodes, 0, num_nodes, num_nodes]
        timing = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = self._call(predictor, [1, 2], predicted_parents, timing, num_nodes)
        node2_entry = next(e for e in result if e["node_id"] == 2)
        assert node2_entry["pred_parent_id"] is None

    def test_actual_parent_minus_one_is_none(self, predictor):
        """actual_label == -1 → node did not actually fail → actual_parent_id is None."""
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        actual_labels = [-1, -1, -1, -1, -1]
        result = self._call(predictor, [0], predicted_parents, timing, num_nodes, actual_labels)
        assert result[0]["actual_parent_id"] is None

    def test_actual_parent_trigger_index_is_none(self, predictor):
        """actual_label == num_nodes (trigger class) → actual_parent_id is None."""
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        actual_labels = [num_nodes, -1, -1, -1, -1]  # node 0 is trigger
        result = self._call(predictor, [0], predicted_parents, timing, num_nodes, actual_labels)
        assert result[0]["actual_parent_id"] is None

    def test_actual_parent_valid_node_is_returned(self, predictor):
        """actual_label == 3 → actual_parent_id == 3."""
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        actual_labels = [3, -1, -1, -1, -1]   # node 0 was caused by node 3
        result = self._call(predictor, [0], predicted_parents, timing, num_nodes, actual_labels)
        assert result[0]["actual_parent_id"] == 3

    def test_no_actual_labels_gives_none(self, predictor):
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        result = self._call(predictor, [0, 1], predicted_parents, timing, num_nodes, None)
        for entry in result:
            assert entry["actual_parent_id"] is None

    def test_sorted_by_pred_time_ascending(self, predictor):
        num_nodes = 5
        predicted_parents = [num_nodes] * num_nodes
        timing = [5.0, 1.0, 3.0, 2.0, 4.0]
        result = self._call(predictor, [0, 1, 2, 3, 4], predicted_parents, timing, num_nodes)
        times = [e["pred_time_minutes"] for e in result]
        assert times == sorted(times)

    def test_order_assigned_ascending_from_one(self, predictor):
        num_nodes = 4
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0, 2.0, 3.0, 4.0]
        result = self._call(predictor, [0, 1, 2, 3], predicted_parents, timing, num_nodes)
        orders = [e["order"] for e in result]
        assert orders == [1, 2, 3, 4]

    def test_all_risky_nodes_present_in_result(self, predictor):
        risky = [0, 2, 4]
        num_nodes = 6
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        result = self._call(predictor, risky, predicted_parents, timing, num_nodes)
        assert {e["node_id"] for e in result} == set(risky)

    def test_result_contains_required_keys(self, predictor):
        num_nodes = 3
        predicted_parents = [num_nodes] * num_nodes
        timing = [1.0] * num_nodes
        result = self._call(predictor, [0], predicted_parents, timing, num_nodes)
        assert all(
            k in result[0]
            for k in ("node_id", "pred_parent_id", "actual_parent_id",
                      "pred_time_minutes", "order")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. print_report
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrintReport:
    """Tests for cascade_prediction.inference.reporting.print_report()."""

    CASCADE_THRESH = 0.25
    NODE_THRESH = 0.25

    def _run(self, res):
        return _capture_print(print_report, res, self.CASCADE_THRESH, self.NODE_THRESH)

    def test_true_positive_message(self):
        res = _make_result(cascade_detected=True, is_cascade=True,
                           failed_nodes=[0, 1], high_risk_nodes=[0, 1])
        out = self._run(res)
        assert "Correctly detected" in out

    def test_true_negative_message(self):
        res = _make_result(cascade_detected=False, is_cascade=False)
        out = self._run(res)
        assert "Correctly identified" in out

    def test_false_positive_message(self):
        res = _make_result(cascade_detected=True, is_cascade=False,
                           high_risk_nodes=[3])
        out = self._run(res)
        assert "FALSE POSITIVE" in out

    def test_false_negative_message(self):
        res = _make_result(cascade_detected=False, is_cascade=True,
                           failed_nodes=[2])
        out = self._run(res)
        assert "FALSE NEGATIVE" in out

    def test_output_contains_inference_time(self):
        res = _make_result()
        out = self._run(res)
        assert "Inference Time" in out

    def test_output_contains_risk_assessment_dimensions(self):
        res = _make_result(cascade_detected=True, is_cascade=True,
                           high_risk_nodes=[0])
        out = self._run(res)
        assert "Risk Assessment" in out

    def test_output_contains_cascade_path_header(self):
        res = _make_result()
        out = self._run(res)
        assert "Cascade Path" in out

    def test_output_contains_causal_parent_header(self):
        res = _make_result()
        out = self._run(res)
        assert "Causal Parent" in out

    def test_tp_fp_fn_counts_in_output(self):
        res = _make_result(
            cascade_detected=True, is_cascade=True,
            failed_nodes=[0, 1, 2],
            high_risk_nodes=[0, 1, 3],   # TP=2, FP=1, FN=1
        )
        out = self._run(res)
        assert "(TP)" in out
        assert "(FN)" in out
        assert "(FP)" in out

    def test_threshold_values_shown(self):
        res = _make_result(cascade_detected=True, is_cascade=True)
        out = self._run(res)
        assert "0.250" in out or "0.25" in out

    def test_risk_level_labels_present(self):
        res = _make_result(cascade_detected=True, is_cascade=True,
                           risk_assessment=[0.9, 0.7, 0.4, 0.2, 0.0, 0.85, 0.6])
        out = self._run(res)
        # At least one severity label should appear
        assert any(l in out for l in ("Critical", "Severe", "Medium", "Low"))

    def test_empty_risk_assessment_does_not_crash(self):
        res = _make_result(cascade_detected=False, is_cascade=False,
                           risk_assessment=[])
        out = self._run(res)
        assert len(out) > 0

    def test_with_causal_sequence_shows_match_column(self):
        seq = [
            {"order": 1, "node_id": 0, "pred_parent_id": None,
             "actual_parent_id": None, "pred_time_minutes": 1.0},
            {"order": 2, "node_id": 1, "pred_parent_id": 0,
             "actual_parent_id": 0, "pred_time_minutes": 3.0},
        ]
        res = _make_result(
            cascade_detected=True, is_cascade=True,
            high_risk_nodes=[0, 1],
            cascade_sequence=seq,
        )
        out = self._run(res)
        assert "Match" in out or "✓" in out or "✗" in out

    def test_no_failing_nodes_predicted_message(self):
        res = _make_result(cascade_detected=False, is_cascade=False)
        res["cascade_sequence"] = []
        out = self._run(res)
        assert "No failing nodes predicted" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Package exports
# ═══════════════════════════════════════════════════════════════════════════════

class TestPackageExports:
    """Verify cascade_prediction.inference __all__ contract."""

    def test_cascade_predictor_exported(self):
        assert hasattr(inference_pkg, "CascadePredictor")

    def test_print_report_exported(self):
        assert hasattr(inference_pkg, "print_report")

    def test_cascade_predictor_is_class(self):
        import inspect
        assert inspect.isclass(inference_pkg.CascadePredictor)

    def test_print_report_is_callable(self):
        assert callable(inference_pkg.print_report)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
