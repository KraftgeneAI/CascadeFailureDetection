"""
Tests for cascade_prediction utility modules
=============================================
Covers:
  - NumpyEncoder (utils/json_encoder.py)
  - print_report (inference/reporting.py)
  - apply_calibrated_weights (training/calibration.py)
"""

import io
import json
import sys
import pytest
import numpy as np

from cascade_prediction.utils.json_encoder import NumpyEncoder
from cascade_prediction.inference.reporting import print_report
from cascade_prediction.training.calibration import apply_calibrated_weights


# ---------------------------------------------------------------------------
# NumpyEncoder
# ---------------------------------------------------------------------------

class TestNumpyEncoder:
    """Tests for custom JSON encoder that handles NumPy types."""

    def _encode(self, obj):
        return json.loads(json.dumps(obj, cls=NumpyEncoder))

    # ── integer types ───────────────────────────────────────────────────────

    def test_numpy_int32(self):
        result = self._encode(np.int32(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_int64(self):
        result = self._encode(np.int64(-7))
        assert result == -7
        assert isinstance(result, int)

    def test_numpy_uint8(self):
        result = self._encode(np.uint8(255))
        assert result == 255

    # ── float types ─────────────────────────────────────────────────────────

    def test_numpy_float32(self):
        result = self._encode(np.float32(3.14))
        assert result == pytest.approx(3.14, abs=1e-5)
        assert isinstance(result, float)

    def test_numpy_float64(self):
        result = self._encode(np.float64(2.718281828))
        assert result == pytest.approx(2.718281828, rel=1e-9)
        assert isinstance(result, float)

    # ── boolean type ────────────────────────────────────────────────────────

    def test_numpy_bool_true(self):
        result = self._encode(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_bool_false(self):
        result = self._encode(np.bool_(False))
        assert result is False

    # ── ndarray ─────────────────────────────────────────────────────────────

    def test_numpy_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = self._encode(arr)
        assert result == pytest.approx([1.0, 2.0, 3.0])
        assert isinstance(result, list)

    def test_numpy_2d_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = self._encode(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_empty_array(self):
        arr = np.array([])
        result = self._encode(arr)
        assert result == []

    def test_numpy_array_of_ints(self):
        arr = np.array([10, 20, 30], dtype=np.int64)
        result = self._encode(arr)
        assert result == [10, 20, 30]

    # ── nested structures ────────────────────────────────────────────────────

    def test_numpy_in_dict(self):
        data = {"score": np.float32(0.95), "count": np.int64(42)}
        result = self._encode(data)
        assert result["score"] == pytest.approx(0.95, abs=1e-5)
        assert result["count"] == 42

    def test_numpy_in_list(self):
        data = [np.int32(1), np.float64(2.5), np.bool_(True)]
        result = self._encode(data)
        assert result == pytest.approx([1, 2.5, True])

    def test_numpy_array_in_dict(self):
        data = {"weights": np.array([0.1, 0.9])}
        result = self._encode(data)
        assert result["weights"] == pytest.approx([0.1, 0.9])

    # ── native Python types pass through unchanged ───────────────────────────

    def test_native_int_unchanged(self):
        result = self._encode({"x": 5})
        assert result == {"x": 5}

    def test_native_float_unchanged(self):
        result = self._encode(3.14)
        assert result == pytest.approx(3.14)

    def test_native_string_unchanged(self):
        result = self._encode("hello")
        assert result == "hello"

    def test_native_none_unchanged(self):
        result = self._encode(None)
        assert result is None

    # ── non-serialisable type still raises ──────────────────────────────────

    def test_non_serialisable_raises(self):
        class Unserializable:
            pass
        with pytest.raises(TypeError):
            json.dumps(Unserializable(), cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

def _make_result(
    cascade_detected=True,
    cascade_probability=0.87,
    is_cascade=True,
    high_risk_nodes=None,
    failed_nodes=None,
    risk_assessment=None,
    pred_timing=None,
    act_timing=None,
):
    """Build a minimal result dict that print_report expects."""
    high_risk_nodes = high_risk_nodes if high_risk_nodes is not None else [0, 1]
    failed_nodes    = failed_nodes    if failed_nodes    is not None else [0, 2]
    risk_assessment = risk_assessment if risk_assessment is not None else [0.9, 0.7, 0.8, 0.6, 0.4, 0.3, 0.5]
    pred_timing     = pred_timing     if pred_timing     is not None else [
        {"order": 1, "node_id": 0, "ranking_score": 0.87, "pred_time_minutes": 10.0},
        {"order": 2, "node_id": 1, "ranking_score": 0.72, "pred_time_minutes": 20.0},
    ]
    act_timing = act_timing if act_timing is not None else [
        {"node_id": 0, "time_minutes": 12.0},
        {"node_id": 2, "time_minutes": 22.0},
    ]
    return {
        "inference_time": 0.042,
        "cascade_detected": cascade_detected,
        "cascade_probability": cascade_probability,
        "ground_truth": {
            "is_cascade": is_cascade,
            "failed_nodes": failed_nodes,
            "cascade_path": act_timing,
            "ground_truth_risk": None,
        },
        "high_risk_nodes": high_risk_nodes,
        "risk_assessment": risk_assessment,
        "top_nodes": pred_timing,
        "cascade_path": pred_timing,
        "system_state": {
            "frequency": 59.5,
            "voltages": [0.95, 1.02, 0.88, 1.0],
        },
    }


class TestPrintReport:
    """Tests for print_report console output function."""

    def _capture(self, res, cascade_thresh=0.5, node_thresh=0.3):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_report(res, cascade_thresh, node_thresh)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_no_exception_on_valid_input(self):
        """print_report should not raise on a complete, valid result dict."""
        res = _make_result()
        self._capture(res)  # just ensure no exception

    def test_true_positive_message(self):
        """TP: cascade predicted and actual."""
        out = self._capture(_make_result(cascade_detected=True, is_cascade=True))
        assert "Correctly detected a cascade" in out

    def test_true_negative_message(self):
        """TN: no cascade predicted, none actual."""
        res = _make_result(
            cascade_detected=False, is_cascade=False,
            high_risk_nodes=[], failed_nodes=[], pred_timing=[], act_timing=[]
        )
        out = self._capture(res)
        assert "Correctly identified a normal scenario" in out

    def test_false_positive_message(self):
        """FP: cascade predicted but none actual."""
        res = _make_result(cascade_detected=True, is_cascade=False)
        out = self._capture(res)
        assert "FALSE POSITIVE" in out

    def test_false_negative_message(self):
        """FN: cascade actual but not predicted."""
        res = _make_result(
            cascade_detected=False, is_cascade=True,
            high_risk_nodes=[], pred_timing=[]
        )
        out = self._capture(res)
        assert "FALSE NEGATIVE" in out or "Missed" in out

    def test_threshold_displayed(self):
        out = self._capture(_make_result(), cascade_thresh=0.55, node_thresh=0.35)
        assert "0.55" in out
        assert "0.35" in out

    def test_system_frequency_in_output(self):
        out = self._capture(_make_result())
        assert "59.5" in out or "Frequency" in out

    def test_voltage_range_in_output(self):
        out = self._capture(_make_result())
        assert "0.88" in out or "Voltage" in out

    def test_risk_assessment_section_present(self):
        out = self._capture(_make_result())
        assert "Risk" in out

    def test_cascade_path_section_present(self):
        out = self._capture(_make_result())
        assert "Cascade Path" in out

    def test_empty_risk_assessment(self):
        """print_report must not crash when risk_assessment has < 7 entries."""
        res = _make_result(risk_assessment=[0.5, 0.3])
        self._capture(res)  # should not raise

    def test_empty_voltages_list(self):
        """No voltages → voltage range section should be skipped gracefully."""
        res = _make_result()
        res["system_state"]["voltages"] = []
        self._capture(res)  # should not raise

    def test_output_contains_separator_line(self):
        """Report should have a clear separator line."""
        out = self._capture(_make_result())
        assert "=" * 40 in out

    def test_inference_time_displayed(self):
        out = self._capture(_make_result())
        assert "0.042" in out or "Inference Time" in out

    def test_tp_fp_fn_counts(self):
        """Node-level TP/FP/FN counts should appear in output."""
        res = _make_result(
            high_risk_nodes=[0, 1],
            failed_nodes=[0, 2],
        )
        out = self._capture(res)
        # node 0 is TP, node 1 is FP, node 2 is FN
        assert "TP" in out or "Correctly Identified" in out
        assert "FP" in out or "False Alarms" in out
        assert "FN" in out or "Missed" in out


# ---------------------------------------------------------------------------
# apply_calibrated_weights
# ---------------------------------------------------------------------------

class TestApplyCalibratedWeights:
    """Tests for the lambda-merging utility in calibration.py."""

    def test_calibrated_overrides_base(self):
        base       = {"lambda_prediction": 1.0, "lambda_risk": 0.5}
        calibrated = {"lambda_prediction": 10.0}
        result = apply_calibrated_weights(base, calibrated)
        assert result["lambda_prediction"] == pytest.approx(10.0)

    def test_base_kept_when_not_in_calibrated(self):
        base       = {"lambda_prediction": 1.0, "lambda_risk": 0.5}
        calibrated = {"lambda_prediction": 10.0}
        result = apply_calibrated_weights(base, calibrated)
        assert result["lambda_risk"] == pytest.approx(0.5)

    def test_scaling_factor_applied(self):
        base       = {"lambda_prediction": 2.0}
        calibrated = {"lambda_prediction": 10.0}
        result = apply_calibrated_weights(base, calibrated, scaling_factor=0.5)
        assert result["lambda_prediction"] == pytest.approx(5.0)

    def test_scaling_factor_on_base_key(self):
        base       = {"lambda_risk": 1.0}
        calibrated = {}   # risk not calibrated
        result = apply_calibrated_weights(base, calibrated, scaling_factor=3.0)
        assert result["lambda_risk"] == pytest.approx(3.0)

    def test_empty_calibrated_returns_scaled_base(self):
        base = {"lambda_prediction": 1.0, "lambda_risk": 2.0}
        result = apply_calibrated_weights(base, {}, scaling_factor=2.0)
        assert result["lambda_prediction"] == pytest.approx(2.0)
        assert result["lambda_risk"]       == pytest.approx(4.0)

    def test_empty_base_returns_empty(self):
        result = apply_calibrated_weights({}, {"lambda_prediction": 5.0})
        assert result == {}

    def test_all_keys_present_in_result(self):
        base = {
            "lambda_prediction": 1.0, "lambda_risk": 0.1,
            "lambda_timing": 0.2, "lambda_voltage": 0.3
        }
        calibrated = {"lambda_prediction": 50.0, "lambda_timing": 0.5}
        result = apply_calibrated_weights(base, calibrated)
        assert set(result.keys()) == set(base.keys())

    def test_default_scaling_factor_is_one(self):
        base       = {"lambda_prediction": 2.0}
        calibrated = {"lambda_prediction": 5.0}
        result_explicit = apply_calibrated_weights(base, calibrated, scaling_factor=1.0)
        result_default  = apply_calibrated_weights(base, calibrated)
        assert result_explicit == result_default

    def test_zero_scaling_makes_all_zero(self):
        base       = {"lambda_prediction": 5.0, "lambda_risk": 3.0}
        calibrated = {"lambda_prediction": 10.0}
        result = apply_calibrated_weights(base, calibrated, scaling_factor=0.0)
        for v in result.values():
            assert v == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
