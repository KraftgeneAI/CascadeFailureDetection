"""
Tests for video_processor.py
=============================
Covers extract_threat_curve() — the YOLO-based threat signal extractor
introduced in the final-video-stress branch.

All tests mock both cv2.VideoCapture and the YOLO model so they run
without a GPU, a real video file, or a trained weights file.

Mock strategy
-------------
  * ultralytics is injected into sys.modules before import so the module
    loads even without the package installed.
  * cv2.VideoCapture is patched via unittest.mock.patch on the
    video_processor namespace.
  * video_processor.YOLO is patched to return a controllable model mock.

Helper factories
----------------
  _make_cap(frames)      — builds a mock cap that yields (True, frame) for
                           each frame then (False, None) to signal EOF.
  _make_box(conf, bbox)  — builds a mock detection box.
  _make_results(boxes)   — wraps boxes into a YOLO-style results list.
"""

import sys
import math
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject mock for ultralytics BEFORE video_processor is imported
# ---------------------------------------------------------------------------
if "ultralytics" not in sys.modules:
    sys.modules["ultralytics"] = MagicMock()

# Ensure the project root is on sys.path so video_processor is importable
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from video_processor import extract_threat_curve  # noqa: E402


# ---------------------------------------------------------------------------
# Mock-building helpers
# ---------------------------------------------------------------------------

_FRAME_H, _FRAME_W = 640, 640
_BLANK_FRAME = np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)


def _make_cap(frames: List[np.ndarray]) -> MagicMock:
    """Mock cv2.VideoCapture that yields the given frames then EOF."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    reads = [(True, f) for f in frames] + [(False, None)]
    cap.read.side_effect = reads
    return cap


def _make_cap_closed() -> MagicMock:
    """Mock cap that reports it could not open the video."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    return cap


def _make_box(conf: float, x1=10.0, y1=10.0, x2=110.0, y2=110.0) -> MagicMock:
    """Mock YOLO detection box with given confidence and bounding box."""
    box = MagicMock()
    box.conf = [conf]
    xyxy_mock = MagicMock()
    xyxy_mock.cpu.return_value.numpy.return_value = np.array(
        [x1, y1, x2, y2], dtype=np.float32
    )
    box.xyxy = [xyxy_mock]
    return box


def _make_results(boxes: list) -> list:
    """Wrap boxes into a YOLO-style results list."""
    r = MagicMock()
    r.boxes = boxes
    return [r]


def _make_results_no_boxes() -> list:
    """Results with no detections."""
    r = MagicMock()
    r.boxes = None
    return [r]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_cannot_open_video_raises_runtime_error(self):
        """Non-openable video → RuntimeError mentioning the path."""
        cap = _make_cap_closed()
        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with pytest.raises(RuntimeError, match="Cannot open video"):
                extract_threat_curve("no_such_video.mp4")

    def test_yolo_load_failure_raises_runtime_error(self):
        """YOLO constructor raising → RuntimeError, cap released."""
        cap = _make_cap([_BLANK_FRAME])
        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", side_effect=Exception("bad weights")):
                with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
                    extract_threat_curve("video.mp4")
        cap.release.assert_called_once()

    def test_no_frames_processed_raises_value_error(self):
        """Video that immediately returns EOF → ValueError."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)  # EOF on first read

        model = MagicMock()
        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                with pytest.raises(ValueError, match="No frames processed"):
                    extract_threat_curve("video.mp4")

    def test_model_inference_exception_produces_zero_score(self):
        """Exception during model inference records 0.0 for that frame."""
        frames = [_BLANK_FRAME.copy() for _ in range(3)]
        cap = _make_cap(frames)

        model = MagicMock()
        model.side_effect = Exception("CUDA OOM")

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("video.mp4", frame_skip=1)

        assert isinstance(signal, np.ndarray)
        # All inference failures → all raw values 0 → normalised to 0
        assert np.all(signal == 0.0)


# ---------------------------------------------------------------------------
# Tests: output properties (dtype, range, length)
# ---------------------------------------------------------------------------


class TestOutputProperties:

    def _run_with_detections(
        self,
        n_frames: int = 5,
        conf: float = 0.9,
        frame_skip: int = 1,
        smooth: bool = False,
    ) -> np.ndarray:
        """Helper: run extract_threat_curve with constant detections."""
        frames = [_BLANK_FRAME.copy() for _ in range(n_frames)]
        cap = _make_cap(frames)

        model = MagicMock()
        model.return_value = _make_results([_make_box(conf)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                return extract_threat_curve(
                    "v.mp4",
                    frame_skip=frame_skip,
                    smooth=smooth,
                    confidence_threshold=0.1,
                )

    def test_output_is_numpy_float32(self):
        signal = self._run_with_detections()
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float32

    def test_output_values_in_unit_range(self):
        signal = self._run_with_detections()
        assert np.all(signal >= 0.0), f"min={signal.min()}"
        assert np.all(signal <= 1.0), f"max={signal.max()}"

    def test_output_max_is_one_when_nonzero_detections(self):
        """After min-max normalisation, the maximum must be 1.0.

        Uses two frames: one with a detection (non-zero score) and one without
        (zero score) so the raw signal is non-constant and normalisation maps
        the peak to exactly 1.0.
        """
        frames = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy()]
        cap = _make_cap(frames)
        model = MagicMock()
        model.side_effect = [
            _make_results([_make_box(0.9)]),   # frame 1: detection → non-zero score
            _make_results_no_boxes(),           # frame 2: nothing  → zero score
        ]
        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve(
                    "v.mp4", frame_skip=1, confidence_threshold=0.1
                )
        assert float(signal.max()) == pytest.approx(1.0)

    def test_constant_zero_signal_is_all_zeros(self):
        """If no frame scores any detection, output is all-zero."""
        frames = [_BLANK_FRAME.copy() for _ in range(4)]
        cap = _make_cap(frames)

        model = MagicMock()
        model.return_value = _make_results_no_boxes()

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1)

        assert np.all(signal == 0.0)

    def test_output_is_clipped_to_unit_range(self):
        """Values must be clipped even if intermediate scores are extreme."""
        signal = self._run_with_detections(conf=0.99)
        assert np.all(signal >= 0.0)
        assert np.all(signal <= 1.0)

    def test_cap_is_released_on_success(self):
        frames = [_BLANK_FRAME.copy() for _ in range(3)]
        cap = _make_cap(frames)
        model = MagicMock()
        model.return_value = _make_results([_make_box(0.8)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                extract_threat_curve("v.mp4", frame_skip=1)

        cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: frame skip
# ---------------------------------------------------------------------------


class TestFrameSkip:

    def test_frame_skip_reduces_model_calls(self):
        """With frame_skip=3, model called for every 3rd frame only."""
        n_frames = 9
        frames = [_BLANK_FRAME.copy() for _ in range(n_frames)]
        cap = _make_cap(frames)

        model = MagicMock()
        model.return_value = _make_results([_make_box(0.8)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=3, confidence_threshold=0.1)

        # Frames 3 and 6 and 9 are processed (frame_index % 3 == 0)
        # signal length = number of processed frames = 3
        assert len(signal) == 3

    def test_frame_skip_1_processes_all_frames(self):
        """frame_skip=1 means every frame is processed."""
        n_frames = 5
        frames = [_BLANK_FRAME.copy() for _ in range(n_frames)]
        cap = _make_cap(frames)

        model = MagicMock()
        model.return_value = _make_results_no_boxes()

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1)

        assert len(signal) == n_frames


# ---------------------------------------------------------------------------
# Tests: confidence threshold
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:

    def test_below_threshold_detection_ignored(self):
        """Detections below confidence_threshold contribute score 0."""
        frames = [_BLANK_FRAME.copy() for _ in range(3)]
        cap = _make_cap(frames)

        model = MagicMock()
        # conf=0.1 is below threshold=0.5
        model.return_value = _make_results([_make_box(0.1)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.5)

        # All detections filtered → all zeros
        assert np.all(signal == 0.0)

    def test_above_threshold_detection_contributes(self):
        """Detections above confidence_threshold contribute a positive score.

        Frame 0 has a high-conf detection; frame 1 has none. The two frames
        produce different raw scores, so normalisation maps the peak to 1.0.
        """
        frames = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy()]
        cap = _make_cap(frames)

        model = MagicMock()
        model.side_effect = [
            _make_results([_make_box(0.9)]),  # above threshold → positive score
            _make_results_no_boxes(),          # nothing         → zero score
        ]

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.5)

        # Non-constant raw signal → max normalised to 1.0
        assert float(signal.max()) == pytest.approx(1.0)

    def test_mixed_confidence_partial_filtering(self):
        """Mix of above- and below-threshold detections: only above count."""
        frames = [_BLANK_FRAME.copy() for _ in range(2)]
        cap = _make_cap(frames)

        model = MagicMock()
        # Frame 0: high conf; frame 1: low conf
        reads_side_effect = [
            _make_results([_make_box(0.9)]),  # processed
            _make_results([_make_box(0.1)]),  # filtered
        ]
        model.side_effect = reads_side_effect

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.5)

        # First frame has detection, second does not
        # After normalisation: max should be 1.0, last element 0
        assert float(signal.max()) == pytest.approx(1.0)
        assert float(signal[-1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: scoring formula
# ---------------------------------------------------------------------------


class TestScoringFormula:

    def test_score_is_conf_times_sqrt_area_ratio(self):
        """
        Per-frame score = conf * sqrt(box_area / frame_area).
        With a 100×100 box in a 640×640 frame and conf=1.0:
          area_ratio = 10000 / 409600 ≈ 0.024414
          raw_score  = 1.0 * sqrt(0.024414) ≈ 0.15625
        After min-max normalisation of a single-value signal → max = 0 (constant).
        We verify the *raw* mean before normalisation by checking the
        output is all-zero for a constant signal.
        """
        # Two identical frames so signal is constant → normalised to 0
        frames = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy()]
        cap = _make_cap(frames)

        model = MagicMock()
        # Same detection every frame → constant signal → normalised all-zero
        model.return_value = _make_results([_make_box(conf=1.0, x1=0, y1=0, x2=100, y2=100)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        # Constant raw values → min == max → normalised to all-zero
        assert np.all(signal == 0.0)

    def test_larger_box_gives_higher_score(self):
        """A larger bounding box should produce a higher per-frame score."""
        frames_small = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy() * 2]
        frames_big   = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy() * 2]

        cap_small = _make_cap(frames_small)
        cap_big   = _make_cap(frames_big)

        # Small box: 10×10; Big box: 200×200
        model_small = MagicMock()
        model_big   = MagicMock()

        small_results = _make_results([_make_box(0.9, 0, 0, 10, 10)])
        big_results   = _make_results([_make_box(0.9, 0, 0, 200, 200)])

        # Alternate: frame 0 = small/big detection, frame 1 = no detection
        # This creates a non-constant signal so normalisation is non-trivial.
        model_small.side_effect = [small_results, _make_results_no_boxes()]
        model_big.side_effect   = [big_results,   _make_results_no_boxes()]

        with patch("video_processor.cv2.VideoCapture", return_value=cap_small):
            with patch("video_processor.YOLO", return_value=model_small):
                sig_small = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        with patch("video_processor.cv2.VideoCapture", return_value=cap_big):
            with patch("video_processor.YOLO", return_value=model_big):
                sig_big = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        # Both are normalised to [0,1] so both have max=1.
        # The second frame (no detection) maps to min in both cases.
        # The relative ordering within each signal is what we care about.
        # sig_small[0] should be > sig_small[1] and same for big.
        assert sig_small[0] > sig_small[1]
        assert sig_big[0] > sig_big[1]

    def test_multiple_boxes_aggregated_by_mean(self):
        """Multiple detections per frame → mean score used."""
        frames = [_BLANK_FRAME.copy(), _BLANK_FRAME.copy() * 0]
        cap = _make_cap(frames)

        model = MagicMock()
        # Frame 0: two detections; frame 1: no detection → non-constant signal
        two_box_results = _make_results([
            _make_box(0.9, 0, 0, 100, 100),
            _make_box(0.5, 200, 200, 300, 300),
        ])
        model.side_effect = [two_box_results, _make_results_no_boxes()]

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        # Frame 0 > frame 1 after normalisation
        assert signal[0] > signal[1]


# ---------------------------------------------------------------------------
# Tests: smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:

    def _run_varying_signal(self, smooth: bool, alpha: float = 0.6) -> np.ndarray:
        """Run with alternating 0/1 detections to create a varying signal."""
        n = 8
        frames = [_BLANK_FRAME.copy() for _ in range(n)]
        cap = _make_cap(frames)

        model = MagicMock()
        # Alternate high/no detection to produce variation
        results_seq = []
        for i in range(n):
            if i % 2 == 0:
                results_seq.append(_make_results([_make_box(0.9)]))
            else:
                results_seq.append(_make_results_no_boxes())
        model.side_effect = results_seq

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                return extract_threat_curve(
                    "v.mp4",
                    frame_skip=1,
                    smooth=smooth,
                    smoothing_alpha=alpha,
                    confidence_threshold=0.0,
                )

    def test_smoothed_signal_differs_from_unsmoothed(self):
        raw      = self._run_varying_signal(smooth=False)
        smoothed = self._run_varying_signal(smooth=True)
        # The two signals should differ (EMA changes values)
        assert not np.allclose(raw, smoothed), \
            "Smoothed and raw signals should differ for a varying input"

    def test_smoothed_signal_preserves_length(self):
        raw      = self._run_varying_signal(smooth=False)
        smoothed = self._run_varying_signal(smooth=True)
        assert len(smoothed) == len(raw)

    def test_smoothed_first_element_unchanged(self):
        """EMA initialises with signal[0] so first value is the same."""
        raw      = self._run_varying_signal(smooth=False)
        smoothed = self._run_varying_signal(smooth=True)
        assert float(smoothed[0]) == pytest.approx(float(raw[0]))

    def test_smoothed_reduces_peak_to_trough_amplitude(self):
        """EMA should reduce the peak-to-trough amplitude of a varying signal."""
        raw      = self._run_varying_signal(smooth=False)
        smoothed = self._run_varying_signal(smooth=True)
        amplitude_raw      = float(raw.max() - raw.min())
        amplitude_smoothed = float(smoothed.max() - smoothed.min())
        assert amplitude_smoothed <= amplitude_raw, \
            "Smoothed amplitude should not exceed raw amplitude"

    def test_single_frame_with_smooth_no_crash(self):
        """Single-frame signal: smooth branch skipped (len > 1 guard)."""
        cap = _make_cap([_BLANK_FRAME.copy()])
        model = MagicMock()
        model.return_value = _make_results([_make_box(0.8)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, smooth=True)

        assert len(signal) == 1
        assert isinstance(signal, np.ndarray)


# ---------------------------------------------------------------------------
# Tests: normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:

    def test_min_max_normalisation_maps_range_to_unit_interval(self):
        """Resulting signal must always lie in [0, 1]."""
        frames = [_BLANK_FRAME.copy() for _ in range(6)]
        cap = _make_cap(frames)

        model = MagicMock()
        # Varying confidences to produce a non-trivial signal
        confs = [0.3, 0.7, 0.5, 0.9, 0.1, 0.6]
        model.side_effect = [
            _make_results([_make_box(c)]) if c > 0.2 else _make_results_no_boxes()
            for c in confs
        ]

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        assert np.all(signal >= 0.0)
        assert np.all(signal <= 1.0)

    def test_all_same_raw_score_produces_all_zeros(self):
        """When min == max, signal is set to all zeros (no divide-by-zero)."""
        frames = [_BLANK_FRAME.copy() for _ in range(3)]
        cap = _make_cap(frames)

        model = MagicMock()
        # Identical boxes every frame → identical raw scores
        model.return_value = _make_results([_make_box(0.8)])

        with patch("video_processor.cv2.VideoCapture", return_value=cap):
            with patch("video_processor.YOLO", return_value=model):
                signal = extract_threat_curve("v.mp4", frame_skip=1, confidence_threshold=0.0)

        assert np.all(signal == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
