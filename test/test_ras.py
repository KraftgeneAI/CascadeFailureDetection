"""
Tests for Phase 4 — Remedial Action Schemes (MISO data gap plan)
================================================================
Covers assign_ras_schemes (placement on top corridors, target validity,
determinism) and RASEngine (Available/Armed/Triggered/Operated transitions,
single-shot latch, action emission, telemetry stream). Torch-free.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.ras import RASEngine, assign_ras_schemes
from cascade_prediction.data.generator.config import Settings

R = Settings.RAS


def ring_grid(num_nodes=20):
    src, dst = [], []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src += [i, j]
        dst += [j, i]
    ei = np.array([src, dst], dtype=np.int64)
    nt = np.zeros(num_nodes); nt[0] = 1; nt[1] = 1; nt[10] = 2
    limits = np.arange(1, ei.shape[1] + 1, dtype=float)  # ratings rise with idx
    return ei, nt, limits


class TestAssignment:
    def test_monitors_top_rated_corridors(self):
        ei, nt, limits = ring_grid()
        s = assign_ras_schemes(ei, nt, limits, rng=np.random.default_rng(1))
        assert len(s['ras_monitored_edge']) == R.N_SCHEMES
        # monitored reps must be among the highest-rated representative edges
        mon_ratings = limits[s['ras_monitored_edge']]
        assert mon_ratings.min() > np.median(limits)

    def test_targets_match_action_types(self):
        ei, nt, limits = ring_grid()
        s = assign_ras_schemes(ei, nt, limits, rng=np.random.default_rng(1))
        for act, tgt in zip(s['ras_action_type'], s['ras_action_target']):
            if act == R.ACTION_TRIP_GEN:
                assert nt[tgt] == 1
            elif act == R.ACTION_SHED_LOAD:
                assert nt[tgt] == 0
            else:
                assert 0 <= tgt < ei.shape[1]
                assert tgt not in s['ras_monitored_edge']

    def test_deterministic(self):
        ei, nt, limits = ring_grid()
        a = assign_ras_schemes(ei, nt, limits, rng=np.random.default_rng(5))
        b = assign_ras_schemes(ei, nt, limits, rng=np.random.default_rng(5))
        for k in a:
            assert (a[k] == b[k]).all(), k


def make_engine(monitored=(3,), action=(R.ACTION_SHED_LOAD,), target=(7,)):
    return RASEngine({
        'ras_monitored_edge': np.array(monitored),
        'ras_action_type': np.array(action),
        'ras_action_target': np.array(target),
    })


class TestStateMachine:
    def test_available_below_arm(self):
        eng = make_engine()
        loading = np.zeros(10); loading[3] = 0.5
        assert eng.update(loading) == []
        assert eng.status[0] == R.STATUS_AVAILABLE

    def test_arms_and_disarms(self):
        eng = make_engine()
        loading = np.zeros(10); loading[3] = 0.9
        assert eng.update(loading) == []
        assert eng.status[0] == R.STATUS_ARMED
        loading[3] = 0.5
        eng.update(loading)
        assert eng.status[0] == R.STATUS_AVAILABLE

    def test_trigger_fires_action_once_and_latches(self):
        eng = make_engine()
        loading = np.zeros(10); loading[3] = 1.2
        fired = eng.update(loading)
        assert fired == [(R.ACTION_SHED_LOAD, 7)]
        assert eng.status[0] == R.STATUS_TRIGGERED
        # stays overloaded — but single-shot: no second firing, latches
        assert eng.update(loading) == []
        assert eng.status[0] == R.STATUS_OPERATED
        # even after loading recovers, Operated is latched
        loading[3] = 0.1
        eng.update(loading)
        assert eng.status[0] == R.STATUS_OPERATED

    def test_reset_rearms(self):
        eng = make_engine()
        loading = np.zeros(10); loading[3] = 1.2
        eng.update(loading)
        eng.update(loading)
        eng.reset()
        assert eng.status[0] == R.STATUS_AVAILABLE
        assert eng.update(loading) == [(R.ACTION_SHED_LOAD, 7)]

    def test_zero_schemes_noop(self):
        eng = RASEngine({
            'ras_monitored_edge': np.array([], dtype=np.int64),
            'ras_action_type': np.array([], dtype=np.int64),
            'ras_action_target': np.array([], dtype=np.int64),
        })
        assert eng.update(np.zeros(10)) == []
        assert eng.status_stream(np.zeros(10)).shape == (0, 3)


class TestTelemetry:
    def test_status_stream_layout(self):
        eng = make_engine(monitored=(3, 5),
                          action=(R.ACTION_TRIP_GEN, R.ACTION_OPEN_LINE),
                          target=(1, 8))
        loading = np.zeros(10); loading[3] = 0.9; loading[5] = 0.2
        eng.update(loading)
        frame = eng.status_stream(loading)
        assert frame.shape == (2, R.RAS_FEATURES)
        assert frame[0, 0] == R.STATUS_ARMED
        assert frame[1, 0] == R.STATUS_AVAILABLE
        assert frame[0, 1] == pytest.approx(0.9)
        assert frame[0, 2] == R.ACTION_TRIP_GEN
        assert frame[1, 2] == R.ACTION_OPEN_LINE
