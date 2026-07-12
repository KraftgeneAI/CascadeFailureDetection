"""
Tests for Phase 1 — multi-rating lines (MISO data gap plan)
===========================================================
Emergency (LEP) and load-dump tiers as per-edge thresholds on the
normal-rating loading ratio, plus mirror consistency of tier trips.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.cascade import LineTripMonitor
from cascade_prediction.data.generator.config import Settings


@pytest.fixture
def mirrored_edge_index():
    """4-node ring as mirrored directed pairs (8 directed edges)."""
    src = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    dst = np.array([1, 0, 2, 1, 3, 2, 0, 3])
    return np.stack([src, dst])


@pytest.fixture
def tiered_monitor(mirrored_edge_index):
    """Monitor with per-edge tier thresholds like the simulator builds:
    emergency = 1.2 × normal, load-dump = 1.45 × normal."""
    emergency_ratio = np.full(8, 1.2)
    load_dump_ratio = np.full(8, 1.45)
    return LineTripMonitor(
        mirrored_edge_index,
        trip_ratio=emergency_ratio,
        sustain_steps=3,
        fast_trip_ratio=load_dump_ratio,
        fast_trip_steps=1,
    )


class TestEmergencyTier:
    def test_between_normal_and_emergency_never_trips(self, tiered_monitor):
        """Overload above normal but below emergency rides through — the
        core Phase 1 behavior change vs. instant tripping."""
        loading = np.ones(8) * 1.15  # >1.0 normal, <1.2 emergency
        for _ in range(20):
            assert tiered_monitor.update(loading, set()) == []

    def test_above_emergency_trips_after_sustain(self, tiered_monitor):
        loading = np.ones(8) * 0.5
        loading[[2, 3]] = 1.3  # above emergency, below load-dump
        assert tiered_monitor.update(loading, set()) == []
        assert tiered_monitor.update(loading, set()) == []
        assert tiered_monitor.update(loading, set()) == [2, 3]

    def test_per_edge_thresholds(self, mirrored_edge_index):
        """Each edge can carry its own emergency ratio."""
        ratios = np.full(8, 1.2)
        ratios[[4, 5]] = 1.10  # line (2,3) has a tighter emergency margin
        monitor = LineTripMonitor(mirrored_edge_index, trip_ratio=ratios,
                                  sustain_steps=1)
        loading = np.ones(8) * 1.15  # only trips where threshold is 1.10
        assert monitor.update(loading, set()) == [4, 5]


class TestLoadDumpTier:
    def test_above_load_dump_trips_in_one_step(self, tiered_monitor):
        """Load-dump rating tolerates only minutes — one timestep here."""
        loading = np.ones(8) * 0.5
        loading[[0, 1]] = 1.6  # above load-dump 1.45
        assert tiered_monitor.update(loading, set()) == [0, 1]

    def test_emergency_timer_unaffected_by_other_lines_fast_trip(self, tiered_monitor):
        """A fast trip elsewhere doesn't reset another line's emergency timer."""
        loading = np.ones(8) * 0.5
        loading[[2, 3]] = 1.3   # emergency tier, needs 3 steps
        loading[[6, 7]] = 1.6   # load-dump tier, trips now
        assert tiered_monitor.update(loading, set()) == [6, 7]
        loading[[6, 7]] = 0.0   # tripped lines carry no flow
        assert tiered_monitor.update(loading, {6, 7}) == []
        assert tiered_monitor.update(loading, {6, 7}) == [2, 3]

    def test_fast_tier_disabled_by_default(self, mirrored_edge_index):
        """Phase 0 behavior preserved when no load-dump ratings supplied."""
        monitor = LineTripMonitor(mirrored_edge_index, trip_ratio=1.2,
                                  sustain_steps=3)
        loading = np.ones(8) * 5.0  # extreme — but still needs 3 sustained steps
        assert monitor.update(loading, set()) == []
        assert monitor.update(loading, set()) == []
        assert len(monitor.update(loading, set())) == 8


class TestMirrorConsistencyTiers:
    def test_fast_trip_takes_whole_mirrored_pair(self, tiered_monitor):
        loading = np.ones(8) * 0.5
        loading[4] = 1.6  # only forward row of line (2,3) above load-dump
        assert tiered_monitor.update(loading, set()) == [4, 5]

    def test_reset_clears_both_timers(self, tiered_monitor):
        loading = np.ones(8) * 1.3
        tiered_monitor.update(loading, set())
        tiered_monitor.update(loading, set())
        tiered_monitor.reset()
        assert tiered_monitor.update(loading, set()) == []


class TestPhase1Config:
    def test_rating_factor_ordering(self):
        """normal < emergency < load-dump must hold for the tier logic."""
        c = Settings.Cascade
        assert 1.0 < c.EMERGENCY_RATING_FACTOR_MIN <= c.EMERGENCY_RATING_FACTOR_MAX
        assert c.EMERGENCY_RATING_FACTOR_MAX < c.LOAD_DUMP_RATING_FACTOR_MIN
        assert c.LOAD_DUMP_RATING_FACTOR_MIN <= c.LOAD_DUMP_RATING_FACTOR_MAX

    def test_edge_features_widened(self):
        assert Settings.Model.EDGE_FEATURES == 10
