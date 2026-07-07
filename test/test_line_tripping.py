"""
Tests for Phase 0 — independent line tripping (MISO data gap plan)
==================================================================
Validation items (e) and (f) from the plan:
(e) a sustained line overload trips the line even when no attached node has failed
(f) mirrored edge pairs carry identical trip status
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.cascade import LineTripMonitor
from cascade_prediction.data.generator.config import Settings


@pytest.fixture
def mirrored_edge_index():
    """4-node ring stored as mirrored directed pairs (8 directed edges).

    Physical lines: (0,1), (1,2), (2,3), (3,0)
    Directed columns: even idx = forward, odd idx = mirror.
    """
    src = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    dst = np.array([1, 0, 2, 1, 3, 2, 0, 3])
    return np.stack([src, dst])


def _make_monitor(edge_index, sustain_steps=3):
    return LineTripMonitor(edge_index, trip_ratio=1.2, sustain_steps=sustain_steps)


class TestSustainedOverloadTrip:
    def test_sustained_overload_trips_without_node_failure(self, mirrored_edge_index):
        """(e) Line trips after sustain_steps consecutive overloaded steps,
        with no failed nodes involved at all."""
        monitor = _make_monitor(mirrored_edge_index, sustain_steps=3)
        loading = np.ones(8) * 0.5
        loading[[2, 3]] = 1.5  # physical line (1,2) overloaded, both rows

        assert monitor.update(loading, set()) == []          # step 1
        assert monitor.update(loading, set()) == []          # step 2
        assert monitor.update(loading, set()) == [2, 3]      # step 3 → trip

    def test_overload_blip_resets_timer(self, mirrored_edge_index):
        """A dip below threshold resets the sustain timer — no instant trips."""
        monitor = _make_monitor(mirrored_edge_index, sustain_steps=3)
        hot = np.ones(8) * 0.5
        hot[[2, 3]] = 1.5
        cool = np.ones(8) * 0.5

        assert monitor.update(hot, set()) == []
        assert monitor.update(hot, set()) == []
        assert monitor.update(cool, set()) == []   # blip resets
        assert monitor.update(hot, set()) == []
        assert monitor.update(hot, set()) == []
        assert monitor.update(hot, set()) == [2, 3]  # needs full 3 again

    def test_below_threshold_never_trips(self, mirrored_edge_index):
        monitor = _make_monitor(mirrored_edge_index)
        loading = np.ones(8) * 1.19  # just under trip_ratio=1.2
        for _ in range(20):
            assert monitor.update(loading, set()) == []

    def test_already_failed_lines_ignored(self, mirrored_edge_index):
        """Lines already out of service (node-derived failures) never re-trip."""
        monitor = _make_monitor(mirrored_edge_index, sustain_steps=1)
        loading = np.ones(8) * 2.0  # everything overloaded
        already_failed = {0, 1, 2, 3}
        tripped = monitor.update(loading, already_failed)
        assert set(tripped).isdisjoint(already_failed)
        assert tripped == [4, 5, 6, 7]


class TestMirrorConsistency:
    def test_one_row_overloaded_trips_both_rows(self, mirrored_edge_index):
        """(f) Even if only one directed row registers the overload, the whole
        mirrored pair trips together."""
        monitor = _make_monitor(mirrored_edge_index, sustain_steps=2)
        loading = np.ones(8) * 0.5
        loading[4] = 1.5  # only forward row of line (2,3)

        assert monitor.update(loading, set()) == []
        assert monitor.update(loading, set()) == [4, 5]  # mirror row 5 included

    def test_groups_cover_all_edges(self, mirrored_edge_index):
        monitor = _make_monitor(mirrored_edge_index)
        for i in range(8):
            group = monitor._group_of[i]
            assert len(group) == 2
            assert i in group


class TestDefaults:
    def test_defaults_come_from_settings(self, mirrored_edge_index):
        monitor = LineTripMonitor(mirrored_edge_index)
        assert monitor.trip_ratio == Settings.Cascade.LINE_TRIP_LOADING_RATIO
        assert monitor.sustain_steps == Settings.Cascade.LINE_TRIP_SUSTAIN_STEPS

    def test_reset_clears_timers(self, mirrored_edge_index):
        monitor = _make_monitor(mirrored_edge_index, sustain_steps=3)
        loading = np.ones(8) * 2.0
        monitor.update(loading, set())
        monitor.update(loading, set())
        monitor.reset()
        assert monitor.update(loading, set()) == []  # timer restarted
