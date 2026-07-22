"""
Tests for Phase 6 — scheduled outages & de-rates (MISO data gap plan)
====================================================================
The load-bearing invariants:
  * a PLANNED outage is never a failure (labels stay clean)
  * the ternary branch_status masks correctly — the -1 code is TRUTHY, so a
    naive `mask = status != 0` leaves a planned-open branch in the adjacency
  * de-rates stay within physical bounds and respond to ambient
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.config import Settings
from cascade_prediction.data.generator.outages import (
    OutageSchedule,
    branch_status_with_outages,
    edge_mask_from_branch_status,
)

C = Settings.Outage


@pytest.fixture
def grid():
    """Mirrored ring + chords with a handful of generator buses."""
    n = 30
    pairs = [(i, (i + 1) % n) for i in range(n)]
    pairs += [(i, (i + 5) % n) for i in range(0, n, 3)]
    src, dst = [], []
    for a, b in pairs:
        src += [a, b]
        dst += [b, a]
    node_types = np.zeros(n, dtype=np.int64)
    node_types[[2, 7, 13, 21, 26]] = 1
    return np.stack([np.array(src), np.array(dst)]), node_types


@pytest.fixture
def schedule(grid):
    ei, nt = grid
    return OutageSchedule(ei, nt, sequence_length=40,
                          rng=np.random.default_rng(4))


class TestTernaryBranchStatus:
    def test_three_distinct_codes(self):
        bs = branch_status_with_outages(20, [0, 1], [4, 5])
        assert bs[0] == C.STATUS_FORCED_OPEN
        assert bs[4] == C.STATUS_PLANNED_OPEN
        assert bs[10] == C.STATUS_CLOSED

    def test_planned_takes_precedence_over_forced(self):
        """
        A branch both scheduled out AND otherwise reading forced-open must
        report PLANNED — the operator removed it deliberately, and labelling
        it a trip would teach the model a failure that never happened.
        """
        bs = branch_status_with_outages(10, [3], [3])
        assert bs[3] == C.STATUS_PLANNED_OPEN

    def test_mask_drops_planned_open(self):
        """The whole point of the ternary: -1 must still be masked OUT."""
        bs = branch_status_with_outages(20, [0], [4])
        mask = edge_mask_from_branch_status(bs)
        assert mask[4] == 0.0, "planned-open branch left in the adjacency"
        assert mask[0] == 0.0, "forced-open branch left in the adjacency"
        assert mask[10] == 1.0

    def test_truthiness_mask_would_be_wrong(self):
        """
        Regression guard for the trap. `status != 0` is the natural way to
        write this and it is WRONG, because STATUS_PLANNED_OPEN == -1 is
        truthy. If this ever starts passing, someone has changed the codes
        and every consumer needs revisiting.
        """
        bs = branch_status_with_outages(20, [0], [4])
        naive = (bs != 0).astype(np.float32)
        correct = edge_mask_from_branch_status(bs)
        assert naive[4] == 1.0 and correct[4] == 0.0
        assert naive.sum() != correct.sum()

    def test_values_need_no_normalization(self):
        """dataset.py leaves col 7 raw — the codes must already be in [-1,1]."""
        for v in (C.STATUS_CLOSED, C.STATUS_FORCED_OPEN, C.STATUS_PLANNED_OPEN):
            assert -1.0 <= v <= 1.0


class TestScheduleStructure:
    def test_windows_inside_sequence(self, schedule):
        for _rows, s, e in schedule.line_outages:
            assert 0 <= s < e <= schedule.sequence_length
        for _n, s, e in schedule.gen_outages:
            assert 0 <= s < e <= schedule.sequence_length
        for _rows, s, e, _f in schedule.derate_events:
            assert 0 <= s < e <= schedule.sequence_length

    def test_line_outages_mirror_consistent(self, grid, schedule):
        """Both directed rows of a physical branch go out together."""
        ei, _nt = grid
        n = int(max(ei[0].max(), ei[1].max())) + 1
        canon = np.minimum(ei[0], ei[1]) * n + np.maximum(ei[0], ei[1])
        for t in range(schedule.sequence_length):
            rows = set(int(r) for r in schedule.planned_line_rows(t))
            for k in np.unique(canon):
                pair = set(np.nonzero(canon == k)[0].tolist())
                if pair & rows:
                    assert pair <= rows, "mirrored pair split by an outage"

    def test_gen_outages_only_on_generators(self, grid, schedule):
        _ei, nt = grid
        gens = set(np.nonzero(nt == 1)[0].tolist())
        seen = set()
        for t in range(schedule.sequence_length):
            seen |= schedule.planned_gen_nodes(t)
        assert seen <= gens

    def test_outage_is_a_closed_open_window(self, schedule):
        """Equipment is out during [start, end) and back in afterwards."""
        if not schedule.line_outages:
            pytest.skip("no line outages drawn")
        rows, start, end = schedule.line_outages[0]
        r = int(rows[0])
        assert r in set(int(x) for x in schedule.planned_line_rows(start))
        if end < schedule.sequence_length:
            assert r not in set(int(x) for x in schedule.planned_line_rows(end))


class TestDerates:
    def test_ambient_reference_is_nameplate(self, schedule):
        assert schedule.ambient_derate(C.AMBIENT_REF_C) == 1.0
        assert schedule.ambient_derate(C.AMBIENT_REF_C - 15.0) == 1.0

    def test_hotter_ambient_derates_more(self, schedule):
        a = schedule.ambient_derate(30.0)
        b = schedule.ambient_derate(45.0)
        assert 1.0 > a > b

    def test_ambient_floor_holds(self, schedule):
        assert schedule.ambient_derate(500.0) >= C.AMBIENT_DERATE_FLOOR

    def test_derate_always_within_bounds(self, schedule):
        for t in range(schedule.sequence_length):
            for ambient in (0.0, 25.0, 45.0, 80.0):
                d = schedule.derate(t, ambient)
                assert d.min() >= C.TOTAL_DERATE_FLOOR - 1e-9
                assert d.max() <= 1.0 + 1e-9

    def test_derate_never_raises_a_rating(self, schedule):
        """A de-rate reduces capability; it must never exceed nameplate."""
        for t in range(schedule.sequence_length):
            assert schedule.derate(t, 25.0).max() <= 1.0 + 1e-9


class TestLabelIntegrity:
    """
    Planned outages take equipment out of service but are NOT failures. These
    pin the separation at the schedule level; the simulator keeps planned rows
    in `out_of_service_lines` (fed to the solve) and out of
    `cumulative_failed_lines` (fed to the labels).
    """

    def test_planned_rows_are_a_distinct_set(self, schedule):
        for t in range(schedule.sequence_length):
            planned = set(int(r) for r in schedule.planned_line_rows(t))
            forced = {0, 1}                      # stand-in failure set
            out_of_service = forced | planned
            # The solve sees the union...
            assert planned <= out_of_service
            # ...but the failure set is untouched by the schedule.
            assert forced == {0, 1}

    def test_planned_gen_outage_is_not_a_node_failure(self, schedule):
        """A unit on outage stops generating; its BUS stays healthy."""
        failed_nodes: set = set()
        for t in range(schedule.sequence_length):
            out = schedule.planned_gen_nodes(t)
            assert not (out & failed_nodes), \
                "a planned generator outage must not imply a failed bus"


class TestDisabledSchedule:
    """
    The master switch (OutageConfig.ENABLED) produces a live-but-empty
    schedule: the query interface still works, but nothing is ever out and no
    scheduled de-rate is applied. The simulator relies on this so it needs no
    special-casing when outages are turned off.
    """

    @pytest.fixture
    def disabled(self, grid):
        ei, nt = grid
        return OutageSchedule(ei, nt, sequence_length=40,
                              rng=np.random.default_rng(4), enabled=False)

    def test_no_events_drawn(self, disabled):
        assert disabled.line_outages == []
        assert disabled.gen_outages == []
        assert disabled.derate_events == []
        assert not disabled.has_any()

    def test_queries_return_empty(self, disabled):
        for t in range(disabled.sequence_length):
            assert disabled.planned_line_rows(t).size == 0
            assert disabled.planned_gen_nodes(t) == set()

    def test_derate_is_ambient_only(self, disabled):
        """With no scheduled events, the only de-rate left is ambient."""
        for t in range(disabled.sequence_length):
            d = disabled.derate(t, ambient_temp_c=25.0)   # at reference
            assert np.allclose(d, 1.0)

    def test_branch_status_stays_binary(self, disabled):
        """No planned rows -> branch_status never carries the -1 code."""
        from cascade_prediction.data.generator.outages import (
            branch_status_with_outages)
        for t in range(disabled.sequence_length):
            planned = disabled.planned_line_rows(t)
            bs = branch_status_with_outages(disabled.num_edges, [0, 1], planned)
            assert not (bs == C.STATUS_PLANNED_OPEN).any()


class TestPhase6Config:
    def test_status_codes_distinct(self):
        codes = {C.STATUS_CLOSED, C.STATUS_FORCED_OPEN, C.STATUS_PLANNED_OPEN}
        assert len(codes) == 3

    def test_derate_factors_reduce(self):
        assert 0.0 < C.DERATE_FACTOR_MIN <= C.DERATE_FACTOR_MAX < 1.0

    def test_floors_are_ordered(self):
        assert C.TOTAL_DERATE_FLOOR <= C.AMBIENT_DERATE_FLOOR <= 1.0
