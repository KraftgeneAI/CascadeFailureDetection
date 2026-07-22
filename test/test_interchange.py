"""
Tests for Phase 5 — inter-area exchange & storage (MISO data gap plan)
=====================================================================
Covers the plan's named validation items:
  (a) net actual interchange == sum of tie flows
  (b) SOC stays within [0, capacity]
plus the mirror invariant on tie designation, droop sign/deadband, and the
schedule block profile.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.config import Settings
from cascade_prediction.data.generator.interchange import (
    InterchangeScheduler,
    StorageController,
    assign_storage_nodes,
    assign_tie_lines,
    net_actual_interchange,
    tie_representative_rows,
)


@pytest.fixture
def ring_grid():
    """Mirrored ring + chords, with positions, as the simulator builds them."""
    rng = np.random.default_rng(7)
    n = 40
    positions = rng.uniform(-100, 100, (n, 2))
    pairs = [(i, (i + 1) % n) for i in range(n)]
    pairs += [(i, (i + 7) % n) for i in range(0, n, 3)]
    src, dst = [], []
    for a, b in pairs:
        src += [a, b]
        dst += [b, a]
    return np.stack([np.array(src), np.array(dst)]), positions


def _canon(edge_index, num_nodes):
    s, d = edge_index[0], edge_index[1]
    return np.minimum(s, d) * num_nodes + np.maximum(s, d)


class TestTieLines:
    def test_mirror_consistent(self, ring_grid):
        """Both directed rows of a tie share the flag (plan item (f) style)."""
        ei, pos = ring_grid
        tie = assign_tie_lines(ei, pos, np.random.default_rng(0))
        canon = _canon(ei, pos.shape[0])
        for k in np.unique(canon):
            vals = set(tie[canon == k].tolist())
            assert len(vals) == 1, f"tie flag split across mirrored pair {k}"

    def test_tie_count_matches_config(self, ring_grid):
        ei, pos = ring_grid
        tie = assign_tie_lines(ei, pos, np.random.default_rng(0))
        rows = tie_representative_rows(ei, tie)
        assert len(rows) == Settings.Interchange.N_TIE_LINES

    def test_ties_are_peripheral(self, ring_grid):
        """Ties sit at the boundary — farther from the centroid than average."""
        ei, pos = ring_grid
        tie = assign_tie_lines(ei, pos, np.random.default_rng(0))
        centroid = pos.mean(axis=0)
        mid = 0.5 * (pos[ei[0]] + pos[ei[1]])
        dist = np.linalg.norm(mid - centroid, axis=1)
        assert dist[tie == 1].mean() > dist[tie == 0].mean()


class TestNetInterchange:
    def test_equals_sum_of_tie_flows(self, ring_grid):
        """Plan validation item (a)."""
        ei, pos = ring_grid
        tie = assign_tie_lines(ei, pos, np.random.default_rng(0))
        rows = tie_representative_rows(ei, tie)

        rng = np.random.default_rng(1)
        flows = rng.normal(0, 50, ei.shape[1])
        # Enforce the antisymmetry a real solve produces.
        canon = _canon(ei, pos.shape[0])
        for k in np.unique(canon):
            idx = np.nonzero(canon == k)[0]
            flows[idx[1]] = -flows[idx[0]]

        expected = sum(flows[r] for r in rows)
        assert net_actual_interchange(flows, rows) == pytest.approx(expected)

    def test_both_rows_would_cancel(self, ring_grid):
        """Why representative rows exist: summing both rows is degenerate."""
        ei, pos = ring_grid
        tie = assign_tie_lines(ei, pos, np.random.default_rng(0))
        rng = np.random.default_rng(2)
        flows = rng.normal(0, 50, ei.shape[1])
        canon = _canon(ei, pos.shape[0])
        for k in np.unique(canon):
            idx = np.nonzero(canon == k)[0]
            flows[idx[1]] = -flows[idx[0]]
        assert flows[tie == 1].sum() == pytest.approx(0.0, abs=1e-9)

    def test_no_ties_gives_zero(self):
        assert net_actual_interchange(np.array([1.0, 2.0]),
                                      np.array([], dtype=np.int64)) == 0.0


class TestStorageSOC:
    @pytest.fixture
    def controller(self):
        st = assign_storage_nodes(np.zeros(30, dtype=np.int64),
                                  np.random.default_rng(5))
        return StorageController(st['storage_power_mw'],
                                 st['storage_capacity_mwh'],
                                 st['storage_qmax_mvar']), st

    def test_soc_stays_in_bounds(self, controller):
        """Plan validation item (b) — hammered with sustained extremes."""
        sc, st = controller
        cap = st['storage_capacity_mwh']
        for freq in (59.0, 61.0, 58.5, 61.5):
            for _ in range(300):
                sc.update(freq, dt_minutes=2.0)
                assert sc.soc_mwh.min() >= -1e-9
                assert np.all(sc.soc_mwh <= cap + 1e-9)

    def test_soc_fraction_range(self, controller):
        sc, _ = controller
        for freq in (59.0, 61.0):
            for _ in range(50):
                sc.update(freq, dt_minutes=2.0)
                f = sc.soc_fraction()
                assert np.all(f >= -1e-9) and np.all(f <= 1.0 + 1e-9)

    def test_low_frequency_discharges(self, controller):
        sc, _ = controller
        flow, _q = sc.update(59.5, dt_minutes=2.0)
        assert flow[sc.is_storage].min() > 0, "low frequency must discharge (+)"

    def test_high_frequency_charges(self, controller):
        sc, _ = controller
        flow, _q = sc.update(60.5, dt_minutes=2.0)
        assert flow[sc.is_storage].max() < 0, "high frequency must charge (-)"

    def test_deadband_no_response(self, controller):
        sc, _ = controller
        flow, _q = sc.update(
            Settings.Frequency.BASE_FREQUENCY
            + 0.5 * Settings.Storage.DEADBAND_HZ, dt_minutes=2.0)
        assert np.abs(flow).max() == 0.0

    def test_non_storage_buses_stay_zero(self, controller):
        sc, _ = controller
        flow, q = sc.update(59.0, dt_minutes=2.0)
        assert np.all(flow[~sc.is_storage] == 0.0)
        assert np.all(q[~sc.is_storage] == 0.0)

    def test_failed_bus_contributes_nothing(self, controller):
        sc, _ = controller
        target = int(np.nonzero(sc.is_storage)[0][0])
        flow, _q = sc.update(59.0, dt_minutes=2.0, failed_nodes={target})
        assert flow[target] == 0.0

    def test_empty_storage_cannot_discharge(self, controller):
        """
        Energy limit binds: a drained ESR delivers nothing, however large the
        frequency deviation. Drive SOC to empty directly rather than waiting
        for the droop to drain it — DROOP_MW_PER_HZ is deliberately tiny (it
        is sized to the frequency model's loop gain, see StorageConfig), so
        draining by droop alone would take thousands of steps and the test
        would silently be asserting the wrong thing.
        """
        sc, _ = controller
        sc.soc_mwh[:] = 0.0
        flow, _q = sc.update(55.0, dt_minutes=2.0)   # extreme underfrequency
        assert np.abs(flow).max() == pytest.approx(0.0, abs=1e-9)
        assert sc.soc_mwh.min() >= -1e-9

    def test_full_storage_cannot_charge(self, controller):
        """Mirror case: a full ESR absorbs nothing on overfrequency."""
        sc, st = controller
        sc.soc_mwh[:] = st['storage_capacity_mwh']
        flow, _q = sc.update(65.0, dt_minutes=2.0)   # extreme overfrequency
        assert np.abs(flow).max() == pytest.approx(0.0, abs=1e-9)
        cap = st['storage_capacity_mwh']
        assert np.all(sc.soc_mwh <= cap + 1e-9)

    def test_reset_restores_initial_soc(self, controller):
        sc, st = controller
        for _ in range(50):
            sc.update(59.0, dt_minutes=2.0)
        sc.reset()
        expected = st['storage_capacity_mwh'] * Settings.Storage.SOC_INIT_FRACTION
        assert np.allclose(sc.soc_mwh, expected)


class TestFrequencyLoopStability:
    """
    Storage MW joins the generation sum the frequency model reads, so the
    droop closes a feedback loop. At the original 120 MW/Hz the loop gain was
    ~127 and the grid collapsed on underfrequency at t=1. These pin the gain.
    """

    @staticmethod
    def _loop_gain(system_base_mw=8500.0):
        sens = (Settings.Frequency.BASE_FREQUENCY
                / (Settings.Frequency.D_TOTAL * system_base_mw))
        return (sens * Settings.Storage.N_STORAGE_NODES
                * Settings.Storage.DROOP_MW_PER_HZ)

    def test_loop_gain_is_damped(self):
        assert self._loop_gain() < 0.5, "droop would oscillate or diverge"

    def test_closed_loop_settles(self):
        """Replay the frequency<->storage loop; it must converge, not cycle."""
        f0 = Settings.Frequency.BASE_FREQUENCY
        sens = f0 / (Settings.Frequency.D_TOTAL * 8500.0)
        a = 1.0 - np.exp(-1.0 / 0.5)
        f, hist = 59.90, []
        for _ in range(30):
            dev = f0 - f
            if abs(dev) < Settings.Storage.DEADBAND_HZ:
                dev = 0.0
            agg = (Settings.Storage.DROOP_MW_PER_HZ * dev
                   * Settings.Storage.N_STORAGE_NODES)
            f = f + ((f0 + sens * agg) - f) * a
            hist.append(f)
        hist = np.array(hist)
        assert not ((hist < 55.0) | (hist > 65.0)).any(), "left frequency envelope"
        assert np.ptp(hist[-5:]) < 1e-3, "sustained limit cycle"

    def test_aggregate_cap_bounds_gain(self):
        """The structural cap holds even if the droop is retuned upward."""
        st = assign_storage_nodes(np.zeros(30, dtype=np.int64),
                                  np.random.default_rng(5))
        sc = StorageController(st['storage_power_mw'],
                               st['storage_capacity_mwh'],
                               st['storage_qmax_mvar'])
        base = 8500.0
        sens = (Settings.Frequency.BASE_FREQUENCY
                / (Settings.Frequency.D_TOTAL * base))
        flow, _q = sc.update(55.0, dt_minutes=2.0, system_base_mw=base)
        assert abs(flow.sum()) * sens <= Settings.Storage.MAX_LOOP_GAIN + 1e-9


class TestSchedule:
    def test_holds_then_steps(self):
        sch = InterchangeScheduler(np.random.default_rng(11))
        vals = [sch.step() for _ in range(40)]
        blocks = 1 + sum(1 for i in range(1, len(vals)) if vals[i] != vals[i - 1])
        assert 2 <= blocks <= 8, "schedule should be blocky, not per-step noise"

    def test_within_configured_span(self):
        C = Settings.Interchange
        sch = InterchangeScheduler(np.random.default_rng(12))
        for _ in range(200):
            v = sch.step()
            assert abs(v - C.SCHEDULE_BASE_MW) <= C.SCHEDULE_SPAN_MW + 1e-9


class TestPhase5FeatureWidths:
    # EDGE_FEATURES lives on ModelConfig; the SCADA/node dims on
    # EmbeddingConfig. Keep the lookups where they actually are.
    def test_widths_grew_append_only(self):
        assert Settings.Model.EDGE_FEATURES >= 14
        assert Settings.Embedding.INFRA_SCADA_FEATURES >= 28

    def test_node_feature_dim_consistent(self):
        E = Settings.Embedding
        base = E.NODE_FEATURE_BASE_DIM
        assert base == E.INFRA_SCADA_FEATURES + 8 + 10 + 2
        assert E.NODE_FEATURE_DIM == base * 3 + 1 + 4 + 5
