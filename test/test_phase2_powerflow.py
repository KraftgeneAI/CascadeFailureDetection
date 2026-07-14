"""
Phase 2 integration tests — transformers/PARs inside the AC power flow.
Validation item (d) from the MISO data gap plan: a PAR tap change SHIFTS
flow. Requires PyPSA (skipped if unavailable).
"""

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")

from cascade_prediction.data.generator.physics import PowerFlowSimulator
from cascade_prediction.data.generator.config import Settings


def build_sim(branch_type=None):
    """
    4-bus meshed grid: gen at bus 0, loads elsewhere. Two parallel paths
    0-1-2 and 0-3-2 so a PAR on one path can visibly reroute flow.
    """
    pairs = [(0, 1), (1, 2), (0, 3), (3, 2)]
    src = [p[0] for p in pairs] + [p[1] for p in pairs]
    dst = [p[1] for p in pairs] + [p[0] for p in pairs]
    edge_index = np.array([src, dst], dtype=np.int64)
    e = edge_index.shape[1]

    # Mirror per-pair params onto both directed rows
    def mirrored(vals):
        return np.array(list(vals) + list(vals), dtype=float)

    bt = np.zeros(e, dtype=np.int64)
    if branch_type is not None:
        bt = branch_type

    return PowerFlowSimulator(
        num_nodes=4,
        edge_index=edge_index,
        positions=np.array([[0, 0], [1, 0], [2, 0], [1, 1]], dtype=float),
        node_types=np.array([1, 0, 0, 0]),          # bus 0 = generator
        gen_capacity=np.array([300.0, 0, 0, 0]),
        line_reactance=mirrored([10.0, 10.0, 10.0, 10.0]),
        line_resistance=mirrored([1.0, 1.0, 1.0, 1.0]),
        line_susceptance=mirrored([1e-6] * 4),
        line_conductance=mirrored([0.0] * 4),
        thermal_limits=mirrored([200.0] * 4),
        branch_type=bt,
    ), edge_index


def test_par_tap_change_shifts_flow():
    """Plan validation (d): changing a PAR's phase shift reroutes real power."""
    e = 8
    bt = np.zeros(e, dtype=np.int64)
    bt[0] = bt[4] = Settings.Branch.TYPE_PAR      # pair (0,1) is the PAR
    sim, ei = build_sim(branch_type=bt)

    generation = np.array([150.0, 0, 0, 0])
    load = np.array([0.0, 40.0, 70.0, 40.0])

    neutral = np.zeros(e)
    v0, a0, p_before, q0_, qq0, ok0 = sim.compute_power_flow(
        generation, load, phase_shifts=neutral)
    assert ok0, "base case should converge"

    shifted = np.zeros(e)
    shifted[0] = 10.0                              # 10° shift on the PAR
    v1, a1, p_after, q1_, qq1, ok1 = sim.compute_power_flow(
        generation, load, phase_shifts=shifted)
    assert ok1, "shifted case should converge"

    # The PAR's own flow must change materially, and the parallel path
    # (pair (0,3), rep edge 2) must pick up / shed the difference.
    assert abs(p_after[0] - p_before[0]) > 1.0, \
        f"PAR flow unchanged: {p_before[0]:.2f} -> {p_after[0]:.2f}"
    assert abs(p_after[2] - p_before[2]) > 1.0, \
        "parallel path flow should shift when the PAR reroutes power"


def test_transformer_branch_solves_and_carries_flow():
    """A typed LTC transformer participates in the solve with sane flows."""
    e = 8
    bt = np.zeros(e, dtype=np.int64)
    bt[1] = bt[5] = Settings.Branch.TYPE_TRANSFORMER   # pair (1,2)
    sim, ei = build_sim(branch_type=bt)

    generation = np.array([150.0, 0, 0, 0])
    load = np.array([0.0, 40.0, 70.0, 40.0])
    v, a, p, nq, q, ok = sim.compute_power_flow(generation, load)
    assert ok
    assert abs(p[1]) > 1.0, "transformer branch should carry real power"
    # Mirrored row reports the same physical flow with opposite orientation
    assert p[1] == pytest.approx(-p[5], rel=1e-6)


def test_tap_ratio_changes_voltage():
    """LTC tap ratio measurably moves the regulated-side voltage."""
    e = 8
    bt = np.zeros(e, dtype=np.int64)
    bt[1] = bt[5] = Settings.Branch.TYPE_TRANSFORMER
    sim, ei = build_sim(branch_type=bt)

    generation = np.array([150.0, 0, 0, 0])
    load = np.array([0.0, 40.0, 70.0, 40.0])

    ones = np.ones(e)
    _, _, _, _, _, ok0 = sim.compute_power_flow(generation, load, tap_ratios=ones)
    v_base = sim.network.buses_t.v_mag_pu.loc["now"].copy()

    stepped = np.ones(e)
    stepped[1] = 1.05                              # +8 taps worth of ratio
    _, _, _, _, _, ok1 = sim.compute_power_flow(generation, load, tap_ratios=stepped)
    v_tap = sim.network.buses_t.v_mag_pu.loc["now"]
    assert ok0 and ok1
    assert float(np.abs(v_tap.values - v_base.values).max()) > 1e-3, \
        "tap ratio change should move bus voltages"
