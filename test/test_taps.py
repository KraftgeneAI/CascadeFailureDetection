"""
Tests for Phase 2 — branch typing & tap devices (MISO data gap plan)
====================================================================
Covers assign_branch_types (fractions, mirror consistency, determinism)
and TapController (LTC voltage feedback, PAR flow relief, limits,
mirror-consistent state, dead-bus hold). Torch-free.

Validation item (d) from the plan — "PAR tap change shifts flow" — needs
PyPSA and lives in the integration path: build a PowerFlowSimulator with a
PAR branch and compare flows at phase_shift 0° vs 10°.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.taps import (
    TapController, assign_branch_types,
)
from cascade_prediction.data.generator.config import Settings

B = Settings.Branch


def ring_edge_index(num_nodes=10):
    src, dst = [], []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src += [i, j]
        dst += [j, i]
    return np.array([src, dst], dtype=np.int64)


class TestAssignBranchTypes:
    def test_mirror_consistent(self):
        ei = ring_edge_index(40)
        bt = assign_branch_types(ei, rng=np.random.default_rng(7))
        # both directed rows of a pair share the type (pairs are [2k, 2k+1])
        assert (bt[0::2] == bt[1::2]).all()

    def test_fractions_roughly_respected(self):
        ei = ring_edge_index(100)   # 100 physical pairs
        bt = assign_branch_types(ei, rng=np.random.default_rng(7))
        n_pairs = len(bt) // 2
        n_trafo = int((bt[0::2] == B.TYPE_TRANSFORMER).sum())
        n_par = int((bt[0::2] == B.TYPE_PAR).sum())
        assert n_trafo == round(n_pairs * B.TRANSFORMER_FRACTION)
        assert n_par == round(n_pairs * B.PAR_FRACTION)

    def test_deterministic_with_seed(self):
        ei = ring_edge_index(50)
        a = assign_branch_types(ei, rng=np.random.default_rng(42))
        b = assign_branch_types(ei, rng=np.random.default_rng(42))
        assert (a == b).all()


def make_controller(num_nodes=6, trafo_pairs=(0,), par_pairs=(2,)):
    """Ring grid; selected pairs typed as transformer / PAR."""
    ei = ring_edge_index(num_nodes)
    bt = np.zeros(ei.shape[1], dtype=np.int64)
    for p in trafo_pairs:
        bt[2 * p] = bt[2 * p + 1] = B.TYPE_TRANSFORMER
    for p in par_pairs:
        bt[2 * p] = bt[2 * p + 1] = B.TYPE_PAR
    return TapController(ei, bt), ei, bt


class TestLTC:
    def test_low_voltage_raises_tap(self):
        ctl, ei, bt = make_controller()
        volts = np.ones(6)
        volts[ei[1, 0]] = 0.95    # regulated bus below setpoint band
        flows = np.zeros(ei.shape[1]); loading = np.zeros(ei.shape[1])
        ratio, _ = ctl.update(volts, flows, loading)
        assert ctl.tap_position[0] == 1
        assert ctl.tap_position[1] == 1          # mirror row identical
        assert ratio[0] == pytest.approx(1 + B.LTC_STEP_PU)

    def test_high_voltage_lowers_tap(self):
        ctl, ei, _ = make_controller()
        volts = np.ones(6); volts[ei[1, 0]] = 1.05
        ctl.update(volts, np.zeros(ei.shape[1]), np.zeros(ei.shape[1]))
        assert ctl.tap_position[0] == -1

    def test_deadband_holds(self):
        ctl, ei, _ = make_controller()
        volts = np.ones(6)
        volts[ei[1, 0]] = 1.005   # inside the ±LTC_DEADBAND_PU band
        ctl.update(volts, np.zeros(ei.shape[1]), np.zeros(ei.shape[1]))
        assert ctl.tap_position[0] == 0

    def test_one_step_per_timestep_and_limit(self):
        """
        Tap moves at most one step per timestep and clamps at LTC_TAP_MAX.

        The regulated voltage must actually RESPOND, or anti-windup correctly
        blocks the device before it ever reaches the limit (see
        test_anti_windup_blocks_unresponsive_bus). Here the bus creeps upward
        by more than LTC_MIN_V_RESPONSE_PU each step while staying below the
        setpoint band, so the controller keeps raising and hits the clamp.
        """
        ctl, ei, _ = make_controller()
        flows = np.zeros(ei.shape[1])
        loading = np.zeros(ei.shape[1])
        for k in range(50):                           # more steps than range
            volts = np.ones(6)
            volts[ei[1, 0]] = 0.80 + 0.001 * k        # responds, still low
            ctl.update(volts, flows, loading)
        assert ctl.tap_position[0] == B.LTC_TAP_MAX   # clamped, not runaway

    def test_anti_windup_blocks_unresponsive_bus(self):
        """
        A tap with no authority must stop stepping, not ratchet to its limit.

        A frozen regulated voltage is the pathological case that caused the
        production bug: the LTC was regulating a generator bus pinned by its
        AVR, so the tap moved one step EVERY timestep, rerouting flow until
        lines tripped at constant load. Anti-windup blocks after
        LTC_STALL_STEPS of no voltage response.
        """
        ctl, ei, _ = make_controller()
        volts = np.ones(6)
        volts[ei[1, 0]] = 0.8         # far below setpoint, and never responds
        for _ in range(50):
            ctl.update(volts, np.zeros(ei.shape[1]), np.zeros(ei.shape[1]))
        assert abs(int(ctl.tap_position[0])) <= B.LTC_STALL_STEPS
        assert abs(int(ctl.tap_position[0])) < B.LTC_TAP_MAX

    def test_load_side_bus_is_regulated(self):
        """
        Given node types, the LTC regulates the NON-generator end. A generator
        bus is pinned at its AVR setpoint, so a tap has no authority over it.
        """
        from cascade_prediction.data.generator.taps import TapController
        # Two mirrored transformer branches: 0<->1 and 1<->2, bus 1 = generator
        ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        bt = np.array([B.TYPE_TRANSFORMER] * 4)
        node_types = np.array([0, 1, 0])          # bus 1 is a generator
        ctl = TapController(ei, bt, node_types)
        assert 1 not in set(int(b) for b in ctl._rep_dst), \
            "LTC must not regulate a generator bus"

    def test_dead_bus_holds_tap(self):
        ctl, ei, _ = make_controller()
        volts = np.ones(6); volts[ei[1, 0]] = 0.0    # de-energized bus
        ctl.update(volts, np.zeros(ei.shape[1]), np.zeros(ei.shape[1]))
        assert ctl.tap_position[0] == 0


class TestPAR:
    def test_overload_steps_against_flow(self):
        ctl, ei, _ = make_controller()
        rep = 4                                       # pair 2 → rep edge 4
        flows = np.zeros(ei.shape[1]); flows[rep] = 100.0   # positive flow
        loading = np.zeros(ei.shape[1]); loading[rep] = 1.1
        _, shift = ctl.update(np.ones(6), flows, loading)
        assert shift[rep] == pytest.approx(-B.PAR_STEP_DEG)
        assert shift[rep + 1] == shift[rep]           # mirror row identical

    def test_below_target_holds(self):
        ctl, ei, _ = make_controller()
        flows = np.zeros(ei.shape[1]); flows[4] = 100.0
        loading = np.zeros(ei.shape[1]); loading[4] = 0.5
        _, shift = ctl.update(np.ones(6), flows, loading)
        assert shift[4] == 0.0

    def test_angle_clamped(self):
        ctl, ei, _ = make_controller()
        flows = np.zeros(ei.shape[1]); flows[4] = 100.0
        loading = np.zeros(ei.shape[1]); loading[4] = 2.0
        for _ in range(50):
            ctl.update(np.ones(6), flows, loading)
        assert ctl.tap_angle[4] == B.PAR_ANGLE_MIN_DEG

    def test_reset_clears_state(self):
        ctl, ei, _ = make_controller()
        volts = np.ones(6); volts[ei[1, 0]] = 0.9
        flows = np.zeros(ei.shape[1]); flows[4] = 100.0
        loading = np.zeros(ei.shape[1]); loading[4] = 1.5
        ctl.update(volts, flows, loading)
        ctl.reset()
        assert (ctl.tap_position == 0).all()
        assert (ctl.tap_angle == 0.0).all()


class TestNonDevices:
    def test_plain_lines_never_move(self):
        ctl, ei, bt = make_controller()
        volts = np.zeros(6)                     # extreme conditions
        flows = np.full(ei.shape[1], 999.0)
        loading = np.full(ei.shape[1], 9.9)
        ratio, shift = ctl.update(volts, flows, loading)
        lines = bt == B.TYPE_LINE
        assert (ratio[lines] == 1.0).all()
        assert (shift[lines] == 0.0).all()
