"""
Tests for Phase 3 — reactive support & voltage regulation (MISO data gap plan)
==============================================================================
Covers assign_shunt_devices (placement rules, AVR setpoints, determinism)
and ShuntController (cap/reactor hysteresis, SVC droop + saturation,
dead-bus behavior, reset). Torch-free.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.shunts import (
    ShuntController, assign_shunt_devices,
)
from cascade_prediction.data.generator.config import Settings

S = Settings.Shunt


def node_types(n_load=20, n_gen=5, n_sub=10):
    return np.array([0] * n_load + [1] * n_gen + [2] * n_sub)


class TestAssignment:
    def test_devices_on_correct_bus_types(self):
        nt = node_types()
        d = assign_shunt_devices(nt, rng=np.random.default_rng(3))
        assert (d['shunt_cap_mvar'][nt != 0] == 0).all(), "caps only on load buses"
        assert (d['shunt_reactor_mvar'][nt != 2] == 0).all(), "reactors only on subs"
        assert (d['avr_setpoint'][nt != 1] == 0).all(), "AVR only on generators"
        assert (d['avr_setpoint'][nt == 1] >= S.AVR_SETPOINT_MIN_PU).all()
        assert (d['avr_setpoint'][nt == 1] <= S.AVR_SETPOINT_MAX_PU).all()

    def test_fractions(self):
        nt = node_types(n_load=40, n_gen=5, n_sub=20)
        d = assign_shunt_devices(nt, rng=np.random.default_rng(3))
        assert (d['shunt_cap_mvar'] > 0).sum() == round(40 * S.CAP_BANK_FRACTION)
        assert (d['shunt_reactor_mvar'] > 0).sum() == round(20 * S.REACTOR_FRACTION)
        assert (d['svc_qmax_mvar'] > 0).sum() == round(65 * S.SVC_FRACTION)

    def test_deterministic(self):
        nt = node_types()
        a = assign_shunt_devices(nt, rng=np.random.default_rng(9))
        b = assign_shunt_devices(nt, rng=np.random.default_rng(9))
        for k in a:
            assert (a[k] == b[k]).all(), k


def make_controller(n=5):
    cap = np.zeros(n); cap[0] = 40.0
    rea = np.zeros(n); rea[1] = 30.0
    svc = np.zeros(n); svc[2] = 50.0
    return ShuntController(cap, rea, svc)


class TestCapacitorBank:
    def test_switches_in_on_low_voltage(self):
        c = make_controller()
        v = np.ones(5); v[0] = 0.96
        q = c.update(v)
        assert c.cap_status[0] == 1
        assert q[0] == pytest.approx(40.0)

    def test_hysteresis_holds_in_band(self):
        c = make_controller()
        v = np.ones(5); v[0] = 0.96
        c.update(v)                        # switched in
        v[0] = 1.00                        # recovered but below OUT threshold
        q = c.update(v)
        assert c.cap_status[0] == 1, "stays in until voltage exceeds OUT level"
        v[0] = 1.03
        c.update(v)
        assert c.cap_status[0] == 0, "switches out above OUT level"

    def test_dead_bus_disconnects(self):
        c = make_controller()
        v = np.ones(5); v[0] = 0.96
        c.update(v)
        v[0] = 0.0                         # bus de-energized
        q = c.update(v)
        assert c.cap_status[0] == 0
        assert q[0] == 0.0


class TestReactorBank:
    def test_switches_in_on_high_voltage_and_absorbs(self):
        c = make_controller()
        v = np.ones(5); v[1] = 1.04
        q = c.update(v)
        assert c.reactor_status[1] == 1
        assert q[1] == pytest.approx(-30.0), "reactor ABSORBS Q (negative injection)"

    def test_switches_out_when_voltage_falls(self):
        c = make_controller()
        v = np.ones(5); v[1] = 1.04
        c.update(v)
        v[1] = 0.97
        c.update(v)
        assert c.reactor_status[1] == 0


class TestSVC:
    def test_droop_response_sign_and_magnitude(self):
        c = make_controller()
        v = np.ones(5); v[2] = 0.99        # 0.01 pu below setpoint
        q = c.update(v)
        assert q[2] == pytest.approx(0.01 * S.SVC_GAIN_MVAR_PER_PU)

    def test_saturates_at_qmax(self):
        c = make_controller()
        v = np.ones(5); v[2] = 0.80        # far below → demand >> Qmax
        q = c.update(v)
        assert q[2] == pytest.approx(50.0), "clipped at the SVC's Qmax"

    def test_absorbs_when_voltage_high(self):
        c = make_controller()
        v = np.ones(5); v[2] = 1.10
        q = c.update(v)
        assert q[2] == pytest.approx(-50.0)

    def test_no_device_no_q(self):
        c = make_controller()
        v = np.zeros(5) + 0.9              # every bus sagging
        q = c.update(v)
        assert q[3] == 0.0 and q[4] == 0.0


class TestReset:
    def test_reset_clears_everything(self):
        c = make_controller()
        v = np.ones(5); v[0] = 0.9; v[1] = 1.1; v[2] = 0.9
        c.update(v)
        c.reset()
        assert (c.cap_status == 0).all()
        assert (c.reactor_status == 0).all()
        assert (c.svc_q == 0).all()
        assert (c.update(np.ones(5)) == 0).all()
