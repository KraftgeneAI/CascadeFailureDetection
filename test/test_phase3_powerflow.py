"""
Phase 3 integration tests — reactive support inside the AC power flow.
Requires PyPSA (skipped if unavailable).
"""

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")

from cascade_prediction.data.generator.physics import PowerFlowSimulator


def build_sim(avr_setpoint=None):
    """4-bus meshed grid: gen at bus 0, loads elsewhere (as in Phase 2 tests)."""
    pairs = [(0, 1), (1, 2), (0, 3), (3, 2)]
    src = [p[0] for p in pairs] + [p[1] for p in pairs]
    dst = [p[1] for p in pairs] + [p[0] for p in pairs]
    edge_index = np.array([src, dst], dtype=np.int64)

    def mirrored(vals):
        return np.array(list(vals) + list(vals), dtype=float)

    return PowerFlowSimulator(
        num_nodes=4,
        edge_index=edge_index,
        positions=np.array([[0, 0], [1, 0], [2, 0], [1, 1]], dtype=float),
        node_types=np.array([1, 0, 0, 0]),
        gen_capacity=np.array([300.0, 0, 0, 0]),
        line_reactance=mirrored([10.0] * 4),
        line_resistance=mirrored([1.0] * 4),
        line_susceptance=mirrored([1e-6] * 4),
        line_conductance=mirrored([0.0] * 4),
        thermal_limits=mirrored([200.0] * 4),
        avr_setpoint=avr_setpoint,
    )


def test_shunt_q_injection_raises_voltage():
    """Plan validation: capacitive injection at a bus props its voltage up."""
    sim = build_sim()
    generation = np.array([150.0, 0, 0, 0])
    load = np.array([0.0, 40.0, 70.0, 40.0])

    _, _, _, _, _, ok0 = sim.compute_power_flow(generation, load)
    assert ok0
    v_base = sim.network.buses_t.v_mag_pu.loc["now"].copy()

    shunt_q = np.zeros(4)
    shunt_q[2] = 40.0                       # 40 MVar capacitive at bus 2
    _, _, _, _, _, ok1 = sim.compute_power_flow(generation, load, shunt_q=shunt_q)
    assert ok1
    v_shunt = sim.network.buses_t.v_mag_pu.loc["now"]

    assert float(v_shunt["bus_2"]) > float(v_base["bus_2"]) + 1e-4, \
        "capacitive shunt injection should raise the local bus voltage"


def test_reactor_absorption_lowers_voltage():
    sim = build_sim()
    generation = np.array([150.0, 0, 0, 0])
    load = np.array([0.0, 40.0, 70.0, 40.0])

    _, _, _, _, _, _ = sim.compute_power_flow(generation, load)
    v_base = sim.network.buses_t.v_mag_pu.loc["now"].copy()

    shunt_q = np.zeros(4)
    shunt_q[2] = -40.0                      # inductive absorption at bus 2
    _, _, _, _, _, ok = sim.compute_power_flow(generation, load, shunt_q=shunt_q)
    assert ok
    v_shunt = sim.network.buses_t.v_mag_pu.loc["now"]
    assert float(v_shunt["bus_2"]) < float(v_base["bus_2"]) - 1e-4


def test_avr_setpoint_respected_on_pv_generator():
    """A PV generator regulates to its telemetered AVR setpoint."""
    # Two generators so bus 1 gets a PV (bus 0 is always slack)
    pairs = [(0, 1), (1, 2), (0, 3), (3, 2)]
    src = [p[0] for p in pairs] + [p[1] for p in pairs]
    dst = [p[1] for p in pairs] + [p[0] for p in pairs]
    edge_index = np.array([src, dst], dtype=np.int64)

    def mirrored(vals):
        return np.array(list(vals) + list(vals), dtype=float)

    avr = np.array([1.0, 1.025, 0.0, 0.0])
    sim = PowerFlowSimulator(
        num_nodes=4,
        edge_index=edge_index,
        positions=np.array([[0, 0], [1, 0], [2, 0], [1, 1]], dtype=float),
        node_types=np.array([1, 1, 0, 0]),      # gens at 0 (slack) and 1 (PV)
        gen_capacity=np.array([200.0, 100.0, 0, 0]),
        line_reactance=mirrored([10.0] * 4),
        line_resistance=mirrored([1.0] * 4),
        line_susceptance=mirrored([1e-6] * 4),
        line_conductance=mirrored([0.0] * 4),
        thermal_limits=mirrored([200.0] * 4),
        avr_setpoint=avr,
    )
    assert float(sim.network.generators.at["gen_1", "v_set"]) == pytest.approx(1.025)

    generation = np.array([80.0, 60.0, 0, 0])
    load = np.array([0.0, 0.0, 80.0, 50.0])
    _, _, _, _, _, ok = sim.compute_power_flow(generation, load)
    assert ok
    v1 = float(sim.network.buses_t.v_mag_pu.loc["now", "bus_1"])
    assert v1 == pytest.approx(1.025, abs=5e-3), \
        "PV bus should hold (near) its AVR setpoint"
