"""
Builders for scenarios that satisfy the data contract, for tests.

Values are drawn inside every column's expected range (or from its category
set), so a freshly built scenario has zero structural errors and zero range
violations. Tests then break one thing at a time.
"""

import numpy as np

from cascade_prediction import contract as C


def _fill(spec, rows, rng):
    out = np.zeros((rows, spec.width), dtype=np.float32)
    for j, col in enumerate(spec.columns):
        if col.categories:
            out[:, j] = rng.choice(np.asarray(col.categories, dtype=np.float32), size=rows)
        elif col.valid_range:
            lo, hi = col.valid_range
            lo = -1.0 if lo is None else lo
            hi = lo + 2.0 if hi is None else hi
            # stay well inside the band
            out[:, j] = rng.uniform(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo), size=rows)
        else:
            out[:, j] = rng.standard_normal(rows)
        if col.scope == "broadcast":
            out[:, j] = out[0, j]
    return out


def make_timestep(num_nodes=12, num_edges=20, n_ras=6, rng=None):
    rng = rng or np.random.default_rng(0)
    ts = {
        C.SCADA.key: _fill(C.SCADA, num_nodes, rng),
        C.PMU.key: _fill(C.PMU, num_nodes, rng),
        C.EQUIPMENT.key: _fill(C.EQUIPMENT, num_nodes, rng),
        C.EDGE_ATTR.key: _fill(C.EDGE_ATTR, num_edges, rng),
        C.RAS_STATUS.key: _fill(C.RAS_STATUS, n_ras, rng),
        C.NODE_LABELS.key: np.zeros(num_nodes, np.float32),
        C.CASCADE_TIMING.key: -np.ones(num_nodes, np.float32),
        C.POWER_INJECTION.key: rng.standard_normal(num_nodes).astype(np.float32),
        C.REACTIVE_INJECTION.key: rng.standard_normal(num_nodes).astype(np.float32),
        C.CONDUCTANCE.key: np.zeros(num_edges, np.float32),
        C.SUSCEPTANCE.key: np.zeros(num_edges, np.float32),
        C.THERMAL_LIMITS.key: rng.uniform(50, 500, num_edges).astype(np.float32),
        "cascade_events": [],
    }
    return ts


def make_scenario(num_nodes=12, num_edges=20, T=3, stamp=True, seed=0):
    rng = np.random.default_rng(seed)
    edge_index = rng.integers(0, num_nodes, size=(2, num_edges)).astype(np.int64)
    meta = {"is_cascade": False, "num_nodes": num_nodes, "num_edges": num_edges,
            "base_mva": 1000.0, "cascade_start_time": -1}
    if stamp:
        meta.update(C.contract_stamp())
    return {
        "sequence": [make_timestep(num_nodes, num_edges, rng=rng) for _ in range(T)],
        "edge_index": edge_index,
        "metadata": meta,
    }
