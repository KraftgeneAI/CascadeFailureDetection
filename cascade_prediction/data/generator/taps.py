"""
Tap Devices Module (Phase 2 — MISO data gap plan)
=================================================
LTC transformer and phase-angle regulator (PAR) tap controllers.

[MISO RTO-SPEC-006: "Transformer flows + LTC tap positions" and
 "Phase shifter (PAR) flows + tap positions"]

Design notes
------------
- Branches are stored as mirrored directed pairs; tap state is kept per
  directed edge but always identical across a pair (updates run on the
  representative edge of each undirected group and are broadcast).
- Controllers act on the PREVIOUS timestep's power-flow results (voltages,
  flows, loadings) — one mechanical step per timestep. This physical delay
  is exactly the "LTC tap creep during voltage decay" precursor signal the
  cascade model should learn.
- Torch-free (numpy only) so backend services and tests can use it without
  the ML stack.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import Settings


def assign_branch_types(edge_index: np.ndarray,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Assign a device type to every directed edge, mirror-consistent.

    A fraction of PHYSICAL (undirected) branches become LTC transformers and
    a smaller fraction become PARs; the rest stay plain lines.

    Returns:
        int array [num_edges]: Settings.Branch.TYPE_LINE / TYPE_TRANSFORMER /
        TYPE_PAR, identical on both directed rows of each pair.
    """
    rng = rng or np.random.default_rng()
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    num_edges = src.shape[0]
    num_nodes = int(max(src.max(), dst.max())) + 1
    canon = np.minimum(src, dst) * num_nodes + np.maximum(src, dst)
    uniq_keys = np.unique(canon)
    n_pairs = len(uniq_keys)

    n_trafo = int(round(n_pairs * Settings.Branch.TRANSFORMER_FRACTION))
    n_par = int(round(n_pairs * Settings.Branch.PAR_FRACTION))
    chosen = rng.choice(n_pairs, size=n_trafo + n_par, replace=False)
    pair_type = np.full(n_pairs, Settings.Branch.TYPE_LINE, dtype=np.int64)
    pair_type[chosen[:n_trafo]] = Settings.Branch.TYPE_TRANSFORMER
    pair_type[chosen[n_trafo:]] = Settings.Branch.TYPE_PAR

    key_to_type = {int(k): int(t) for k, t in zip(uniq_keys, pair_type)}
    return np.array([key_to_type[int(k)] for k in canon], dtype=np.int64)


class TapController:
    """
    Steps LTC taps and PAR angles once per timestep from previous-step
    power-flow results, mirror-consistently.

    LTC: regulates the representative edge's DESTINATION bus voltage toward
    LTC_V_SETPOINT_PU. Voltage below the setpoint band → raise tap (raises
    regulated-side ratio); above → lower. De-energized buses (V < DEAD_BUS)
    hold their tap (a real LTC can't regulate a dead bus).

    PAR: relieves its own overload. Above PAR_LOADING_TARGET apparent
    loading, step the phase shift AGAINST the active-flow direction
    (DC approximation: P ≈ (Δθ + shift)/x, so reducing the shift in the flow
    direction reduces through-flow).
    """

    def __init__(self, edge_index: np.ndarray, branch_type: np.ndarray):
        self.branch_type = np.asarray(branch_type, dtype=np.int64)
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        self.num_edges = src.shape[0]
        num_nodes = int(max(src.max(), dst.max())) + 1
        canon = np.minimum(src, dst) * num_nodes + np.maximum(src, dst)

        groups: Dict[int, List[int]] = {}
        for i, k in enumerate(canon):
            groups.setdefault(int(k), []).append(i)
        # Representative directed edge per group = lowest index; its (src,dst)
        # orientation defines the regulated bus (dst) and flow sign reference.
        self._groups = [np.array(v, dtype=np.int64) for v in groups.values()]
        self._rep = np.array([g.min() for g in self._groups], dtype=np.int64)
        self._rep_dst = dst[self._rep]

        self.tap_position = np.zeros(self.num_edges, dtype=np.int64)   # LTC steps
        self.tap_angle = np.zeros(self.num_edges, dtype=np.float64)    # PAR degrees

    def reset(self) -> None:
        """Zero all tap state (call at the start of each scenario)."""
        self.tap_position[:] = 0
        self.tap_angle[:] = 0.0

    def update(self,
               prev_voltages: np.ndarray,
               prev_line_flows: np.ndarray,
               prev_loading: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance every device one control step from previous-step results.

        Returns:
            (tap_ratio, phase_shift_deg) per directed edge:
            tap_ratio = 1 + tap_position * LTC_STEP_PU  (1.0 for non-LTC),
            phase_shift_deg = tap_angle                  (0.0 for non-PAR).
        """
        B = Settings.Branch
        for g, rep, reg_bus in zip(self._groups, self._rep, self._rep_dst):
            btype = self.branch_type[rep]

            if btype == B.TYPE_TRANSFORMER:
                v = float(prev_voltages[reg_bus])
                if v < B.LTC_DEAD_BUS_V_PU:
                    continue  # dead/islanded bus — hold tap
                tap = int(self.tap_position[rep])
                if v < B.LTC_V_SETPOINT_PU - B.LTC_DEADBAND_PU and tap < B.LTC_TAP_MAX:
                    tap += 1
                elif v > B.LTC_V_SETPOINT_PU + B.LTC_DEADBAND_PU and tap > B.LTC_TAP_MIN:
                    tap -= 1
                self.tap_position[g] = tap

            elif btype == B.TYPE_PAR:
                if float(prev_loading[rep]) > B.PAR_LOADING_TARGET:
                    flow = float(prev_line_flows[rep])
                    if flow != 0.0:
                        angle = self.tap_angle[rep] - np.sign(flow) * B.PAR_STEP_DEG
                        self.tap_angle[g] = float(np.clip(
                            angle, B.PAR_ANGLE_MIN_DEG, B.PAR_ANGLE_MAX_DEG))

        tap_ratio = 1.0 + self.tap_position * B.LTC_STEP_PU
        return tap_ratio, self.tap_angle.copy()
