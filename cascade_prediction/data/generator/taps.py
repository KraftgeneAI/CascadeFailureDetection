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

    LTC: regulates the LOAD-SIDE bus voltage toward LTC_V_SETPOINT_PU.
    Voltage below the setpoint band → raise tap (raises regulated-side
    ratio); above → lower. De-energized buses (V < DEAD_BUS) hold their tap
    (a real LTC can't regulate a dead bus).

    Regulated-bus selection matters. A generator (PV/slack) bus has its
    voltage PINNED by PyPSA at v_mag_pu_set, so a tap has no authority over
    it whatsoever. Regulating such a bus makes the controller see a constant
    out-of-band error and ratchet one step every timestep until it clips —
    which silently reroutes flow onto neighbouring corridors and trips lines
    at constant load. Real LTCs regulate the load side for exactly this
    reason, so we pick the non-generator end and disable control outright
    when both ends are voltage-regulated.

    Anti-windup backs that up: if the tap has moved but the regulated
    voltage has not responded, the device has no authority in practice and
    stepping is blocked (real LTC controls have equivalent runback/blocking
    logic).

    PAR: relieves its own overload. Above PAR_LOADING_TARGET apparent
    loading, step the phase shift AGAINST the active-flow direction
    (DC approximation: P ≈ (Δθ + shift)/x, so reducing the shift in the flow
    direction reduces through-flow).
    """

    def __init__(self, edge_index: np.ndarray, branch_type: np.ndarray,
                 node_types: Optional[np.ndarray] = None):
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
        # orientation defines the flow sign reference.
        self._groups = [np.array(v, dtype=np.int64) for v in groups.values()]
        self._rep = np.array([g.min() for g in self._groups], dtype=np.int64)

        # --- Regulated bus: prefer the LOAD-side (non-generator) end --------
        # node_types == 1 marks a generator bus (PV/slack) whose voltage is
        # pinned by its AVR setpoint and cannot be moved by a tap.
        _is_gen = (
            np.asarray(node_types, dtype=np.int64) == 1
            if node_types is not None
            else np.zeros(num_nodes, dtype=bool)
        )
        rep_src = src[self._rep]
        rep_dst = dst[self._rep]
        reg = np.where(_is_gen[rep_dst] & ~_is_gen[rep_src], rep_src, rep_dst)
        self._rep_dst = reg
        # No authority when BOTH ends are generator buses — hold the tap.
        self._ltc_blocked = _is_gen[rep_src] & _is_gen[rep_dst]

        self.tap_position = np.zeros(self.num_edges, dtype=np.int64)   # LTC steps
        self.tap_angle = np.zeros(self.num_edges, dtype=np.float64)    # PAR degrees

        # Anti-windup state, per GROUP (index-aligned with self._groups)
        self._prev_reg_v = np.full(len(self._groups), np.nan)
        self._stall_count = np.zeros(len(self._groups), dtype=np.int64)

    def reset(self) -> None:
        """Zero all tap state (call at the start of each scenario)."""
        self.tap_position[:] = 0
        self.tap_angle[:] = 0.0
        self._prev_reg_v[:] = np.nan
        self._stall_count[:] = 0

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
        for gi, (g, rep, reg_bus) in enumerate(
                zip(self._groups, self._rep, self._rep_dst)):
            btype = self.branch_type[rep]

            if btype == B.TYPE_TRANSFORMER:
                if self._ltc_blocked[gi]:
                    continue  # both ends voltage-regulated — no authority
                v = float(prev_voltages[reg_bus])
                if v < B.LTC_DEAD_BUS_V_PU:
                    continue  # dead/islanded bus — hold tap
                tap = int(self.tap_position[rep])

                # Anti-windup: if the tap moved last step but the regulated
                # voltage barely responded, this device has no practical
                # authority — block it instead of ratcheting to the limit.
                v_prev = self._prev_reg_v[gi]
                if not np.isnan(v_prev) and tap != 0:
                    if abs(v - v_prev) < B.LTC_MIN_V_RESPONSE_PU:
                        self._stall_count[gi] += 1
                    else:
                        self._stall_count[gi] = 0
                self._prev_reg_v[gi] = v
                if self._stall_count[gi] >= B.LTC_STALL_STEPS:
                    continue  # blocked — voltage is not responding to taps

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
