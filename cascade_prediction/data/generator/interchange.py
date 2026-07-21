"""
Inter-area Exchange & Storage Module (Phase 5 — MISO data gap plan)
==================================================================
Tie lines to a synthetic neighbouring RC, net actual interchange, dynamic
interchange schedules (ACE), and energy storage resources (ESR).

[MISO RTO-SPEC-006: "Tie-line flows & net actual interchange",
 "Dynamic schedules (interchange MW)",
 "Energy storage resources: SOC, MW, MVAr"]

Design notes
------------
- TIE LINES are ordinary branches *designated* as boundary ties. There is no
  explicit neighbouring network: net actual interchange is the signed sum of
  tie flows, taken on one fixed representative direction per physical tie so
  the sum is orientation-stable. Ties are picked farthest from the network
  centroid, because real ties sit at the edge of a footprint.
- The SCHEDULE is a block profile (hold for a dwell, then re-step), not smooth
  drift — interchange is scheduled in blocks, and a step change is a genuine
  disturbance. ACE = schedule - actual is system-wide, NOT per-node; it is
  broadcast into SCADA exactly as current_frequency already is.
- STORAGE is a separate per-node attribute, never a new node_types value:
  node_types == 1 (generator) is load-bearing in the power flow, the LTC
  regulated-bus choice, RAS targeting and dispatch. A bus is storage iff
  storage_power_mw > 0.
- The storage controller acts on the PREVIOUS timestep's frequency, mirroring
  the tap/shunt controllers, and integrates SOC with a hard clamp to
  [0, capacity] so the energy budget can never go non-physical.
- Torch-free (numpy only) so backend services and tests can import it.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .config import Settings


# ---------------------------------------------------------------------------
# Assignment (static — rides in the topology block)
# ---------------------------------------------------------------------------
def assign_tie_lines(edge_index: np.ndarray,
                     positions: np.ndarray,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Designate physical branches as boundary tie lines.

    Ties are the branches whose midpoint lies farthest from the network
    centroid — the geographic boundary of the footprint. Both directed rows of
    a physical branch receive the same flag (mirror invariant).

    Returns:
        int array [E]: 1 where the directed edge belongs to a tie line, else 0.
    """
    rng = rng or np.random.default_rng()
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    pos = np.asarray(positions, dtype=np.float64)
    num_edges = src.shape[0]
    num_nodes = pos.shape[0]

    canon = np.minimum(src, dst) * num_nodes + np.maximum(src, dst)
    uniq = np.unique(canon)

    centroid = pos.mean(axis=0)
    # Distance of each physical branch's midpoint from the centroid.
    rep_idx = np.array([int(np.nonzero(canon == k)[0][0]) for k in uniq])
    mid = 0.5 * (pos[src[rep_idx]] + pos[dst[rep_idx]])
    dist = np.linalg.norm(mid - centroid, axis=1)

    n_ties = int(min(Settings.Interchange.N_TIE_LINES, len(uniq)))
    tie_pairs = set(int(uniq[i]) for i in np.argsort(-dist)[:n_ties])

    is_tie = np.zeros(num_edges, dtype=np.int64)
    for i in range(num_edges):
        if int(canon[i]) in tie_pairs:
            is_tie[i] = 1
    return is_tie


def tie_representative_rows(edge_index: np.ndarray,
                            is_tie_line: np.ndarray) -> np.ndarray:
    """
    One directed row per physical tie, for orientation-stable flow summation.

    Taking BOTH mirrored rows would sum to zero (they carry equal and opposite
    flow), so interchange must be computed from exactly one row per tie.

    Returns:
        int array [n_ties]: directed edge indices (lowest index per pair).
    """
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    is_tie = np.asarray(is_tie_line, dtype=np.int64)
    num_nodes = int(max(src.max(), dst.max())) + 1 if src.size else 0

    reps: Dict[int, int] = {}
    for i in np.nonzero(is_tie)[0]:
        i = int(i)
        k = int(min(src[i], dst[i]) * num_nodes + max(src[i], dst[i]))
        if k not in reps:
            reps[k] = i
    return np.array(sorted(reps.values()), dtype=np.int64)


def assign_storage_nodes(node_types: np.ndarray,
                         rng: Optional[np.random.Generator] = None
                         ) -> Dict[str, np.ndarray]:
    """
    Fit a subset of buses with energy storage resources.

    Storage is deliberately NOT a node_types value — see module docstring.
    A bus is a storage bus iff storage_power_mw > 0.

    Returns dict of per-node arrays (0 on non-storage buses):
        storage_power_mw, storage_capacity_mwh, storage_qmax_mvar
    """
    rng = rng or np.random.default_rng()
    S = Settings.Storage
    n = int(np.asarray(node_types).shape[0])

    power = np.zeros(n, dtype=np.float64)
    capacity = np.zeros(n, dtype=np.float64)
    qmax = np.zeros(n, dtype=np.float64)

    k = int(min(S.N_STORAGE_NODES, n))
    if k > 0:
        chosen = rng.choice(n, size=k, replace=False)
        p = rng.uniform(S.POWER_MW_MIN, S.POWER_MW_MAX, k)
        dur = rng.uniform(S.DURATION_H_MIN, S.DURATION_H_MAX, k)
        power[chosen] = p
        capacity[chosen] = p * dur
        qmax[chosen] = p * S.QMAX_FRACTION

    return {
        'storage_power_mw': power,
        'storage_capacity_mwh': capacity,
        'storage_qmax_mvar': qmax,
    }


# ---------------------------------------------------------------------------
# Runtime controllers (dynamic state — reset per scenario, NOT in the block)
# ---------------------------------------------------------------------------
class InterchangeScheduler:
    """
    Emits the scheduled interchange (MW, positive = export) as a block
    profile: hold for a dwell of timesteps, then re-draw.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.schedule_mw = 0.0
        self._hold = 0
        self.reset()

    def reset(self) -> None:
        C = Settings.Interchange
        self.schedule_mw = float(
            C.SCHEDULE_BASE_MW
            + self.rng.uniform(-C.SCHEDULE_SPAN_MW, C.SCHEDULE_SPAN_MW))
        self._hold = int(self.rng.integers(
            C.SCHEDULE_DWELL_MIN, C.SCHEDULE_DWELL_MAX + 1))

    def step(self) -> float:
        """Advance one timestep; re-steps the block when the dwell expires."""
        C = Settings.Interchange
        self._hold -= 1
        if self._hold <= 0:
            self.schedule_mw = float(
                C.SCHEDULE_BASE_MW
                + self.rng.uniform(-C.SCHEDULE_SPAN_MW, C.SCHEDULE_SPAN_MW))
            self._hold = int(self.rng.integers(
                C.SCHEDULE_DWELL_MIN, C.SCHEDULE_DWELL_MAX + 1))
        return self.schedule_mw


def net_actual_interchange(line_flows: np.ndarray,
                           tie_rows: np.ndarray) -> float:
    """
    Net actual interchange (MW) = signed sum of tie flows on the
    representative rows. Validation item (a) in the plan.
    """
    if tie_rows.size == 0:
        return 0.0
    return float(np.asarray(line_flows, dtype=np.float64)[tie_rows].sum())


class StorageController:
    """
    Energy storage: frequency-droop power response with SOC integration.

    Sign convention: positive flow_mw = DISCHARGING (injecting MW into the
    grid, depleting SOC). Frequency below nominal -> discharge to support it.

    SOC is hard-clamped to [0, capacity] and the delivered power is clipped to
    what the remaining energy actually allows, so the energy budget can never
    go non-physical (validation item (b) in the plan).
    """

    def __init__(self,
                 storage_power_mw: np.ndarray,
                 storage_capacity_mwh: np.ndarray,
                 storage_qmax_mvar: np.ndarray):
        self.power_mw = np.asarray(storage_power_mw, dtype=np.float64)
        self.capacity_mwh = np.asarray(storage_capacity_mwh, dtype=np.float64)
        self.qmax_mvar = np.asarray(storage_qmax_mvar, dtype=np.float64)
        self.is_storage = self.power_mw > 0.0
        self.num_nodes = self.power_mw.shape[0]
        self.soc_mwh = np.zeros(self.num_nodes, dtype=np.float64)
        self.flow_mw = np.zeros(self.num_nodes, dtype=np.float64)
        self.flow_mvar = np.zeros(self.num_nodes, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset SOC to its starting fraction (call per scenario)."""
        self.soc_mwh = self.capacity_mwh * Settings.Storage.SOC_INIT_FRACTION
        self.flow_mw[:] = 0.0
        self.flow_mvar[:] = 0.0

    def update(self,
               frequency_hz: float,
               dt_minutes: float,
               failed_nodes: Optional[set] = None,
               system_base_mw: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one control step from the PREVIOUS timestep's frequency.

        Args:
            frequency_hz:   system frequency last step
            dt_minutes:     timestep length (minutes)
            failed_nodes:   buses out of service — their ESR contributes nothing
            system_base_mw: total load, used to bound the frequency-loop gain.
                            When omitted the cap is skipped (unit tests).

        Returns:
            (flow_mw, flow_mvar) per node; positive MW = discharging.
        """
        S = Settings.Storage
        f_nom = Settings.Frequency.BASE_FREQUENCY
        dev = float(f_nom) - float(frequency_hz)     # >0 when frequency is LOW
        if abs(dev) < S.DEADBAND_HZ:
            dev = 0.0

        # Droop request, saturated at the power rating.
        req = np.where(
            self.is_storage,
            np.clip(S.DROOP_MW_PER_HZ * dev, -self.power_mw, self.power_mw),
            0.0,
        )

        # --- Frequency-loop stability cap ---------------------------------
        # Storage MW joins the generation sum the frequency model integrates,
        # so this droop closes a feedback loop whose gain is
        #     (BASE_FREQUENCY / (D_TOTAL * system_base)) * sum(response)
        # Scale the aggregate response down so that gain never exceeds
        # MAX_LOOP_GAIN. Without this the loop diverges in a single step and
        # collapses the grid on underfrequency — see StorageConfig.
        if system_base_mw and system_base_mw > 0:
            sens_hz_per_mw = (Settings.Frequency.BASE_FREQUENCY
                              / (Settings.Frequency.D_TOTAL
                                 * float(system_base_mw)))
            max_aggregate = S.MAX_LOOP_GAIN / max(sens_hz_per_mw, 1e-12)
            aggregate = float(np.abs(req).sum())
            if aggregate > max_aggregate:
                req = req * (max_aggregate / aggregate)

        dt_h = float(dt_minutes) / 60.0
        # Energy limits: cannot discharge below 0 or charge above capacity.
        # Discharge (req > 0) draws soc; charge (req < 0) adds soc * efficiency.
        max_discharge_mw = np.where(dt_h > 0, self.soc_mwh / max(dt_h, 1e-9), 0.0)
        head_room = np.maximum(self.capacity_mwh - self.soc_mwh, 0.0)
        max_charge_mw = np.where(
            dt_h > 0, head_room / max(dt_h * S.CHARGE_EFFICIENCY, 1e-9), 0.0)

        flow = np.where(req >= 0,
                        np.minimum(req, max_discharge_mw),
                        -np.minimum(-req, max_charge_mw))

        if failed_nodes:
            idx = np.fromiter(failed_nodes, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < self.num_nodes)]
            flow[idx] = 0.0

        # Integrate SOC: discharging depletes 1:1, charging credits at
        # round-trip efficiency.
        delta = np.where(flow >= 0, -flow * dt_h,
                         -flow * dt_h * S.CHARGE_EFFICIENCY)
        self.soc_mwh = np.clip(self.soc_mwh + delta, 0.0, self.capacity_mwh)

        # Reactive output: inverter-coupled, held proportional to the MW
        # utilisation and capped at Qmax.
        util = np.divide(flow, np.maximum(self.power_mw, 1e-9),
                         out=np.zeros_like(flow), where=self.is_storage)
        self.flow_mw = flow
        self.flow_mvar = np.clip(util * self.qmax_mvar,
                                 -self.qmax_mvar, self.qmax_mvar)
        return self.flow_mw, self.flow_mvar

    def soc_fraction(self) -> np.ndarray:
        """SOC as a 0-1 fraction of capacity (0 on non-storage buses)."""
        return np.divide(self.soc_mwh, self.capacity_mwh,
                         out=np.zeros_like(self.soc_mwh),
                         where=self.capacity_mwh > 0)
