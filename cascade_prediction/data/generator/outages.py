"""
Scheduled Outages & De-rates Module (Phase 6 — MISO data gap plan)
=================================================================
Pre-scheduled generator and transmission outages, plus time-varying rating
de-rates driven by both maintenance schedules and ambient temperature.

[MISO RTO-SPEC-006: "Transmission/generation outage schedules + de-rates"]

Design notes
------------
- PLANNED != FORCED. A scheduled outage is an operator decision known in
  advance; a forced trip is the failure the model exists to predict. They are
  distinguished in telemetry by the TERNARY branch_status
  (1 closed / 0 forced-open / -1 planned-open) and, critically, a planned
  outage NEVER enters the failure labels — exactly like a RAS action.
- A scheduled GENERATOR outage zeroes the unit's output but leaves its BUS
  energised. The bus is not failed, its load is still served, and it must not
  appear in node_labels.
- The schedule is SCENARIO state, not grid parameterization, so it is drawn
  per scenario and deliberately kept OUT of the topology block (which
  describes what the grid IS, not what is happening to it).
- De-rates combine two independent drivers, multiplicatively:
      derate(t) = scheduled_event_factor(t) * ambient_factor(ambient_t)
  Ambient matters physically: a thermal rating is defined at a reference
  ambient, so a hot afternoon genuinely eats headroom.
- Torch-free (numpy only).
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from .config import Settings


class OutageSchedule:
    """
    Per-scenario schedule of planned outages and de-rate events.

    All queries are by timestep; the schedule itself is immutable once drawn.
    """

    def __init__(self,
                 edge_index: np.ndarray,
                 node_types: np.ndarray,
                 sequence_length: int,
                 rng: Optional[np.random.Generator] = None,
                 thermal_limits: Optional[np.ndarray] = None,
                 enabled: bool = True):
        # enabled=False builds a live-but-empty schedule: the query interface
        # works (returns no outages / no de-rate events, only the ambient
        # factor), so callers need no special-casing. The simulator passes
        # Settings.Outage.ENABLED here; unit tests default to True to exercise
        # the drawing logic directly.
        self.enabled = bool(enabled)
        self.rng = rng or np.random.default_rng()
        C = Settings.Outage
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        self.num_edges = int(src.shape[0])
        self.num_nodes = int(np.asarray(node_types).shape[0])
        self.sequence_length = int(sequence_length)

        # Group directed rows into physical branches so an outage takes both
        # rows of a pair (mirror invariant).
        canon = np.minimum(src, dst) * self.num_nodes + np.maximum(src, dst)
        groups: Dict[int, List[int]] = {}
        for i, k in enumerate(canon):
            groups.setdefault(int(k), []).append(i)
        self._pair_rows = [np.array(v, dtype=np.int64) for v in groups.values()]

        # --- Outage screening -------------------------------------------
        # Real outage coordination STUDIES a proposed outage and rejects it if
        # removing that branch creates a thermal violation. Drawing uniformly
        # at random does not: a scheduled outage can land on a backbone
        # corridor whose loss dumps its flow onto a floor-rated parallel path
        # and drives it to 6x rating within one timestep — observed exactly
        # that, and it detonated the grid from a PLANNED event.
        # Ratings were calibrated from base-case flow, so thermal_limits is a
        # direct proxy for how much a branch normally carries. Exclude the
        # most heavily-loaded corridors from outage candidacy, which is what
        # an operator's study would effectively do.
        self._outage_candidates = list(range(len(self._pair_rows)))
        if thermal_limits is not None and self._pair_rows:
            tl = np.asarray(thermal_limits, dtype=np.float64)
            pair_rating = np.array([tl[rows[0]] for rows in self._pair_rows])
            keep = int(round(len(pair_rating)
                             * (1.0 - Settings.Outage.EXCLUDE_TOP_RATED_FRAC)))
            keep = max(1, min(keep, len(pair_rating)))
            # Ascending rating -> keep the lightly-loaded branches.
            self._outage_candidates = [
                int(i) for i in np.argsort(pair_rating)[:keep]]

        # (rows, start, end) — end exclusive
        self.line_outages: List[Tuple[np.ndarray, int, int]] = []
        # (node, start, end)
        self.gen_outages: List[Tuple[int, int, int]] = []
        # (rows, start, end, factor)
        self.derate_events: List[Tuple[np.ndarray, int, int, float]] = []

        if self.enabled:
            self._draw_line_outages(C)
            self._draw_gen_outages(C, np.asarray(node_types))
            self._draw_derate_events(C)

    # -- drawing ---------------------------------------------------------
    def _window(self, C, dur_min: int, dur_max: int) -> Tuple[int, int]:
        start = int(self.rng.uniform(C.START_FRACTION_MIN,
                                     C.START_FRACTION_MAX)
                    * max(self.sequence_length, 1))
        dur = int(self.rng.integers(dur_min, dur_max + 1))
        return start, min(start + dur, self.sequence_length)

    def _draw_line_outages(self, C) -> None:
        cand = self._outage_candidates
        if self.rng.random() > C.LINE_OUTAGE_PROBABILITY or not cand:
            return
        n = int(self.rng.integers(C.N_LINE_OUTAGES_MIN,
                                  C.N_LINE_OUTAGES_MAX + 1))
        n = min(n, len(cand))
        # Draw only from SCREENED candidates (see __init__).
        for gi in self.rng.choice(cand, size=n, replace=False):
            start, end = self._window(C, C.LINE_OUTAGE_DUR_MIN,
                                      C.LINE_OUTAGE_DUR_MAX)
            self.line_outages.append((self._pair_rows[int(gi)], start, end))

    def _draw_gen_outages(self, C, node_types: np.ndarray) -> None:
        gens = np.nonzero(node_types == 1)[0]
        if self.rng.random() > C.GEN_OUTAGE_PROBABILITY or gens.size == 0:
            return
        n = int(self.rng.integers(C.N_GEN_OUTAGES_MIN,
                                  C.N_GEN_OUTAGES_MAX + 1))
        n = min(n, gens.size)
        for node in self.rng.choice(gens, size=n, replace=False):
            start, end = self._window(C, C.GEN_OUTAGE_DUR_MIN,
                                      C.GEN_OUTAGE_DUR_MAX)
            self.gen_outages.append((int(node), start, end))

    def _draw_derate_events(self, C) -> None:
        if self.rng.random() > C.DERATE_PROBABILITY or not self._pair_rows:
            return
        n = int(self.rng.integers(C.N_DERATE_EVENTS_MIN,
                                  C.N_DERATE_EVENTS_MAX + 1))
        n = min(n, len(self._pair_rows))
        for gi in self.rng.choice(len(self._pair_rows), size=n, replace=False):
            start, end = self._window(C, C.DERATE_DUR_MIN, C.DERATE_DUR_MAX)
            factor = float(self.rng.uniform(C.DERATE_FACTOR_MIN,
                                            C.DERATE_FACTOR_MAX))
            self.derate_events.append(
                (self._pair_rows[int(gi)], start, end, factor))

    # -- queries ---------------------------------------------------------
    def planned_line_rows(self, t: int) -> np.ndarray:
        """Directed edge indices out on a PLANNED outage at timestep t."""
        out: List[int] = []
        for rows, start, end in self.line_outages:
            if start <= t < end:
                out.extend(int(r) for r in rows)
        return np.array(sorted(set(out)), dtype=np.int64)

    def planned_gen_nodes(self, t: int) -> Set[int]:
        """Generator buses whose unit is out on a PLANNED outage at t."""
        return {node for node, start, end in self.gen_outages
                if start <= t < end}

    def ambient_derate(self, ambient_temp_c: float) -> float:
        """
        Rating multiplier from ambient temperature alone.

        Nameplate holds at or below the reference ambient; above it the rating
        falls linearly and is floored so a heat wave cannot zero a line.
        """
        C = Settings.Outage
        excess = max(0.0, float(ambient_temp_c) - C.AMBIENT_REF_C)
        return float(np.clip(1.0 - C.AMBIENT_DERATE_PER_C * excess,
                             C.AMBIENT_DERATE_FLOOR, 1.0))

    def derate(self, t: int, ambient_temp_c: float) -> np.ndarray:
        """
        Per-edge rating multiplier at timestep t.

        Scheduled maintenance de-rates and the ambient de-rate stack
        multiplicatively, clamped at TOTAL_DERATE_FLOOR.
        """
        C = Settings.Outage
        d = np.full(self.num_edges, self.ambient_derate(ambient_temp_c),
                    dtype=np.float64)
        for rows, start, end, factor in self.derate_events:
            if start <= t < end:
                d[rows] *= factor
        return np.clip(d, C.TOTAL_DERATE_FLOOR, 1.0)

    # -- reporting -------------------------------------------------------
    def summary(self) -> str:
        return (f"{len(self.line_outages)} line outage(s), "
                f"{len(self.gen_outages)} generator outage(s), "
                f"{len(self.derate_events)} de-rate event(s)")

    def has_any(self) -> bool:
        return bool(self.line_outages or self.gen_outages
                    or self.derate_events)


def branch_status_with_outages(num_edges: int,
                               forced_open_rows,
                               planned_open_rows) -> np.ndarray:
    """
    Build the TERNARY branch_status telemetry column.

        1.0  closed
        0.0  forced open  (tripped / node-derived — a FAILURE)
       -1.0  planned open (scheduled outage — NOT a failure)

    Planned is written last so that a branch which is both scheduled out and
    would otherwise read as forced-open reports as planned: the operator took
    it out deliberately, and it must not be presented to the model as a trip.
    """
    C = Settings.Outage
    status = np.full(num_edges, C.STATUS_CLOSED, dtype=np.float32)
    if forced_open_rows is not None and len(forced_open_rows) > 0:
        status[np.asarray(forced_open_rows, dtype=np.int64)] = C.STATUS_FORCED_OPEN
    if planned_open_rows is not None and len(planned_open_rows) > 0:
        status[np.asarray(planned_open_rows, dtype=np.int64)] = C.STATUS_PLANNED_OPEN
    return status


def edge_mask_from_branch_status(branch_status: np.ndarray) -> np.ndarray:
    """
    Edge-driven mask: 1.0 where the branch is IN SERVICE, else 0.0.

    Tests equality against STATUS_CLOSED rather than truthiness. This is the
    trap the ternary encoding creates: STATUS_PLANNED_OPEN is -1.0, which is
    truthy, so `mask = node_mask AND branch_status` (the formulation in the
    plan's cross-cutting notes) would leave a planned-open branch in the GNN
    adjacency even though it is physically disconnected.
    """
    bs = np.asarray(branch_status)
    return (bs == Settings.Outage.STATUS_CLOSED).astype(np.float32)
