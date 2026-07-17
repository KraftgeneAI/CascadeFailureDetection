"""
Remedial Action Schemes Module (Phase 4 — MISO data gap plan)
=============================================================
RAS objects: monitored-line trigger conditions → pre-programmed corrective
actions (trip generator / shed load / open relief line), with the telemetered
Armed/Available/Triggered/Operated arming status.

[MISO RTO-SPEC-006: "RAS arming status (Armed/Available/Triggered/Operated)"]

Design notes
------------
- Each scheme watches ONE monitored line (a high-rating corridor). Above
  ARM_LOADING the scheme arms; above TRIGGER_LOADING it fires once and
  latches Operated for the rest of the scenario — real schemes are
  single-shot until re-set by operators.
- Actions are CONTROLLED interventions, not failures: a RAS-tripped
  generator or shed load must never appear in failure labels. The simulator
  applies actions to its dispatch state; this module only decides.
- The trigger threshold (1.05) sits below the Phase-0/1 emergency-trip band
  (sustained ≥ 1.15×): a RAS exists to act BEFORE line protection does.
- Torch-free (numpy only).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import Settings


def assign_ras_schemes(edge_index: np.ndarray,
                       node_types: np.ndarray,
                       thermal_limits: np.ndarray,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Create RAS scheme definitions for a grid.

    Monitored lines are drawn from the highest-rated corridors (that's where
    real schemes live). Actions and their targets are drawn per scheme:
      - TRIP_GEN:  a generator bus
      - SHED_LOAD: a load bus
      - OPEN_LINE: another (relief) line, distinct from the monitored one

    Returns dict of per-scheme arrays (length N_SCHEMES):
        ras_monitored_edge — representative directed edge index watched
        ras_action_type    — ACTION_TRIP_GEN / ACTION_SHED_LOAD / ACTION_OPEN_LINE
        ras_action_target  — node idx (trip/shed) or directed edge idx (open)
    """
    rng = rng or np.random.default_rng()
    R = Settings.RAS
    node_types = np.asarray(node_types)
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    num_edges = src.shape[0]
    num_nodes = int(max(src.max(), dst.max())) + 1

    # Representative directed edge per physical pair (lowest index)
    canon = np.minimum(src, dst) * num_nodes + np.maximum(src, dst)
    _, rep_idx = np.unique(canon, return_index=True)

    # Monitor the top corridors by rating
    order = rep_idx[np.argsort(-np.asarray(thermal_limits)[rep_idx])]
    n = min(R.N_SCHEMES, len(order))
    monitored = order[:n].astype(np.int64)

    gens = np.nonzero(node_types == 1)[0]
    loads = np.nonzero(node_types == 0)[0]

    action_type = np.zeros(n, dtype=np.int64)
    action_target = np.zeros(n, dtype=np.int64)
    for k in range(n):
        choices = []
        if len(gens):
            choices.append(R.ACTION_TRIP_GEN)
        if len(loads):
            choices.append(R.ACTION_SHED_LOAD)
        if len(rep_idx) > n:
            choices.append(R.ACTION_OPEN_LINE)
        act = int(rng.choice(choices))
        action_type[k] = act
        if act == R.ACTION_TRIP_GEN:
            action_target[k] = int(rng.choice(gens))
        elif act == R.ACTION_SHED_LOAD:
            action_target[k] = int(rng.choice(loads))
        else:
            relief = np.setdiff1d(rep_idx, monitored)
            action_target[k] = int(rng.choice(relief))

    return {
        'ras_monitored_edge': monitored,
        'ras_action_type': action_type,
        'ras_action_target': action_target,
    }


class RASEngine:
    """
    Per-timestep RAS state machine.

    Status transitions (per scheme, evaluated on current loading ratios):
        Available ↔ Armed        (monitored loading crosses ARM_LOADING)
        Armed     → Triggered    (crosses TRIGGER_LOADING; fires ONCE)
        Triggered → Operated     (latched until reset())
    """

    def __init__(self, schemes: Dict[str, np.ndarray]):
        self.monitored = np.asarray(schemes['ras_monitored_edge'], dtype=np.int64)
        self.action_type = np.asarray(schemes['ras_action_type'], dtype=np.int64)
        self.action_target = np.asarray(schemes['ras_action_target'], dtype=np.int64)
        self.n = len(self.monitored)
        self.status = np.zeros(self.n, dtype=np.int64)

    def reset(self) -> None:
        """All schemes back to Available (call at the start of each scenario)."""
        self.status[:] = Settings.RAS.STATUS_AVAILABLE

    def update(self, loading_ratios: np.ndarray) -> List[Tuple[int, int]]:
        """
        Advance the state machine one timestep.

        Args:
            loading_ratios: per-directed-edge apparent loading (vs normal rating)

        Returns:
            Actions fired THIS step: list of (action_type, action_target).
            Empty on most steps. The caller applies them to its dispatch
            state starting next timestep (relay/mechanical delay).
        """
        R = Settings.RAS
        fired: List[Tuple[int, int]] = []
        for k in range(self.n):
            if self.status[k] in (R.STATUS_TRIGGERED, R.STATUS_OPERATED):
                # single-shot: Triggered latches into Operated
                self.status[k] = R.STATUS_OPERATED
                continue
            loading = float(loading_ratios[self.monitored[k]])
            if loading >= R.TRIGGER_LOADING:
                self.status[k] = R.STATUS_TRIGGERED
                fired.append((int(self.action_type[k]), int(self.action_target[k])))
            elif loading >= R.ARM_LOADING:
                self.status[k] = R.STATUS_ARMED
            else:
                self.status[k] = R.STATUS_AVAILABLE
        return fired

    def status_stream(self, loading_ratios: np.ndarray) -> np.ndarray:
        """
        Telemetry frame [N_SCHEMES, RAS_FEATURES]:
        [arming_status, monitored line loading, action_type].
        [MISO RTO-SPEC-006: RAS arming status — the data item itself]
        """
        return np.column_stack([
            self.status.astype(np.float32),
            np.asarray(loading_ratios, dtype=np.float32)[self.monitored],
            self.action_type.astype(np.float32),
        ]).astype(np.float32)
