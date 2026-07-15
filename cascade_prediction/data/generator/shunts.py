"""
Reactive Support Module (Phase 3 — MISO data gap plan)
======================================================
Switchable capacitor/reactor banks, SVCs, and AVR setpoints.

[MISO RTO-SPEC-006: "Voltage-regulation equipment (AVR, capacitor/reactor,
 SVC) status"]

Design notes
------------
- Devices are per-NODE. Banks are discrete (whole bank in or out, one
  switching action per timestep — mechanical/controller delay); SVCs are
  continuous and respond within one step, saturating at ±Qmax.
- Controllers act on the PREVIOUS timestep's voltages, mirroring the tap
  controllers (Phase 2). Exhausted reactive support — caps all in, SVCs at
  ceiling, voltage still sagging — is the classic voltage-collapse precursor
  the cascade model should learn.
- Torch-free (numpy only).
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .config import Settings


def assign_shunt_devices(node_types: np.ndarray,
                         rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Assign reactive-support devices to nodes.

    - Capacitor banks on a fraction of LOAD buses (type 0)
    - Reactor banks on a fraction of SUBSTATION buses (type 2)
    - SVCs on a fraction of ALL buses
    - AVR setpoints on GENERATOR buses (type 1); 0 elsewhere

    Returns dict of per-node arrays (0 where a node has no such device):
        shunt_cap_mvar, shunt_reactor_mvar, svc_qmax_mvar, avr_setpoint
    """
    rng = rng or np.random.default_rng()
    S = Settings.Shunt
    node_types = np.asarray(node_types)
    n = len(node_types)

    def pick(candidates: np.ndarray, fraction: float) -> np.ndarray:
        k = int(round(len(candidates) * fraction))
        if k == 0 or len(candidates) == 0:
            return np.array([], dtype=np.int64)
        return rng.choice(candidates, size=min(k, len(candidates)), replace=False)

    cap = np.zeros(n)
    for i in pick(np.nonzero(node_types == 0)[0], S.CAP_BANK_FRACTION):
        cap[i] = rng.uniform(S.CAP_BANK_MVAR_MIN, S.CAP_BANK_MVAR_MAX)

    reactor = np.zeros(n)
    for i in pick(np.nonzero(node_types == 2)[0], S.REACTOR_FRACTION):
        reactor[i] = rng.uniform(S.REACTOR_MVAR_MIN, S.REACTOR_MVAR_MAX)

    svc = np.zeros(n)
    for i in pick(np.arange(n), S.SVC_FRACTION):
        svc[i] = rng.uniform(S.SVC_QMAX_MVAR_MIN, S.SVC_QMAX_MVAR_MAX)

    avr = np.zeros(n)
    gen_idx = np.nonzero(node_types == 1)[0]
    avr[gen_idx] = rng.uniform(S.AVR_SETPOINT_MIN_PU, S.AVR_SETPOINT_MAX_PU,
                               size=len(gen_idx))

    return {
        'shunt_cap_mvar': cap,
        'shunt_reactor_mvar': reactor,
        'svc_qmax_mvar': svc,
        'avr_setpoint': avr,
    }


class ShuntController:
    """
    Steps bank switching (hysteresis, one action/step) and SVC output
    (continuous, saturating) from the previous timestep's voltages.

    Sign convention: positive Q = injection into the bus (capacitive).
    Reactor banks ABSORB (negative injection).
    """

    def __init__(self, cap_mvar: np.ndarray, reactor_mvar: np.ndarray,
                 svc_qmax_mvar: np.ndarray):
        self.cap_mvar = np.asarray(cap_mvar, dtype=np.float64)
        self.reactor_mvar = np.asarray(reactor_mvar, dtype=np.float64)
        self.svc_qmax = np.asarray(svc_qmax_mvar, dtype=np.float64)
        n = len(self.cap_mvar)
        self.cap_status = np.zeros(n, dtype=np.int64)      # 1 = switched in
        self.reactor_status = np.zeros(n, dtype=np.int64)
        self.svc_q = np.zeros(n, dtype=np.float64)         # current MVar output

    def reset(self) -> None:
        """All banks out, SVCs idle (call at the start of each scenario)."""
        self.cap_status[:] = 0
        self.reactor_status[:] = 0
        self.svc_q[:] = 0.0

    def update(self, prev_voltages: np.ndarray) -> np.ndarray:
        """
        One control step from previous-step voltages.

        Returns:
            shunt_q [num_nodes]: net reactive injection in MVar
            (+ = capacitive injection, − = inductive absorption).
        """
        S = Settings.Shunt
        v = np.asarray(prev_voltages, dtype=np.float64)
        alive = v >= S.DEAD_BUS_V_PU

        # Capacitor banks: in when voltage sags, out when it recovers high.
        has_cap = self.cap_mvar > 0
        switch_in = has_cap & alive & (v < S.CAP_IN_V_PU) & (self.cap_status == 0)
        switch_out = has_cap & ((v > S.CAP_OUT_V_PU) | ~alive) & (self.cap_status == 1)
        self.cap_status[switch_in] = 1
        self.cap_status[switch_out] = 0

        # Reactor banks: in when voltage rises, out when it falls back.
        has_rea = self.reactor_mvar > 0
        r_in = has_rea & alive & (v > S.REACTOR_IN_V_PU) & (self.reactor_status == 0)
        r_out = has_rea & ((v < S.REACTOR_OUT_V_PU) | ~alive) & (self.reactor_status == 1)
        self.reactor_status[r_in] = 1
        self.reactor_status[r_out] = 0

        # SVC: continuous droop response, saturating at ±Qmax; dead bus → 0.
        has_svc = self.svc_qmax > 0
        q = S.SVC_GAIN_MVAR_PER_PU * (S.SVC_V_SETPOINT_PU - v)
        self.svc_q = np.where(
            has_svc & alive,
            np.clip(q, -self.svc_qmax, self.svc_qmax),
            0.0,
        )

        return (self.cap_status * self.cap_mvar
                - self.reactor_status * self.reactor_mvar
                + self.svc_q)
