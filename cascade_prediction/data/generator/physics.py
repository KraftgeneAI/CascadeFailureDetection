"""
Physics Module
=============
Physics-based simulation models for power grid dynamics.

This module implements:
- Power flow calculations (AC power flow via PyPSA)
- Frequency dynamics
- Thermal dynamics
- Voltage stability
"""

import logging

import numpy as np
import pypsa
from typing import Tuple, Optional, List, Dict

from .config import Settings

# PyPSA emits an INFO line for EVERY sub-network on EVERY solve
# ("Balancing power on single-bus sub-network ...", "Performing non-linear
# load-flow ..."). During a cascade the grid fragments into dozens of islands
# and each node failure triggers a fresh solve, so a single collapse timestep
# floods the console with hundreds of these lines — the console I/O alone is a
# large share of wall-clock time. Silence INFO; keep WARNING (non-convergence)
# and above, which the simulator handles and reports itself.
logging.getLogger("pypsa").setLevel(logging.WARNING)


class PowerFlowSimulator:
    """
    AC power flow simulator using PyPSA.
    
    Provides accurate power flow calculations based on proper
    AC power flow equations (Newton-Raphson method).
    """
    
    def __init__(
        self,
        num_nodes: int,
        edge_index: np.ndarray,
        positions: np.ndarray,
        node_types: np.ndarray,
        gen_capacity: np.ndarray,
        line_reactance: np.ndarray,
        line_resistance: np.ndarray,
        line_susceptance: np.ndarray,
        line_conductance: np.ndarray,
        thermal_limits: np.ndarray,
        branch_type: Optional[np.ndarray] = None,
        avr_setpoint: Optional[np.ndarray] = None,
    ):
        """
        Initialize power flow simulator.

        Args:
            num_nodes: Number of nodes
            edge_index: Edge connectivity [2, num_edges]
            positions: Node positions [num_nodes, 2]
            node_types: Node types (0=load, 1=gen, 2=sub)
            gen_capacity: Generator capacities
            line_reactance: Line reactances
            line_resistance: Line resistances
            line_susceptance: Line susceptances
            line_conductance: Line conductances
            thermal_limits: Line thermal limits
            branch_type: Optional per-edge device type (Phase 2): 0=line,
                1=LTC transformer, 2=phase-shifting transformer (PAR).
                None → all plain lines (pre-Phase-2 behavior).
            avr_setpoint: Optional per-node AVR voltage setpoints (Phase 3) —
                PV generators regulate to these instead of internal randoms,
                making the setpoint a telemetered, reproducible quantity.
                [MISO RTO-SPEC-006: AVR status]
        """
        self.num_nodes = num_nodes
        self.num_edges = edge_index.shape[1]
        self.node_types = node_types
        # Keep the (bidirectional) edge index so result extraction can map each
        # directed edge back to its single physical PyPSA line (see Fix #2).
        self.edge_index = np.asarray(edge_index)

        # Per-node power factors based on node type (tan(phi) = Q/P)
        # Load buses (0): mixed residential/commercial/industrial
        #   - residential: pf ~0.95 → tan(phi) ~0.33
        #   - industrial:  pf ~0.85 → tan(phi) ~0.62
        #   - commercial:  pf ~0.90 → tan(phi) ~0.48
        # Generator buses (1): generators supply/absorb Q via AVR, not load-driven
        # Substation buses (2): large transformer loads, pf ~0.90
        pf = np.where(
            node_types == 1,  # Generator bus: minimal reactive load
            np.random.uniform(Settings.Node.PF_GENERATOR_MIN, Settings.Node.PF_GENERATOR_MAX, num_nodes),
            np.where(
                node_types == 2,  # Substation: transformer magnetizing current
                np.random.uniform(Settings.Node.PF_SUBSTATION_MIN, Settings.Node.PF_SUBSTATION_MAX, num_nodes),
                np.random.uniform(Settings.Node.PF_LOAD_MIN, Settings.Node.PF_LOAD_MAX, num_nodes)  # Load bus: mixed load types
            )
        )
        # tan(phi) = sqrt(1 - pf²) / pf  → Q = P * tan(phi)
        self.tan_phi = np.sqrt(1.0 - pf**2) / pf  # per-node Q/P ratio
        
        # Phase 3 — AVR setpoints (None → legacy internal randoms)
        self.avr_setpoint = avr_setpoint

        # Phase 2 — branch typing (None → all plain lines)
        self.branch_type = (
            np.asarray(branch_type, dtype=np.int64)
            if branch_type is not None
            else np.zeros(self.num_edges, dtype=np.int64)
        )

        # Initialize PyPSA network
        self.network = self._create_pypsa_network(
            edge_index, positions, gen_capacity,
            line_reactance, line_resistance,
            line_susceptance, line_conductance,
            thermal_limits,
        )
    
    def _create_pypsa_network(
        self,
        edge_index: np.ndarray,
        positions: np.ndarray,
        gen_capacity: np.ndarray,
        line_reactance: np.ndarray,
        line_resistance: np.ndarray,
        line_susceptance: np.ndarray,
        line_conductance: np.ndarray,
        thermal_limits: np.ndarray,
    ) -> pypsa.Network:
        """
        Create PyPSA network from grid parameters.
        
        Returns:
            Initialized PyPSA Network
        """
        network = pypsa.Network()
        network.set_snapshots(["now"])
        # Set system base to 100 MVA so that MW-scale loads/generators are ~1 pu.
        # PyPSA uses sn_mva as the system MVA base for per-unit conversion.
        network.sn_mva = Settings.PowerSystem.SN_MVA

        # Add buses (nodes)
        for i in range(self.num_nodes):
            network.add(
                "Bus",
                f"bus_{i}",
                v_nom=Settings.PowerSystem.V_NOM_KV,  # 138 kV transmission
                x=positions[i, 0],
                y=positions[i, 1]
            )

        # Add generators
        # Slack bus (node 0): sets voltage angle reference and absorbs P imbalance
        # PV buses (other generators): regulate voltage magnitude at their bus
        # PQ buses: fixed P/Q injection — NOT used for generators (causes divergence)
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            if idx == 0:
                control = "Slack"
                v_set = Settings.PowerFlow.SLACK_V_SET
            else:
                control = "PV"   # Voltage-regulating generator
                # Phase 3: the AVR setpoint IS the PV target — telemetered
                # (SCADA col 21) instead of an internal random draw.
                if self.avr_setpoint is not None and self.avr_setpoint[idx] > 0:
                    v_set = float(self.avr_setpoint[idx])
                else:
                    v_set = np.random.uniform(Settings.PowerFlow.PV_V_SET_MIN, Settings.PowerFlow.PV_V_SET_MAX)
            network.add(
                "Generator",
                f"gen_{idx}",
                bus=f"bus_{idx}",
                p_nom=gen_capacity[idx],
                control=control,
                p_set=0.0,
                v_set=v_set   # bookkeeping only — see bus target below
            )
            # PyPSA reads the PV/Slack voltage TARGET from the BUS attribute
            # v_mag_pu_set, not from any generator attribute; a bare `v_set`
            # kwarg is stored as an inert custom column. Without this line
            # every PV bus silently regulated to the 1.0 default and AVR
            # setpoints (Phase 3) had no effect on the solve.
            network.buses.at[f"bus_{idx}", "v_mag_pu_set"] = v_set
        
        # Add loads
        for i in range(self.num_nodes):
            network.add(
                "Load",
                f"load_{i}",
                bus=f"bus_{i}",
                p_set=0.0,
                q_set=0.0
            )
        
        # Add transmission lines.
        # Fix (#2): edge_index is bidirectional, so add ONE PyPSA Line per
        # *undirected* pair instead of one per directed edge (which created two
        # parallel branches per line, halving reactance and doubling capacity).
        # Build a mapping from every directed edge -> (physical line name, sign)
        # so flows can be mapped back to all directed edges during extraction.
        src, dst = edge_index
        self.edge_to_line = {}   # directed edge idx -> (line_name, sign)
        self.line_buses = {}     # name -> (bus0_idx, bus1_idx)  (bus0 < bus1)
        # Phase 2: physical branches may be Lines OR Transformers (LTC/PAR).
        # branch_is_trafo: name -> True if PyPSA Transformer component.
        # trafo_rep_edge:  transformer name -> representative directed edge
        #                  index (used to look up per-call tap state).
        self.branch_is_trafo = {}
        self.trafo_rep_edge = {}
        self._ghost_map = {}   # trafo name -> ghost Line name (fallback solves only)
        self._prefer_ghost_solve = False  # sticky: skip the main solve while the
                                          # PyPSA transformer-less-island bug bites
        self.bus_branches = {}  # bus idx -> set of incident physical branch names
        seen = {}                # (min,max) pair -> name
        line_counter = 0
        trafo_counter = 0
        v_nom = Settings.PowerSystem.V_NOM_KV
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            key = (min(s, d), max(s, d))
            if key not in seen:
                if self.branch_type[i] != 0:
                    # LTC transformer or PAR → PyPSA Transformer.
                    # [MISO RTO-SPEC-006: transformer / PAR flows + taps]
                    # PyPSA Transformer r/x are per-unit on the SYSTEM base
                    # (network.sn_mva), the same convention Line ohms reduce to
                    # internally — so convert ohms → system p.u. by dividing by
                    # z_base_sys = v_nom²/sn_mva. (The earlier ×s_nom/v_nom²
                    # form assumed an own-s_nom base PyPSA does NOT use here,
                    # which made transformers ~10× under-impedance and turned
                    # them into near-short-circuit reactive-current sinks.)
                    name = f"trafo_{trafo_counter}"
                    s_nom = float(thermal_limits[i])
                    z_base_sys = v_nom ** 2 / Settings.PowerSystem.SN_MVA
                    network.add(
                        "Transformer",
                        name,
                        bus0=f"bus_{key[0]}",
                        bus1=f"bus_{key[1]}",
                        x=max(float(line_reactance[i]) / z_base_sys, 1e-4),
                        r=float(line_resistance[i]) / z_base_sys,
                        s_nom=s_nom,
                        tap_ratio=1.0,
                        phase_shift=0.0,
                        model="pi",
                    )
                    self.branch_is_trafo[name] = True
                    self.trafo_rep_edge[name] = i
                    trafo_counter += 1
                else:
                    name = f"line_{line_counter}"
                    network.add(
                        "Line",
                        name,
                        bus0=f"bus_{key[0]}",
                        bus1=f"bus_{key[1]}",
                        x=line_reactance[i],
                        r=line_resistance[i],
                        b=line_susceptance[i],
                        g=line_conductance[i],
                        s_nom=thermal_limits[i],
                        length=1.0
                    )
                    self.branch_is_trafo[name] = False
                    line_counter += 1
                seen[key] = name
                self.line_buses[name] = key
                self.bus_branches.setdefault(key[0], set()).add(name)
                self.bus_branches.setdefault(key[1], set()).add(name)
            name = seen[key]
            # +1 if this directed edge runs bus0->bus1 (PyPSA p0 convention), else -1
            sign = 1.0 if (s, d) == key else -1.0
            self.edge_to_line[i] = (name, sign)

        self.num_phys_lines = line_counter
        self.num_phys_trafos = trafo_counter
        return network

    def _branch_df(self, name: str):
        """Return the component dataframe (lines or transformers) for a branch."""
        return (self.network.transformers if self.branch_is_trafo.get(name)
                else self.network.lines)

    # Static attributes preserved when a failed transformer is removed
    _TRAFO_PARAMS = ('bus0', 'bus1', 'x', 'r', 's_nom',
                     'tap_ratio', 'phase_shift', 'model')

    def _deactivate_branch(self, name: str, restore: Dict) -> None:
        """
        Take a physical branch out of service for this solve.

        Lines: flip 'active' (cheap, restorable).
        Transformers: REMOVE from the network and re-add on restore. PyPSA's
        sub-network pf raises KeyError('Transformer') when a sub-network's
        transformers are all inactive — the apparent-power frame drops the
        'Transformer' column level while component iteration still requests
        it. Physically removing the component keeps the frames consistent.
        """
        if self.branch_is_trafo.get(name):
            if (name not in restore['removed_trafos']
                    and name in self.network.transformers.index):
                row = self.network.transformers.loc[name]
                restore['removed_trafos'][name] = {
                    k: row[k] for k in self._TRAFO_PARAMS}
                self.network.remove("Transformer", name)
        else:
            if name not in restore['line_active']:
                restore['line_active'][name] = bool(
                    self.network.lines.at[name, "active"])
                self.network.lines.at[name, "active"] = False
    
    def compute_power_flow(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None,
        tap_ratios: Optional[np.ndarray] = None,
        phase_shifts: Optional[np.ndarray] = None,
        shunt_q: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute AC power flow using PyPSA with proper bus isolation.
        
        Args:
            generation: Generation at each node [num_nodes]
            load: Load at each node [num_nodes]
            failed_lines: List of failed line indices
            failed_nodes: List of failed node indices
            
        Returns:
            Tuple of:
            - voltages: Voltage magnitudes [num_nodes]
            - angles: Voltage angles [num_nodes]
            - line_flows_p: Active power flows [num_edges]
            - node_reactive: Reactive power at nodes [num_nodes]
            - line_flows_q: Reactive power flows [num_edges]
            - is_stable: Whether power flow converged
        """
        # Calculate reactive load with 50% power-factor correction applied.
        # Equivalent to shunt capacitor compensation at every load bus, scaled
        # with actual load so there is no over/under-compensation at any stress level.
        q_load = load * self.tan_phi * Settings.PowerFlow.Q_LOAD_FACTOR

        # Phase 3 — reactive-support injection (cap/reactor banks + SVC).
        # Positive shunt_q injects Q at the bus → reduces the net reactive
        # demand seen by the network, propping the local voltage — the
        # mechanism by which exhausted reactive reserves become visible.
        # [MISO RTO-SPEC-006: voltage-regulation equipment status]
        if shunt_q is not None:
            q_load = q_load - shunt_q

        failed_node_set = set(int(n) for n in failed_nodes) if failed_nodes else set()

        # Record everything we mutate so we can restore the pristine base network.
        restore = {
            'gen_p_set': {}, 'gen_control': {}, 'gen_active': {},
            'load_p_set': {}, 'load_q_set': {}, 'load_active': {},
            'line_active': {}, 'dummy_gens': [],
            'removed_trafos': {},   # Phase 2: failed trafos are REMOVED, not deactivated
        }

        # 1. Generator active-power setpoints (failed generators contribute 0).
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            gen_name = f"gen_{idx}"
            restore['gen_p_set'][gen_name] = self.network.generators.at[gen_name, "p_set"]
            self.network.generators.at[gen_name, "p_set"] = (
                0.0 if idx in failed_node_set else generation[idx]
            )

        # 2. Load setpoints (failed loads contribute 0).
        for i in range(self.num_nodes):
            load_name = f"load_{i}"
            restore['load_p_set'][load_name] = self.network.loads.at[load_name, "p_set"]
            restore['load_q_set'][load_name] = self.network.loads.at[load_name, "q_set"]
            if i in failed_node_set:
                self.network.loads.at[load_name, "p_set"] = 0.0
                self.network.loads.at[load_name, "q_set"] = 0.0
            else:
                self.network.loads.at[load_name, "p_set"] = load[i]
                self.network.loads.at[load_name, "q_set"] = q_load[i]

        # 3. PROPERLY remove failed nodes from the solved network (Fix #3).
        # PyPSA has no 'in_service' attribute (that is pandapower); it uses
        # 'active' on generators/loads/lines.  Deactivating a component truly
        # removes it from the power-flow sub-network, so flow redistributes onto
        # the remaining corridors — the physics a cascade depends on.
        if failed_node_set:
            for idx in failed_node_set:
                gen_name = f"gen_{idx}"
                if gen_name in self.network.generators.index:
                    restore['gen_active'][gen_name] = bool(self.network.generators.at[gen_name, "active"])
                    self.network.generators.at[gen_name, "active"] = False
                load_name = f"load_{idx}"
                restore['load_active'][load_name] = bool(self.network.loads.at[load_name, "active"])
                self.network.loads.at[load_name, "active"] = False

            # Take every physical branch (line OR transformer) touching a
            # failed bus out of service.
            for name, (b0, b1) in self.line_buses.items():
                if b0 in failed_node_set or b1 in failed_node_set:
                    self._deactivate_branch(name, restore)

            # A failed bus is now an isolated single-bus sub-network with no
            # reference, which PyPSA cannot solve.  Give each one a tiny dummy
            # slack so the solve succeeds; its output is masked to 0 anyway.
            for idx in failed_node_set:
                dummy = f"_failslack_{idx}"
                if dummy in self.network.generators.index:
                    self.network.remove("Generator", dummy)
                self.network.add(
                    "Generator", dummy, bus=f"bus_{idx}",
                    control="Slack", p_set=0.0, v_set=1.0, p_nom=1e-3,
                )
                restore['dummy_gens'].append(dummy)

        # 4. Explicitly failed lines (map directed edge idx -> physical branch;
        # Phase 2: the branch may be a Line or a Transformer component).
        if failed_lines:
            for e_idx in failed_lines:
                mapped = self.edge_to_line.get(int(e_idx))
                if mapped is None:
                    continue
                self._deactivate_branch(mapped[0], restore)

        # 4b. EVERY electrical island must contain a slack source. Branch
        # outages (line trips, RAS openings, failed-bus disconnections) can
        # split off generator-less islands of ANY size: PyPSA then either
        # crashes in its single-bus path (slack_generator=None → KeyError:
        # None) or builds a singular Jacobian for multi-bus islands ("Matrix
        # is exactly singular" → non-convergence → KeyError in extraction).
        # Walk the connected components of the ACTIVE-branch graph and give
        # each generator-less component a dummy slack at its first bus.
        out_of_service = set(restore['line_active']) | set(restore['removed_trafos'])
        if out_of_service or failed_node_set:
            adj = {b: [] for b in range(self.num_nodes)}
            for name, (b0, b1) in self.line_buses.items():
                if name in out_of_service:
                    continue
                adj[b0].append(b1)
                adj[b1].append(b0)
            seen: set = set()
            for start in range(self.num_nodes):
                if start in seen:
                    continue
                comp = [start]
                seen.add(start)
                stack = [start]
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            comp.append(v)
                            stack.append(v)
                has_slack = any(
                    (self.node_types[b] == 1 and b not in failed_node_set)
                    or f"_failslack_{b}" in self.network.generators.index
                    for b in comp
                )
                if not has_slack:
                    b = comp[0]
                    dummy = f"_failslack_{b}"
                    if dummy not in self.network.generators.index:
                        self.network.add(
                            "Generator", dummy, bus=f"bus_{b}",
                            control="Slack", p_set=0.0, v_set=1.0, p_nom=1e-3,
                        )
                        restore['dummy_gens'].append(dummy)

        # 5. Phase 2 — apply tap state to transformer components.
        # tap_ratio (LTC voltage ratio) and phase_shift (PAR degrees) come
        # from TapController via per-directed-edge arrays; each transformer
        # reads its representative edge. When None, taps keep their previous
        # values — cascade-internal recomputes deliberately inherit the main
        # loop's tap state.  [MISO RTO-SPEC-006: LTC/PAR tap positions]
        if tap_ratios is not None or phase_shifts is not None:
            for name, rep in self.trafo_rep_edge.items():
                if name not in self.network.transformers.index:
                    continue   # removed this solve (failed branch)
                if tap_ratios is not None:
                    self.network.transformers.at[name, "tap_ratio"] = float(tap_ratios[rep])
                if phase_shifts is not None:
                    self.network.transformers.at[name, "phase_shift"] = float(phase_shifts[rep])

        # If the slack bus (node 0) is failed, reassign slack to the next active generator.
        # Without a slack bus PyPSA has no angle reference and will crash.
        slack_reassigned = None
        if 0 in failed_node_set:
            for candidate in gen_indices:
                if candidate not in failed_node_set:
                    gen_name = f"gen_{candidate}"
                    restore['gen_control'][gen_name] = self.network.generators.at[gen_name, "control"]
                    self.network.generators.at[gen_name, "control"] = "Slack"
                    slack_reassigned = candidate
                    break
            if slack_reassigned is None:
                # No generators left — full collapse
                self._restore_network_state(restore)
                return self._get_default_results()

        # Fast path: while the grid is fragmented in a way that triggers the
        # PyPSA transformer-less-island KeyError, the main solve is guaranteed
        # to fail — go straight to the ghost-line solve instead of paying for
        # a doomed full solve every step. The flag clears once a normal solve
        # succeeds again (healed topology / next scenario's early steps).
        if self._prefer_ghost_solve and (failed_lines or failed_nodes):
            fb = self._pf_transformers_as_lines(failed_lines, failed_nodes)
            self._restore_network_state(restore)
            if fb == 'unstable':
                return self._get_default_results()
            if fb is not None:
                return fb
            return self._get_default_results()

        try:
            # Flat start (use_seed=False) is more robust than warm start when
            # network state changes significantly between calls.
            # distribute_slack spreads imbalance across all PV generators,
            # reducing voltage stress on the single slack bus at high load.
            status = self.network.pf(use_seed=False, distribute_slack=True)
            converged = status.get("converged")
            if hasattr(converged, "values"):   # DataFrame: snapshots × sub-networks
                # ALL sub-networks must converge (an edited grid with a
                # decommissioned node has >1 sub-network; reading only
                # [0, 0] checked an arbitrary one).
                is_stable = bool(np.asarray(converged.values).astype(bool).all())
            elif isinstance(converged, dict):
                inner = next(iter(converged.values()), {})
                is_stable = bool(next(iter(inner.values()), False)) if isinstance(inner, dict) else bool(inner)
            else:
                is_stable = bool(converged) if converged is not None else False

            if not is_stable:
                self._restore_network_state(restore)
                return self._get_default_results()

            # Extract results
            voltages, angles, node_reactive = self._extract_bus_results(failed_nodes)
            line_flows_p, line_flows_q = self._extract_line_results(failed_lines, failed_nodes)

            # Restore network state
            self._restore_network_state(restore)

            self._prefer_ghost_solve = False   # normal solve works again
            return voltages, angles, line_flows_p, node_reactive, line_flows_q, True

        except Exception as e:
            # PyPSA sub-network pf bug: when a cascade splits off an AC
            # island that contains NO transformers (while the network has
            # them globally), branch-flow extraction raises
            # KeyError('Transformer'). Re-solve this one step with every
            # transformer modeled as an equivalent Line — tap ratio and
            # phase shift are approximated away for the step, which is
            # acceptable in an already-fragmenting grid state and far
            # better than returning all-zero telemetry.
            if isinstance(e, KeyError) and 'Transformer' in str(e):
                print("  [PF-FALLBACK] transformer-less AC island hit PyPSA "
                      "KeyError('Transformer') — re-solving with transformers "
                      "as equivalent lines for this step")
                # Once the grid has fragmented into a state that triggers
                # this PyPSA bug, every solve until the topology heals will
                # trigger it too — skip the doomed main solve next time.
                self._prefer_ghost_solve = True
                fb = self._pf_transformers_as_lines(failed_lines, failed_nodes)
                self._restore_network_state(restore)
                if fb == 'unstable':
                    # Fallback solved cleanly; the grid state itself doesn't
                    # converge (singular fragments in deep collapse). This is
                    # a physical outcome, not an error — no traceback.
                    return self._get_default_results()
                if fb is not None:
                    return fb
                import traceback
                print(f"  [WARNING] Power flow exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                return self._get_default_results()
            import traceback
            print(f"  [WARNING] Power flow exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            self._restore_network_state(restore)
            return self._get_default_results()

    def _pf_transformers_as_lines(self, failed_lines, failed_nodes):
        """
        Fallback solve with all (still-active) transformers temporarily
        replaced by impedance-equivalent Lines (system-p.u. reactance/resistance
        converted back to ohms on the system base). Returns the standard
        6-tuple, or None if the fallback also fails. The base network is fully
        restored.
        """
        v_nom = Settings.PowerSystem.V_NOM_KV
        z_base_sys = v_nom ** 2 / Settings.PowerSystem.SN_MVA
        converted = {}
        for name in list(self.network.transformers.index):
            row = self.network.transformers.loc[name]
            ghost = f"_ghost_{name}"
            s_nom = float(row['s_nom'])
            # Transformer x/r are system-p.u.; a Line wants ohms → × z_base_sys.
            self.network.add(
                "Line", ghost,
                bus0=row['bus0'], bus1=row['bus1'],
                x=max(float(row['x']) * z_base_sys, 1e-6),
                r=float(row['r']) * z_base_sys,
                s_nom=s_nom,
            )
            converted[name] = ({k: row[k] for k in self._TRAFO_PARAMS}, ghost)
            self.network.remove("Transformer", name)
        # Extraction reads ghost line flows for transformer-typed branches.
        self._ghost_map = {n: g for n, (_, g) in converted.items()}
        try:
            status = self.network.pf(use_seed=False, distribute_slack=True)
            converged = status.get("converged")
            if hasattr(converged, "values"):
                is_stable = bool(np.asarray(converged.values).astype(bool).all())
            else:
                is_stable = bool(converged) if converged is not None else False
            if not is_stable:
                return 'unstable'   # solved, but grid state doesn't converge
            voltages, angles, node_reactive = self._extract_bus_results(failed_nodes)
            p, q = self._extract_line_results(failed_lines, failed_nodes)
            return voltages, angles, p, node_reactive, q, True
        except Exception as e2:
            print(f"  [PF-FALLBACK] also failed: {type(e2).__name__}: {e2}")
            return None
        finally:
            self._ghost_map = {}
            for name, (params, ghost) in converted.items():
                if ghost in self.network.lines.index:
                    self.network.remove("Line", ghost)
                if name not in self.network.transformers.index:
                    self.network.add("Transformer", name, **params)

    @staticmethod
    def _pf_converged(status) -> bool:
        """Robustly read the convergence flag from network.pf() across PyPSA versions."""
        conv = None
        if hasattr(status, "converged"):
            conv = status.converged              # namedtuple-style result
        elif isinstance(status, dict):
            conv = status.get("converged")
        if conv is None:
            return False
        if hasattr(conv, "iloc"):                # DataFrame
            return bool(conv.iloc[0, 0])
        if isinstance(conv, dict):
            inner = next(iter(conv.values()), {})
            return bool(next(iter(inner.values()), False)) if isinstance(inner, dict) else bool(inner)
        try:
            return bool(np.all(conv))
        except Exception:
            return bool(conv)
    
    def _extract_bus_results(self, failed_nodes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract voltage, angle, and reactive power from buses.
        
        Args:
            failed_nodes: List of failed node indices (will have default values)
        
        Returns:
            Tuple of (voltages, angles, node_reactive)
        """
        voltages = np.zeros(self.num_nodes)
        angles = np.zeros(self.num_nodes)
        node_reactive = np.zeros(self.num_nodes)
        
        failed_node_set = set(failed_nodes) if failed_nodes else set()
        
        for i in range(self.num_nodes):
            bus_name = f"bus_{i}"
            
            if i in failed_node_set:
                # Failed nodes: use default low voltage
                voltages[i] = 0.0  # Zero voltage for failed buses
                angles[i] = 0.0
                node_reactive[i] = 0.0
            else:
                # Active nodes: extract from power flow results
                try:
                    voltages[i] = self.network.buses_t.v_mag_pu.loc["now", bus_name]
                    angles[i] = np.radians(self.network.buses_t.v_ang.loc["now", bus_name])
                    node_reactive[i] = self.network.buses_t.q.loc["now", bus_name]
                except (KeyError, AttributeError):
                    # Bus might be in isolated island - use default
                    voltages[i] = 0.95
                    angles[i] = 0.0
                    node_reactive[i] = 0.0
        
        return voltages, angles, node_reactive
    
    def _extract_line_results(
        self,
        failed_lines: Optional[List[int]],
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract active and reactive power flows from lines.
        
        Args:
            failed_lines: List of explicitly failed line indices
            failed_nodes: List of failed node indices (lines connected to these are also failed)
        
        Returns:
            Tuple of (line_flows_p, line_flows_q)
        """
        line_flows_p = np.zeros(self.num_edges)
        line_flows_q = np.zeros(self.num_edges)

        failed_line_set = set(int(l) for l in failed_lines) if failed_lines else set()
        failed_node_set = set(int(n) for n in failed_nodes) if failed_nodes else set()

        src, dst = self.edge_index
        p0 = getattr(self.network.lines_t, "p0", None)
        q0 = getattr(self.network.lines_t, "q0", None)
        # Phase 2: transformer/PAR flows live in transformers_t, not lines_t.
        # [MISO RTO-SPEC-006: "Transformer flows" / "PAR flows" telemetry]
        tp0 = getattr(self.network.transformers_t, "p0", None)
        tq0 = getattr(self.network.transformers_t, "q0", None)

        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])

            if i in failed_line_set or s in failed_node_set or d in failed_node_set:
                # Failed / out-of-service edges carry zero flow
                continue

            # Map this directed edge back to its single physical PyPSA branch.
            # 'sign' restores the source->destination orientation of this edge
            # relative to the branch's bus0->bus1 convention (Fix #2).
            name, sign = self.edge_to_line[i]
            if self.branch_is_trafo.get(name):
                # During the transformer-less-island fallback, transformer
                # flows live in the ghost Line (same bus0/bus1 → same sign).
                ghost = getattr(self, '_ghost_map', {}).get(name)
                if ghost is not None:
                    fp, fq, name = p0, q0, ghost
                else:
                    fp, fq = tp0, tq0
            else:
                fp, fq = p0, q0
            try:
                line_flows_p[i] = float(fp.at["now", name]) * sign
                line_flows_q[i] = float(fq.at["now", name]) * sign
            except (KeyError, AttributeError, TypeError):
                # Branch in an isolated island / not solved — leave as zero flow
                continue

        return line_flows_p, line_flows_q

    def _remove_dummy_gens(self, names: List[str]) -> None:
        """Remove the temporary _failslack_ generators added for failed buses."""
        for name in names:
            if name in self.network.generators.index:
                self.network.remove("Generator", name)

    def _restore_network_state(self, restore: Dict):
        """
        Restore the network to its pristine base state after a power-flow call.

        Reverses every mutation recorded in ``restore`` (setpoints, 'active'
        flags, slack reassignment) and removes any temporary dummy slack
        generators that were added to isolated failed buses.

        Args:
            restore: State dictionary populated by ``compute_power_flow``.
        """
        # Remove temporary dummy slack generators first.
        for name in restore.get('dummy_gens', []):
            if name in self.network.generators.index:
                self.network.remove("Generator", name)

        # Reactivate components.
        for gen_name, val in restore.get('gen_active', {}).items():
            self.network.generators.at[gen_name, "active"] = val
        for load_name, val in restore.get('load_active', {}).items():
            self.network.loads.at[load_name, "active"] = val
        for line_name, val in restore.get('line_active', {}).items():
            # Phase 2: the branch may be a Line or a Transformer component.
            self._branch_df(line_name).at[line_name, "active"] = val

        # Phase 2: re-add transformers removed for this solve (failed
        # branches) so the base network is pristine for the next call.
        for name, params in restore.get('removed_trafos', {}).items():
            if name not in self.network.transformers.index:
                self.network.add("Transformer", name, **params)

        # Restore slack reassignment.
        for gen_name, val in restore.get('gen_control', {}).items():
            self.network.generators.at[gen_name, "control"] = val

        # Restore setpoints.
        for gen_name, val in restore.get('gen_p_set', {}).items():
            self.network.generators.at[gen_name, "p_set"] = val
        for load_name, val in restore.get('load_p_set', {}).items():
            self.network.loads.at[load_name, "p_set"] = val
        for load_name, val in restore.get('load_q_set', {}).items():
            self.network.loads.at[load_name, "q_set"] = val
    
    def _get_default_results(self) -> Tuple:
        """Return default results when power flow fails."""
        return (
            np.ones(self.num_nodes) * 0.95,
            np.zeros(self.num_nodes),
            np.zeros(self.num_edges),
            np.zeros(self.num_nodes),
            np.zeros(self.num_edges),
            False
        )


class FrequencyDynamicsSimulator:
    """
    System frequency dynamics simulator.
    
    Models frequency response to generation/load imbalance
    with inertia and damping effects.
    """
    
    def __init__(
        self,
        num_nodes: int,
        node_types: np.ndarray,
        gen_capacity: np.ndarray,
        base_frequency: float = Settings.Frequency.BASE_FREQUENCY
    ):
        """
        Initialize frequency dynamics simulator.

        Args:
            num_nodes: Number of nodes
            node_types: Node types
            gen_capacity: Generator capacities
            base_frequency: Base frequency (Hz)
        """
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.base_frequency = base_frequency

        # Initialize inertia constants
        self.inertia = self._initialize_inertia(gen_capacity)
        self.damping = np.random.uniform(Settings.Frequency.DAMPING_MIN, Settings.Frequency.DAMPING_MAX, num_nodes)
        # Under-frequency load shedding (UFLS) settings
        self.ufls_stages = [dict(s) for s in Settings.UFLS.STAGES]
        # UFLS state: each stage is a one-shot relay; cumulative shed tracked here
        self.ufls_activated = [False] * len(self.ufls_stages)
        self.ufls_shed_factor = 1.0  # multiplier applied to load after UFLS

    def reset_ufls(self):
        """Reset UFLS relay state at the start of each scenario."""
        self.ufls_activated = [False] * len(self.ufls_stages)
        self.ufls_shed_factor = 1.0
    
    def _initialize_inertia(self, gen_capacity: np.ndarray) -> np.ndarray:
        """
        Initialize generator inertia constants.
        
        Args:
            gen_capacity: Generator capacities
            
        Returns:
            Inertia constants (H in seconds)
        """
        inertia = np.zeros(self.num_nodes)
        gen_indices = np.where(self.node_types == 1)[0]
        
        for idx in gen_indices:
            # Larger generators have more inertia
            capacity_mw = gen_capacity[idx]
            if capacity_mw > Settings.Frequency.INERTIA_LARGE_MW_THRESHOLD:
                inertia[idx] = np.random.uniform(Settings.Frequency.INERTIA_LARGE_MIN, Settings.Frequency.INERTIA_LARGE_MAX)
            elif capacity_mw > Settings.Frequency.INERTIA_MEDIUM_MW_THRESHOLD:
                inertia[idx] = np.random.uniform(Settings.Frequency.INERTIA_MEDIUM_MIN, Settings.Frequency.INERTIA_MEDIUM_MAX)
            else:
                inertia[idx] = np.random.uniform(Settings.Frequency.INERTIA_SMALL_MIN, Settings.Frequency.INERTIA_SMALL_MAX)
        
        return inertia
    
    def update_frequency(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        current_frequency: float,
        dt: float = 1.0  # timestep in minutes
    ) -> Tuple[float, np.ndarray]:
        """
        Update system frequency based on power imbalance.

        Uses a steady-state droop model rather than integrating the raw swing
        equation over a full minute. The swing equation describes sub-second
        transients; over a 1-minute timestep, governor and AGC have already
        acted, so the appropriate model is:

            Δf_ss = -f0 * ΔP / (D_total * S_base)   [Hz]

        where D_total is the combined governor droop + load damping coefficient.

        We then apply a first-order lag so frequency doesn't jump instantly:
            f[t] = f[t-1] + (f_target - f[t-1]) * (1 - exp(-dt / tau_gov))

        Args:
            generation: Generation at each node (MW)
            load: Load at each node (MW)
            current_frequency: Current frequency (Hz)
            dt: Time step in minutes (default 1.0 min)

        Returns:
            Tuple of (new_frequency, adjusted_load)
        """
        total_gen = generation.sum()
        total_load = load.sum()
        imbalance = total_gen - total_load  # MW, positive = over-generation

        if total_gen == 0:
            return 0.0, load  # System collapsed

        # Steady-state frequency deviation from droop model
        # Governor droop R = 5% (standard) → D_gov = S_base / (R * f0)
        # Combined droop + load damping coefficient (% of load per % freq change)
        # Typical value: D_total ≈ 0.02-0.05 per unit (2-5% load change per 1% freq)
        D_total = Settings.Frequency.D_TOTAL
        system_base = max(total_load, 1.0)

        # Δf_ss = f0 * ΔP / (D_total * S_base)
        # e.g. 100 MW imbalance on 10,000 MW system → Δf = 60*100/(0.04*10000) = 1.5 Hz
        delta_f_ss = self.base_frequency * imbalance / (D_total * system_base)

        # Clamp steady-state target to ±DELTA_F_MAX_HZ (physical limit before collapse)
        delta_f_ss = np.clip(delta_f_ss, -Settings.Frequency.DELTA_F_MAX_HZ, Settings.Frequency.DELTA_F_MAX_HZ)
        f_target = self.base_frequency + delta_f_ss

        # First-order lag: governor response time constant
        tau_gov = Settings.Frequency.TAU_GOV_MIN  # minutes
        alpha = 1.0 - np.exp(-dt / tau_gov)
        new_frequency = current_frequency + alpha * (f_target - current_frequency)
        new_frequency = np.clip(new_frequency, Settings.Frequency.FREQ_MIN_HZ, Settings.Frequency.FREQ_MAX_HZ)

        # Under-frequency load shedding (one-shot per stage)
        for i, stage in enumerate(self.ufls_stages):
            if new_frequency < stage['frequency'] and not self.ufls_activated[i]:
                self.ufls_shed_factor *= (1.0 - stage['load_shed'])
                self.ufls_activated[i] = True

        adjusted_load = load * self.ufls_shed_factor
        return new_frequency, adjusted_load


class ThermalDynamicsSimulator:
    """
    Equipment thermal dynamics simulator.
    
    Models heating and cooling of equipment based on loading.
    """
    
    def __init__(
        self,
        num_nodes: int,
        thermal_time_constant: np.ndarray,
        thermal_capacity: np.ndarray,
        cooling_effectiveness: np.ndarray,
        ambient_temperature: float = Settings.Thermal.AMBIENT_TEMP_C
    ):
        """
        Initialize thermal dynamics simulator.

        Args:
            num_nodes: Number of nodes
            thermal_time_constant: Time constants (minutes)
            thermal_capacity: Thermal capacities
            cooling_effectiveness: Cooling effectiveness
            ambient_temperature: Ambient temperature (°C)
        """
        self.num_nodes = num_nodes
        self.thermal_time_constant = thermal_time_constant
        self.thermal_capacity = thermal_capacity
        self.cooling_effectiveness = cooling_effectiveness
        self.ambient_temperature = ambient_temperature
        
        # Initialize temperatures
        self.temperatures = np.full(num_nodes, ambient_temperature)
    
    def update_temperatures(
        self,
        heat_generation: np.ndarray,
        dt: float = Settings.Thermal.DT_MINUTES
    ) -> np.ndarray:
        """
        Update equipment temperatures based on loading.
        
        Args:
            heat_generation: heat generation at each node
            dt: Time step (minutes)
            
        Returns:
            Updated temperatures
        """

        # Heat dissipation (proportional to temperature difference)
        temp_diff = self.temperatures - self.ambient_temperature
        heat_dissipation = (
            self.cooling_effectiveness * temp_diff / self.thermal_time_constant
        )
        
        # Temperature change
        dT = (heat_generation / self.thermal_capacity - heat_dissipation) * dt
        
        # Update temperatures
        self.temperatures += dT
        self.temperatures += np.random.normal(0, Settings.Thermal.TEMP_NOISE_STD, self.num_nodes)

        # Clip to reasonable range
        self.temperatures = np.clip(
            self.temperatures,
            self.ambient_temperature - Settings.Thermal.AMBIENT_BUFFER_C,
            Settings.Thermal.TEMP_MAX_C,
        )
        
        return self.temperatures
    
    def reset_temperatures(self):
        """Reset all temperatures to ambient."""
        self.temperatures = np.full(self.num_nodes, self.ambient_temperature)
