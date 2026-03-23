"""
Physics-Based Grid Simulator Module
====================================

PURPOSE:
--------
Orchestrates the complete physics-based simulation of power grid cascade failures.
This module integrates topology, physics, cascade, environmental, and robotic
components to generate realistic multi-modal training scenarios.

SIMULATION PROCESS:
-------------------
1. Initialize grid topology and properties
2. Determine scenario type based on stress level
3. Check for initial failures using physics simulation
4. Propagate cascade if failures occur
5. Generate time series data for all timesteps
6. Compute ground truth labels

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import pypsa

from .topology import GridTopologyGenerator, NodePropertyInitializer
from .physics import PowerFlowSimulator, FrequencyDynamicsSimulator, ThermalDynamicsSimulator
from .cascade import CascadeSimulator, create_adjacency_list
from .environmental import EnvironmentalDataGenerator
from .robotic import RoboticDataGenerator
from .utils import get_failed_lines_from_nodes
from .config import Settings


class PhysicsBasedGridSimulator:
    """
    Complete physics-based power grid simulator for cascade failure generation.
    
    This class orchestrates all simulation components to generate realistic
    cascade failure scenarios with multi-modal data (infrastructure, environmental,
    and robotic sensor data).
    
    SCENARIO TYPES:
    ---------------
    - NORMAL   (stress 0.00–0.55): Grid operates safely, no failures expected
    - STRESSED (stress 0.55–0.72): High load but no cascade (near-miss scenarios)
    - CASCADE  (stress 0.72–1.00): Failures propagate through the grid
    
    MULTI-MODAL DATA:
    -----------------
    1. Infrastructure (SCADA/PMU): Real physics measurements
    2. Environmental (Satellite/Weather): Correlated synthetic data
    3. Robotic (Drone sensors): Equipment condition indicators
    """
    
    def __init__(
        self,
        num_nodes: int = Settings.Scenario.DEFAULT_NUM_NODES,
        seed: int = Settings.Scenario.DEFAULT_SEED,
        topology_file: Optional[str] = None
    ):
        """
        Initialize the physics-based grid simulator.
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the power grid (default: 118)
        seed : int
            Random seed for reproducibility
        topology_file : str, optional
            Path to saved topology file (if None, generates new topology)
        """
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize topology
        print(f"Initializing grid topology...")
        topo_gen = GridTopologyGenerator(num_nodes, seed)
        
        if topology_file:
            topo_data = topo_gen.load_topology(topology_file)
            self.adjacency_matrix = topo_data['adjacency_matrix']
            self.edge_index = topo_data['edge_index']
            self.positions = topo_data['positions']
            self.num_nodes = self.adjacency_matrix.shape[0]
        else:
            topo_data = topo_gen.generate_topology()
            self.adjacency_matrix = topo_data['adjacency_matrix']
            self.edge_index = topo_data['edge_index']
            self.positions = topo_data['positions']
        
        self.num_edges = self.edge_index.shape[1]
        
        # Initialize node properties
        print(f"Initializing node properties...")
        node_init = NodePropertyInitializer(self.num_nodes, seed)
        node_props = node_init.initialize_properties()
        
        # Store node properties
        self.node_types = node_props['node_types']
        self.base_load = node_props['base_load']
        self.gen_capacity = node_props['gen_capacity']
        self.equipment_age = node_props['equipment_age']
        self.equipment_condition = node_props['equipment_condition']
        
        # Failure thresholds
        self.loading_failure_threshold = node_props['loading_failure_threshold']
        self.loading_damage_threshold = node_props['loading_damage_threshold']
        self.voltage_failure_threshold = node_props['voltage_failure_threshold']
        self.voltage_damage_threshold = node_props['voltage_damage_threshold']
        self.temperature_failure_threshold = node_props['temperature_failure_threshold']
        self.temperature_damage_threshold = node_props['temperature_damage_threshold']
        self.frequency_failure_threshold = node_props['frequency_failure_threshold']
        self.frequency_damage_threshold = node_props['frequency_damage_threshold']
        
        # Thermal properties
        self.thermal_capacity = node_props['thermal_capacity']
        self.cooling_effectiveness = node_props['cooling_effectiveness']
        self.thermal_time_constant = node_props['thermal_time_constant']
        
        # Initialize edge properties
        print(f"Initializing edge properties...")
        src, dst = self.edge_index
        distances = np.linalg.norm(
            self.positions[src] - self.positions[dst], axis=1
        )
        
        # PyPSA uses sn_mva=100 MVA as system base.
        # z_base = v_nom^2 / sn_mva = 138^2 / 100 = 190.44 Ω
        # Target x_pu in 0.05–0.20 range for well-conditioned Newton-Raphson.
        # Normalize distances to [0,1] so shorter lines get lower reactance.
        dist_norm = distances / (distances.max() + 1e-6)  # 0..1

        # x_pu: short lines ~0.03, long lines ~0.15 (realistic transmission range)
        x_pu = (
            np.random.uniform(Settings.LineImpedance.X_PU_BASE_MIN, Settings.LineImpedance.X_PU_BASE_MAX, self.num_edges)
            + dist_norm * np.random.uniform(Settings.LineImpedance.X_PU_DIST_MIN, Settings.LineImpedance.X_PU_DIST_MAX, self.num_edges)
        )
        x_pu = np.clip(x_pu, Settings.LineImpedance.X_PU_CLIP_MIN, Settings.LineImpedance.X_PU_CLIP_MAX)

        z_base = Settings.PowerSystem.Z_BASE  # 190.44 Ω  (sn_mva = 100 MVA)
        self.line_reactance = x_pu * z_base          # Ω, ~4–38 Ω range
        self.line_resistance = self.line_reactance * np.random.uniform(
            Settings.LineImpedance.R_X_RATIO_MIN, Settings.LineImpedance.R_X_RATIO_MAX, self.num_edges
        )

        # Shunt susceptance: b_pu ~ 1e-4 per unit  →  b_siemens = b_pu / z_base
        b_pu = np.random.uniform(Settings.LineImpedance.B_PU_MIN, Settings.LineImpedance.B_PU_MAX, self.num_edges)
        self.line_susceptance = b_pu / z_base        # Siemens
        self.line_conductance = np.zeros(self.num_edges)
        total_load = self.base_load.sum()
        avg_flow_per_line = total_load / self.num_edges  # Average flow
        
        self.thermal_limits = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            
            # Estimate flow based on distance and connected nodes
            # Shorter lines carry more power (distribution)
            # Longer lines carry less power (transmission)
            if distances[i] < Settings.LineImpedance.THERMAL_SHORT_DIST_KM:
                # Short lines: high capacity (distribution)
                base_capacity = avg_flow_per_line * np.random.uniform(
                    Settings.LineImpedance.THERMAL_SHORT_MIN, Settings.LineImpedance.THERMAL_SHORT_MAX)
            elif distances[i] < Settings.LineImpedance.THERMAL_MEDIUM_DIST_KM:
                # Medium lines: moderate capacity
                base_capacity = avg_flow_per_line * np.random.uniform(
                    Settings.LineImpedance.THERMAL_MEDIUM_MIN, Settings.LineImpedance.THERMAL_MEDIUM_MAX)
            else:
                # Long lines: lower capacity (but still adequate)
                base_capacity = avg_flow_per_line * np.random.uniform(
                    Settings.LineImpedance.THERMAL_LONG_MIN, Settings.LineImpedance.THERMAL_LONG_MAX)

            # Add margin for convergence
            margin = np.random.uniform(Settings.LineImpedance.THERMAL_MARGIN_MIN, Settings.LineImpedance.THERMAL_MARGIN_MAX)
            self.thermal_limits[i] = base_capacity * margin

        # Ensure minimum thermal limits
        min_thermal_limit = total_load * Settings.LineImpedance.THERMAL_MIN_FRACTION
        self.thermal_limits = np.maximum(self.thermal_limits, min_thermal_limit)
        
        # Initialize physics simulators
        print(f"Initializing physics simulators...")
        self.power_flow_sim = PowerFlowSimulator(
            self.num_nodes, self.edge_index.numpy(), self.positions, self.node_types, self.gen_capacity,
            self.line_reactance, self.line_resistance,
            self.line_susceptance, self.line_conductance, self.thermal_limits,
        )
        
        self.frequency_sim = FrequencyDynamicsSimulator(
            self.num_nodes, self.node_types, self.gen_capacity
        )
        
        self.thermal_sim = ThermalDynamicsSimulator(
            self.num_nodes, self.thermal_time_constant, self.thermal_capacity, self.cooling_effectiveness
        )
        
        # Initialize cascade simulator
        print(f"Initializing cascade simulator...")
        adjacency_list = create_adjacency_list(
            self.edge_index.numpy(), self.node_types
        )
        
        self.cascade_sim = CascadeSimulator(
            self.num_nodes, adjacency_list,
            self.loading_failure_threshold, self.loading_damage_threshold,
            self.voltage_failure_threshold, self.voltage_damage_threshold,
            self.temperature_failure_threshold, self.temperature_damage_threshold,
            self.frequency_failure_threshold, self.frequency_damage_threshold
        )
        
        # Initialize environmental and robotic generators
        print(f"Initializing environmental and robotic generators...")
        self.env_gen = EnvironmentalDataGenerator(
            self.num_nodes, self.positions, self.edge_index.numpy()
        )
        
        self.robot_gen = RoboticDataGenerator(
            self.num_nodes, self.equipment_age, self.equipment_condition
        )
        
        print(f"[OK] Initialized grid: {self.num_nodes} nodes, {self.num_edges} edges")
    
    def generate_scenario(
        self,
        stress_level: float,
        sequence_length: int = 30
    ) -> Optional[Dict]:
        """
        Generate a complete power grid scenario with multi-modal data.
        
        This is the main function that orchestrates the entire simulation process.
        
        PROCESS:
        --------
        1. Determine scenario type (normal/stressed/cascade)
        2. Check for initial failures
        3. Propagate cascade if failures occur
        4. Generate time series for all timesteps
        5. Compute ground truth labels
        
        Parameters:
        -----------
        stress_level : float
            Grid stress level (0.0 to 1.0)
            - 0.00–0.55: Normal operation
            - 0.55–0.72: Stressed operation (near-miss)
            - 0.72–1.00: Critical stress (cascade likely)
        sequence_length : int
            Number of timesteps to simulate (default: 30)
        
        Returns:
        --------
        scenario : Dict or None
            Complete scenario data if successful, None if generation failed
        """
        print(f"  [INPUT] Generating scenario with stress_level: {stress_level:.3f}")

        # Pick a random ambient temperature base for this scenario
        ambient_temp_base = (
            Settings.Simulation.AMBIENT_BASE_MIN_C
            + (Settings.Simulation.AMBIENT_BASE_MAX_C - Settings.Simulation.AMBIENT_BASE_MIN_C)
            * np.random.rand()
        )

        # Generate time series — failures emerge dynamically from physics
        scenario_data = self._generate_time_series(
            stress_level, sequence_length, ambient_temp_base
        )

        if scenario_data is None:
            return None

        # Extract dynamically discovered failures from time series
        failed_nodes     = scenario_data.pop('failed_nodes')
        failure_times    = scenario_data.pop('failure_times')
        failure_reasons  = scenario_data.pop('failure_reasons')
        cascade_start_time = scenario_data.pop('actual_cascade_start')
        is_cascade = len(failed_nodes) > 0

        # Compute risk vector
        if is_cascade:
            ground_truth_risk = self._compute_risk_vector(
                stress_level,
                failure_reasons[0] if failure_reasons else 'none',
                len(failed_nodes)
            )
        else:
            if stress_level > Settings.Scenario.STRESSED_STRESS_MAX:
                print(f"  [STRESSED] No failures at stress={stress_level:.3f}")
            else:
                print(f"  [NORMAL] No failures")
            ground_truth_risk = np.array([
                stress_level, stress_level*0.7, stress_level*0.5,
                0.1, 0.1, 0.1, stress_level
            ], dtype=np.float32)

        scenario_data['metadata'] = {
            'cascade_start_time': cascade_start_time,
            'failed_nodes': failed_nodes,
            'failure_times': failure_times,
            'failure_reasons': failure_reasons,
            'ground_truth_risk': ground_truth_risk,
            'is_cascade': is_cascade,
            'stress_level': stress_level,
            'num_nodes': self.num_nodes,
            'num_edges': len(self.edge_index[0]),
            'base_mva': Settings.PowerSystem.SN_MVA,
        }

        return scenario_data
    
    def _get_heat_generation(self, loading_ratios: np.ndarray) -> np.ndarray:
        """
        Compute per-node heat generation from line loading ratios.

        Heat is proportional to I²R.  We use (apparent loading ratio)² as a
        dimensionless current² proxy so the scale stays consistent with the
        thermal model tuning regardless of the chosen resistance units.
        Each line's heat is split equally between its two endpoint nodes.
        """
        src, dst = self.edge_index
        heat_per_line = loading_ratios ** 2          # dimensionless, 0–(several) range
        heat_generation = np.zeros(self.num_nodes)
        np.add.at(heat_generation, src.numpy(), heat_per_line / 2)
        np.add.at(heat_generation, dst.numpy(), heat_per_line / 2)
        return heat_generation

    def _generate_time_series(
        self,
        stress_level: float,
        sequence_length: int,
        ambient_temp_base: float
    ) -> Optional[Dict]:
        """
        Generate time series data for all timesteps.
        Failures emerge dynamically from physics — no pre-computed cascade sequence.
        """
        sequence = []
        current_frequency = Settings.Frequency.BASE_FREQUENCY
        self.thermal_sim.ambient_temperature = ambient_temp_base
        self.thermal_sim.reset_temperatures()
        self.frequency_sim.reset_ufls()

        # Draw once per scenario so precursor signals are temporally coherent
        # across all timesteps (same window edge for every call this scenario).
        scenario_precursor_duration = int(np.random.randint(8, 20))

        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        cumulative_failed_nodes: set = set()
        prev_voltages: Optional[np.ndarray] = None

        # failure_record: node → (absolute timestep of failure, reason string)
        failure_record: Dict[int, Tuple[int, str]] = {}
        cascade_start_time = -1

        # Stress ramp: rise from 60% to 100% of stress_level over first RAMP_FRACTION of sequence
        earlest_cascade_time = max(1, int(sequence_length * (np.random.uniform(Settings.Simulation.RAMP_FRACTION_MIN, Settings.Simulation.RAMP_FRACTION_MAX))))
        alpha = (0.75/stress_level - 0.6) / earlest_cascade_time
        if alpha * (sequence_length - 1) + 0.6 < 0.75:
            alpha = 0.16 / (sequence_length - 1)
        for t in range(sequence_length):
            # Current stress level
            if stress_level > Settings.Scenario.CASCADE_STRESS_MIN:
                current_stress = stress_level * (0.6 + (t * alpha))
                if current_stress > stress_level >= 0.75:
                    current_stress = stress_level
            else:
                current_stress = stress_level

            # Load calculation
            load_noise = (
                Settings.Simulation.LOAD_NOISE_HIGH_STRESS
                if stress_level > Settings.Scenario.CASCADE_STRESS_MIN
                else Settings.Simulation.LOAD_NOISE_LOW_STRESS
            )
            load_multiplier = (
                Settings.Simulation.LOAD_MULT_HIGH_BASE + current_stress * Settings.Simulation.LOAD_MULT_HIGH_SLOPE
                if stress_level > Settings.Scenario.CASCADE_STRESS_MIN
                else Settings.Simulation.LOAD_MULT_LOW_BASE + current_stress * Settings.Simulation.LOAD_MULT_LOW_SLOPE
            )
            noise = np.clip(
                np.random.normal(0, load_noise, self.num_nodes),
                -Settings.Simulation.LOAD_NOISE_CLIP_FACTOR * load_noise,
                Settings.Simulation.LOAD_NOISE_CLIP_FACTOR * load_noise,
            )
            load_values = self.base_load * load_multiplier * (1 + noise)
            load_values *= self.frequency_sim.ufls_shed_factor

            # Zero out failed nodes
            for n in cumulative_failed_nodes:
                load_values[n] = 0.0

            # Size generation to match load — only from active (non-failed) generators
            total_load = load_values.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            active_gen_indices = [idx for idx in gen_indices if idx not in cumulative_failed_nodes]
            active_capacity = self.gen_capacity[active_gen_indices].sum() if len(active_gen_indices) > 0 else 0.0
            generation[:] = 0.0  # reset all first
            for idx in active_gen_indices:
                if active_capacity > 0:
                    generation[idx] = (self.gen_capacity[idx] / active_capacity) * total_load * Settings.Simulation.GENERATION_MARGIN

            failed_nodes_t = list(cumulative_failed_nodes)
            failed_lines_t = get_failed_lines_from_nodes(
                self.edge_index.numpy(), cumulative_failed_nodes
            )
            # Run power flow
            voltages, angles, line_flows, node_reactive, line_flows_q, is_stable = (
                self.power_flow_sim.compute_power_flow(
                    generation, load_values, failed_lines_t, failed_nodes_t
                )
            )

            # Track last stable voltages for voltage-collapse fallback
            if is_stable:
                prev_voltages = voltages.copy()

            failure_ratio = len(cumulative_failed_nodes) / self.num_nodes
            if not is_stable:
                if failure_ratio >= Settings.Simulation.COLLAPSE_FAILURE_RATIO:
                    print(f"  [COMPLETE] Grid collapse ({len(cumulative_failed_nodes)}/{self.num_nodes} nodes). Generating final timestep.")
                elif failure_ratio > 0:
                    print(f"  [UNSTABLE] t={t}, {len(cumulative_failed_nodes)}/{self.num_nodes} failed. Continuing...")
                elif t > 0 and prev_voltages is not None:
                    # Voltage collapse at high stress with no prior failures.
                    # Degrade last stable voltages to trigger voltage-threshold failures
                    # so the cascade propagates naturally rather than rejecting the scenario.
                    voltages = prev_voltages * Settings.PowerFlow.VOLTAGE_COLLAPSE_PROXY
                    print(f"  [VCOLLAPSE] t={t}, stress={current_stress:.4f}: "
                          f"voltage collapse, proxy min={voltages.min():.3f}")
                else:
                    print(f"  [REJECT] Power flow unstable at t={t} with no prior state. Rejecting.")
                    return None

            # Update physics — use apparent power (S = √(P²+Q²)) for loading
            # ratios so both active and reactive current contribute to thermal
            # loading, consistent with how propagate_cascade_physics works.
            loading_ratios = np.sqrt(line_flows**2 + line_flows_q**2) / (self.thermal_limits + 1e-6)
            heat_generation = self._get_heat_generation(loading_ratios)

            current_frequency, load_values = self.frequency_sim.update_frequency(
                generation, load_values, current_frequency, 1.0
            )

            ambient_temp = ambient_temp_base + Settings.Thermal.DIURNAL_AMPLITUDE_C * np.sin(2 * np.pi * ((t / 60.0) - 6) / 24)
            self.thermal_sim.ambient_temperature = ambient_temp
            equipment_temps = self.thermal_sim.update_temperatures(heat_generation, 1.0)

            # ----------------------------------------------------------------
            # Step 1 — Identify trigger failures.
            # Use the load-ratio proxy (actual load / base load) as the
            # initial threshold check.  This is fast and well-calibrated with
            # the existing NodeConfig failure thresholds.
            # ----------------------------------------------------------------
            node_line_loading = load_values / (self.base_load + 1e-6)

            new_failures: List[Tuple[int, str]] = []
            for n in range(self.num_nodes):
                if n in cumulative_failed_nodes:
                    continue
                state, reason = self.cascade_sim.check_node_state(
                    n, node_line_loading[n], voltages[n], equipment_temps[n], current_frequency
                )
                if state == 2:
                    new_failures.append((n, reason))

            # ----------------------------------------------------------------
            # Step 2 — Physics-based cascade propagation.
            # When trigger failures exist, recompute AC power flow after each
            # new failure to find every downstream node that overloads next.
            # This models real-world cascade speed (many nodes can trip within
            # the same simulation minute) and is consistent with the power flow
            # physics used throughout the rest of the simulation.
            # ----------------------------------------------------------------
            if new_failures:
                target_max_failures = min(
                    len(new_failures) + int(self.num_nodes * Settings.Simulation.CASCADE_MAX_SPREAD_FRACTION),
                    self.num_nodes
                )
                cascade_sequence = self.cascade_sim.propagate_cascade_physics(
                    initial_failed_nodes=new_failures,
                    generation=generation.copy(),  # propagator uses internal copies
                    load=load_values.copy(),
                    current_temperature=equipment_temps,
                    current_frequency=current_frequency,
                    target_num_failures=target_max_failures,
                    power_flow_simulator=self.power_flow_sim,
                    edge_index=self.edge_index.numpy(),
                    thermal_limits=self.thermal_limits
                )
                # Record every failure from this cascade wave (initial + propagated)
                for fail_node, fail_time_offset, fail_reason in cascade_sequence:
                    if fail_node not in cumulative_failed_nodes:
                        if cascade_start_time < 0:
                            cascade_start_time = t
                        failure_record[fail_node] = (t, fail_reason)
                        cumulative_failed_nodes.add(fail_node)
                        generation[fail_node] = 0.0
                        load_values[fail_node] = 0.0
                        print(f"  [FAIL] Node {fail_node} failed at t={t} "
                              f"(reason: {fail_reason}, cascade offset: {fail_time_offset:.2f} min)")

            # Generate multi-modal data — pass the scenario-fixed precursor
            # duration so all timesteps share the same signal window edge.
            sat_data, weather_seq, threat_ind = self.env_gen.generate_correlated_environmental_data(
                list(cumulative_failed_nodes), failed_lines_t, t, cascade_start_time, current_stress,
                precursor_duration=scenario_precursor_duration
            )
            vis_data, thermal_data, sensor_data = self.robot_gen.generate_correlated_robotic_data(
                list(cumulative_failed_nodes), failed_lines_t, t, cascade_start_time, equipment_temps,
                precursor_duration=scenario_precursor_duration
            )

            current_cascade_timing = self._compute_cascade_timing(t, failure_record)

            timestep_data = self._package_timestep_data(
                t, current_stress, voltages, angles, generation, node_reactive,
                load_values, equipment_temps, current_frequency, loading_ratios,
                line_flows, line_flows_q, sat_data, weather_seq, threat_ind,
                vis_data, thermal_data, sensor_data, cumulative_failed_nodes,
                current_cascade_timing, sequence_length
            )
            sequence.append(timestep_data)

        # Collect failure info sorted by failure time
        failed_nodes_out    = sorted(failure_record.keys(), key=lambda n: failure_record[n][0])
        failure_times_out   = [failure_record[n][0] for n in failed_nodes_out]
        failure_reasons_out = [failure_record[n][1] for n in failed_nodes_out]

        if failed_nodes_out:
            print(f"  [CASCADE] {len(failed_nodes_out)} nodes failed. First failure at t={cascade_start_time}.")

        result = self._package_scenario(sequence, failed_nodes_out)
        if result is not None:
            result['failed_nodes']        = failed_nodes_out
            result['failure_times']       = failure_times_out
            result['failure_reasons']     = failure_reasons_out
            result['actual_cascade_start'] = cascade_start_time
        return result
    
    def _compute_risk_vector(
        self,
        stress_level: float,
        initial_reason: str,
        num_failed: int
    ) -> np.ndarray:
        """Compute 7-dimensional risk vector."""
        risk_vec = np.zeros(7, dtype=np.float32)
        risk_vec[0] = stress_level  # threat_severity
        
        if 'loading' in initial_reason or 'temperature' in initial_reason:
            risk_vec[1] = 0.8  # vulnerability
            risk_vec[2] = 0.7  # operational_impact
            risk_vec[6] = 0.7  # urgency
        elif 'voltage' in initial_reason or 'frequency' in initial_reason:
            risk_vec[1] = 0.7  # vulnerability
            risk_vec[2] = 0.9  # operational_impact
            risk_vec[6] = 0.9  # urgency
        
        risk_vec[3] = 0.5 + 0.5 * (num_failed / self.num_nodes)  # cascade_probability
        
        return risk_vec
    
    def _compute_cascade_timing(
        self,
        t: int,
        failure_record: Dict[int, Tuple[int, str]]
    ) -> np.ndarray:
        """
        Time since failure for each node.
        -1.0 = not yet failed; 0.0 = failed this timestep; positive = timesteps since failure.
        """
        timing = np.full(self.num_nodes, -1.0, dtype=np.float32)
        for n, (ft, _) in failure_record.items():
            timing[n] = float(t - ft)
        return timing
    
    def _package_timestep_data(
        self, t, current_stress, voltages, angles, generation, node_reactive,
        load_values, equipment_temps, current_frequency, loading_ratios,
        line_flows, line_flows_q, sat_data, weather_seq, threat_ind,
        vis_data, thermal_data, sensor_data, cumulative_failed_nodes,
        current_cascade_timing, sequence_length
    ) -> Dict:
        """Package all data for a single timestep."""
        return {
            'satellite_data': sat_data.astype(np.float32),
            'weather_sequence': weather_seq.astype(np.float32),
            'threat_indicators': threat_ind.astype(np.float32),

            # SCADA Data Format (18 features per node)
            # CRITICAL: Keep this order synchronized with dataset.py extraction!
            # Index 0: voltages (p.u.)
            # Index 1: angles (radians)
            # Index 2: generation (MW)
            # Index 3: node_reactive (MVAr)
            # Index 4: load_values (MW)
            # Index 5: equipment_temps (°C) ← Used for ground_truth_temperature
            # Index 6: current_frequency (Hz) ← NOT temperature!
            # Index 7: equipment_age (years)
            # Index 8: equipment_condition (0-1)
            # Index 9: gen_capacity (MW)
            # Index 10: base_load (MW)
            # Index 11: node_types (0=load, 1=gen, 2=sub)
            # Index 12: time_ratio (0-1)
            # Index 13: stress_level (0-1)
            # Index 14: voltage_ratio (voltage / voltage_failure_threshold) - >1 = safe, <1 = danger
            # Index 15: temp_ratio (temp / temp_failure_threshold) - <1 = safe, >1 = danger
            # Index 16: freq_ratio (freq / freq_failure_threshold) - >1 = safe, <1 = danger
            # Index 17: loading_ratio (loading / loading_failure_threshold) - <1 = safe, >1 = danger
            'scada_data': np.column_stack([
                voltages,                                    # 0
                angles,                                      # 1
                generation,                                  # 2
                node_reactive,                               # 3
                load_values,                                 # 4
                equipment_temps,                             # 5 ← TEMPERATURE
                np.full(self.num_nodes, current_frequency), # 6 ← FREQUENCY
                self.equipment_age,                          # 7
                self.equipment_condition,                    # 8
                self.gen_capacity,                           # 9
                self.base_load,                              # 10
                self.node_types,                             # 11
                np.full(self.num_nodes, t / sequence_length), # 12
                np.full(self.num_nodes, current_stress),     # 13
                # FAILURE PROXIMITY RATIOS (CRITICAL for prediction!)
                voltages / self.voltage_failure_threshold,   # 14: >1 = safe, <1 = danger
                equipment_temps / self.temperature_failure_threshold,  # 15: <1 = safe, >1 = danger
                np.full(self.num_nodes, current_frequency) / self.frequency_failure_threshold,  # 16: >1 = safe, <1 = danger
                load_values / self.base_load / self.loading_failure_threshold,  # 17: <1 = safe, >1 = danger
            ]).astype(np.float32),

            'pmu_sequence': np.column_stack([
                voltages,
                angles,
                generation,
                load_values,
                equipment_temps,
                np.full(self.num_nodes, current_frequency),
                loading_ratios.mean() * np.ones(self.num_nodes),
                node_reactive,
            ]).astype(np.float32),

            'equipment_status': np.column_stack([
                self.equipment_age,
                self.equipment_condition,
                equipment_temps,
                self.thermal_capacity,
                self.cooling_effectiveness,
                self.thermal_time_constant / 30.0,
                (equipment_temps / self.temperature_failure_threshold),
                self.node_types,
                self.gen_capacity / (self.gen_capacity.max() + 1e-6),
                load_values / (self.base_load + 1e-6),
            ]).astype(np.float32),

            'visual_data': vis_data.astype(np.float16),
            'thermal_data': thermal_data.astype(np.float16),
            'sensor_data': sensor_data.astype(np.float16),

            'edge_attr': np.column_stack([
                self.line_reactance, self.thermal_limits, self.line_resistance,
                self.line_susceptance, self.line_conductance,
                line_flows, line_flows_q,
            ]).astype(np.float32),

            'node_labels': np.array([
                1.0 if node in cumulative_failed_nodes else 0.0
                for node in range(self.num_nodes)
            ], dtype=np.float32),

            'cascade_timing': current_cascade_timing,

            'conductance': self.line_conductance.astype(np.float32),
            'susceptance': self.line_susceptance.astype(np.float32),
            'thermal_limits': self.thermal_limits.astype(np.float32),
            'power_injection': (generation - load_values).astype(np.float32),
            'reactive_injection': (node_reactive).astype(np.float32),
        }

    
    def _package_scenario(self, sequence: List[Dict], failed_nodes: List[int]) -> Dict:
        """
        Package time series into final scenario format.
        
        Returns the same format as the original multimodal_data_generator.py:
        {
            'sequence': [list of timestep dicts],
            'edge_index': edge connectivity,
            'metadata': {metadata dict}
        }
        """
        return {
            'sequence': sequence,
            'edge_index': self.edge_index.numpy(),
        }

