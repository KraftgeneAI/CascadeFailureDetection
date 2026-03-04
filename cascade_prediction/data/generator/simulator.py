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


class PhysicsBasedGridSimulator:
    """
    Complete physics-based power grid simulator for cascade failure generation.
    
    This class orchestrates all simulation components to generate realistic
    cascade failure scenarios with multi-modal data (infrastructure, environmental,
    and robotic sensor data).
    
    SCENARIO TYPES:
    ---------------
    - NORMAL (stress 0.3-0.7): Grid operates safely, no failures
    - STRESSED (stress 0.8-0.9): High load but no failures (near-miss scenarios)
    - CASCADE (stress > 0.9): Failures propagate through grid
    
    MULTI-MODAL DATA:
    -----------------
    1. Infrastructure (SCADA/PMU): Real physics measurements
    2. Environmental (Satellite/Weather): Correlated synthetic data
    3. Robotic (Drone sensors): Equipment condition indicators
    """
    
    def __init__(
        self,
        num_nodes: int = 118,
        seed: int = 42,
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
        self.line_reactance = np.random.uniform(0.0003, 0.0005, self.num_edges) * distances
        self.line_reactance = np.maximum(self.line_reactance, 1e-6)  # Minimum value
        self.line_resistance = self.line_reactance * 0.1
        self.line_resistance = np.maximum(self.line_resistance, 1e-7)  # Minimum value
        self.line_susceptance = np.random.uniform(1e-6, 3e-6, self.num_edges) * distances
        self.line_conductance = np.zeros(self.num_edges)
        total_load = self.base_load.sum()
        avg_flow_per_line = total_load / self.num_edges  # Average flow
        
        self.thermal_limits = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            
            # Estimate flow based on distance and connected nodes
            # Shorter lines carry more power (distribution)
            # Longer lines carry less power (transmission)
            if distances[i] < 30:
                # Short lines: high capacity (distribution)
                base_capacity = avg_flow_per_line * np.random.uniform(1.5, 2.5)
            elif distances[i] < 60:
                # Medium lines: moderate capacity
                base_capacity = avg_flow_per_line * np.random.uniform(1.0, 1.8)
            else:
                # Long lines: lower capacity (but still adequate)
                base_capacity = avg_flow_per_line * np.random.uniform(0.8, 1.5)
            
            # Add margin for convergence (150-200% of expected flow)
            margin = np.random.uniform(1.5, 2.0)
            self.thermal_limits[i] = base_capacity * margin

        # Initialize physics simulators
        print(f"Initializing physics simulators...")
        self.power_flow_sim = PowerFlowSimulator(
            self.num_nodes, self.edge_index.numpy(), self.positions, self.node_types, self.gen_capacity,
            self.line_reactance, self.line_resistance,
            self.line_susceptance, self.line_conductance, self.thermal_limits
        )
        
        self.frequency_sim = FrequencyDynamicsSimulator(
            self.num_nodes, self.node_types, self.gen_capacity
        )
        
        self.thermal_sim = ThermalDynamicsSimulator(
            self.num_nodes, self.thermal_capacity, self.cooling_effectiveness,
            self.thermal_time_constant
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
            - 0.3-0.7: Normal operation
            - 0.8-0.9: Stressed operation
            - 0.9-1.0: Critical stress (cascade likely)
        sequence_length : int
            Number of timesteps to simulate (default: 30)
        
        Returns:
        --------
        scenario : Dict or None
            Complete scenario data if successful, None if generation failed
        """
        print(f"  [INPUT] Generating scenario with stress_level: {stress_level:.3f}")
        
        # Determine cascade start time
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        # Initialize generation and load
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        
        # Set load based on stress level
        load_multiplier = 0.7 + stress_level * 0.4
        load_noise = 0.05
        load_values = self.base_load * load_multiplier * (
            1 + np.random.normal(0, load_noise, self.num_nodes)
        )
        
        # Size generation to match load
        total_load = load_values.sum()
        gen_indices = np.where(self.node_types == 1)[0]
        total_capacity = self.gen_capacity.sum()
        for idx in gen_indices:
            if total_capacity > 0:
                generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
        
        # Run initial power flow (no failures yet)
        voltages, angles, line_flows, node_reactive, line_flows_q, is_stable = (
            self.power_flow_sim.compute_power_flow(generation, load_values, [], [])
        )
        
        # Calculate loading ratios
        loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
        
        # Initialize thermal state
        ambient_temp_base = 25 + 10 * np.random.rand()
        self.thermal_sim.ambient_temperature = ambient_temp_base
        self.thermal_sim.reset_temperatures()

        src, dst = self.edge_index
        
        # Calculate heat generation per node
        heat_generation = self._get_heat_generation(loading_ratios)

        equipment_temps = self.thermal_sim.update_temperatures(heat_generation, 1.0)
        
        # Calculate node loading
        node_loading = load_values / (self.base_load + 1e-6)
        
        # Calculate frequency
        current_frequency = 60.0 - (node_loading.mean() - 0.9) * 5
        current_frequency = np.clip(current_frequency, 58.0, 60.5)
        
        # Check for initial failures
        initial_failed_nodes = []
        failed_reasons = []
        
        for n in range(self.num_nodes):
            state, reason = self.cascade_sim.check_node_state(
                n, node_loading[n], voltages[n], equipment_temps[n], current_frequency
            )
            
            if state == 2:  # Full failure
                initial_failed_nodes.append(n)
                failed_reasons.append(reason)
                generation[n]=0
                load_values[n]=0
        
        # Determine if cascade occurs
        if len(initial_failed_nodes) > 0:
            is_cascade = True
            print(f"  [CASCADE] Nodes {initial_failed_nodes} FAILED. Reasons: {failed_reasons}")
            
            # Propagate cascade
            failure_sequence = self.cascade_sim.propagate_cascade_physics(
                list(zip(initial_failed_nodes, failed_reasons)),
                generation, load_values, equipment_temps, current_frequency,
                int(self.num_nodes * 0.6),
                self.power_flow_sim, self.edge_index.numpy(), self.thermal_limits
            )
            
            failed_nodes = [node for node, _, _ in failure_sequence]
            failure_times = [time for _, time, _ in failure_sequence]
            failure_reasons = [r for _, _, r in failure_sequence]
            
            # Compute risk vector
            risk_vec = self._compute_risk_vector(
                stress_level, failed_reasons[0] if failed_reasons else "none",
                len(failed_nodes)
            )
            ground_truth_risk = risk_vec
            
        else:
            is_cascade = False
            if stress_level > 0.8:
                print(f"  [STRESSED] No failures (Voltage: {voltages.min():.3f})")
            else:
                print(f"  [NORMAL] No failures")
            
            failed_nodes, failure_times, failure_reasons = [], [], []
            cascade_start_time = -1
            ground_truth_risk = np.array([
                stress_level, stress_level*0.7, stress_level*0.5,
                0.1, 0.1, 0.1, stress_level
            ], dtype=np.float32)
        
        # Generate time series
        scenario_data = self._generate_time_series(
            stress_level, sequence_length, cascade_start_time,
            failed_nodes, failure_times, ambient_temp_base
        )
        
        if scenario_data is None:
            return None
        
        # Add metadata in the same format as original multimodal_data_generator.py
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
            'base_mva': 100.0,
        }
        
        return scenario_data
    
    ## Helper function to get heat generation from loading ratios of lines
    def _get_heat_generation(self, loading_ratios: np.ndarray) -> np.ndarray:
        src, dst = self.edge_index
        
        # Calculate heat generation per node
        heat_generation = np.zeros(self.num_nodes)
        
        for i in range(self.num_edges):
            s, d = src[i].item(), dst[i].item()
            heat = (loading_ratios[i] ** 2) * self.line_resistance[i] * 100
            heat_generation[s] += heat / 2
            heat_generation[d] += heat / 2
        return heat_generation

    def _generate_time_series(
        self,
        stress_level: float,
        sequence_length: int,
        cascade_start_time: int,
        failed_nodes: List[int],
        failure_times: List[float],
        ambient_temp_base: float
    ) -> Optional[Dict]:
        """
        Generate time series data for all timesteps.
        
        This method simulates the grid evolution over time, applying failures
        at the correct timesteps and generating all multi-modal data.
        """
        # Build timestep to failed nodes mapping
        timestep_to_failed_nodes = {}
        for i, node in enumerate(failed_nodes):
            failure_time = failure_times[i]
            if cascade_start_time >= 0:
                failure_timestep = cascade_start_time + int(failure_time if failure_time else 0)
            else:
                failure_timestep = -1
            
            if failure_timestep >= sequence_length:
                continue
            
            if failure_timestep not in timestep_to_failed_nodes:
                timestep_to_failed_nodes[failure_timestep] = []
            timestep_to_failed_nodes[failure_timestep].append(node)
        
        # Initialize storage
        sequence = []
        current_frequency = 60.0
        self.thermal_sim.ambient_temperature = ambient_temp_base
        self.thermal_sim.reset_temperatures()
        
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        cumulative_failed_nodes = set()
        
        is_cascade = len(failed_nodes) > 0
        
        # Simulate each timestep
        for t in range(sequence_length):
            # Determine current stress (ramp up before cascade)
            if is_cascade and t < cascade_start_time:
                ramp_factor = 0.6 + 0.4 * (t / max(1, cascade_start_time - 1))
                current_stress = stress_level * ramp_factor
            else:
                current_stress = stress_level
            
            # Set load
            if is_cascade:
                load_multiplier = 0.7 + current_stress * 0.4
                load_noise = 0.05
            else:
                load_multiplier = 0.5 + current_stress * 0.4
                load_noise = 0.02
            
            load_values = self.base_load * load_multiplier * (
                1 + np.random.normal(0, load_noise, self.num_nodes)
            )
            
            # Size generation
            total_load = load_values.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            total_capacity = self.gen_capacity.sum()
            for idx in gen_indices:
                if total_capacity > 0:
                    generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
            
            # Apply failures at this timestep
            if t in timestep_to_failed_nodes:
                cumulative_failed_nodes.update(timestep_to_failed_nodes[t])
            
            failed_nodes_t = list(cumulative_failed_nodes)
            
            # Calculate failed lines from failed nodes
            failed_lines_t = get_failed_lines_from_nodes(
                self.edge_index.numpy(), cumulative_failed_nodes
            )
            
            # Run power flow
            voltages, angles, line_flows, node_reactive, line_flows_q, is_stable = (
                self.power_flow_sim.compute_power_flow(
                    generation, load_values, failed_lines_t, failed_nodes_t
                )
            )
            
            num_failed = len(failed_nodes_t)
            failure_ratio = num_failed / self.num_nodes

            # Check stability
            if not is_stable:
                if is_cascade:
                    if failure_ratio >= 0.9:
                        print(f"  [COMPLETE] Grid collapse complete ({num_failed}/{self.num_nodes} nodes failed = {failure_ratio*100:.1f}%). Generating final timestep.")
                    else:
                        print(f"  [UNSTABLE] Power flow unstable at timestep {t} ({num_failed}/{self.num_nodes} nodes failed = {failure_ratio*100:.1f}%). Continuing to capture cascade progression...")
                else:
                    print(f"  [REJECT] Power flow unstable in NORMAL scenario at timestep {t}. This should not happen. Rejecting scenario.")
                    return None
            
            # Update physics
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            heat_generation = self._get_heat_generation(loading_ratios)

            current_frequency, load_values = self.frequency_sim.update_frequency(
                generation, load_values, current_frequency, 1.0
            )
            
            ambient_temp = ambient_temp_base + 8 * np.sin(2 * np.pi * ((t / 60.0) - 6) / 24)
            self.thermal_sim.ambient_temperature = ambient_temp
            equipment_temps = self.thermal_sim.update_temperatures(heat_generation, 1.0)
            
            # Generate multi-modal data
            sat_data, weather_seq, threat_ind = self.env_gen.generate_correlated_environmental_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, current_stress
            )
            
            vis_data, thermal_data, sensor_data = self.robot_gen.generate_correlated_robotic_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, equipment_temps
            )
            
            # Compute cascade timing
            current_cascade_timing = self._compute_cascade_timing(
                t, cascade_start_time, failed_nodes, failure_times, cumulative_failed_nodes
            )
            
            # Store timestep data
            timestep_data = self._package_timestep_data(
                t, current_stress, voltages, angles, generation, node_reactive,
                load_values, equipment_temps, current_frequency, loading_ratios,
                line_flows, line_flows_q, sat_data, weather_seq, threat_ind,
                vis_data, thermal_data, sensor_data, cumulative_failed_nodes,
                current_cascade_timing, sequence_length
            )
            
            sequence.append(timestep_data)
        
        # Package scenario
        return self._package_scenario(sequence, failed_nodes)
    
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
        cascade_start_time: int,
        failed_nodes: List[int],
        failure_times: List[float],
        cumulative_failed_nodes: set
    ) -> np.ndarray:
        """Compute cascade timing for current timestep."""
        if cascade_start_time >= 0:
            current_cascade_timing = np.array([
                (failure_times[failed_nodes.index(node)] - (t - cascade_start_time)
                 if t >= cascade_start_time else failure_times[failed_nodes.index(node)])
                if node in failed_nodes else -1.0
                for node in range(self.num_nodes)
            ], dtype=np.float32)
        else:
            current_cascade_timing = np.full(self.num_nodes, -1.0, dtype=np.float32)
        
        # Set timing to 0 for nodes that just failed
        for node in cumulative_failed_nodes:
            if node in failed_nodes:
                idx = failed_nodes.index(node)
                if cascade_start_time >= 0:
                    failure_timestep = cascade_start_time + int(failure_times[idx])
                    if t == failure_timestep:
                        current_cascade_timing[node] = 0.0
        
        return current_cascade_timing
    
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

            'scada_data': np.column_stack([
                voltages, 
                angles, 
                generation,
                node_reactive,
                load_values,
                equipment_temps,
                np.full(self.num_nodes, current_frequency),
                self.equipment_age, 
                self.equipment_condition,
                self.gen_capacity,
                self.base_load, 
                self.node_types,
                np.full(self.num_nodes, t / sequence_length),
                np.full(self.num_nodes, current_stress),
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

