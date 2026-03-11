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

import numpy as np
import pypsa
from typing import Tuple, Optional, List, Dict


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
        thermal_limits: np.ndarray
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
        """
        self.num_nodes = num_nodes
        self.num_edges = edge_index.shape[1]
        self.node_types = node_types
        
        # Initialize PyPSA network
        self.network = self._create_pypsa_network(
            edge_index, positions, gen_capacity,
            line_reactance, line_resistance,
            line_susceptance, line_conductance,
            thermal_limits
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
        thermal_limits: np.ndarray
    ) -> pypsa.Network:
        """
        Create PyPSA network from grid parameters.
        
        Returns:
            Initialized PyPSA Network
        """
        network = pypsa.Network()
        
        # Add buses (nodes)
        for i in range(self.num_nodes):
            network.add(
                "Bus",
                f"bus_{i}",
                v_nom=138.0,  # 138 kV transmission
                x=positions[i, 0],
                y=positions[i, 1]
            )
        
        # Add generators
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            control = "Slack" if idx == 0 else "PQ"
            network.add(
                "Generator",
                f"gen_{idx}",
                bus=f"bus_{idx}",
                p_nom=gen_capacity[idx],
                control=control,
                p_set=0.0
            )
        
        # Add loads
        for i in range(self.num_nodes):
            network.add(
                "Load",
                f"load_{i}",
                bus=f"bus_{i}",
                p_set=0.0,
                q_set=0.0
            )
        
        # Add transmission lines
        src, dst = edge_index
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            network.add(
                "Line",
                f"line_{i}",
                bus0=f"bus_{s}",
                bus1=f"bus_{d}",
                x=line_reactance[i],
                r=line_resistance[i],
                b=line_susceptance[i],
                g=line_conductance[i],
                s_nom=thermal_limits[i],
                length=1.0
            )
        
        return network
    
    def compute_power_flow(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
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
        # Calculate reactive load (0.95 power factor)
        q_load = load * 0.33
        
        # Store original states for restoration
        original_bus_states = {}
        original_gen_states = {}
        original_load_states = {}
        original_line_states = {}
        
        # Handle failed nodes - PROPERLY ISOLATE BUSES
        if failed_nodes:
            failed_node_set = set(failed_nodes)
            
            # 1. Disable failed buses
            for node_idx in failed_nodes:
                bus_name = f"bus_{node_idx}"
                if 'in_service' in self.network.buses.columns:
                    original_bus_states[bus_name] = self.network.buses.loc[bus_name, "in_service"]
                    self.network.buses.loc[bus_name, "in_service"] = False
                
                # 2. Disable generators at failed buses
                gen_name = f"gen_{node_idx}"
                if gen_name in self.network.generators.index:
                    original_gen_states[gen_name] = {
                        'p_set': self.network.generators.loc[gen_name, "p_set"]
                    }
                    if 'in_service' in self.network.generators.columns:
                        original_gen_states[gen_name]['in_service'] = self.network.generators.loc[gen_name, "in_service"]
                        self.network.generators.loc[gen_name, "in_service"] = False
                    else:
                        # Fallback: set p_set to 0
                        self.network.generators.loc[gen_name, "p_set"] = 0.0
                
                # 3. Disable loads at failed buses
                load_name = f"load_{node_idx}"
                if 'in_service' in self.network.loads.columns:
                    original_load_states[load_name] = self.network.loads.loc[load_name, "in_service"]
                    self.network.loads.loc[load_name, "in_service"] = False
                else:
                    # Fallback: set p_set and q_set to 0
                    original_load_states[load_name] = {
                        'p_set': self.network.loads.loc[load_name, "p_set"],
                        'q_set': self.network.loads.loc[load_name, "q_set"]
                    }
                    self.network.loads.loc[load_name, "p_set"] = 0.0
                    self.network.loads.loc[load_name, "q_set"] = 0.0
            
            # 4. Disable lines connected to failed buses
            for line_idx in range(self.num_edges):
                line_name = f"line_{line_idx}"
                src, dst = int(self.network.lines.loc[line_name, "bus0"].split("_")[1]), \
                           int(self.network.lines.loc[line_name, "bus1"].split("_")[1])
                
                if src in failed_node_set or dst in failed_node_set:
                    if line_name not in original_line_states:
                        if 'in_service' in self.network.lines.columns:
                            original_line_states[line_name] = self.network.lines.loc[line_name, "in_service"]
                            self.network.lines.loc[line_name, "in_service"] = False
        
        # Update generator setpoints for active nodes
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            if failed_nodes and idx in failed_nodes:
                continue  # Already disabled above
            
            gen_name = f"gen_{idx}"
            if gen_name not in original_gen_states:
                original_gen_states[gen_name] = {
                    'p_set': self.network.generators.loc[gen_name, "p_set"]
                }
                if 'in_service' in self.network.generators.columns:
                    original_gen_states[gen_name]['in_service'] = self.network.generators.loc[gen_name, "in_service"]
            self.network.generators.loc[gen_name, "p_set"] = generation[idx]
        
        # Update load setpoints for active nodes
        for i in range(self.num_nodes):
            if failed_nodes and i in failed_nodes:
                continue  # Already disabled above
            
            load_name = f"load_{i}"
            if load_name not in original_load_states:
                if 'in_service' in self.network.loads.columns:
                    original_load_states[load_name] = self.network.loads.loc[load_name, "in_service"]
                else:
                    original_load_states[load_name] = {
                        'p_set': self.network.loads.loc[load_name, "p_set"],
                        'q_set': self.network.loads.loc[load_name, "q_set"]
                    }
            self.network.loads.loc[load_name, "p_set"] = load[i]
            self.network.loads.loc[load_name, "q_set"] = q_load[i]
        
        # Handle explicitly failed lines (in addition to those connected to failed buses)
        if failed_lines:
            for line_idx in failed_lines:
                line_name = f"line_{line_idx}"
                if line_name not in original_line_states:
                    if 'in_service' in self.network.lines.columns:
                        original_line_states[line_name] = self.network.lines.loc[line_name, "in_service"]
                        self.network.lines.loc[line_name, "in_service"] = False
        
        try:
            # Run power flow
            status = self.network.pf()
            is_stable = status.get("converged", {}).get("0", {}).get("now", False)
            
            if not is_stable:
                self._restore_network_state(original_bus_states, original_gen_states, 
                                           original_load_states, original_line_states)
                return self._get_default_results()
            
            # Extract results
            voltages, angles, node_reactive = self._extract_bus_results(failed_nodes)
            line_flows_p, line_flows_q = self._extract_line_results(failed_lines, failed_nodes)
            
            # Restore network state
            self._restore_network_state(original_bus_states, original_gen_states, 
                                       original_load_states, original_line_states)
            
            return voltages, angles, line_flows_p, node_reactive, line_flows_q, True
            
        except Exception as e:
            print(f"  [WARNING] Power flow exception: {e}")
            self._restore_network_state(original_bus_states, original_gen_states, 
                                       original_load_states, original_line_states)
            return self._get_default_results()
    
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
        
        failed_line_set = set(failed_lines) if failed_lines else set()
        failed_node_set = set(failed_nodes) if failed_nodes else set()
        
        for i in range(self.num_edges):
            line_name = f"line_{i}"
            
            # Check if line is connected to failed nodes
            src = int(self.network.lines.loc[line_name, "bus0"].split("_")[1])
            dst = int(self.network.lines.loc[line_name, "bus1"].split("_")[1])
            
            if i in failed_line_set or src in failed_node_set or dst in failed_node_set:
                # Failed lines have zero flow
                line_flows_p[i] = 0.0
                line_flows_q[i] = 0.0
            else:
                # Active lines: extract from power flow results
                try:
                    line_flows_p[i] = self.network.lines_t.p0.loc["now", line_name]
                    line_flows_q[i] = self.network.lines_t.q0.loc["now", line_name]
                except (KeyError, AttributeError):
                    # Line might be in isolated island - use zero flow
                    line_flows_p[i] = 0.0
                    line_flows_q[i] = 0.0
        
        return line_flows_p, line_flows_q
    
    def _restore_network_state(
        self,
        original_bus_states: Dict,
        original_gen_states: Dict,
        original_load_states: Dict,
        original_line_states: Dict
    ):
        """
        Restore network to original state after power flow simulation.
        
        Args:
            original_bus_states: Original bus in_service states
            original_gen_states: Original generator states (in_service, p_set)
            original_load_states: Original load in_service states or {p_set, q_set}
            original_line_states: Original line in_service states
        """
        # Restore buses
        for bus_name, state in original_bus_states.items():
            if 'in_service' in self.network.buses.columns:
                self.network.buses.loc[bus_name, "in_service"] = state
        
        # Restore generators
        for gen_name, states in original_gen_states.items():
            if gen_name in self.network.generators.index:
                if 'in_service' in states:
                    if 'in_service' in self.network.generators.columns:
                        self.network.generators.loc[gen_name, "in_service"] = states['in_service']
                self.network.generators.loc[gen_name, "p_set"] = states['p_set']
        
        # Restore loads
        for load_name, state in original_load_states.items():
            if isinstance(state, dict):
                # Fallback mode: restore p_set and q_set
                self.network.loads.loc[load_name, "p_set"] = state['p_set']
                self.network.loads.loc[load_name, "q_set"] = state['q_set']
            else:
                # Normal mode: restore in_service
                if 'in_service' in self.network.loads.columns:
                    self.network.loads.loc[load_name, "in_service"] = state
        
        # Restore lines
        for line_name, state in original_line_states.items():
            if 'in_service' in self.network.lines.columns:
                self.network.lines.loc[line_name, "in_service"] = state
    
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
        base_frequency: float = 60.0
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
        self.damping = np.random.uniform(1.0, 2.0, num_nodes)
        # Under-frequency load shedding (UFLS) settings
        self.ufls_stages = [
            {'frequency': 59.3, 'load_shed': 0.10},  # Shed 10% at 59.3 Hz
            {'frequency': 59.0, 'load_shed': 0.15},  # Shed 15% at 59.0 Hz
            {'frequency': 58.7, 'load_shed': 0.20},  # Shed 20% at 58.7 Hz
        ]
    
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
            if capacity_mw > 400:
                inertia[idx] = np.random.uniform(4.0, 6.0)  # Large
            elif capacity_mw > 150:
                inertia[idx] = np.random.uniform(2.5, 4.0)  # Medium
            else:
                inertia[idx] = np.random.uniform(1.5, 2.5)  # Small
        
        return inertia
    
    def update_frequency(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        current_frequency: float,
        dt: float = 2.0 # timestep in seconds
    ) -> Tuple[float, np.ndarray]:
        """
        Update system frequency based on power imbalance.
        
        Args:
            generation: Generation at each node
            load: Load at each node
            current_frequency: Current frequency (Hz)
            dt: Time step (minutes)
            
        Returns:
            Tuple of (new_frequency, adjusted_load)
        """
        # Calculate power imbalance
        total_gen = generation.sum()
        total_load = load.sum()
        imbalance = total_gen - total_load
        
        if total_gen == 0:
            return 0.0, load  # System collapsed
        # Calculate total system inertia
        total_inertia = self.inertia.sum()
        if total_inertia == 0:
            total_inertia = 1.0
        
        # Frequency rate of change
        system_base = 10000  # 10 GW base
        df_dt = imbalance / (2 * total_inertia * system_base) * 60
        
        # Load damping effect
        load_damping_effect = np.sum(self.damping * load) * (current_frequency - 60) / 60
        df_dt += load_damping_effect / (2 * total_inertia * system_base) * 60
        
        # Update frequency
        new_frequency = current_frequency + df_dt * dt
        new_frequency = np.clip(new_frequency, 55.0, 65.0)
        
        # Under-frequency load shedding
        adjusted_load = load.copy()
        for stage in self.ufls_stages:
            if new_frequency < stage['frequency']:
                shed_amount = stage['load_shed']
                adjusted_load *= (1 - shed_amount)
                break
        
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
        ambient_temperature: float = 25.0
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
        dt: float = 2.0
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
        dT = (heat_generation - heat_dissipation) * dt / self.thermal_capacity
        
        # Update temperatures
        self.temperatures += dT
        self.temperatures += np.random.normal(0,0.5)
        
        # Clip to reasonable range
        self.temperatures = np.clip(self.temperatures, self.ambient_temperature - 5, 150.0)
        
        return self.temperatures
    
    def reset_temperatures(self):
        """Reset all temperatures to ambient."""
        self.temperatures = np.full(self.num_nodes, self.ambient_temperature)
