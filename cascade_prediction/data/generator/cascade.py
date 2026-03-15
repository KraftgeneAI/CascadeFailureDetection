"""
Cascade Module
=============
Cascade failure propagation logic.

This module implements:
- Cascade propagation through the grid
- Failure state checking (OK, Damaged, Failed)
- Physics-based cascade with power flow recomputation
- Timing and sequencing of failures
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from .utils import get_failed_lines_from_nodes
from .config import Settings


class CascadeSimulator:
    """
    Simulates cascade failure propagation through power grid.
    
    Implements physics-based cascade propagation where failures
    spread through the network based on actual power flow conditions.
    """
    
    def __init__(
        self,
        num_nodes: int,
        adjacency_list: List[List[Tuple[int, int, float]]],
        loading_failure_threshold: np.ndarray,
        loading_damage_threshold: np.ndarray,
        voltage_failure_threshold: np.ndarray,
        voltage_damage_threshold: np.ndarray,
        temperature_failure_threshold: np.ndarray,
        temperature_damage_threshold: np.ndarray,
        frequency_failure_threshold: np.ndarray,
        frequency_damage_threshold: np.ndarray
    ):
        """
        Initialize cascade simulator.
        
        Args:
            num_nodes: Number of nodes in grid
            adjacency_list: Directed adjacency list with propagation weights
            loading_failure_threshold: Loading thresholds for failure
            loading_damage_threshold: Loading thresholds for damage
            voltage_failure_threshold: Voltage thresholds for failure
            voltage_damage_threshold: Voltage thresholds for damage
            temperature_failure_threshold: Temperature thresholds for failure
            temperature_damage_threshold: Temperature thresholds for damage
            frequency_failure_threshold: Frequency thresholds for failure
            frequency_damage_threshold: Frequency thresholds for damage
        """
        self.num_nodes = num_nodes
        self.adjacency_list = adjacency_list
        
        # Failure thresholds
        self.loading_failure_threshold = loading_failure_threshold
        self.loading_damage_threshold = loading_damage_threshold
        self.voltage_failure_threshold = voltage_failure_threshold
        self.voltage_damage_threshold = voltage_damage_threshold
        self.temperature_failure_threshold = temperature_failure_threshold
        self.temperature_damage_threshold = temperature_damage_threshold
        self.frequency_failure_threshold = frequency_failure_threshold
        self.frequency_damage_threshold = frequency_damage_threshold
    
    def check_node_state(
        self,
        node_id: int,
        loading: float,
        voltage: float,
        temperature: float,
        frequency: float
    ) -> Tuple[int, str]:
        """
        Check node failure state based on physics conditions.
        
        Args:
            node_id: Node index
            loading: Loading ratio
            voltage: Voltage (p.u.)
            temperature: Temperature (°C)
            frequency: Frequency (Hz)
            
        Returns:
            Tuple of (state, reason) where:
            - state: 0=OK, 1=Damaged, 2=Failed
            - reason: Failure reason string
        """
        # Check for full failure
        if loading > self.loading_failure_threshold[node_id]:
            return 2, "overload"
        if voltage < self.voltage_failure_threshold[node_id]:
            return 2, "voltage_collapse"
        if temperature > self.temperature_failure_threshold[node_id]:
            return 2, "overheating"
        if frequency < self.frequency_failure_threshold[node_id]:
            return 2, "underfrequency"
        
        # Check for damage (partial failure)
        if loading > self.loading_damage_threshold[node_id]:
            return 1, "overload_damage"
        if voltage < self.voltage_damage_threshold[node_id]:
            return 1, "voltage_stress"
        if temperature > self.temperature_damage_threshold[node_id]:
            return 1, "thermal_stress"
        if frequency < self.frequency_damage_threshold[node_id]:
            return 1, "frequency_stress"
        
        return 0, "none"
    
    def propagate_cascade_simple(
        self,
        initial_failed_nodes: List[int],
        current_loading: np.ndarray,
        current_voltage: np.ndarray,
        current_temperature: np.ndarray,
        current_frequency: float
    ) -> List[Tuple[int, float, str]]:
        """
        Propagate cascade using simplified stress multipliers.
        
        Fast but less accurate - uses stress multipliers instead of
        recomputing power flow after each failure.
        
        Args:
            initial_failed_nodes: Initially failed node IDs
            current_loading: Loading ratios
            current_voltage: Voltages
            current_temperature: Temperatures
            current_frequency: Frequency
            
        Returns:
            List of (node_id, failure_time, reason) tuples
        """
        failed_nodes = set(initial_failed_nodes)
        failure_sequence = []
        
        # Queue: (node_id, failure_time, accumulated_stress)
        queue = [(node, 0.0, 1.0) for node in initial_failed_nodes]
        visited = set(initial_failed_nodes)
        
        while queue:
            current_node, current_time, accumulated_stress = queue.pop(0)
            
            # Check neighbors
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                
                # Calculate stress multiplier
                stress_multiplier = accumulated_stress * propagation_weight
                
                # Apply stress to neighbor conditions
                neighbor_loading = current_loading[neighbor] * (1.0 + stress_multiplier * Settings.Cascade.STRESS_TO_LOADING_FACTOR)
                neighbor_voltage = current_voltage[neighbor] * (1.0 - stress_multiplier * Settings.Cascade.STRESS_TO_VOLTAGE_FACTOR)
                neighbor_temperature = current_temperature[neighbor] + stress_multiplier * Settings.Cascade.STRESS_TO_TEMP_FACTOR
                
                # Check failure state
                failure_state, reason = self.check_node_state(
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if failure_state == 2:  # Full failure
                    failure_time = current_time + np.random.uniform(
                        Settings.Cascade.FAILURE_DELAY_MIN, Settings.Cascade.FAILURE_DELAY_MAX)
                    failure_sequence.append((neighbor, failure_time, reason))
                    failed_nodes.add(neighbor)
                    visited.add(neighbor)

                    # Add to queue for further propagation
                    queue.append((neighbor, failure_time, stress_multiplier * Settings.Cascade.STRESS_DECAY))
                    
                    print(f"    [CASCADE] Node {current_node} -> Node {neighbor} "
                          f"(reason: {reason}, time: {failure_time:.2f}min)")
                
                elif failure_state == 1:  # Partial failure (damaged)
                    print(f"    [PARTIAL] Node {current_node} -> Node {neighbor} "
                          f"DAMAGED (reason: {reason}) - Cascade stops")
                    visited.add(neighbor)
                
                else:  # OK
                    visited.add(neighbor)
        
        return failure_sequence
    
    def propagate_cascade_physics(
        self,
        initial_failed_nodes: List[Tuple[int, str]],
        generation: np.ndarray,
        load: np.ndarray,
        current_temperature: np.ndarray,
        current_frequency: float,
        target_num_failures: int,
        power_flow_simulator,
        edge_index: np.ndarray,
        thermal_limits: np.ndarray
    ) -> List[Tuple[int, float, str]]:
        """
        Propagate cascade with physics-based power flow recomputation.
        
        More accurate - recomputes power flow after each failure to get
        actual voltages and loadings.
        
        Args:
            initial_failed_nodes: List of (node_id, reason) tuples
            generation: Generation at each node
            load: Load at each node
            current_temperature: Temperatures
            current_frequency: Frequency
            target_num_failures: Maximum failures to simulate
            power_flow_simulator: PowerFlowSimulator instance
            edge_index: Edge connectivity
            thermal_limits: Line thermal limits
            
        Returns:
            List of (node_id, failure_time, reason) tuples
        """
        # Initialize
        failed_nodes = set(node[0] for node in initial_failed_nodes)
        failed_reasons = [node[1] for node in initial_failed_nodes]
        failure_sequence = [
            (fail_node, 0.0, fail_reason)
            for fail_node, fail_reason in zip(failed_nodes, failed_reasons)
        ]
        
        queue = [(node[0], 0.0) for node in initial_failed_nodes]
        visited = set(node[0] for node in initial_failed_nodes)
        
        # Calculate failed lines from failed nodes
        failed_lines = get_failed_lines_from_nodes(edge_index, failed_nodes)
        
        # Recompute power flow after initial failures
        print(f"    [POWER FLOW] Recomputing after initial failures: {list(failed_nodes)}")
        print(f"    [POWER FLOW] Failed lines connected to failed nodes: {failed_lines}")
        voltages, angles, line_flows, node_reactive, line_flows_q, is_stable = \
            power_flow_simulator.compute_power_flow(
                generation, load,
                failed_lines=failed_lines,
                failed_nodes=list(failed_nodes)
            )
        
        if not is_stable:
            print(f"    [WARNING] Power flow unstable after initial failures")
            ## Diverge at initial, do not need to continue the propagete
            return failure_sequence
        
        # Calculate loading ratios from power flow
        apparent_power = np.sqrt(line_flows**2 + line_flows_q**2)
        loading_ratios = apparent_power / (thermal_limits + 1e-6)
        
        # Map line loadings to nodes
        node_loading = self._calculate_node_loading(
            edge_index, loading_ratios
        )
        
        print(f"    [POWER FLOW] Voltage: {voltages.min():.3f}-{voltages.max():.3f} p.u., "
              f"Max loading: {loading_ratios.max():.3f}")
        
        # Propagate cascade
        while queue and len(failed_nodes) < target_num_failures:
            current_node, current_time = queue.pop(0)
            
            # Check neighbors
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                # Use actual power flow values
                neighbor_loading = node_loading[neighbor]
                neighbor_voltage = voltages[neighbor]
                neighbor_temperature = current_temperature[neighbor]
                
                # Check failure state
                failure_state, reason = self.check_node_state(
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if failure_state == 2:  # Full failure
                    physical_delay = np.random.uniform(Settings.Cascade.FAILURE_DELAY_MIN, Settings.Cascade.FAILURE_DELAY_MAX)
                    failure_time = current_time + physical_delay
                    
                    failure_sequence.append((neighbor, failure_time, reason))
                    failed_nodes.add(neighbor)
                    queue.append((neighbor, failure_time))
                    
                    print(f"    [CASCADE] Node {current_node} -> {neighbor} FAILS: "
                          f"{reason}, t={failure_time:.2f}min, V={neighbor_voltage:.3f}, "
                          f"L={neighbor_loading:.3f}")
                    
                    # Recompute power flow after this failure
                    generation[neighbor]=0
                    load[neighbor]=0
                    
                    # Calculate failed lines from all failed nodes
                    failed_lines = get_failed_lines_from_nodes(edge_index, failed_nodes)
                    
                    print(f"    [POWER FLOW] Recomputing after node {neighbor} failure...")
                    print(f"    [POWER FLOW] Total failed nodes: {len(failed_nodes)}, Failed lines: {len(failed_lines)}")
                    voltages, angles, line_flows, node_reactive, line_flows_q, is_stable = \
                        power_flow_simulator.compute_power_flow(
                            generation, load,
                            failed_lines=failed_lines,
                            failed_nodes=list(failed_nodes)
                        )
                    
                    if not is_stable:
                        print(f"    [WARNING] Power flow unstable after {len(failed_nodes)} failures")
                        ## Diverge at initial, do not need to continue the propagete
                        return failure_sequence
                    
                    # Update loading ratios
                    apparent_power = np.sqrt(line_flows**2 + line_flows_q**2)
                    loading_ratios = apparent_power / (thermal_limits + 1e-6)
                    node_loading = self._calculate_node_loading(edge_index, loading_ratios)
                    
                    print(f"    [POWER FLOW] New voltage: {voltages.min():.3f}-{voltages.max():.3f} p.u., "
                          f"Max loading: {loading_ratios.max():.3f}")
                    
                    if len(failed_nodes) >= target_num_failures:
                        print(f"Maximum number of failed nodes({len(failed_nodes)}) is reached, stopping...")
                        break
                
                elif failure_state == 1:  # Partial failure
                    print(f"    [PARTIAL] Node {current_node} -> {neighbor} DAMAGED: "
                          f"{reason}")
            
            if len(failed_nodes) >= target_num_failures:
                print(f"Maximum number of failed nodes({len(failed_nodes)}) is reached, stopping...")
                break
        
        return failure_sequence
    
    def _calculate_node_loading(
        self,
        edge_index: np.ndarray,
        loading_ratios: np.ndarray
    ) -> np.ndarray:
        """
        Calculate node loading from line loading ratios.
        
        Each node's loading is the maximum loading of its connected lines.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            loading_ratios: Line loading ratios
            
        Returns:
            Node loading ratios
        """
        node_loading = np.zeros(self.num_nodes)
        src, dst = edge_index
        
        for i in range(len(loading_ratios)):
            s, d = int(src[i]), int(dst[i])
            node_loading[s] = max(node_loading[s], loading_ratios[i])
            node_loading[d] = max(node_loading[d], loading_ratios[i])
        
        return node_loading



def create_adjacency_list(
    edge_index: np.ndarray,
    node_types: np.ndarray,
    propagation_weights: Optional[np.ndarray] = None
) -> List[List[Tuple[int, int, float]]]:
    """
    Create directed adjacency list with propagation weights.
    
    Cascade propagation follows grid hierarchy:
    - Generator → Substation (high weight)
    - Substation → Load (medium weight)
    - Load → Load (low weight)
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        node_types: Node types (0=load, 1=gen, 2=sub)
        propagation_weights: Optional custom weights
        
    Returns:
        Adjacency list: adjacency_list[node] = [(neighbor, edge_idx, weight), ...]
    """
    num_nodes = node_types.shape[0]
    num_edges = edge_index.shape[1]
    
    adjacency_list = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    
    for i in range(num_edges):
        s, d = int(src[i]), int(dst[i])
        
        # Calculate propagation weight based on node types
        if propagation_weights is not None:
            weight = propagation_weights[i]
        else:
            # Generator → Substation: high weight
            if node_types[s] == 1 and node_types[d] == 2:
                weight = Settings.Cascade.WEIGHT_GEN_TO_SUB
            # Substation → Load: medium weight
            elif node_types[s] == 2 and node_types[d] == 0:
                weight = Settings.Cascade.WEIGHT_SUB_TO_LOAD
            # Generator → Load: medium weight
            elif node_types[s] == 1 and node_types[d] == 0:
                weight = Settings.Cascade.WEIGHT_GEN_TO_LOAD
            # Load → Load: low weight
            elif node_types[s] == 0 and node_types[d] == 0:
                weight = Settings.Cascade.WEIGHT_LOAD_TO_LOAD
            # Default
            else:
                weight = Settings.Cascade.WEIGHT_DEFAULT
        
        # Add directed edge
        adjacency_list[s].append((d, i, weight))
    
    return adjacency_list
