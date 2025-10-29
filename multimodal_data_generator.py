"""
Rule-Based Multi-Modal Data Generator for Cascade Failure Detection
====================================================================
Generates data based on CONSISTENT RULES that the model can learn:
- Node properties (loading, voltage, temperature, frequency)
- Failure thresholds (when does a node fail?)
- Cascade propagation (which nodes fail when another fails?)
- Temporal patterns (gradual degradation before failure)

The model learns to recognize:
1. When node properties violate thresholds → node fails
2. When node A fails → connected nodes B, C, D fail (based on edges)
3. Temporal patterns before failure (loading increases, voltage drops, etc.)

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime
import gc
import psutil
import warnings
from scipy.ndimage import gaussian_filter # Added import for gaussian_filter
import os # Added import for os
import argparse # Added import for argparse


class MemoryMonitor:
    """Monitor memory usage."""
    
    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_threshold(threshold_mb=8000):
        current = MemoryMonitor.get_memory_usage()
        if current > threshold_mb:
            warnings.warn(f"High memory: {current:.1f} MB")
            return True
        return False


class PhysicsBasedGridSimulator:
    """
    Rule-based power grid simulator with CONSISTENT PATTERNS.
    """
    
    def __init__(self, num_nodes: int = 118, seed: int = 42, topology_file: str = None):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if topology_file and os.path.exists(topology_file):
            print(f"Loading grid topology from {topology_file}...")
            with open(topology_file, 'rb') as f:
                topology_data = pickle.load(f)
                self.adjacency_matrix = topology_data['adjacency_matrix']
                self.edge_index = topology_data['edge_index']
                self.positions = topology_data['positions']
                self.num_nodes = self.adjacency_matrix.shape[0]
                self.num_edges = self.edge_index.shape[1]
                print(f"  Loaded topology: {self.num_nodes} nodes, {self.num_edges} edges")
        else:
            # Generate realistic topology
            self.adjacency_matrix = self._generate_realistic_topology()
            self.edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
            self.num_edges = self.edge_index.shape[1]
            
            # Geographic positions (for environmental correlation)
            self.positions = self._generate_geographic_positions()
        
        self._initialize_node_properties_and_rules()
        
        self._initialize_edge_features_and_cascade_rules()
        
        print(f"Initialized grid: {self.num_nodes} nodes, {self.num_edges} edges")
        print(f"  Node failure rules: loading, voltage, temperature, frequency thresholds")
        print(f"  Cascade propagation: graph-based (A→B→C)")
    
    def _generate_realistic_topology(self) -> np.ndarray:
        """Generate realistic meshed grid topology."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        # Create 4 zones (like regional transmission areas)
        num_zones = 4
        nodes_per_zone = self.num_nodes // num_zones
        
        # Intra-zone connections (meshed within zone)
        for zone in range(num_zones):
            start = zone * nodes_per_zone
            end = start + nodes_per_zone if zone < num_zones - 1 else self.num_nodes
            
            # Each node connects to 2-4 neighbors in same zone
            for i in range(start, end):
                num_connections = np.random.randint(2, 5)
                possible_neighbors = list(range(start, end))
                possible_neighbors.remove(i)
                neighbors = np.random.choice(
                    possible_neighbors,
                    size=min(num_connections, len(possible_neighbors)),
                    replace=False
                )
                for j in neighbors:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Inter-zone tie lines (fewer, critical connections)
        for zone in range(num_zones - 1):
            zone_end = (zone + 1) * nodes_per_zone
            next_zone_start = zone_end
            # 2-3 tie lines between adjacent zones
            for _ in range(np.random.randint(2, 4)):
                i = np.random.randint(zone * nodes_per_zone, zone_end)
                j = np.random.randint(next_zone_start,
                                     min(next_zone_start + nodes_per_zone, self.num_nodes))
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _generate_geographic_positions(self) -> np.ndarray:
        """Generate realistic geographic positions (for environmental correlation)."""
        # Cluster nodes in zones
        positions = []
        num_zones = 4
        nodes_per_zone = self.num_nodes // num_zones
        
        zone_centers = [
            (-50, -50), (50, -50), (-50, 50), (50, 50)
        ]
        
        for zone_idx, (cx, cy) in enumerate(zone_centers):
            start = zone_idx * nodes_per_zone
            end = start + nodes_per_zone if zone_idx < num_zones - 1 else self.num_nodes
            num_in_zone = end - start
            
            # Nodes clustered around zone center
            zone_positions = np.random.randn(num_in_zone, 2) * 20 + np.array([cx, cy])
            positions.append(zone_positions)
        
        return np.vstack(positions)
    
    def _initialize_node_properties_and_rules(self):
        """
        Initialize node properties and FAILURE RULES.
        These are the patterns the model will learn!
        """
        # Node types
        self.node_types = np.zeros(self.num_nodes, dtype=int)
        num_generators = int(self.num_nodes * 0.22)
        gen_indices = np.random.choice(self.num_nodes, num_generators, replace=False)
        self.node_types[gen_indices] = 1
        
        num_substations = int(self.num_nodes * 0.10)
        sub_indices = np.random.choice(
            [i for i in range(self.num_nodes) if i not in gen_indices],
            num_substations, replace=False
        )
        self.node_types[sub_indices] = 2
        
        # Generator capacity
        self.gen_capacity = np.zeros(self.num_nodes)
        for idx in gen_indices:
            gen_type = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
            if gen_type == 'small':
                self.gen_capacity[idx] = np.random.uniform(50, 150)
            elif gen_type == 'medium':
                self.gen_capacity[idx] = np.random.uniform(150, 400)
            else:
                self.gen_capacity[idx] = np.random.uniform(400, 800)
        
        # Base load
        self.base_load = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if self.node_types[i] == 1:
                self.base_load[i] = np.random.uniform(5, 20)
            elif self.node_types[i] == 2:
                self.base_load[i] = np.random.uniform(50, 150)
            else:
                self.base_load[i] = np.random.uniform(30, 200)
        
        # Loading threshold: node fails if loading > threshold
        self.loading_failure_threshold = np.random.uniform(1.05, 1.15, self.num_nodes)
        
        # Voltage threshold: node fails if voltage < threshold
        self.voltage_failure_threshold = np.random.uniform(0.88, 0.92, self.num_nodes)
        
        # Temperature threshold: node fails if temperature > threshold
        self.temperature_failure_threshold = np.random.uniform(85, 95, self.num_nodes)
        
        # Frequency threshold: node fails if frequency < threshold
        self.frequency_failure_threshold = np.random.uniform(58.5, 59.2, self.num_nodes)
        
        # Equipment age and condition
        self.equipment_age = np.random.uniform(0, 40, self.num_nodes)
        self.equipment_condition = np.clip(
            1.0 - 0.008 * self.equipment_age + np.random.normal(0, 0.05, self.num_nodes),
            0.6, 1.0
        )
        
        # Thermal properties
        self.thermal_time_constant = np.random.uniform(10, 30, self.num_nodes)
        self.thermal_capacity = np.random.uniform(0.8, 1.2, self.num_nodes)
        self.cooling_effectiveness = np.random.uniform(0.7, 1.0, self.num_nodes)
        self.equipment_temperatures = np.full(self.num_nodes, 25.0)
        
        print(f"  Defined failure thresholds:")
        print(f"    Loading: {self.loading_failure_threshold.mean():.2f} ± {self.loading_failure_threshold.std():.2f}")
        print(f"    Voltage: {self.voltage_failure_threshold.mean():.2f} ± {self.voltage_failure_threshold.std():.2f}")
        print(f"    Temperature: {self.temperature_failure_threshold.mean():.1f} ± {self.temperature_failure_threshold.std():.1f}°C")
        print(f"    Frequency: {self.frequency_failure_threshold.mean():.2f} ± {self.frequency_failure_threshold.std():.2f} Hz")
    
    def _initialize_edge_features_and_cascade_rules(self):
        """
        Initialize edge features and CASCADE PROPAGATION RULES.
        These define how failures propagate through the graph!
        """
        src, dst = self.edge_index
        distances = np.linalg.norm(
            self.positions[src] - self.positions[dst], axis=1
        )
        
        # Line properties
        self.line_reactance = np.random.uniform(0.3, 0.5, self.num_edges) * distances / 100.0
        self.line_resistance = self.line_reactance * 0.1
        self.line_susceptance = 1.0 / (self.line_reactance + 1e-6)
        self.line_conductance = 1.0 / (self.line_resistance + 1e-6)
        
        # Thermal limits
        self.thermal_limits = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            if distances[i] < 30:
                self.thermal_limits[i] = np.random.uniform(300, 600)
            elif distances[i] < 60:
                self.thermal_limits[i] = np.random.uniform(200, 400)
            else:
                self.thermal_limits[i] = np.random.uniform(100, 300)
        
        # When node A fails, how much does it affect connected node B?
        # Higher weight = stronger cascade effect
        self.cascade_propagation_weight = np.random.uniform(0.6, 0.9, self.num_edges)
        
        # Build adjacency list for fast cascade propagation
        self.adjacency_list = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            self.adjacency_list[s].append((d, i, self.cascade_propagation_weight[i]))
            self.adjacency_list[d].append((s, i, self.cascade_propagation_weight[i]))
        
        print(f"  Cascade propagation weights: {self.cascade_propagation_weight.mean():.2f} ± {self.cascade_propagation_weight.std():.2f}")
    
    def _initialize_realistic_grid_properties(self):
        """Initialize grid with REALISTIC electrical parameters."""
        
        self.node_types = np.zeros(self.num_nodes, dtype=int)  # 0=load, 1=generator, 2=substation
        
        # 20-25% generators (realistic for transmission grid)
        num_generators = int(self.num_nodes * 0.22)
        gen_indices = np.random.choice(self.num_nodes, num_generators, replace=False)
        self.node_types[gen_indices] = 1
        
        # 10% substations (high connectivity nodes)
        num_substations = int(self.num_nodes * 0.10)
        sub_indices = np.random.choice(
            [i for i in range(self.num_nodes) if i not in gen_indices],
            num_substations, replace=False
        )
        self.node_types[sub_indices] = 2
        
        self.gen_capacity = np.zeros(self.num_nodes)
        for idx in gen_indices:
            # Mix of small (50-150 MW), medium (150-400 MW), large (400-800 MW)
            gen_type = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
            if gen_type == 'small':
                self.gen_capacity[idx] = np.random.uniform(50, 150)
            elif gen_type == 'medium':
                self.gen_capacity[idx] = np.random.uniform(150, 400)
            else:
                self.gen_capacity[idx] = np.random.uniform(400, 800)
        
        self.base_load = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if self.node_types[i] == 1:  # Generators have minimal load
                self.base_load[i] = np.random.uniform(5, 20)
            elif self.node_types[i] == 2:  # Substations have medium load
                self.base_load[i] = np.random.uniform(50, 150)
            else:  # Load buses
                self.base_load[i] = np.random.uniform(30, 200)
        
        src, dst = self.edge_index
        distances = np.linalg.norm(
            self.positions[src.numpy()] - self.positions[dst.numpy()], axis=1
        )
        
        # Reactance: 0.3-0.5 ohms/km for typical transmission lines
        self.line_reactance = np.random.uniform(0.3, 0.5, self.num_edges) * distances / 100.0
        self.line_resistance = self.line_reactance * 0.1  # R/X ratio ~ 0.1 for transmission
        
        # Susceptance (inverse of reactance)
        self.line_susceptance = 1.0 / (self.line_reactance + 1e-6)
        self.line_conductance = 1.0 / (self.line_resistance + 1e-6)
        
        self.thermal_limits = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            if distances[i] < 30:  # Short lines: higher capacity
                self.thermal_limits[i] = np.random.uniform(300, 600)
            elif distances[i] < 60:  # Medium lines
                self.thermal_limits[i] = np.random.uniform(200, 400)
            else:  # Long lines: lower capacity
                self.thermal_limits[i] = np.random.uniform(100, 300)
        
        self.equipment_age = np.random.uniform(0, 40, self.num_nodes)
        # Condition degrades with age: 1.0 (new) to 0.6 (old)
        self.equipment_condition = np.clip(
            1.0 - 0.008 * self.equipment_age + np.random.normal(0, 0.05, self.num_nodes),
            0.6, 1.0
        )
        
        # Failure probability increases with age and poor condition
        self.base_failure_prob = (1.0 - self.equipment_condition) * 0.01
    
    def _initialize_protection_settings(self):
        """Initialize deterministic protection relay settings."""
        src, dst = self.edge_index
        
        self.oc_relay_pickup = np.random.uniform(1.00, 1.10, self.num_edges)  # Trip at 100-110% loading
        
        self.relay_time_dial = np.random.uniform(0.1, 0.5, self.num_edges)  # Very fast (was 0.3-1.0)
        
        # Distance relay settings (impedance zones)
        self.zone1_reach = self.line_reactance * 0.85  # Zone 1: 85% of line (instantaneous)
        self.zone2_reach = self.line_reactance * 1.20  # Zone 2: 120% of line (0.3-0.5s delay)
        
        # Differential relay settings for nodes (instantaneous for internal<bos> faults)
        self.diff_relay_pickup = np.random.uniform(0.2, 0.4, self.num_nodes)  # 20-40% differential current
        
        self.uv_relay_pickup = np.random.uniform(0.93, 0.96, self.num_nodes)  # 93-96% voltage
        self.uv_relay_delay = np.random.uniform(0.2, 0.8, self.num_nodes)  # Faster (was 0.5-1.5s)
        
        # Under-frequency relay settings
        self.uf_relay_pickup = np.random.uniform(59.0, 59.5, self.num_nodes)  # 59.0-59.5 Hz
        self.uf_relay_delay = np.random.uniform(0.5, 2.0, self.num_nodes)  # 0.5-2 second delay
    
    def _initialize_frequency_dynamics(self):
        """Initialize frequency dynamics parameters."""
        # Generator inertia constants (H in seconds)
        self.generator_inertia = np.zeros(self.num_nodes)
        gen_indices = np.where(self.node_types == 1)[0]
        
        for idx in gen_indices:
            # Larger generators have higher inertia
            if self.gen_capacity[idx] > 400:  # Large units
                self.generator_inertia[idx] = np.random.uniform(4.0, 6.0)
            elif self.gen_capacity[idx] > 150:  # Medium units
                self.generator_inertia[idx] = np.random.uniform(2.5, 4.0)
            else:  # Small units
                self.generator_inertia[idx] = np.random.uniform(1.5, 2.5)
        
        # Load frequency sensitivity (% load change per % frequency change)
        self.load_damping = np.random.uniform(1.0, 2.0, self.num_nodes)
        
        # Under-frequency load shedding (UFLS) settings
        self.ufls_stages = [
            {'frequency': 59.3, 'load_shed': 0.10},  # Shed 10% at 59.3 Hz
            {'frequency': 59.0, 'load_shed': 0.15},  # Shed 15% at 59.0 Hz
            {'frequency': 58.7, 'load_shed': 0.20},  # Shed 20% at 58.7 Hz
        ]
    
    def _initialize_thermal_dynamics(self):
        """Initialize per-node thermal dynamics parameters."""
        # Thermal time constants (minutes) - how fast equipment heats/cools
        self.thermal_time_constant = np.random.uniform(10, 30, self.num_nodes)
        
        # Thermal capacity (how much heat equipment can store)
        self.thermal_capacity = np.random.uniform(0.8, 1.2, self.num_nodes)
        
        # Cooling effectiveness (depends on ambient conditions)
        self.cooling_effectiveness = np.random.uniform(0.7, 1.0, self.num_nodes)
        
        # Initial equipment temperatures (start at ambient)
        self.equipment_temperatures = np.full(self.num_nodes, 25.0)
        
        # Maximum safe operating temperature
        self.max_safe_temp = np.random.uniform(90, 110, self.num_nodes)
    
    def _compute_realistic_power_flow(
        self, 
        generation: np.ndarray, 
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute REALISTIC DC power flow with proper physics.
        Returns: voltages, angles, line_flows, is_stable
        """
        
        gen = generation.copy()
        ld = load.copy()
        if failed_nodes:
            for node in failed_nodes:
                gen[node] = 0.0
                ld[node] = 0.0
        
        # Net power injection at each bus
        P_net = gen - ld
        
        B = np.zeros((self.num_nodes, self.num_nodes))
        src, dst = self.edge_index
        
        active_lines = []
        for i in range(self.num_edges):
            if failed_lines is not None and i in failed_lines:
                continue
            active_lines.append(i)
            
            s, d = src[i].item(), dst[i].item()
            b = self.line_susceptance[i]
            
            # Build B matrix: B_ii = sum of susceptances, B_ij = -susceptance
            B[s, s] += b
            B[d, d] += b
            B[s, d] -= b
            B[d, s] -= b
        
        # Use slack bus (bus 0) as reference (theta_0 = 0)
        B_reduced = B[1:, 1:]
        P_reduced = P_net[1:]
        
        try:
            # Solve for voltage angles
            theta_reduced = np.linalg.solve(B_reduced, P_reduced)
            theta = np.zeros(self.num_nodes)
            theta[1:] = theta_reduced
            
            if np.max(np.abs(theta)) > np.radians(15):  # >15 degrees is unstable
                is_stable = False
            else:
                is_stable = True
                
        except np.linalg.LinAlgError:
            # Singular matrix = islanded system = unstable
            theta = np.zeros(self.num_nodes)
            is_stable = False
        
        line_flows = np.zeros(self.num_edges)
        for i in active_lines:
            s, d = src[i].item(), dst[i].item()
            line_flows[i] = self.line_susceptance[i] * (theta[s] - theta[d]) * 3.0
        
        # Voltage drops with heavy loading
        voltages = np.ones(self.num_nodes)
        for i in range(self.num_nodes):
            # Loading based on load and connections, plus some noise
            num_connections = len(self.adjacency_list[i])
            node_loading_factor = load[i] / (self.base_load[i] + 1e-6) * (1.0 + num_connections * 0.05)
            voltage_drop = 0.08 * node_loading_factor
            voltages[i] = 1.0 - voltage_drop + np.random.normal(0, 0.005)
        
        # Clip to realistic range
        voltages = np.clip(voltages, 0.85, 1.15)
        
        if np.any(voltages < 0.94) or np.any(voltages > 1.06):
            is_stable = False
        
        return voltages, theta, line_flows, is_stable
    
    def _run_ac_power_flow(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None,
        max_iterations: int = 20,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute REALISTIC AC power flow with voltage collapse and reactive power.
        Uses Newton-Raphson method for solving nonlinear power flow equations.
        
        Returns: voltages, angles, real_power_flows, reactive_power_flows, reactive_generation, is_stable
        """
        
        gen = generation.copy()
        ld = load.copy()
        if failed_nodes:
            for node in failed_nodes:
                gen[node] = 0.0
                ld[node] = 0.0
        
        # Net real power injection
        P_net = gen - ld
        
        # Reactive power: assume power factor 0.95 (typical for transmission)
        Q_load = ld * 0.33  # tan(acos(0.95)) ≈ 0.33
        Q_gen = gen * 0.33  # Generators provide reactive power
        Q_net = Q_gen - Q_load
        
        # Build admittance matrix (Y = G + jB)
        Y = np.zeros((self.num_nodes, self.num_nodes), dtype=complex)
        src, dst = self.edge_index
        
        active_lines = []
        for i in range(self.num_edges):
            if failed_lines is not None and i in failed_lines:
                continue
            active_lines.append(i)
            
            s, d = src[i].item(), dst[i].item()
            g = self.line_conductance[i]
            b = self.line_susceptance[i]
            y = g + 1j * b
            
            # Build Y matrix
            Y[s, s] += y
            Y[d, d] += y
            Y[s, d] -= y
            Y[d, s] -= y
        
        # Initialize voltage magnitudes and angles
        V = np.ones(self.num_nodes)  # Start at 1.0 p.u.
        theta = np.zeros(self.num_nodes)  # Start at 0 radians
        
        # Newton-Raphson iteration
        is_stable = True
        for iteration in range(max_iterations):
            # Calculate power mismatches
            P_calc = np.zeros(self.num_nodes)
            Q_calc = np.zeros(self.num_nodes)
            
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    Y_ij = Y[i, j]
                    G_ij = Y_ij.real
                    B_ij = Y_ij.imag
                    theta_ij = theta[i] - theta[j]
                    
                    P_calc[i] += V[i] * V[j] * (G_ij * np.cos(theta_ij) + B_ij * np.sin(theta_ij))
                    Q_calc[i] += V[i] * V[j] * (G_ij * np.sin(theta_ij) - B_ij * np.cos(theta_ij))
            
            # Power mismatches
            dP = P_net - P_calc
            dQ = Q_net - Q_calc
            
            # Check convergence
            max_mismatch = max(np.max(np.abs(dP)), np.max(np.abs(dQ)))
            if max_mismatch < tolerance:
                break
            
            # Build Jacobian matrix (simplified - full Jacobian is complex)
            # For speed, use simplified decoupled power flow approximation
            # dP/dtheta ≈ B (susceptance matrix)
            # dQ/dV ≈ B (susceptance matrix)
            
            B_matrix = Y.imag
            
            # Solve for angle corrections (skip slack bus)
            try:
                B_reduced = B_matrix[1:, 1:]
                dP_reduced = dP[1:]
                dtheta_reduced = np.linalg.solve(B_reduced, dP_reduced)
                theta[1:] += dtheta_reduced * 0.5  # Damping factor for stability
            except np.linalg.LinAlgError:
                is_stable = False
                break
            
            # Solve for voltage corrections
            try:
                dQ_reduced = dQ[1:]
                dV_reduced = np.linalg.solve(B_reduced, dQ_reduced)
                V[1:] += dV_reduced * 0.3  # Smaller damping for voltage
            except np.linalg.LinAlgError:
                is_stable = False
                break
            
            # Check for voltage collapse
            if np.any(V < 0.7) or np.any(V > 1.3):
                is_stable = False
                break
            
            # Check for angle instability
            if np.max(np.abs(theta)) > np.radians(30):
                is_stable = False
                break
        
        # If didn't converge, mark as unstable
        if iteration >= max_iterations - 1:
            is_stable = False
        
        # Calculate line flows (both real and reactive)
        real_power_flows = np.zeros(self.num_edges)
        reactive_power_flows = np.zeros(self.num_edges)
        
        for i in active_lines:
            s, d = src[i].item(), dst[i].item()
            g = self.line_conductance[i]
            b = self.line_susceptance[i]
            
            V_s, V_d = V[s], V[d]
            theta_sd = theta[s] - theta[d]
            
            # Real power flow: P = V_s * V_d * (G * cos(theta) + B * sin(theta))
            real_power_flows[i] = V_s * V_d * (g * np.cos(theta_sd) + b * np.sin(theta_sd))
            
            # Reactive power flow: Q = V_s * V_d * (G * sin(theta) - B * cos(theta))
            reactive_power_flows[i] = V_s * V_d * (g * np.sin(theta_sd) - b * np.sin(theta_sd))
        
        # Clip voltages to realistic range
        V = np.clip(V, 0.7, 1.3)
        
        # Check stability criteria
        if np.any(V < 0.85) or np.any(V > 1.15):
            is_stable = False
        
        # Calculate reactive power generation (for generator limits)
        reactive_generation = Q_gen.copy()
        
        return V, theta, real_power_flows, reactive_power_flows, reactive_generation, is_stable

    def _calculate_relay_operation_time(self, line_idx: int, loading_ratio: float) -> float:
        """
        Calculate deterministic relay operating time based on protection settings.
        Returns time in seconds, or -1 if relay doesn't operate.
        """
        if loading_ratio < self.oc_relay_pickup[line_idx]:
            return -1  # No operation
        
        # Inverse-time overcurrent characteristic: t = TD * K / (I^2 - 1)
        I_pu = loading_ratio / self.oc_relay_pickup[line_idx]
        K = 0.14  # Standard inverse curve constant
        
        if I_pu <= 1.0:
            return -1
        
        operating_time = self.relay_time_dial[line_idx] * K / (I_pu**2 - 1)
        
        # Instantaneous element for very high currents (>8x pickup)
        if loading_ratio > self.oc_relay_pickup[line_idx] * 8:
            operating_time = 0.05  # 50ms instantaneous trip
        
        return operating_time
    
    def _update_frequency_dynamics(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        failed_nodes: List[int],
        current_frequency: float,
        dt: float = 2.0  # timestep in seconds
    ) -> Tuple[float, np.ndarray]:
        """
        Update system frequency based on generation-load imbalance.
        Returns: new_frequency, adjusted_load (after UFLS)
        """
        # Calculate total system inertia
        active_gens = [i for i in range(self.num_nodes) 
                      if self.node_types[i] == 1 and i not in failed_nodes]
        
        if len(active_gens) == 0:
            return 0.0, load  # System collapsed
        
        total_inertia = np.sum(self.generator_inertia[active_gens])
        
        # Power imbalance
        total_gen = np.sum(generation)
        total_load = np.sum(load)
        power_imbalance = total_gen - total_load
        
        # Frequency rate of change: df/dt = (P_gen - P_load) / (2 * H * S_base)
        # Simplified: df/dt ≈ power_imbalance / (2 * total_inertia * system_base)
        system_base = 10000  # 10 GW base
        df_dt = power_imbalance / (2 * total_inertia * system_base) * 60  # Convert to Hz/s
        
        # Load damping effect (loads reduce with frequency)
        load_damping_effect = np.sum(self.load_damping * load) * (current_frequency - 60) / 60
        df_dt += load_damping_effect / (2 * total_inertia * system_base) * 60
        
        # Update frequency
        new_frequency = current_frequency + df_dt * dt
        new_frequency = np.clip(new_frequency, 55.0, 65.0)  # Physical limits
        
        # Under-frequency load shedding
        adjusted_load = load.copy()
        for stage in self.ufls_stages:
            if new_frequency < stage['frequency']:
                # Shed load proportionally across all buses
                shed_amount = stage['load_shed']
                adjusted_load *= (1 - shed_amount)
                print(f"  [UFLS] Frequency {new_frequency:.2f} Hz → Shed {shed_amount*100:.0f}% load")
                break
        
        return new_frequency, adjusted_load
    
    def _update_thermal_dynamics(
        self,
        loading_ratios: np.ndarray,
        ambient_temp: float,
        dt: float = 2.0  # timestep in seconds
    ) -> np.ndarray:
        """
        Update per-node equipment temperatures based on loading and thermal dynamics.
        Returns: updated equipment temperatures
        """
        src, dst = self.edge_index
        
        # Calculate heat generation per node based on connected line loadings
        heat_generation = np.zeros(self.num_nodes)
        
        for i in range(self.num_edges):
            s, d = src[i].item(), dst[i].item()
            # Heat proportional to I^2 * R losses
            heat = (loading_ratios[i] ** 2) * self.line_resistance[i] * 100  # Scaled
            heat_generation[s] += heat / 2
            heat_generation[d] += heat / 2
        
        # Thermal dynamics: dT/dt = (heat_in - heat_out) / thermal_capacity
        # heat_out = cooling_effectiveness * (T - T_ambient) / time_constant
        
        for node in range(self.num_nodes):
            heat_in = heat_generation[node]
            heat_out = (self.cooling_effectiveness[node] * 
                       (self.equipment_temperatures[node] - ambient_temp) / 
                       (self.thermal_time_constant[node] * 60))  # Convert minutes to seconds
            
            dT_dt = (heat_in - heat_out) / self.thermal_capacity[node]
            
            # Update temperature
            self.equipment_temperatures[node] += dT_dt * dt
            
            # Add some thermal noise
            self.equipment_temperatures[node] += np.random.normal(0, 0.5)
            
            # Physical limits
            self.equipment_temperatures[node] = np.clip(
                self.equipment_temperatures[node], 
                ambient_temp - 5, 
                150  # Maximum physical temperature
            )
        
        return self.equipment_temperatures.copy()
    
    # Check node failure based on rules
    def _check_node_failure(
        self,
        node_idx: int,
        loading: float,
        voltage: float,
        temperature: float,
        frequency: float
    ) -> Tuple[bool, str]:
        """
        Check if a node should fail based on RULES.
        Returns: (should_fail, reason)
        """
        # Rule 1: Loading exceeds threshold
        if loading > self.loading_failure_threshold[node_idx]:
            return True, "loading"
        
        # Rule 2: Voltage below threshold
        if voltage < self.voltage_failure_threshold[node_idx]:
            return True, "voltage"
        
        # Rule 3: Temperature exceeds threshold
        if temperature > self.temperature_failure_threshold[node_idx]:
            return True, "temperature"
        
        # Rule 4: Frequency below threshold
        if frequency < self.frequency_failure_threshold[node_idx]:
            return True, "frequency"
        
        return False, "none"
    
    # Propagate cascade based on rules
    def _propagate_cascade(
        self,
        initial_failed_nodes: List[int],
        current_loading: np.ndarray,
        current_voltage: np.ndarray,
        current_temperature: np.ndarray,
        current_frequency: float
    ) -> List[Tuple[int, float, str]]:
        """
        Propagate cascade through the graph based on RULES.
        When node A fails, connected nodes B, C, D experience stress and may fail.
        Returns: list of (node_id, failure_time, reason)
        """
        failed_nodes = set(initial_failed_nodes)
        failure_sequence = []
        
        # Queue: (node_id, failure_time, accumulated_stress)
        queue = [(node, 0.0, 1.0) for node in initial_failed_nodes]
        visited = set(initial_failed_nodes)
        
        while queue:
            current_node, current_time, accumulated_stress = queue.pop(0)
            
            # Check all neighbors connected by edges
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                
                # Calculate stress propagation: failures propagate through edges
                # Higher propagation_weight = stronger cascade effect
                stress_multiplier = accumulated_stress * propagation_weight
                
                # Neighbor experiences increased stress from failed node
                neighbor_loading = current_loading[neighbor] * (1.0 + stress_multiplier * 0.4)
                neighbor_voltage = current_voltage[neighbor] * (1.0 - stress_multiplier * 0.15)
                neighbor_temperature = current_temperature[neighbor] + stress_multiplier * 25
                
                # Check if neighbor should fail based on rules
                should_fail, reason = self._check_node_failure(
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if should_fail:
                    # Neighbor fails! Add to cascade sequence
                    failure_time = current_time + np.random.uniform(1.0, 3.0)  # Propagation delay
                    failure_sequence.append((neighbor, failure_time, reason))
                    failed_nodes.add(neighbor)
                    visited.add(neighbor)
                    
                    # Add to queue to propagate further (A→B→C)
                    queue.append((neighbor, failure_time, stress_multiplier * 0.8))
                    
                    print(f"    [CASCADE] Node {current_node} → Node {neighbor} (reason: {reason}, time: {failure_time:.1f}s)")
                else:
                    # Mark as visited even if didn't fail (to avoid re-checking)
                    visited.add(neighbor)
        
        return failure_sequence
    
    # Simulate cascade based on CONSISTENT RULES
    def _simulate_rule_based_cascade(
        self,
        stress_level: float,
        sequence_length: int = 60,
        target_failure_percentage: Optional[float] = None  # Added parameter to control cascade severity
    ) -> Tuple[List[int], List[float], List[str], int]:
        """
        Simulate cascade based on CONSISTENT RULES.
        GUARANTEED to produce a cascade with DIVERSE failure types and CONTROLLABLE severity!
        
        Args:
            target_failure_percentage: Target percentage of nodes to fail (0.2-1.0)
                                      If None, randomly chosen for diversity
        """
        if target_failure_percentage is None:
            # Create diverse cascades: 20%, 40%, 60%, 80%, or 100% node failures
            target_failure_percentage = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0], p=[0.2, 0.25, 0.25, 0.2, 0.1])
        
        target_num_failures = int(self.num_nodes * target_failure_percentage)
        print(f"  [TARGET] Aiming for {target_num_failures}/{self.num_nodes} node failures ({target_failure_percentage*100:.0f}%)")
        
        # Cascade starts at 65-85% through the sequence
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        # Choose initial trigger node (high centrality node for better propagation)
        node_degrees = np.array([len(self.adjacency_list[i]) for i in range(self.num_nodes)])
        high_degree_nodes = np.where(node_degrees > np.percentile(node_degrees, 60))[0]
        initial_trigger_node = np.random.choice(high_degree_nodes)
        
        # Calculate operating conditions at cascade start
        load_multiplier = 0.7 + stress_level * 0.4
        load = self.base_load * load_multiplier
        
        # Calculate loading for each node
        node_loading = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            num_connections = len(self.adjacency_list[i])
            node_loading[i] = load[i] / (self.base_load[i] + 1e-6) * (1.0 + num_connections * 0.05)
        
        # Voltage (decreases with loading)
        node_voltage = 1.0 - (node_loading - 1.0) * 0.15
        node_voltage = np.clip(node_voltage, 0.85, 1.05)
        
        # Temperature (increases with loading)
        ambient_temp = 30.0
        node_temperature = ambient_temp + (node_loading - 0.8) * 40
        node_temperature = np.clip(node_temperature, 25, 100)
        
        # Frequency (decreases with high loading)
        system_frequency = 60.0 - (node_loading.mean() - 0.9) * 5
        system_frequency = np.clip(system_frequency, 58.0, 60.5)
        
        failure_type = np.random.choice(['loading', 'voltage', 'temperature', 'frequency', 'environmental'])
        
        if failure_type == 'loading':
            # Overload failure: increase loading beyond threshold
            node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
            reason = "loading"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Loading overload: {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
            
        elif failure_type == 'voltage':
            # Voltage collapse: decrease voltage below threshold
            node_voltage[initial_trigger_node] = self.voltage_failure_threshold[initial_trigger_node] * np.random.uniform(0.85, 0.95)
            reason = "voltage"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Voltage collapse: {node_voltage[initial_trigger_node]:.3f} < {self.voltage_failure_threshold[initial_trigger_node]:.3f}")
            
        elif failure_type == 'temperature':
            # Thermal overload: increase temperature beyond threshold
            node_temperature[initial_trigger_node] = self.temperature_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
            reason = "temperature"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Thermal overload: {node_temperature[initial_trigger_node]:.1f}°C > {self.temperature_failure_threshold[initial_trigger_node]:.1f}°C")
            
        elif failure_type == 'frequency':
            # Frequency instability: decrease frequency below threshold
            system_frequency = self.frequency_failure_threshold[initial_trigger_node] * np.random.uniform(0.95, 0.99)
            reason = "frequency"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Frequency instability: {system_frequency:.2f} Hz < {self.frequency_failure_threshold[initial_trigger_node]:.2f} Hz")
            
        else:  # environmental
            # Environmental threat: randomly affect loading, voltage, or temperature
            env_effect = np.random.choice(['wildfire', 'storm', 'flooding', 'extreme_cold'])
            if env_effect == 'wildfire':
                # Wildfire increases temperature
                node_temperature[initial_trigger_node] = self.temperature_failure_threshold[initial_trigger_node] * np.random.uniform(1.1, 1.2)
                reason = "temperature"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Wildfire: Temperature {node_temperature[initial_trigger_node]:.1f}°C > {self.temperature_failure_threshold[initial_trigger_node]:.1f}°C")
            elif env_effect == 'storm':
                # Storm damages lines, increases loading
                node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.1, 1.2)
                reason = "loading"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Storm damage: Loading {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
            elif env_effect == 'flooding':
                # Flooding causes voltage issues
                node_voltage[initial_trigger_node] = self.voltage_failure_threshold[initial_trigger_node] * np.random.uniform(0.8, 0.9)
                reason = "voltage"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Flooding: Voltage {node_voltage[initial_trigger_node]:.3f} < {self.voltage_failure_threshold[initial_trigger_node]:.3f}")
            else:  # extreme_cold
                # Extreme cold reduces capacity, increases loading
                node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
                reason = "loading"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Extreme cold: Loading {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
        
        failure_sequence = self._propagate_cascade_controlled(
            [initial_trigger_node],
            node_loading,
            node_voltage,
            node_temperature,
            system_frequency,
            target_num_failures=target_num_failures  # Control how many nodes fail
        )
        
        # Build final lists
        failed_nodes = [initial_trigger_node] + [node for node, _, _ in failure_sequence]
        failure_times = [0.0] + [time for _, time, _ in failure_sequence]
        failure_reasons = [reason] + [r for _, _, r in failure_sequence]
        
        print(f"  [RESULT] Cascade generated: {len(failed_nodes)}/{self.num_nodes} nodes failed ({len(failed_nodes)/self.num_nodes*100:.1f}%)")
        
        return failed_nodes, failure_times, failure_reasons, cascade_start_time

    def _propagate_cascade_controlled(
        self,
        initial_failed_nodes: List[int],
        current_loading: np.ndarray,
        current_voltage: np.ndarray,
        current_temperature: np.ndarray,
        current_frequency: float,
        target_num_failures: int
    ) -> List[Tuple[int, float, str]]:
        """
        Propagate cascade through the graph with CONTROLLED severity.
        Stops when target number of failures is reached.
        Returns: list of (node_id, failure_time, reason)
        """
        failed_nodes = set(initial_failed_nodes)
        failure_sequence = []
        
        # Queue: (node_id, failure_time, accumulated_stress, priority)
        # Priority: higher stress = higher priority to fail
        queue = [(node, 0.0, 1.0, 1.0) for node in initial_failed_nodes]
        visited = set(initial_failed_nodes)
        
        # Candidates for failure (sorted by stress level)
        failure_candidates = []
        
        while queue and len(failed_nodes) < target_num_failures:
            current_node, current_time, accumulated_stress, priority = queue.pop(0)
            
            # Check all neighbors connected by edges
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                
                # Calculate stress propagation
                stress_multiplier = accumulated_stress * propagation_weight
                
                # Neighbor experiences increased stress from failed node
                neighbor_loading = current_loading[neighbor] * (1.0 + stress_multiplier * 0.4)
                neighbor_voltage = current_voltage[neighbor] * (1.0 - stress_multiplier * 0.15)
                neighbor_temperature = current_temperature[neighbor] + stress_multiplier * 25
                
                # Check if neighbor should fail based on rules
                should_fail, reason = self._check_node_failure(
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if should_fail:
                    # Calculate failure priority (higher stress = higher priority)
                    failure_priority = stress_multiplier
                    failure_time = current_time + np.random.uniform(1.0, 3.0)
                    
                    # Add to candidates list
                    failure_candidates.append((neighbor, failure_time, reason, failure_priority, stress_multiplier))
                    visited.add(neighbor)
        
        # Sort candidates by priority (highest stress fails first)
        failure_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Select top candidates up to target number
        num_to_fail = min(len(failure_candidates), target_num_failures - len(failed_nodes))
        selected_failures = failure_candidates[:num_to_fail]
        
        for neighbor, failure_time, reason, priority, stress_multiplier in selected_failures:
            failure_sequence.append((neighbor, failure_time, reason))
            failed_nodes.add(neighbor)
            print(f"    [CASCADE] Node {neighbor} fails (reason: {reason}, time: {failure_time:.1f}s, stress: {stress_multiplier:.2f})")
        
        return failure_sequence

    def _simulate_normal_operation(
        self,
        stress_level: float,
        sequence_length: int = 60
    ) -> Tuple[List[int], List[float], List[str], int]:
        """
        Simulate NORMAL operation (NO CASCADE).
        Returns empty failure lists to indicate no failures occurred.
        """
        print(f"  [NORMAL] Simulating normal operation at stress level {stress_level:.2f}")
        
        # No failures in normal operation
        return [], [], [], -1  # Empty lists, cascade_start_time = -1 (no cascade)

    def _run_cascade_simulation(
        self,
        initial_trigger: Dict,
        stress_level: float,
        sequence_length: int
    ) -> Tuple[List[int], List[int], List[float], int]:
        """Helper to run the core cascade simulation logic with AGGRESSIVE propagation."""
        load_multiplier = 0.7 + stress_level * 0.4
        load = self.base_load * load_multiplier
        
        total_load = load.sum()
        gen_indices = np.where(self.node_types == 1)[0]
        generation = np.zeros(self.num_nodes)
        total_capacity = self.gen_capacity.sum()
        
        for idx in gen_indices:
            generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
        
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        current_frequency = 60.0
        
        ambient_temp_base = 25 + 10 * np.random.rand()
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base)
        
        if initial_trigger['type'] == 'line_trip':
            initial_failure_line = initial_trigger['line_id']
            failed_lines = [initial_failure_line]
            failed_nodes = []
        elif initial_trigger['type'] == 'generator_trip':
            initial_failure_node = initial_trigger['node_id']
            failed_nodes = [initial_failure_node]
            failed_lines = []
        else:
            failed_nodes = []
            failed_lines = []
            spike_nodes = initial_trigger['affected_nodes']
            for node in spike_nodes:
                load[node] *= 1.5
        
        failure_times = [0.0]
        current_time = 0.0
        
        for iteration in range(25):
            voltages, angles, line_flows, reactive_line_flows, reactive_generation, is_stable = self._run_ac_power_flow(
                generation, load, failed_lines, failed_nodes
            )
            
            if not is_stable:
                num_failed = len(failed_nodes)
                if num_failed >= self.num_nodes * 0.9:  # 90% or more nodes failed
                    print(f"  [COMPLETE] Grid collapse complete ({num_failed}/{self.num_nodes} nodes failed). Scenario generation complete.")
                else:
                    print(f"  [UNSTABLE] AC power flow unstable at timestep {t} ({num_failed}/{self.num_nodes} nodes failed). Stopping scenario generation.")
                break
            
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            current_frequency, load = self._update_frequency_dynamics(
                generation, load, failed_nodes, current_frequency, dt=2.0
            )
            
            ambient_temp = ambient_temp_base + 8 * np.sin(2 * np.pi * ((current_time / 60.0) - 6) / 24)
            self.equipment_temperatures = self._update_thermal_dynamics(
                loading_ratios, ambient_temp, dt=2.0
            )
            
            trips_occurred = False
            
            for i in range(self.num_edges):
                if i in failed_lines:
                    continue
                
                # Trip immediately if overloaded beyond pickup threshold
                if loading_ratios[i] >= self.oc_relay_pickup[i]:
                    failed_lines.append(i)
                    failure_times.append(current_time)
                    trips_occurred = True
                    
                    src, dst = self.edge_index
                    s, d = src[i].item(), dst[i].item()
                    
                    if loading_ratios[i] > 3.0:  # Severe overload
                        if s not in failed_nodes:
                            failed_nodes.append(s)
                            failure_times.append(current_time)
                        if d not in failed_nodes:
                            failed_nodes.append(d)
                            failure_times.append(current_time)
            
            for node in range(self.num_nodes):
                if node in failed_nodes:
                    continue
                
                # Under-voltage trip
                if voltages[node] < self.uv_relay_pickup[node]:
                    failed_nodes.append(node)
                    failure_times.append(current_time)
                    trips_occurred = True
                
                # Under-frequency trip
                elif current_frequency < self.uf_relay_pickup[node]:
                    failed_nodes.append(node)
                    failure_times.append(current_time)
                    trips_occurred = True
                
                # Thermal trip (lowered threshold from 100% to 95%)
                elif self.equipment_temperatures[node] > self.max_safe_temp[node] * 0.95:
                    failed_nodes.append(node)
                    failure_times.append(current_time)
                    trips_occurred = True
            
            if not trips_occurred:
                break
            
            current_time += 0.5  # Faster time progression (was 2.0s)
            
            if len(failed_lines) > self.num_edges * 0.25 or len(failed_nodes) > self.num_nodes * 0.15:
                break
            
            if current_frequency < 58.0:  # Higher threshold (was 57.0)
                print(f"  [COLLAPSE] System frequency collapsed to {current_frequency:.2f} Hz")
                break
        
        return failed_lines, failed_nodes, failure_times, cascade_start_time

    def _force_cascade_scenario(self, sequence_length: int) -> Tuple[List[int], List[int], List[float], int]:
        """Forces a multi-failure scenario if cascade simulation fails."""
        print("  [FORCED CASCADE] Triggering multiple failures...")
        failed_lines = []
        failed_nodes = []
        
        # Force multiple line trips
        num_line_trips = np.random.randint(3, 8)
        failed_lines = np.random.choice(self.num_edges, num_line_trips, replace=False).tolist()
        
        # Force some node trips (e.g., from adjacent to tripped lines or random)
        num_node_trips = np.random.randint(1, 4)
        possible_node_trips = list(range(self.num_nodes))
        for line_idx in failed_lines:
            s, d = self.edge_index[:, line_idx].tolist()
            if s not in possible_node_trips: possible_node_trips.append(s)
            if d not in possible_node_trips: possible_node_trips.append(d)
        
        if len(possible_node_trips) > num_node_trips:
            failed_nodes = np.random.choice(possible_node_trips, num_node_trips, replace=False).tolist()
        else:
            failed_nodes = possible_node_trips # Take all if fewer than required

        # Simulate failure times (simplified)
        failure_times = [0.0] + [float(i) * 0.5 for i in range(1, len(failed_lines) + len(failed_nodes))]
        
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        return failed_lines, failed_nodes, failure_times, cascade_start_time

    def _simulate_realistic_cascade(
        self, 
        initial_trigger: Dict,
        stress_level: float,
        sequence_length: int = 60
    ) -> Tuple[List[int], List[int], List[float], int]:
        """
        Simulate REALISTIC physics-based cascade propagation.
        GUARANTEED to produce a cascade when called.
        """
        
        max_attempts = 3
        for attempt in range(max_attempts):
            # Try to trigger cascade
            failed_lines, failed_nodes, failure_times, cascade_start_time = self._run_cascade_simulation(
                initial_trigger, stress_level, sequence_length
            )
            
            # Check if cascade actually occurred (at least 2 failures)
            if len(failed_lines) + len(failed_nodes) >= 2:
                return failed_lines, failed_nodes, failure_times, cascade_start_time
            
            # Increase stress and try again
            stress_level = min(0.95, stress_level + 0.1)
            print(f"  [RETRY] Cascade didn't propagate, increasing stress to {stress_level:.2f}")
        
        # If still no cascade after max attempts, force a multi-failure scenario
        print(f"  [FORCE] Forcing cascade scenario")
        return self._force_cascade_scenario(sequence_length)
    
    def _generate_correlated_environmental_data(
        self,
        failed_nodes: List[int],
        failed_lines: List[int],
        timestep: int,
        cascade_start: int,
        stress_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate environmental data CORRELATED with infrastructure failures.
        Environmental threats (wildfires, storms) cause infrastructure failures!
        """
        
        satellite_data = np.zeros((self.num_nodes, 12, 16, 16), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Base vegetation/terrain (spatially smooth)
            for band in range(12):
                # Create spatially coherent patterns using Gaussian smoothing
                base_pattern = np.random.randn(16, 16)
                smooth_pattern = gaussian_filter(base_pattern, sigma=2.0)
                satellite_data[node_idx, band] = (smooth_pattern - smooth_pattern.min()) / (smooth_pattern.max() - smooth_pattern.min() + 1e-6)
            
            # Add realistic spectral signatures
            # Bands 0-3: Visible (RGB + NIR) - vegetation has high NIR
            satellite_data[node_idx, 0:4] = 0.3 + 0.3 * satellite_data[node_idx, 0:4]  # Vegetation
            
            # Bands 4-7: SWIR - water absorption bands
            satellite_data[node_idx, 4:8] = 0.2 + 0.2 * satellite_data[node_idx, 4:8]
            
            # Bands 8-9: Moisture indices
            satellite_data[node_idx, 8:10] = 0.4 + 0.2 * satellite_data[node_idx, 8:10]
            
            # Bands 10-11: Thermal - ambient temperature
            satellite_data[node_idx, 10:12] = 0.5 + 0.1 * satellite_data[node_idx, 10:12]
        
        weather_sequence = np.zeros((self.num_nodes, 10, 8), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Temperature (°C): 15-35°C with diurnal cycle
            hour_of_day = (timestep / 60) * 24
            temp_base = 25 + 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            weather_sequence[node_idx, :, 0] = temp_base + np.random.randn(10) * 2
            
            # Humidity (%): inversely correlated with temperature
            weather_sequence[node_idx, :, 1] = 70 - (weather_sequence[node_idx, :, 0] - 25) * 1.5 + np.random.randn(10) * 5
            weather_sequence[node_idx, :, 1] = np.clip(weather_sequence[node_idx, :, 1], 20, 95)
            
            # Wind speed (m/s): higher during high stress
            wind_base = 5 + stress_level * 10
            weather_sequence[node_idx, :, 2] = wind_base + np.random.randn(10) * 2
            weather_sequence[node_idx, :, 2] = np.clip(weather_sequence[node_idx, :, 2], 0, 25)
            
            # Precipitation (mm/h): rare but correlated with humidity
            precip_prob = (weather_sequence[node_idx, :, 1] - 60) / 40  # Higher humidity → more rain
            weather_sequence[node_idx, :, 3] = np.where(
                np.random.rand(10) < np.clip(precip_prob, 0, 0.3),
                np.random.exponential(5, 10),
                0
            )
            
            # Pressure (hPa): 980-1020
            weather_sequence[node_idx, :, 4] = 1000 + np.random.randn(10) * 10
            
            # Solar radiation (W/m²): depends on time of day
            solar_factor = max(0, np.sin(2 * np.pi * (hour_of_day - 6) / 24))
            weather_sequence[node_idx, :, 5] = 800 * solar_factor + np.random.randn(10) * 50
            weather_sequence[node_idx, :, 5] = np.clip(weather_sequence[node_idx, :, 5], 0, 1000)
            
            # Cloud cover (%): inversely correlated with solar radiation
            weather_sequence[node_idx, :, 6] = 100 - weather_sequence[node_idx, :, 5] / 10 + np.random.randn(10) * 15
            weather_sequence[node_idx, :, 6] = np.clip(weather_sequence[node_idx, :, 6], 0, 100)
            
            # Visibility (km): reduced by precipitation and humidity
            weather_sequence[node_idx, :, 7] = 20 - weather_sequence[node_idx, :, 3] * 2 - (weather_sequence[node_idx, :, 1] - 50) / 10
            weather_sequence[node_idx, :, 7] = np.clip(weather_sequence[node_idx, :, 7], 0.5, 20)
        
        threat_indicators = np.zeros((self.num_nodes, 6), dtype=np.float16)
        
        # Base threat level increases with stress
        base_threat = stress_level * 0.2
        threat_indicators += base_threat
        
        if timestep >= cascade_start - 15:  # 15 timesteps (30 seconds) before cascade
            precursor_strength = 1.0 - (cascade_start - timestep) / 15.0  # Grows stronger as cascade approaches
            precursor_strength = max(0, precursor_strength)
            
            # Wildfire threat grows spatially over time
            if failed_nodes:
                fire_center = self.positions[failed_nodes[0]]
                
                for node_idx in range(self.num_nodes):
                    distance = np.linalg.norm(self.positions[node_idx] - fire_center)
                    
                    # Fire threat grows and spreads
                    fire_threat = precursor_strength * 0.8 * np.exp(-distance / 25)
                    threat_indicators[node_idx, 0] += fire_threat
                    
                    # Update satellite thermal bands to show fire
                    if fire_threat > 0.3:
                        # Hot spot in thermal bands
                        center_x, center_y = 8, 8
                        for x in range(16):
                            for y in range(16):
                                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                heat_signature = fire_threat * np.exp(-dist_from_center / 4)
                                satellite_data[node_idx, 10:12, x, y] += heat_signature
                    
                    # Smoke reduces visibility
                    if fire_threat > 0.2:
                        satellite_data[node_idx, 0:4, :, :] *= (1 - fire_threat * 0.3)  # Darkening from smoke
        
        if timestep >= cascade_start and (failed_nodes or failed_lines):
            # Wildfire threat near failed nodes
            for node in failed_nodes:
                threat_indicators[node, 0] += 0.6  # High wildfire risk
                
                # Spatial correlation: nearby nodes also threatened
                distances = np.linalg.norm(self.positions - self.positions[node], axis=1)
                nearby = np.where(distances < 30)[0]
                for nearby_node in nearby:
                    threat_indicators[nearby_node, 0] += 0.3 * np.exp(-distances[nearby_node] / 20)
            
            # Storm/wind threat near failed lines
            src, dst = self.edge_index
            for line in failed_lines:
                s, d = src[line].item(), dst[line].item()
                threat_indicators[s, 5] += 0.5  # Wind threat
                threat_indicators[d, 5] += 0.5
                
                # Thermal signature (equipment overheating before failure)
                if timestep >= cascade_start - 5:  # 5 timesteps before failure
                    satellite_data[s, 10:12, :, :] += 0.3  # Thermal bands
                    satellite_data[d, 10:12, :, :] += 0.3
        
        # Clip to valid range
        threat_indicators = np.clip(threat_indicators, 0, 1)
        
        return satellite_data, weather_sequence, threat_indicators
    
    def _generate_correlated_robotic_data(
        self,
        failed_nodes: List[int],
        failed_lines: List[int],
        timestep: int,
        cascade_start: int,
        equipment_temps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate robotic sensor data CORRELATED with equipment condition.
        Drones detect hot spots, vibration anomalies before failures!
        """
        
        visual_data = np.zeros((self.num_nodes, 3, 32, 32), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Base equipment appearance (gray/metallic)
            visual_data[node_idx, 0, :, :] = 0.5 + np.random.randn(32, 32) * 0.1  # R
            visual_data[node_idx, 1, :, :] = 0.5 + np.random.randn(32, 32) * 0.1  # G
            visual_data[node_idx, 2, :, :] = 0.5 + np.random.randn(32, 32) * 0.1  # B
            
            # Add equipment degradation based on condition
            degradation = 1.0 - self.equipment_condition[node_idx]
            
            # Rust/corrosion (brownish tint)
            if degradation > 0.3:
                visual_data[node_idx, 0, :, :] += degradation * 0.2  # More red
                visual_data[node_idx, 2, :, :] -= degradation * 0.1  # Less blue
            
            # Oil leaks (dark spots)
            if degradation > 0.4:
                num_spots = int(degradation * 5)
                for _ in range(num_spots):
                    x, y = np.random.randint(0, 32, 2)
                    visual_data[node_idx, :, max(0,x-2):min(32,x+3), max(0,y-2):min(32,y+3)] *= 0.5
        
        thermal_data = equipment_temps.reshape(-1, 1, 1, 1) * np.ones((self.num_nodes, 1, 32, 32), dtype=np.float16)
        
        # Add realistic thermal patterns (hot spots at connection points)
        for node_idx in range(self.num_nodes):
            # Create hot spots at equipment connection points
            num_hotspots = np.random.randint(2, 5)
            for _ in range(num_hotspots):
                hx, hy = np.random.randint(4, 28, 2)
                hotspot_temp = equipment_temps[node_idx] + np.random.uniform(5, 15)
                
                # Gaussian hot spot
                for x in range(32):
                    for y in range(32):
                        dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                        thermal_data[node_idx, 0, x, y] += hotspot_temp * np.exp(-dist / 3)
        
        thermal_data += np.random.uniform(-2, 2, (self.num_nodes, 1, 32, 32)).astype(np.float16)
        
        sensor_data = np.zeros((self.num_nodes, 12), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Vibration (3-axis): increases with age and loading
            base_vibration = 0.5 + self.equipment_age[node_idx] * 0.02
            sensor_data[node_idx, 0:3] = base_vibration + np.random.randn(3) * 0.2
            
            # Acoustic (2 channels): corona discharge increases with voltage stress
            sensor_data[node_idx, 3:5] = 0.3 + np.random.randn(2) * 0.1
            
            # Magnetic field (3-axis): proportional to current flow
            sensor_data[node_idx, 5:8] = 1.0 + np.random.randn(3) * 0.3
            
            # Gas sensors (SF6, O2, moisture): degradation indicators
            sensor_data[node_idx, 8] = 0.95 - (1.0 - self.equipment_condition[node_idx]) * 0.2  # SF6 purity
            sensor_data[node_idx, 9] = 0.02 + (1.0 - self.equipment_condition[node_idx]) * 0.05  # O2 contamination
            sensor_data[node_idx, 10] = 0.01 + (1.0 - self.equipment_condition[node_idx]) * 0.08  # Moisture
            
            # Partial discharge: increases dramatically before failure
            sensor_data[node_idx, 11] = (1.0 - self.equipment_condition[node_idx]) * 0.5 + np.random.randn() * 0.1
        
        if timestep >= cascade_start - 10:  # 10 timesteps (20 seconds) before cascade
            precursor_strength = 1.0 - (cascade_start - timestep) / 10.0
            precursor_strength = max(0, precursor_strength)
            
            for node in failed_nodes:
                # Hot spot detected by thermal camera (grows over time)
                thermal_data[node] += 15.0 * precursor_strength  # Up to 15°C hotter
                
                # Vibration anomaly (exponential growth)
                sensor_data[node, 0:3] += 2.0 * precursor_strength ** 2  # Accelerating vibration
                
                # Acoustic anomaly (arcing, corona) - exponential growth
                sensor_data[node, 3:5] += 1.5 * precursor_strength ** 2
                
                # Partial discharge spikes
                sensor_data[node, 11] += 3.0 * precursor_strength ** 2
                
                # Gas contamination increases
                sensor_data[node, 8] -= 0.1 * precursor_strength  # SF6 purity drops
                sensor_data[node, 9] += 0.05 * precursor_strength  # O2 increases
                sensor_data[node_idx, 10] += 0.08 * precursor_strength  # Moisture increases
                
                # Visual signs of stress (discoloration, arcing)
                visual_data[node, 0, :, :] += 0.3 * precursor_strength  # Reddish glow from heat
                visual_data[node, 1:3, :, :] -= 0.2 * precursor_strength  # Darkening
        
        return visual_data, thermal_data, sensor_data
    
    def _generate_scenario_data(
        self,
        stress_level: float,
        sequence_length: int = 30,
        is_cascade: bool = True
    ) -> Optional[Dict]:
        """
        Generate a single scenario (normal or cascade).
        
        Args:
            stress_level: Operating stress level (0.5-0.95)
            sequence_length: Number of timesteps
            is_cascade: If True, generate cascade; if False, generate normal operation
        
        Returns:
            Scenario data dict or None if generation failed
        """
        if is_cascade:
            failed_nodes, failure_times, failure_reasons, cascade_start_time = self._simulate_rule_based_cascade(
                stress_level, sequence_length
            )
        else:
            # Normal operation has stable conditions with stress_level 0.5-0.7 (not 0.6-0.95)
            failed_nodes, failure_times, failure_reasons, cascade_start_time = self._simulate_normal_operation(
                stress_level, sequence_length
            )
        
        # Build a dictionary mapping timestep to failed nodes
        timestep_to_failed_nodes = {}
        for i, node in enumerate(failed_nodes):
            failure_time = failure_times[i]
            # Convert failure time to timestep (assuming cascade_start_time is in timesteps)
            if cascade_start_time >= 0:
                # Ensure failure_time is relative to cascade_start_time if cascade actually starts
                if t_fail := failure_time: # Check if failure_time is not 0
                    failure_timestep = cascade_start_time + int(t_fail)
                else: # if failure_time is 0 (initial trigger)
                    failure_timestep = cascade_start_time
            else:
                failure_timestep = -1  # No failures
            
            # Ensure failure timestep is within sequence length
            if failure_timestep >= sequence_length:
                continue # Skip failures that happen after sequence ends
            
            if failure_timestep not in timestep_to_failed_nodes:
                timestep_to_failed_nodes[failure_timestep] = []
            timestep_to_failed_nodes[failure_timestep].append(node)
        
        # Print summary
        if is_cascade:
            reasons_set = set(failure_reasons)
            print(f"  [CASCADE] Trigger node: {failed_nodes[0] if failed_nodes else 'N/A'}, Total failures: {len(failed_nodes)}, Reasons: {reasons_set}")
        else:
            print(f"  [NORMAL] No failures, stress level: {stress_level:.2f}")
        
        # Instead of flat arrays, create a sequence of timestep dictionaries
        sequence = []
        
        current_frequency = 60.0
        ambient_temp_base = 25 + 10 * np.random.rand()
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base)
        
        # Initialize power flow variables
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        
        cumulative_failed_nodes = set()
        
        # Simulate time steps
        for t in range(sequence_length):
            if is_cascade:
                load_multiplier = 0.7 + stress_level * 0.4  # 0.7-1.08 for cascades
            else:
                load_multiplier = 0.6 + stress_level * 0.2  # 0.6-0.74 for normal (much lower!)
            
            load_values = self.base_load * load_multiplier * (1 + np.random.normal(0, 0.05))
            
            # Update generation
            total_load = load_values.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            total_capacity = self.gen_capacity.sum()
            for idx in gen_indices:
                generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
            
            # Add any nodes that failed at or before this timestep
            if t in timestep_to_failed_nodes:
                cumulative_failed_nodes.update(timestep_to_failed_nodes[t])
            
            failed_nodes_t = list(cumulative_failed_nodes)
            failed_lines_t = []  # No line failures in rule-based approach
            
            # Simulate AC power flow
            voltages, angles, line_flows, reactive_line_flows, reactive_generation, is_stable = self._run_ac_power_flow(
                generation, load_values, failed_lines_t, failed_nodes_t
            )
            
            # If power flow becomes unstable, stop generating data for this scenario
            if not is_stable:
                num_failed = len(failed_nodes_t)
                if num_failed >= self.num_nodes * 0.9:  # 90% or more nodes failed
                    print(f"  [COMPLETE] Grid collapse complete ({num_failed}/{self.num_nodes} nodes failed). Scenario generation complete.")
                else:
                    print(f"  [UNSTABLE] AC power flow unstable at timestep {t} ({num_failed}/{self.num_nodes} nodes failed). Stopping scenario generation.")
                break
            
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            # Update frequency
            current_frequency, load_values = self._update_frequency_dynamics(
                generation, load_values, failed_nodes_t, current_frequency, dt=1.0
            )
            
            # Update thermal dynamics
            ambient_temp = ambient_temp_base + 8 * np.sin(2 * np.pi * ((t / 60.0) - 6) / 24)
            equipment_temps = self._update_thermal_dynamics(loading_ratios, ambient_temp, dt=1.0)
            
            # Generate correlated data
            sat_data, weather_seq, threat_ind = self._generate_correlated_environmental_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, stress_level
            )
            vis_data, thermal_data, sensor_data = self._generate_correlated_robotic_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, equipment_temps
            )
            
            # Calculate cascade timing for this timestep
            if cascade_start_time >= 0:
                # Time until failure for each node that will eventually fail
                current_cascade_timing = np.array([
                    (failure_times[failed_nodes.index(node)] - (t - cascade_start_time) if t >= cascade_start_time else failure_times[failed_nodes.index(node)])
                    if node in failed_nodes else -1.0
                    for node in range(self.num_nodes)
                ], dtype=np.float32)
            else:
                current_cascade_timing = np.full(self.num_nodes, -1.0, dtype=np.float32)
            
            # Set timing to 0 for nodes that fail at this exact timestep
            for node_in_failure_list in timestep_to_failed_nodes.get(t, []):
                current_cascade_timing[node_in_failure_list] = 0.0

            # Ensure negative timing is not set for nodes that have already failed
            current_cascade_timing = np.where(
                np.array([node in cumulative_failed_nodes for node in range(self.num_nodes)]), # If node is in cumulative_failed_nodes
                np.maximum(0.0, current_cascade_timing), # Set to 0 if already failed or calculate remaining time
                current_cascade_timing # Otherwise, keep as -1.0
            )

            timestep_data = {
                # Environmental data
                'satellite_data': sat_data.astype(np.float32),  # [N, 12, 16, 16]
                'weather_sequence': weather_seq.astype(np.float32),  # [N, 10, 8]
                'threat_indicators': threat_ind.astype(np.float32),  # [N, 6]
                
                # Infrastructure data (SCADA, PMU, equipment status)
                'scada_data': np.column_stack([
                    voltages,  # Voltage magnitude
                    angles,  # Voltage angle
                    generation,  # Real power generation
                    reactive_generation,  # Reactive power generation
                    load_values,  # Real power load
                    load_values * 0.33,  # Reactive power load (assume 0.95 power factor)
                    equipment_temps,  # Temperature
                    np.full(self.num_nodes, current_frequency),  # Frequency
                    self.equipment_age,  # Equipment age
                    self.equipment_condition,  # Equipment condition
                    self.gen_capacity,  # Generator capacity
                    self.base_load,  # Base load
                    self.node_types,  # Node type
                    np.full(self.num_nodes, t / sequence_length),  # Time progress
                    np.full(self.num_nodes, stress_level),  # Stress level
                ]).astype(np.float32),  # [N, 15]
                
                'pmu_sequence': np.column_stack([
                    voltages,
                    angles,
                    generation,
                    load_values,
                    equipment_temps,
                    np.full(self.num_nodes, current_frequency),
                    loading_ratios.mean() * np.ones(self.num_nodes),  # Average loading
                    reactive_generation,  # Reactive power
                ]).astype(np.float32),  # [N, 8]
                
                'equipment_status': np.column_stack([
                    self.equipment_age,
                    self.equipment_condition,
                    equipment_temps,
                    self.thermal_capacity,
                    self.cooling_effectiveness,
                    self.thermal_time_constant / 30.0,  # Normalize to 0-1
                    (equipment_temps / self.temperature_failure_threshold),  # Thermal stress
                    self.node_types,
                    self.gen_capacity / (self.gen_capacity.max() + 1e-6),  # Normalized capacity
                    load_values / (self.base_load + 1e-6),  # Loading ratio
                ]).astype(np.float32),  # [N, 10]
                
                # Robotic data
                'visual_data': vis_data.astype(np.float16),  # [N, 3, 32, 32]
                'thermal_data': thermal_data.astype(np.float16),  # [N, 1, 32, 32]
                'sensor_data': sensor_data.astype(np.float16),  # [N, 12]
                
                # Edge features
                'edge_attr': np.column_stack([
                    self.line_reactance,
                    self.thermal_limits,
                    self.line_resistance,
                    self.line_susceptance,
                    self.line_conductance,
                ]).astype(np.float32),  # [E, 5]
                
                'node_labels': np.array([1.0 if node in cumulative_failed_nodes else 0.0 
                                        for node in range(self.num_nodes)], dtype=np.float32),  # [N]
                
                'cascade_timing': current_cascade_timing, # Use calculated timing for this step
                
                # Physics properties for physics-informed loss
                'conductance': self.line_conductance.astype(np.float32),  # [E]
                'susceptance': self.line_susceptance.astype(np.float32),  # [E]
                'thermal_limits': self.thermal_limits.astype(np.float32),  # [E]
                'power_injection': (generation - load_values).astype(np.float32),  # [N]
                'reactive_injection': (reactive_generation - load_values * 0.33).astype(np.float32),  # [N]
            }
            
            sequence.append(timestep_data)
        
        if len(sequence) > 0:
            last_step = sequence[-1]
            num_positive = int(last_step['node_labels'].sum())
            print(f"  [LABELS] Final timestep: {num_positive}/{self.num_nodes} nodes labeled as failed ({num_positive/self.num_nodes*100:.1f}%)")
        
        return {
            'sequence': sequence,  # List of timestep dictionaries
            'edge_index': self.edge_index.numpy(),  # [2, E]
            'metadata': {
                'cascade_start_time': cascade_start_time,
                'failed_nodes': failed_nodes,  # Added for debugging
                'failure_times': failure_times,  # Added for debugging
                'is_cascade': is_cascade,  # Updated to reflect conversion
                'stress_level': stress_level,
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'base_mva': 100.0,  # Base MVA for power system
            }
        }

    def generate_data(self, num_scenarios: int, sequence_length: int = 30):
        """Generates multi-modal data for a specified number of scenarios."""
        
        all_data = []
        
        for scenario_idx in range(num_scenarios):
            print(f"\n--- Generating Scenario {scenario_idx+1}/{num_scenarios} ---")
            
            # Determine if this scenario should be a cascade or normal operation
            # For now, assuming `num_scenarios` is the total to generate,
            # and the split will happen outside this method.
            # This method will generate data for both types.
            
            # For demonstration, let's alternate between normal and cascade for simplicity here
            # In a real application, you'd use the argparse --normal and --cascade counts.
            is_cascade_scenario = scenario_idx % 2 == 1 # Generate cascade every other scenario
            
            # Stress level: lower for normal operation, higher for cascades
            stress_level = np.random.uniform(0.5, 0.85) if not is_cascade_scenario else np.random.uniform(0.6, 0.95)
            
            # Generate scenario data
            scenario_data = self._generate_scenario_data(
                stress_level=stress_level,
                sequence_length=sequence_length,
                is_cascade=is_cascade_scenario
            )
            
            if scenario_data is not None:
                all_data.append(scenario_data)
            else:
                print(f"  Skipping scenario {scenario_idx+1} due to power flow instability during generation.")
                
            gc.collect()
            if MemoryMonitor.check_threshold():
                print("Memory usage high, cleaning up.")
        
        return all_data

def main():
    """Main function to generate dataset."""
    parser = argparse.ArgumentParser(description='Generate multi-modal cascade failure dataset')
    parser.add_argument('--normal', type=int, default=30, help='Number of normal scenarios')
    parser.add_argument('--cascade', type=int, default=20, help='Number of cascade scenarios')
    parser.add_argument('--grid-size', type=int, default=118, help='Number of nodes in grid')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length (timesteps)')
    parser.add_argument('--batch-size', type=int, default=30, help='Batch size for streaming')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topology-file', type=str, default=None, help='Path to grid topology pickle file')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Validate split ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"
    
    # Create output directories
    train_dir = Path(args.output_dir) / 'train_batches'
    val_dir = Path(args.output_dir) / 'val_batches'
    test_dir = Path(args.output_dir) / 'test_batches'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = PhysicsBasedGridSimulator(
        num_nodes=args.grid_size,
        seed=args.seed,
        topology_file=args.topology_file or os.path.join(args.output_dir, 'grid_topology.pkl')
    )
    
    # Save the generated topology if it was created internally
    if not (args.topology_file and os.path.exists(args.topology_file)):
        print(f"Saving grid topology to {args.output_dir}/grid_topology.pkl")
        topology_data = {
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index.numpy(),
            'positions': simulator.positions
        }
        with open(os.path.join(args.output_dir, 'grid_topology.pkl'), 'wb') as f:
            pickle.dump(topology_data, f)
    
    print(f"\n{'='*80}")
    print(f"GENERATING {args.normal} NORMAL SCENARIOS (no cascades)")
    print(f"{'='*80}")
    normal_scenarios = []
    for i in range(args.normal):
        print(f"\n--- Generating Normal Scenario {i+1}/{args.normal} ---")
        stress_level = np.random.uniform(0.5, 0.85)  # Lower stress for normal operation
        
        # Generate normal operation data
        scenario_data = simulator._generate_scenario_data(
            stress_level=stress_level,
            sequence_length=args.sequence_length,
            is_cascade=False  # Normal scenario
        )
        
        if scenario_data is not None:
            normal_scenarios.append(scenario_data)
        else:
            print(f"  Skipping normal scenario {i+1} due to power flow instability during generation.")
    
    print(f"\n{'='*80}")
    print(f"GENERATING {args.cascade} CASCADE SCENARIOS")
    print(f"{'='*80}")
    cascade_scenarios = []
    for i in range(args.cascade):
        print(f"\n--- Generating Cascade Scenario {i+1}/{args.cascade} ---")
        stress_level = np.random.uniform(0.6, 0.95)  # Higher stress for cascades
        
        # Generate cascade data
        scenario_data = simulator._generate_scenario_data(
            stress_level=stress_level,
            sequence_length=args.sequence_length,
            is_cascade=True  # Cascade scenario
        )
        
        if scenario_data is not None:
            cascade_scenarios.append(scenario_data)
        else:
            print(f"  Skipping cascade scenario {i+1} due to power flow instability during generation.")
    
    all_scenarios = normal_scenarios + cascade_scenarios
    np.random.shuffle(all_scenarios)
    
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"  Normal scenarios: {len(normal_scenarios)}")
    print(f"  Cascade scenarios: {len(cascade_scenarios)}")
    print(f"  Total scenarios: {len(all_scenarios)}")
    
    num_scenarios = len(all_scenarios)
    num_train = int(num_scenarios * args.train_ratio)
    num_val = int(num_scenarios * args.val_ratio)
    num_test = num_scenarios - num_train - num_val
    
    train_scenarios = all_scenarios[:num_train]
    val_scenarios = all_scenarios[num_train:num_train+num_val]
    test_scenarios = all_scenarios[num_train+num_val:]
    
    print(f"\nTRAIN/VAL/TEST SPLIT:")
    print(f"  Training: {len(train_scenarios)} scenarios ({len(train_scenarios)/num_scenarios*100:.1f}%)")
    print(f"  Validation: {len(val_scenarios)} scenarios ({len(val_scenarios)/num_scenarios*100:.1f}%)")
    print(f"  Test: {len(test_scenarios)} scenarios ({len(test_scenarios)/num_scenarios*100:.1f}%)")
    
    print(f"\nSaving training data to {train_dir}...")
    for i in range(0, len(train_scenarios), args.batch_size):
        batch_data = train_scenarios[i:i+args.batch_size]
        batch_file = train_dir / f'scenarios_batch_{i//args.batch_size}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"  Saved {len(batch_data)} scenarios to {batch_file}")
    
    print(f"\nSaving validation data to {val_dir}...")
    for i in range(0, len(val_scenarios), args.batch_size):
        batch_data = val_scenarios[i:i+args.batch_size]
        batch_file = val_dir / f'scenarios_batch_{i//args.batch_size}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"  Saved {len(batch_data)} scenarios to {batch_file}")
    
    print(f"\nSaving test data to {test_dir}...")
    for i in range(0, len(test_scenarios), args.batch_size):
        batch_data = test_scenarios[i:i+args.batch_size]
        batch_file = test_dir / f'scenarios_batch_{i//args.batch_size}.pkl'
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"  Saved {len(batch_data)} scenarios to {batch_file}")
    
    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Train batches: {train_dir}")
    print(f"  Val batches: {val_dir}")
    print(f"  Test batches: {test_dir}")
    print(f"\nYou can now train the model using:")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
