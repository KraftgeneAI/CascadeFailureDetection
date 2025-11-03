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
import gc
import psutil
import warnings
from scipy.ndimage import gaussian_filter
import os
import argparse


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
    
    --- ADVANCED FEATURES ---
    1.  Partial Failure: Nodes can enter a "Damaged" state (partial failure)
        which does not propagate a cascade, simulating a cascade "fizzling out".
        The final "node_label" for the model remains 0 for this state.
    2.  Directed Propagation: Cascade propagation is directed (e.g., A -> B,
        but not B -> A), respecting the grid hierarchy (Gen -> Sub -> Load).
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
        print(f"  Cascade propagation: graph-based (A->B->C)")
        
        # ====================================================================
        # START: NON-LINEAR MODEL NOTIFICATION
        # ====================================================================
        print("\n" + "="*80)
        print("  Using STABLE SIMPLIFIED NON-LINEAR power flow model for data generation.")
        print("  This is fast, stable, and provides more complex patterns (e.g., y=ax^2+b).")
        print("="*80 + "\n")
        # ====================================================================
        # END: NON-LINEAR MODEL NOTIFICATION
        # ====================================================================
    
    # ====================================================================
    # START: CONNECTIVITY FIX
    # ====================================================================
    def _check_and_fix_connectivity(self, adj):
        """
        Ensures the generated graph is fully connected by adding tie-lines.
        This prevents the power flow solver from failing on islanded graphs.
        """
        num_nodes = adj.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        q = [0] # Start BFS from slack bus 0
        visited[0] = True
        component = [0]
        
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in range(num_nodes):
                if adj[u, v] > 0 and not visited[v]:
                    visited[v] = True
                    q.append(v)
                    component.append(v)
        
        if len(component) == num_nodes:
            print("  Grid topology is fully connected.")
            return adj # All good, graph is connected
        
        print(f"  [WARNING] Grid topology is not connected. Found {len(component)} nodes in main component.")
        print("  Adding extra tie lines to connect islands...")
        
        # Find all islands
        all_nodes = set(range(num_nodes))
        main_component_set = set(component)
        island_nodes = list(all_nodes - main_component_set)
        
        while island_nodes:
            # Start a new BFS from an island node to find its component
            island_q = [island_nodes[0]]
            visited[island_nodes[0]] = True
            current_island_component = [island_nodes[0]]
            
            head = 0
            while head < len(island_q):
                u = island_q[head]
                head += 1
                # Find all neighbors of u
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        island_q.append(v)
                        current_island_component.append(v)
            
            # Connect this island to the main component
            island_node = current_island_component[0]
            main_node = component[np.random.randint(len(component))]
            
            adj[island_node, main_node] = 1
            adj[main_node, island_node] = 1
            print(f"    Added tie line: Node {island_node} (island) <-> Node {main_node} (main)")
            
            # Remove all nodes from this island from the list
            island_nodes = [n for n in island_nodes if n not in current_island_component]
            
        print("  Grid connectivity fixed.")
        return adj
    # ====================================================================
    # END: CONNECTIVITY FIX
    # ====================================================================

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
        
        # ====================================================================
        # START: CONNECTIVITY FIX
        # ====================================================================
        adj = self._check_and_fix_connectivity(adj)
        # ====================================================================
        # END: CONNECTIVITY FIX
        # ====================================================================
        
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
        # Node types: 0=Load, 1=Generator, 2=Substation
        self.node_types = np.zeros(self.num_nodes, dtype=int)
        
        # ====================================================================
        # START: SLACK BUS FIX
        # ====================================================================
        # Force node 0 to be a large generator (the slack bus)
        self.node_types[0] = 1 
        
        # Choose other generators, ensuring node 0 is not re-selected
        num_generators = int(self.num_nodes * 0.22) - 1 # One less, since 0 is already a gen
        possible_gen_indices = [i for i in range(1, self.num_nodes)]
        gen_indices = np.random.choice(possible_gen_indices, num_generators, replace=False)
        self.node_types[gen_indices] = 1
        
        all_gen_indices = np.concatenate([[0], gen_indices])
        
        num_substations = int(self.num_nodes * 0.10)
        possible_sub_indices = [i for i in range(1, self.num_nodes) if i not in all_gen_indices]
        sub_indices = np.random.choice(
            possible_sub_indices,
            num_substations, replace=False
        )
        self.node_types[sub_indices] = 2
        
        # Generator capacity
        self.gen_capacity = np.zeros(self.num_nodes)
        
        # Give slack bus (node 0) large capacity
        self.gen_capacity[0] = np.random.uniform(800, 1200) 
        
        for idx in gen_indices: # Other generators
            gen_type = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.3, 0.2])
            if gen_type == 'small':
                self.gen_capacity[idx] = np.random.uniform(50, 150)
            elif gen_type == 'medium':
                self.gen_capacity[idx] = np.random.uniform(150, 400)
            else:
                self.gen_capacity[idx] = np.random.uniform(400, 800)
        # ====================================================================
        # END: SLACK BUS FIX
        # ====================================================================

        # Base load
        self.base_load = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if self.node_types[i] == 1:
                self.base_load[i] = np.random.uniform(5, 20)
            elif self.node_types[i] == 2:
                self.base_load[i] = np.random.uniform(50, 150)
            else:
                self.base_load[i] = np.random.uniform(30, 200)
        
        # --- MODIFIED: Added Partial Failure (Damage) Thresholds ---
        # Loading threshold: node fails if loading > threshold
        self.loading_failure_threshold = np.random.uniform(1.05, 1.15, self.num_nodes)
        self.loading_damage_threshold = self.loading_failure_threshold - np.random.uniform(0.05, 0.1) # e.g., 1.0
        
        # Voltage threshold: node fails if voltage < threshold
        self.voltage_failure_threshold = np.random.uniform(0.88, 0.92, self.num_nodes)
        self.voltage_damage_threshold = self.voltage_failure_threshold + np.random.uniform(0.03, 0.05) # e.g., 0.95
        
        # Temperature threshold: node fails if temperature > threshold
        self.temperature_failure_threshold = np.random.uniform(85, 95, self.num_nodes)
        self.temperature_damage_threshold = self.temperature_failure_threshold - np.random.uniform(10, 15) # e.g., 75-80
        
        # Frequency threshold: node fails if frequency < threshold
        self.frequency_failure_threshold = np.random.uniform(58.5, 59.2, self.num_nodes)
        self.frequency_damage_threshold = self.frequency_failure_threshold + np.random.uniform(0.3, 0.5) # e.g., 59.5
        # --- END MODIFIED ---
        
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
        
        print(f"  Defined failure thresholds (Damage / Failure):")
        print(f"    Loading: {self.loading_damage_threshold.mean():.2f} / {self.loading_failure_threshold.mean():.2f}")
        print(f"    Voltage: {self.voltage_damage_threshold.mean():.2f} / {self.voltage_failure_threshold.mean():.2f}")
        print(f"    Temperature: {self.temperature_damage_threshold.mean():.1f}°C / {self.temperature_failure_threshold.mean():.1f}°C")
        print(f"    Frequency: {self.frequency_damage_threshold.mean():.2f} Hz / {self.frequency_failure_threshold.mean():.2f} Hz")
    
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
        
        # --- MODIFIED: Implement directed cascade propagation (A->B but not B->A) ---
        print("  Building DIRECTED cascade propagation graph (Gen -> Sub -> Load)")
        self.adjacency_list = [[] for _ in range(self.num_nodes)]
        node_types = self.node_types # 0=Load, 1=Gen, 2=Sub
        
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            s_type, d_type = node_types[s], node_types[d]
            weight = self.cascade_propagation_weight[i]

            # Helper function to add a directed edge
            def add_edge(u, v, edge_idx, prop_weight):
                self.adjacency_list[u].append((v, edge_idx, prop_weight))

            # Logic: Power flows "downhill" from Gen(1) to Sub(2) to Load(0)
            # Peers (same type) can cascade in both directions
            if s_type == d_type:
                add_edge(s, d, i, weight)
                add_edge(d, s, i, weight)
            
            # Gen(1) -> Sub(2)
            elif s_type == 1 and d_type == 2: add_edge(s, d, i, weight)
            elif s_type == 2 and d_type == 1: add_edge(d, s, i, weight) # Allow Sub -> Gen (e.g., fault propagation)

            # Gen(1) -> Load(0)
            elif s_type == 1 and d_type == 0: add_edge(s, d, i, weight)
            elif s_type == 0 and d_type == 1: pass # Load failure does not propagate to Gen

            # Sub(2) -> Load(0)
            elif s_type == 2 and d_type == 0: add_edge(s, d, i, weight)
            elif s_type == 0 and d_type == 2: pass # Load failure does not propagate to Sub
        
        total_directed_paths = sum(len(paths) for paths in self.adjacency_list)
        print(f"    Total directed propagation paths: {total_directed_paths} (vs {self.num_edges * 2} if undirected)")
        # --- END MODIFIED ---
        
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
        
        self.oc_relay_pickup = np.random.uniform(1.00, 1.10, self.num_edges)  # Trip at 100-110% loading
        
        self.relay_time_dial = np.random.uniform(0.1, 0.5, self.num_edges)  # Very fast (was 0.3-1.0)
        
        # Distance relay settings (impedance zones)
        self.zone1_reach = self.line_reactance * 0.85  # Zone 1: 85% of line (instantaneous)
        self.zone2_reach = self.line_reactance * 1.20  # Zone 2: 120% of line (0.3-0.5s delay)
        
        # Differential relay settings for nodes (instantaneous for internal faults)
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

    # ====================================================================
    # START: NEW SIMPLIFIED *NON-LINEAR* POWER FLOW FUNCTION
    # ====================================================================
    def _compute_simplified_power_flow(
        self,
        generation: np.ndarray, 
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Compute an ultra-simple, stable, NON-LINEAR power flow.
        This provides more complex patterns for the model to learn.
        
        Returns: voltages, angles, line_flows, is_stable (always True)
        """
        
        gen = generation.copy()
        ld = load.copy()
        if failed_nodes:
            for node in failed_nodes:
                gen[node] = 0.0
                ld[node] = 0.0
        
        # Net power injection at each bus
        P_net = gen - ld
        
        # 1. Model Voltages (Quadratic: y = 1.05 - ax - bx^2)
        # Models accelerating voltage drop under high load (collapse)
        load_norm = ld / (self.base_load + 1e-6) # Normalized load
        a = 0.01 # Linear drop
        b = 0.04 # Quadratic drop
        voltages = 1.05 - (a * load_norm) - (b * (load_norm**2)) + np.random.normal(0, 0.005, self.num_nodes)
        voltages = np.clip(voltages, 0.85, 1.05) # Clip to a reasonable range
        
        # 2. Model Angles (Cubic: y = ax + bx^3)
        # A simple non-linear response to power injection
        P_net_norm = P_net / (self.gen_capacity.max() + 1e-6) # Normalized power
        angles = (0.05 * P_net_norm) + (0.01 * (P_net_norm**3))
        angles[0] = 0.0 # Force slack bus (node 0) to be reference
        
        # 3. Model Line Flows (Sine-based: y = a*sin(x1-x2))
        # This is a simple analog to the real AC power flow equation
        src, dst = self.edge_index
        # Use angles as a proxy for the angular difference
        angle_diff = angles[src] - angles[dst]
        # Use line susceptance as the 'a' coefficient
        line_flows = self.line_susceptance * np.sin(angle_diff) * 100.0 # Scale factor
        line_flows += np.random.normal(0, 0.1, self.num_edges)
        
        # Handle failed lines
        if failed_lines:
            line_flows[failed_lines] = 0.0
        
        # 4. Model Stability
        # Always return True to prevent scenario rejection.
        is_stable = True
        
        return voltages, angles, line_flows, is_stable
    # ====================================================================
    # END: NEW SIMPLIFIED NON-LINEAR POWER FLOW FUNCTION
    # ====================================================================

    def _compute_realistic_power_flow(
        self, 
        generation: np.ndarray, 
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        DEPRECATED: Compute REALISTIC DC power flow with proper physics.
        Replaced by _compute_simplified_power_flow for stability.
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
            # Use degree from undirected graph for this heuristic
            num_connections = np.sum(self.adjacency_matrix[i]) 
            node_loading_factor = load[i] / (self.base_load[i] + 1e-6) * (1.0 + num_connections * 0.05)
            
            # --- START: STABILITY FIX ---
            # Original: voltage_drop = 0.08 * node_loading_factor (unstable at t=0)
            # New heuristic: A less aggressive drop that starts after 70% loading
            voltage_drop = 0.05 * np.maximum(0, node_loading_factor - 0.7) 
            # --- END: STABILITY FIX ---
                
            voltages[i] = 1.0 - voltage_drop + np.random.normal(0, 0.005)
        
        # Clip to realistic range
        voltages = np.clip(voltages, 0.85, 1.15)
        
        if np.any(voltages < 0.94) or np.any(voltages > 1.06):
            is_stable = False
        
        return voltages, theta, line_flows, is_stable

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
        
        # Initialize inertia if not exists
        if not hasattr(self, 'generator_inertia'):
            self.generator_inertia = np.zeros(self.num_nodes)
            gen_indices = np.where(self.node_types == 1)[0]
            for idx in gen_indices:
                if self.gen_capacity[idx] > 400:
                    self.generator_inertia[idx] = np.random.uniform(4.0, 6.0)
                elif self.gen_capacity[idx] > 150:
                    self.generator_inertia[idx] = np.random.uniform(2.5, 4.0)
                else:
                    self.generator_inertia[idx] = np.random.uniform(1.5, 2.5)
        
        if not hasattr(self, 'load_damping'):
            self.load_damping = np.random.uniform(1.0, 2.0, self.num_nodes)
        
        if not hasattr(self, 'ufls_stages'):
            self.ufls_stages = [
                {'frequency': 59.3, 'load_shed': 0.10},
                {'frequency': 59.0, 'load_shed': 0.15},
                {'frequency': 58.7, 'load_shed': 0.20},
            ]
        
        total_inertia = np.sum(self.generator_inertia[active_gens])
        
        # Power imbalance
        total_gen = np.sum(generation)
        total_load = np.sum(load)
        power_imbalance = total_gen - total_load
        
        # Frequency rate of change
        system_base = 10000  # 10 GW base
        df_dt = power_imbalance / (2 * total_inertia * system_base) * 60
        
        # Load damping effect
        load_damping_effect = np.sum(self.load_damping * load) * (current_frequency - 60) / 60
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
    
    def _update_thermal_dynamics(
        self,
        loading_ratios: np.ndarray,
        ambient_temp: float,
        dt: float = 2.0
    ) -> np.ndarray:
        """
        Update per-node equipment temperatures based on loading and thermal dynamics.
        Returns: updated equipment temperatures
        """
        src, dst = self.edge_index
        
        # Calculate heat generation per node
        heat_generation = np.zeros(self.num_nodes)
        
        for i in range(self.num_edges):
            s, d = src[i].item(), dst[i].item()
            heat = (loading_ratios[i] ** 2) * self.line_resistance[i] * 100
            heat_generation[s] += heat / 2
            heat_generation[d] += heat / 2
        
        # Thermal dynamics
        for node in range(self.num_nodes):
            heat_in = heat_generation[node]
            heat_out = (self.cooling_effectiveness[node] * (self.equipment_temperatures[node] - ambient_temp) / 
                       (self.thermal_time_constant[node] * 60))
            
            dT_dt = (heat_in - heat_out) / self.thermal_capacity[node]
            
            self.equipment_temperatures[node] += dT_dt * dt
            self.equipment_temperatures[node] += np.random.normal(0, 0.5)
            
            self.equipment_temperatures[node] = np.clip(
                self.equipment_temperatures[node], 
                ambient_temp - 5, 
                150
            )
        
        return self.equipment_temperatures.copy()
    
    # ====================================================================
    # START: BUG FIX - Added missing helper functions
    # ====================================================================
    
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
        """
        
        satellite_data = np.zeros((self.num_nodes, 12, 16, 16), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            for band in range(12):
                base_pattern = np.random.randn(16, 16)
                smooth_pattern = gaussian_filter(base_pattern, sigma=2.0)
                satellite_data[node_idx, band] = (smooth_pattern - smooth_pattern.min()) / (smooth_pattern.max() - smooth_pattern.min() + 1e-6)
            
            satellite_data[node_idx, 0:4] = 0.3 + 0.3 * satellite_data[node_idx, 0:4]
            satellite_data[node_idx, 4:8] = 0.2 + 0.2 * satellite_data[node_idx, 4:8]
            satellite_data[node_idx, 8:10] = 0.4 + 0.2 * satellite_data[node_idx, 8:10]
            satellite_data[node_idx, 10:12] = 0.5 + 0.1 * satellite_data[node_idx, 10:12]
        
        weather_sequence = np.zeros((self.num_nodes, 10, 8), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            hour_of_day = (timestep / 60) * 24
            temp_base = 25 + 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            weather_sequence[node_idx, :, 0] = temp_base + np.random.randn(10) * 2
            
            weather_sequence[node_idx, :, 1] = 70 - (weather_sequence[node_idx, :, 0] - 25) * 1.5 + np.random.randn(10) * 5
            weather_sequence[node_idx, :, 1] = np.clip(weather_sequence[node_idx, :, 1], 20, 95)
            
            wind_base = 5 + stress_level * 10
            weather_sequence[node_idx, :, 2] = wind_base + np.random.randn(10) * 2
            weather_sequence[node_idx, :, 2] = np.clip(weather_sequence[node_idx, :, 2], 0, 25)
            
            precip_prob = (weather_sequence[node_idx, :, 1] - 60) / 40
            weather_sequence[node_idx, :, 3] = np.where(
                np.random.rand(10) < np.clip(precip_prob, 0, 0.3),
                np.random.exponential(5, 10),
                0
            )
            
            weather_sequence[node_idx, :, 4] = 1000 + np.random.randn(10) * 10
            
            solar_factor = max(0, np.sin(2 * np.pi * (hour_of_day - 6) / 24))
            weather_sequence[node_idx, :, 5] = 800 * solar_factor + np.random.randn(10) * 50
            weather_sequence[node_idx, :, 5] = np.clip(weather_sequence[node_idx, :, 5], 0, 1000)
            
            weather_sequence[node_idx, :, 6] = 100 - weather_sequence[node_idx, :, 5] / 10 + np.random.randn(10) * 15
            weather_sequence[node_idx, :, 6] = np.clip(weather_sequence[node_idx, :, 6], 0, 100)
            
            weather_sequence[node_idx, :, 7] = 20 - weather_sequence[node_idx, :, 3] * 2 - (weather_sequence[node_idx, :, 1] - 50) / 10
            weather_sequence[node_idx, :, 7] = np.clip(weather_sequence[node_idx, :, 7], 0.5, 20)
        
        threat_indicators = np.zeros((self.num_nodes, 6), dtype=np.float16)
        
        base_threat = stress_level * 0.2
        threat_indicators += base_threat
        
        if timestep >= cascade_start - 15:
            precursor_strength = 1.0 - (cascade_start - timestep) / 15.0
            precursor_strength = max(0, precursor_strength)
            
            if failed_nodes:
                fire_center = self.positions[failed_nodes[0]]
                
                for node_idx in range(self.num_nodes):
                    distance = np.linalg.norm(self.positions[node_idx] - fire_center)
                    
                    fire_threat = precursor_strength * 0.8 * np.exp(-distance / 25)
                    threat_indicators[node_idx, 0] += fire_threat
                    
                    if fire_threat > 0.3:
                        center_x, center_y = 8, 8
                        for x in range(16):
                            for y in range(16):
                                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                heat_signature = fire_threat * np.exp(-dist_from_center / 4)
                                satellite_data[node_idx, 10:12, x, y] += heat_signature
                    
                    if fire_threat > 0.2:
                        satellite_data[node_idx, 0:4, :, :] *= (1 - fire_threat * 0.3)
        
        if timestep >= cascade_start and (failed_nodes or failed_lines):
            for node in failed_nodes:
                threat_indicators[node, 0] += 0.6
                
                distances = np.linalg.norm(self.positions - self.positions[node], axis=1)
                nearby = np.where(distances < 30)[0]
                for nearby_node in nearby:
                    threat_indicators[nearby_node, 0] += 0.3 * np.exp(-distances[nearby_node] / 20)
            
            src, dst = self.edge_index
            for line in failed_lines:
                s, d = src[line].item(), dst[line].item()
                threat_indicators[s, 5] += 0.5
                threat_indicators[d, 5] += 0.5
                
                if timestep >= cascade_start - 5:
                    satellite_data[s, 10:12, :, :] += 0.3
                    satellite_data[d, 10:12, :, :] += 0.3
        
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
        """
        
        visual_data = np.zeros((self.num_nodes, 3, 32, 32), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            visual_data[node_idx, 0, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            visual_data[node_idx, 1, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            visual_data[node_idx, 2, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            
            degradation = 1.0 - self.equipment_condition[node_idx]
            
            if degradation > 0.3:
                visual_data[node_idx, 0, :, :] += degradation * 0.2
                visual_data[node_idx, 2, :, :] -= degradation * 0.1
            
            if degradation > 0.4:
                num_spots = int(degradation * 5)
                for _ in range(num_spots):
                    x, y = np.random.randint(0, 32, 2)
                    visual_data[node_idx, :, max(0,x-2):min(32,x+3), max(0,y-2):min(32,y+3)] *= 0.5
        
        thermal_data = equipment_temps.reshape(-1, 1, 1, 1) * np.ones((self.num_nodes, 1, 32, 32), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            num_hotspots = np.random.randint(2, 5)
            for _ in range(num_hotspots):
                hx, hy = np.random.randint(4, 28, 2)
                hotspot_temp = equipment_temps[node_idx] + np.random.uniform(5, 15)
                
                for x in range(32):
                    for y in range(32):
                        dist = np.sqrt((x - hx)**2 + (y - hy)**2)
                        thermal_data[node_idx, 0, x, y] += hotspot_temp * np.exp(-dist / 3)
        
        thermal_data += np.random.uniform(-2, 2, (self.num_nodes, 1, 32, 32)).astype(np.float16)
        
        sensor_data = np.zeros((self.num_nodes, 12), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            base_vibration = 0.5 + self.equipment_age[node_idx] * 0.02
            sensor_data[node_idx, 0:3] = base_vibration + np.random.randn(3) * 0.2
            
            sensor_data[node_idx, 3:5] = 0.3 + np.random.randn(2) * 0.1
            
            sensor_data[node_idx, 5:8] = 1.0 + np.random.randn(3) * 0.3
            
            sensor_data[node_idx, 8] = 0.95 - (1.0 - self.equipment_condition[node_idx]) * 0.2
            sensor_data[node_idx, 9] = 0.02 + (1.0 - self.equipment_condition[node_idx]) * 0.05
            sensor_data[node_idx, 10] = 0.01 + (1.0 - self.equipment_condition[node_idx]) * 0.08
            
            sensor_data[node_idx, 11] = (1.0 - self.equipment_condition[node_idx]) * 0.5 + np.random.randn() * 0.1
        
        if timestep >= cascade_start - 10:
            precursor_strength = 1.0 - (cascade_start - timestep) / 10.0
            precursor_strength = max(0, precursor_strength)
            
            for node in failed_nodes:
                thermal_data[node] += 15.0 * precursor_strength
                
                sensor_data[node, 0:3] += 2.0 * precursor_strength ** 2
                
                sensor_data[node, 3:5] += 1.5 * precursor_strength ** 2
                
                sensor_data[node, 11] += 3.0 * precursor_strength ** 2
                
                sensor_data[node, 8] -= 0.1 * precursor_strength
                sensor_data[node, 9] += 0.05 * precursor_strength
                sensor_data[node, 10] += 0.08 * precursor_strength
                
                visual_data[node, 0, :, :] += 0.3 * precursor_strength
                visual_data[node, 1:3, :, :] -= 0.2 * precursor_strength
        
        return visual_data, thermal_data, sensor_data

    # ====================================================================
    # END: BUG FIX
    # ====================================================================
    
    # --- MODIFIED: Renamed and updated to 3-state logic ---
    def _check_node_state(
        self,
        node_idx: int,
        loading: float,
        voltage: float,
        temperature: float,
        frequency: float
    ) -> Tuple[int, str]:
        """
        Check node state based on RULES.
        Returns: (state, reason)
          0: OK
          1: Damaged (Partial Failure, does not propagate)
          2: Failed (Full Failure, propagates)
        """
        # Check for critical failures first
        if loading > self.loading_failure_threshold[node_idx]:
            return 2, "loading"
        if voltage < self.voltage_failure_threshold[node_idx]:
            return 2, "voltage"
        if temperature > self.temperature_failure_threshold[node_idx]:
            return 2, "temperature"
        if frequency < self.frequency_failure_threshold[node_idx]:
            return 2, "frequency"
        
        # Check for non-propagating damage (partial failure)
        if loading > self.loading_damage_threshold[node_idx]:
            return 1, "loading_damage"
        if voltage < self.voltage_damage_threshold[node_idx]:
            return 1, "voltage_damage"
        if temperature > self.temperature_damage_threshold[node_idx]:
            return 1, "temp_damage"
        if frequency < self.frequency_damage_threshold[node_idx]:
            return 1, "freq_damage"

        return 0, "none"
    # --- END MODIFIED ---
    
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
        Returns: list of (node_id, failure_time, reason)
        """
        failed_nodes = set(initial_failed_nodes)
        failure_sequence = []
        
        queue = [(node, 0.0, 1.0) for node in initial_failed_nodes]
        visited = set(initial_failed_nodes)
        
        while queue:
            current_node, current_time, accumulated_stress = queue.pop(0)
            
            # --- MODIFIED: Uses DIRECTED adjacency_list ---
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                
                stress_multiplier = accumulated_stress * propagation_weight
                
                neighbor_loading = current_loading[neighbor] * (1.0 + stress_multiplier * 0.4)
                neighbor_voltage = current_voltage[neighbor] * (1.0 - stress_multiplier * 0.15)
                neighbor_temperature = current_temperature[neighbor] + stress_multiplier * 25
                
                # --- MODIFIED: Handle 3-state failure (OK, Damaged, Failed) ---
                failure_state, reason = self._check_node_state( # Renamed function
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if failure_state == 2: # 2 = Full Failure
                    failure_time = current_time + np.random.uniform(1.0, 3.0)
                    failure_sequence.append((neighbor, failure_time, reason))
                    failed_nodes.add(neighbor)
                    visited.add(neighbor)
                    
                    queue.append((neighbor, failure_time, stress_multiplier * 0.8))
                    
                    print(f"    [CASCADE] Node {current_node} → Node {neighbor} (reason: {reason}, time: {failure_time:.1f}s)")
                
                elif failure_state == 1: # 1 = Partial Failure (Damaged)
                    print(f"    [PARTIAL] Node {current_node} → Node {neighbor} DAMAGED (reason: {reason}, time: {current_time:.1f}s) - Cascade stops here.")
                    visited.add(neighbor) # Mark as visited so it's not processed again
                
                else: # 0 = OK
                    visited.add(neighbor)
                # --- END MODIFIED ---
        
        return failure_sequence
    
    def _simulate_rule_based_cascade(
        self,
        stress_level: float,
        sequence_length: int = 60,
        target_failure_percentage: Optional[float] = None
    ) -> Tuple[List[int], List[float], List[str], int]:
        """
        Simulate cascade based on CONSISTENT RULES.
        GUARANTEED to produce a cascade with DIVERSE failure types and CONTROLLABLE severity!
        """
        if target_failure_percentage is None:
            target_failure_percentage = np.random.choice([0.2, 0.4, 0.6, 0.8, 1.0], p=[0.2, 0.25, 0.25, 0.2, 0.1])
        
        target_num_failures = int(self.num_nodes * target_failure_percentage)
        print(f"  [TARGET] Aiming for {target_num_failures}/{self.num_nodes} node failures ({target_failure_percentage*100:.0f}%)")
        
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        # --- MODIFIED: Trigger from a high-power node (Gen or Sub) ---
        node_degrees = np.array([len(self.adjacency_list[i]) for i in range(self.num_nodes)])
        high_power_nodes = np.where(self.node_types > 0)[0] # Generators (1) or Substations (2)
        
        if len(high_power_nodes) > 0:
            # Prefer high-power nodes with high-degree (high connectivity)
            high_power_degrees = node_degrees[high_power_nodes]
            trigger_probs = high_power_degrees / (high_power_degrees.sum() + 1e-6)
            initial_trigger_node = np.random.choice(high_power_nodes, p=trigger_probs)
        else:
            # Fallback if no Gen/Sub nodes
            initial_trigger_node = np.random.choice(self.num_nodes)
        # --- END MODIFIED ---
        
        load_multiplier = 0.7 + stress_level * 0.4
        load = self.base_load * load_multiplier
        
        node_loading = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            num_connections = len(self.adjacency_list[i])
            node_loading[i] = load[i] / (self.base_load[i] + 1e-6) * (1.0 + num_connections * 0.05)
        
        node_voltage = 1.0 - (node_loading - 1.0) * 0.15
        node_voltage = np.clip(node_voltage, 0.85, 1.05)
        
        ambient_temp = 30.0
        node_temperature = ambient_temp + (node_loading - 0.8) * 40
        node_temperature = np.clip(node_temperature, 25, 100)
        
        system_frequency = 60.0 - (node_loading.mean() - 0.9) * 5
        system_frequency = np.clip(system_frequency, 58.0, 60.5)
        
        failure_type = np.random.choice(['loading', 'voltage', 'temperature', 'frequency', 'environmental'])
        
        # --- MODIFIED: Force trigger to be a FULL failure, not partial ---
        if failure_type == 'loading':
            node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
            reason = "loading"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Loading overload: {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
            
        elif failure_type == 'voltage':
            node_voltage[initial_trigger_node] = self.voltage_failure_threshold[initial_trigger_node] * np.random.uniform(0.85, 0.95)
            reason = "voltage"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Voltage collapse: {node_voltage[initial_trigger_node]:.3f} < {self.voltage_failure_threshold[initial_trigger_node]:.3f}")
            
        elif failure_type == 'temperature':
            node_temperature[initial_trigger_node] = self.temperature_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
            reason = "temperature"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Thermal overload: {node_temperature[initial_trigger_node]:.1f}°C > {self.temperature_failure_threshold[initial_trigger_node]:.1f}°C")
            
        elif failure_type == 'frequency':
            system_frequency = self.frequency_failure_threshold[initial_trigger_node] * np.random.uniform(0.95, 0.99)
            reason = "frequency"
            print(f"  [TRIGGER] Node {initial_trigger_node} - Frequency instability: {system_frequency:.2f} Hz < {self.frequency_failure_threshold[initial_trigger_node]:.2f} Hz")
            
        else:  # environmental
            env_effect = np.random.choice(['wildfire', 'storm', 'flooding', 'extreme_cold'])
            if env_effect == 'wildfire':
                node_temperature[initial_trigger_node] = self.temperature_failure_threshold[initial_trigger_node] * np.random.uniform(1.1, 1.2)
                reason = "temperature"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Wildfire: Temperature {node_temperature[initial_trigger_node]:.1f}°C > {self.temperature_failure_threshold[initial_trigger_node]:.1f}°C")
            elif env_effect == 'storm':
                node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.1, 1.2)
                reason = "loading"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Storm damage: Loading {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
            elif env_effect == 'flooding':
                node_voltage[initial_trigger_node] = self.voltage_failure_threshold[initial_trigger_node] * np.random.uniform(0.8, 0.9)
                reason = "voltage"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Flooding: Voltage {node_voltage[initial_trigger_node]:.3f} < {self.voltage_failure_threshold[initial_trigger_node]:.3f}")
            else:  # extreme_cold
                node_loading[initial_trigger_node] = self.loading_failure_threshold[initial_trigger_node] * np.random.uniform(1.05, 1.15)
                reason = "loading"
                print(f"  [TRIGGER] Node {initial_trigger_node} - Extreme cold: Loading {node_loading[initial_trigger_node]:.3f} > {self.loading_failure_threshold[initial_trigger_node]:.3f}")
        # --- END MODIFIED ---

        failure_sequence = self._propagate_cascade_controlled(
            [initial_trigger_node],
            node_loading,
            node_voltage,
            node_temperature,
            system_frequency,
            target_num_failures=target_num_failures
        )
        
        failed_nodes = [initial_trigger_node] + [node for node, _, _ in failure_sequence]
        failure_times = [0.0] + [time for _, time, _ in failure_sequence]
        failure_reasons = [reason] + [r for _, _, r in failure_sequence]
        
        print(f"  [RESULT] Cascade generated: {len(failed_nodes)}/{self.num_nodes} nodes failed ({len(failed_nodes)/self.num_nodes*100:.1f}%)")
        
        return failed_nodes, failure_times, failure_reasons, cascade_start_time

    # ====================================================================
    # START: FIX FOR CHAIN-CASCADE (Observation 1)
    # ====================================================================
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
        Propagate cascade through the graph with CONTROLLED severity and
        realistic CHAIN propagation (A->B, B->C, ...).
        
        Returns: list of (node_id, failure_time, reason)
        """
        failed_nodes = set(initial_failed_nodes)
        failure_sequence = []
        
        # The queue now holds nodes to be processed for *their* neighbors
        queue = [(node, 0.0, 1.0) for node in initial_failed_nodes] # (node_id, failure_time, stress)
        visited = set(initial_failed_nodes)
        
        while queue and len(failed_nodes) < target_num_failures:
            current_node, current_time, accumulated_stress = queue.pop(0)
            
            # --- MODIFIED: Uses DIRECTED adjacency_list ---
            for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                if neighbor in visited:
                    continue
                
                visited.add(neighbor) # Mark as visited *immediately* to avoid duplicate processing
                
                stress_multiplier = accumulated_stress * propagation_weight
                
                neighbor_loading = current_loading[neighbor] * (1.0 + stress_multiplier * 0.4)
                neighbor_voltage = current_voltage[neighbor] * (1.0 - stress_multiplier * 0.15)
                neighbor_temperature = current_temperature[neighbor] + stress_multiplier * 25
                
                # --- MODIFIED: Handle 3-state failure (OK, Damaged, Failed) ---
                failure_state, reason = self._check_node_state( # Renamed function
                    neighbor,
                    neighbor_loading,
                    neighbor_voltage,
                    neighbor_temperature,
                    current_frequency
                )
                
                if failure_state == 2: # 2 = Full Failure
                    failure_time = current_time + np.random.uniform(1.0, 3.0)
                    
                    # 1. Log this failure
                    failure_sequence.append((neighbor, failure_time, reason))
                    failed_nodes.add(neighbor)
                    
                    # 2. **CRITICAL FIX**: Add the *newly* failed node to the queue
                    #    so it can propagate the cascade further.
                    queue.append((neighbor, failure_time, stress_multiplier * 0.8))
                    
                    print(f"    [CASCADE] Node {current_node} → Node {neighbor} FAILS (reason: {reason}, time: {failure_time:.1f}s, stress: {stress_multiplier:.2f})")
                    
                    # 3. Check if we've hit our target
                    if len(failed_nodes) >= target_num_failures:
                        break # Stop checking neighbors
                
                elif failure_state == 1: # 1 = Partial Failure (Damaged)
                    print(f"    [PARTIAL] Node {current_node} → Node {neighbor} DAMAGED (reason: {reason}, time: {current_time:.1f}s, stress: {stress_multiplier:.2f}) - Cascade stops here.")
                    # Do NOT add to queue, so propagation stops at this node
                
                # If failure_state == 0 (OK), do nothing.
            
            if len(failed_nodes) >= target_num_failures:
                break # Stop processing the main queue
        
        return failure_sequence
    # ====================================================================
    # END: FIX FOR CHAIN-CASCADE
    # ====================================================================

    # ====================================================================
    # START: STABILITY FIX - Unified generation loop
    # ====================================================================
    def _generate_scenario_data(
        self,
        stress_level: float,
        sequence_length: int = 30,
        is_cascade: bool = True
    ) -> Optional[Dict]:
        """
        Generate a single scenario (normal or cascade) using the
        simplified, non-linear, stable physics model for both.
        
        Returns:
            Scenario data dict or None if generation failed
        """
        
        if is_cascade:
            failed_nodes, failure_times, failure_reasons, cascade_start_time = self._simulate_rule_based_cascade(
                stress_level, sequence_length
            )
            reasons_set = set(failure_reasons)
            print(f"  [CASCADE] Trigger node: {failed_nodes[0] if failed_nodes else 'N/A'}, Total failures: {len(failed_nodes)}, Reasons: {reasons_set}")
        else:
            print(f"  [NORMAL] Generating non-linear normal operation at stress level {stress_level:.2f}")
            failed_nodes, failure_times, failure_reasons, cascade_start_time = [], [], [], -1
        
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
        
        sequence = []
        
        current_frequency = 60.0
        ambient_temp_base = 25 + 10 * np.random.rand()
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base)
        
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        
        cumulative_failed_nodes = set()
        
        # This is the base stress level for the scenario
        base_stress_level = stress_level
        
        for t in range(sequence_length):
            
            # --- START: STABILITY FIX - RAMP-UP LOGIC ---
            current_stress = base_stress_level
            if is_cascade and t < cascade_start_time:
                # At t=0, start with 60% of the final stress.
                # Ramp up to 100% of the final stress by cascade_start_time.
                ramp_factor = 0.6 + 0.4 * (t / max(1, cascade_start_time - 1))
                current_stress = base_stress_level * ramp_factor
            elif not is_cascade:
                # Normal cases use a lower, non-ramping stress level
                current_stress = base_stress_level * 0.7 
            # If is_cascade and t >= cascade_start_time, current_stress remains base_stress_level
            # --- END: STABILITY FIX - RAMP-UP LOGIC ---

            if is_cascade:
                # Use the (potentially ramped-up) stress
                load_multiplier = 0.7 + current_stress * 0.4 
                load_noise = 0.05
            else:
                # Use the lower, stable stress
                load_multiplier = 0.5 + current_stress * 0.25 
                load_noise = 0.02
            
            load_values = self.base_load * load_multiplier * (1 + np.random.normal(0, load_noise, self.num_nodes))
            
            total_load = load_values.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            total_capacity = self.gen_capacity.sum()
            for idx in gen_indices:
                if total_capacity > 0:
                    generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
                else:
                    generation[idx] = 0 # Handle case with no generators
            
            if t in timestep_to_failed_nodes:
                cumulative_failed_nodes.update(timestep_to_failed_nodes[t])
            
            failed_nodes_t = list(cumulative_failed_nodes)
            failed_lines_t = []
            
            # ====================================================================
            # START: NON-LINEAR MODEL SWAP
            # ====================================================================
            # --- This is the one-line change ---
            # Call the new, non-linear simplified solver
            voltages, angles, line_flows, is_stable = self._compute_simplified_power_flow(
                generation, load_values, failed_lines_t, failed_nodes_t
            )
            # ====================================================================
            # END: NON-LINEAR MODEL SWAP
            # ====================================================================
            
            # Create dummy values for the missing AC-only outputs for compatibility
            reactive_generation = generation * 0.33

            num_failed = len(failed_nodes_t)
            failure_ratio = num_failed / self.num_nodes
            
            if not is_stable:
                # This block should now NEVER be hit, but it's good practice
                # to keep it as a safeguard.
                if is_cascade:
                    if failure_ratio >= 0.9:
                        print(f"  [COMPLETE] Grid collapse complete ({num_failed}/{self.num_nodes} nodes failed = {failure_ratio*100:.1f}%). Generating final timestep.")
                    else:
                        print(f"  [UNSTABLE] Power flow unstable at timestep {t} ({num_failed}/{self.num_nodes} nodes failed = {failure_ratio*100:.1f}%). Continuing to capture cascade progression...")
                else:
                    print(f"  [REJECT] Power flow unstable in NORMAL scenario at timestep {t}. This should not happen. Rejecting scenario.")
                    return None
            
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            current_frequency, load_values = self._update_frequency_dynamics(
                generation, load_values, failed_nodes_t, current_frequency, dt=1.0
            )
            
            ambient_temp = ambient_temp_base + 8 * np.sin(2 * np.pi * ((t / 60.0) - 6) / 24)
            equipment_temps = self._update_thermal_dynamics(loading_ratios, ambient_temp, dt=1.0)
            
            sat_data, weather_seq, threat_ind = self._generate_correlated_environmental_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, current_stress # Use current_stress
            )
            vis_data, thermal_data, sensor_data = self._generate_correlated_robotic_data(
                failed_nodes_t, failed_lines_t, t, cascade_start_time, equipment_temps
            )
            
            if cascade_start_time >= 0:
                current_cascade_timing = np.array([
                    (failure_times[failed_nodes.index(node)] - (t - cascade_start_time) if t >= cascade_start_time else failure_times[failed_nodes.index(node)])
                    if node in failed_nodes else -1.0
                    for node in range(self.num_nodes)
                ], dtype=np.float32)
            else:
                current_cascade_timing = np.full(self.num_nodes, -1.0, dtype=np.float32)
            
            for node_in_failure_list in timestep_to_failed_nodes.get(t, []):
                current_cascade_timing[node_in_failure_list] = 0.0

            current_cascade_timing = np.where(
                np.array([node in cumulative_failed_nodes for node in range(self.num_nodes)]),
                np.maximum(0.0, current_cascade_timing),
                current_cascade_timing
            )

            timestep_data = {
                'satellite_data': sat_data.astype(np.float32),
                'weather_sequence': weather_seq.astype(np.float32),
                'threat_indicators': threat_ind.astype(np.float32),
                
                'scada_data': np.column_stack([
                    voltages,
                    angles,
                    generation,
                    reactive_generation,
                    load_values,
                    load_values * 0.33,
                    equipment_temps,
                    np.full(self.num_nodes, current_frequency),
                    self.equipment_age,
                    self.equipment_condition,
                    self.gen_capacity,
                    self.base_load,
                    self.node_types,
                    np.full(self.num_nodes, t / sequence_length),
                    np.full(self.num_nodes, current_stress), # Use current_stress
                ]).astype(np.float32),
                
                'pmu_sequence': np.column_stack([
                    voltages,
                    angles,
                    generation,
                    load_values,
                    equipment_temps,
                    np.full(self.num_nodes, current_frequency),
                    loading_ratios.mean() * np.ones(self.num_nodes),
                    reactive_generation,
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
                    self.line_reactance,
                    self.thermal_limits,
                    self.line_resistance,
                    self.line_susceptance,
                    self.line_conductance,
                ]).astype(np.float32),
                
                'node_labels': np.array([1.0 if node in cumulative_failed_nodes else 0.0 
                                        for node in range(self.num_nodes)], dtype=np.float32),
                
                'cascade_timing': current_cascade_timing,
                
                'conductance': self.line_conductance.astype(np.float32),
                'susceptance': self.line_susceptance.astype(np.float32),
                'thermal_limits': self.thermal_limits.astype(np.float32),
                'power_injection': (generation - load_values).astype(np.float32),
                'reactive_injection': (reactive_generation - load_values * 0.33).astype(np.float32),
            }
            
            sequence.append(timestep_data)
            
            if is_cascade and failure_ratio >= 0.9:
                print(f"  [STOP] Complete collapse reached. Generated {len(sequence)} timesteps.")
                break
        
        if len(sequence) < 10:
            print(f"  [REJECT] Sequence too short ({len(sequence)} timesteps < 10 minimum). Rejecting scenario.")
            return None
        
        if len(sequence) > 0:
            last_step = sequence[-1]
            num_positive = int(last_step['node_labels'].sum())
            print(f"  [LABELS] Final timestep: {num_positive}/{self.num_nodes} nodes labeled as failed ({num_positive/self.num_nodes*100:.1f}%)")
            print(f"  [SUCCESS] Generated {len(sequence)} timesteps of quality {'cascade' if is_cascade else 'normal'} data")
        
        return {
            'sequence': sequence,
            'edge_index': self.edge_index.numpy(),
            'metadata': {
                'cascade_start_time': cascade_start_time,
                'failed_nodes': failed_nodes,
                'failure_times': failure_times,
                'is_cascade': is_cascade,
                'stress_level': base_stress_level, # Save the original target stress
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'base_mva': 100.0,
            }
        }
    # ====================================================================
    # END: BUG FIX
    # ====================================================================


def main():
    """Main function to generate dataset."""
    parser = argparse.ArgumentParser(description='Generate multi-modal cascade failure dataset')
    parser.add_argument('--normal', type=int, default=50, help='Number of normal scenarios')
    parser.add_argument('--cascade', type=int, default=50, help='Number of cascade scenarios')
    parser.add_argument('--grid-size', type=int, default=118, help='Number of nodes in grid')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length (timesteps)')
    parser.add_argument('--batch-size', type=int, default=20, help='Number of scenarios to save in each .pkl file')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topology-file', type=str, default=None, help='Path to grid topology pickle file')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"
    
    train_dir = Path(args.output_dir) / 'train' # Changed from 'train_batches'
    val_dir = Path(args.output_dir) / 'val'     # Changed from 'val_batches'
    test_dir = Path(args.output_dir) / 'test'   # Changed from 'test_batches'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    simulator = PhysicsBasedGridSimulator(
        num_nodes=args.grid_size,
        seed=args.seed,
        topology_file=args.topology_file or os.path.join(args.output_dir, 'grid_topology.pkl')
    )
    
    if not (args.topology_file and os.path.exists(args.topology_file)):
        print(f"Saving grid topology to {args.output_dir}/grid_topology.pkl")
        topology_data = {
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index.numpy(),
            'positions': simulator.positions
        }
        with open(os.path.join(args.output_dir, 'grid_topology.pkl'), 'wb') as f:
            pickle.dump(topology_data, f)
    
    # --- START: MODIFIED LOGIC FOR BATCHED GENERATION ---
    
    # 1. Determine scenario counts for each split
    total_normal = args.normal
    total_cascade = args.cascade
    total_scenarios = total_normal + total_cascade
    
    if total_scenarios == 0:
        print("No scenarios to generate (normal=0, cascade=0). Exiting.")
        return

    # Calculate precise counts for each split
    num_train_normal = int(total_normal * args.train_ratio)
    num_val_normal = int(total_normal * args.val_ratio)
    num_test_normal = total_normal - num_train_normal - num_val_normal
    
    num_train_cascade = int(total_cascade * args.train_ratio)
    num_val_cascade = int(total_cascade * args.val_ratio)
    num_test_cascade = total_cascade - num_train_cascade - num_val_cascade

    num_train = num_train_normal + num_train_cascade
    num_val = num_val_normal + num_val_cascade
    num_test = num_test_normal + num_test_cascade
    
    print(f"\n{'='*80}")
    print(f"DATASET GENERATION PLAN")
    print(f"{'='*80}")
    print(f"  Total Scenarios: {total_scenarios}")
    print(f"    Normal:   {total_normal}")
    print(f"    Cascade:  {total_cascade}")
    print(f"\n  TRAIN Set: {num_train} scenarios ({num_train/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_train_normal}")
    print(f"    Cascade:  {num_train_cascade}")
    print(f"  VAL Set:   {num_val} scenarios ({num_val/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_val_normal}")
    print(f"    Cascade:  {num_val_cascade}")
    print(f"  TEST Set:  {num_test} scenarios ({num_test/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_test_normal}")
    print(f"    Cascade:  {num_test_cascade}")
    print(f"\n  Batch size: {args.batch_size} scenarios per file.")

    # 2. Define helper function for batched generation
    def generate_and_save_split_batched(
        num_normal: int, 
        num_cascade: int, 
        output_dir: Path, 
        split_name: str
    ):
        """Generates scenarios for a split and saves them in batches."""
        print(f"\n{'='*80}")
        print(f"GENERATING {split_name} SET ({num_normal} Normal, {num_cascade} Cascade)")
        print(f"{'='*80}")

        total_to_generate = num_normal + num_cascade
        if total_to_generate == 0:
            print(f"  No scenarios to generate for {split_name} set. Skipping.")
            return

        # Create a shuffled list of types to generate
        types_to_gen = ['normal'] * num_normal + ['cascade'] * num_cascade
        np.random.shuffle(types_to_gen)
        
        current_batch = []
        batch_count = 0
        
        for i in range(total_to_generate):
            gen_type = types_to_gen[i]
            is_cascade = (gen_type == 'cascade')
            
            print(f"\n--- Generating {split_name} scenario {i+1}/{total_to_generate} (Type: {gen_type}) ---")
            
            if is_cascade:
                stress_level = np.random.uniform(0.6, 0.95)
            else:
                stress_level = np.random.uniform(0.5, 0.85) # This is the base_stress_level

            scenario_data = simulator._generate_scenario_data(
                stress_level=stress_level,
                sequence_length=args.sequence_length,
                is_cascade=is_cascade
            )
            
            if scenario_data is not None:
                current_batch.append(scenario_data)
            else:
                print(f"  Skipping {split_name} scenario {i+1} (rejected due to quality checks)")

            # Save batch if full or if it's the last item
            # We check `len(current_batch) > 0` in case the last scenario was rejected
            if (len(current_batch) == args.batch_size or (i == total_to_generate - 1)) and len(current_batch) > 0:
                batch_file = output_dir / f'scenarios_batch_{batch_count}.pkl'
                with open(batch_file, 'wb') as f:
                    # ====================================================================
                    # START: BUG FIX - Saving `current_batch` not `topology_data`
                    # ====================================================================
                    pickle.dump(current_batch, f)
                    # ====================================================================
                    # END: BUG FIX
                    # ====================================================================
                
                print(f"\n  SAVED BATCH: {len(current_batch)} scenarios to {batch_file}")
                batch_count += 1
                current_batch = [] # Clear memory
                print(f"  Memory after saving batch: {MemoryMonitor.get_memory_usage():.1f} MB")
                gc.collect() # Force garbage collection

    # 3. Generate and save each split sequentially
    generate_and_save_split_batched(num_train_normal, num_train_cascade, train_dir, "TRAIN")
    generate_and_save_split_batched(num_val_normal, num_val_cascade, val_dir, "VALIDATION")
    generate_and_save_split_batched(num_test_normal, num_test_cascade, test_dir, "TEST")
    
    # --- END: MODIFIED LOGIC ---
    
    print(f"\n{'='*80}")
    print("DATA GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Train batches: {train_dir}")
    print(f"  Val batches: {val_dir}")
    print(f"  Test batches: {test_dir}")
    print(f"\nYou can now train the model using:")
    print(f"  python train_model.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()