"""
Multi-Modal Cascade Failure Data Generator
===========================================

PURPOSE:
--------
Generates realistic power grid cascade failure scenarios with multi-modal data for
training the AI cascade prediction model. Since real cascade failures are rare and
catastrophic, we use physics-based simulation to create synthetic training data.

WHAT IT GENERATES:
------------------
Three types of correlated data that tell the same story from different perspectives:

1. INFRASTRUCTURE DATA (SCADA/PMU sensors):
   - Voltage, frequency, power flow measurements
   - Based on real AC power flow physics
   - 13 features per node per timestep

2. ENVIRONMENTAL DATA (Satellite + Weather):
   - Satellite imagery (12 channels, 16x16 pixels per node)
   - Weather sequences (temperature, humidity, wind, precipitation)
   - Threat indicators (fire, storm, geohazard, flood, ice, equipment damage)
   - Correlated with grid stress and failures

3. ROBOTIC DATA (Drone sensors):
   - Visual feeds (RGB images, 32x32 pixels)
   - Thermal camera (temperature maps, 32x32 pixels)
   - Sensor array (vibration, acoustic, magnetic, oil quality, partial discharge)
   - Shows equipment condition and precursor signals

SCENARIO TYPES:
---------------
- NORMAL (stress 0.3-0.7): Grid operates safely, no failures
- STRESSED (stress 0.8-0.9): High load but no failures (near-miss scenarios)
- CASCADE (stress > 0.9): Failures propagate through grid (A→B→C chain reaction)

HOW IT WORKS:
-------------
1. Create grid topology (118-node IEEE test system)
2. For each scenario:
   a. Set stress level (determines load/generation balance)
   b. Run physics simulation (power flow, thermal, frequency dynamics)
   c. Check if any node exceeds failure thresholds
   d. If failure occurs, propagate cascade through network
   e. Generate correlated environmental and robotic data
   f. Record ground truth labels (which nodes failed, when, why)
3. Save scenarios to train/val/test splits

OUTPUT:
-------
Pickle files containing batches of scenarios, each with:
- Multi-modal input data (infrastructure, environmental, robotic)
- Ground truth labels (failure labels, timing, risk vectors, causal paths)
- Graph structure (topology, edge features)

USAGE:
------
python multimodal_data_generator.py --normal 1000 --cascade 800 --stressed 200 \\
    --output-dir data --topology-file data/grid_topology.pkl

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
import pypsa
        
class MemoryMonitor:
    """
    Monitor memory usage to prevent Out-Of-Memory (OOM) errors.
    
    The data generator creates large arrays (satellite images, thermal maps, etc.)
    that can consume significant memory. This class helps track usage and warn
    before running out of memory.
    
    Methods:
    --------
    get_memory_usage() : float
        Returns current memory usage in MB
        
    check_threshold(threshold_mb) : bool
        Returns True if memory usage exceeds threshold, issues warning
    """
    
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
    Physics-Based Power Grid Simulator for Cascade Failure Generation
    ==================================================================
    
    This class simulates a power grid with realistic physics to generate training
    data for cascade failure prediction. It models:
    
    GRID COMPONENTS:
    ----------------
    - Nodes: Generators, loads, and transmission substations (default: 118 nodes)
    - Edges: Transmission lines connecting nodes (default: ~186 lines)
    - Topology: Based on IEEE 118-bus test system
    
    PHYSICS MODELS:
    ---------------
    1. Power Flow: Simplified AC power flow (voltage, angle, line flows)
       - Voltages drop under heavy load (quadratic model)
       - Angles respond to power injection (cubic model)
       - Line flows follow sin(angle_difference) relationship
    
    2. Frequency Dynamics: System frequency responds to generation/load imbalance
       - Drops when load > generation
       - Rises when generation > load
       - Includes inertia and damping effects
    
    3. Thermal Dynamics: Equipment heats up under load
       - Heat gain proportional to loading²
       - Heat loss proportional to temperature difference
       - Overheating triggers failures
    
    4. Cascade Propagation: Failures spread through network
       - Failed node → power reroutes → neighbors overload → neighbors fail
       - Realistic timing (0.1-0.5 minutes between failures)
       - Stops when stress dissipates or 60% of grid fails
    
    FAILURE MODES:
    --------------
    - Overloading: Line flow > thermal capacity
    - Voltage collapse: Voltage < 0.85 p.u.
    - Overheating: Equipment temperature > 120°C
    - Frequency deviation: Frequency < 58 Hz or > 62 Hz
    
    MULTI-MODAL DATA:
    -----------------
    Generates three correlated data streams:
    1. Infrastructure (SCADA/PMU): Real physics measurements
    2. Environmental (Satellite/Weather): Synthetic but correlated with grid state
    3. Robotic (Drone sensors): Synthetic equipment condition indicators
    
    ADVANCED FEATURES:
    ------------------
    - Partial Failure: Nodes can enter "Damaged" state (cascade fizzles out)
    - Directed Propagation: Cascade follows grid hierarchy (Gen → Sub → Load)
    - Precursor Signals: Warning signs appear 10-15 timesteps before failures
    
    USAGE:
    ------
    simulator = PhysicsBasedGridSimulator(num_nodes=118, seed=42)
    scenario = simulator._generate_scenario_data(stress_level=0.95, sequence_length=30)
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
        # START: PYPSA POWER FLOW INITIALIZATION
        # ====================================================================
        self._initialize_pypsa_network()
        print("\n" + "="*80)
        print("  Using PyPSA AC power flow for accurate physics-based simulation.")
        print("  This provides realistic power flow calculations with proper AC equations.")
        print("="*80 + "\n")
        # ====================================================================
        # END: PYPSA POWER FLOW INITIALIZATION
        # ====================================================================
    
    def _initialize_pypsa_network(self):
        """
        Initialize PyPSA network for accurate AC power flow calculations.
        
        This creates a PyPSA Network object that mirrors the grid topology
        and electrical parameters, enabling realistic power flow computation.
        """
        
        self.pypsa_network = pypsa.Network()
        
        # Add buses (nodes)
        for i in range(self.num_nodes):
            self.pypsa_network.add(
                "Bus",
                f"bus_{i}",
                v_nom=138.0,  # 138 kV transmission voltage
                x=self.positions[i, 0],
                y=self.positions[i, 1]
            )
        
        # Add generators
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            self.pypsa_network.add(
                "Generator",
                f"gen_{idx}",
                bus=f"bus_{idx}",
                p_nom=self.gen_capacity[idx],
                control="PQ",  # PQ control for non-slack generators
                p_set=0.0  # Will be set dynamically during simulation
            )
        
        # Set bus 0 as slack bus
        if 0 in gen_indices:
            self.pypsa_network.generators.loc[f"gen_0", "control"] = "Slack"
        
        # Add loads
        for i in range(self.num_nodes):
            self.pypsa_network.add(
                "Load",
                f"load_{i}",
                bus=f"bus_{i}",
                p_set=0.0  # Will be set dynamically during simulation
            )
        
        # Add transmission lines
        src, dst = self.edge_index
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            self.pypsa_network.add(
                "Line",
                f"line_{i}",
                bus0=f"bus_{s}",
                bus1=f"bus_{d}",
                x=self.line_reactance[i],
                r=self.line_resistance[i],
                b=self.line_susceptance[i],
                g=self.line_conductance[i],
                s_nom=self.thermal_limits[i],
                length=1.0  # Normalized length
            )
        
        print(f"  Initialized PyPSA network: {len(self.pypsa_network.buses)} buses, "
              f"{len(self.pypsa_network.generators)} generators, "
              f"{len(self.pypsa_network.lines)} lines")
    
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
        Initialize node properties and FAILURE RULES with CONVERGENCE GUARANTEE.
        
        This method ensures that generation capacity and load are balanced
        to guarantee PyPSA convergence from the start.
        
        CONVERGENCE STRATEGY:
        ---------------------
        1. Calculate total expected load first
        2. Size generators to provide 150% of load (50% reserve margin)
        3. Ensure slack bus has 30-40% of total capacity
        4. Distribute remaining capacity among other generators
        5. This guarantees sufficient capacity for convergence
        
        NODE TYPES:
        -----------
        - 0 = Load bus (consumes power)
        - 1 = Generator bus (produces power)
        - 2 = Substation bus (transmission hub)
        
        PYPSA MAPPING:
        --------------
        - node_types == 1 → PyPSA Generator attached to bus
        - All nodes → PyPSA Load attached to bus (base_load)
        - Node 0 → Slack bus (voltage/angle reference)
        
        FAILURE THRESHOLDS:
        -------------------
        Each node has thresholds for:
        - Loading (overload failure)
        - Voltage (collapse failure)
        - Temperature (thermal failure)
        - Frequency (instability failure)
        """
        # Node types: 0=Load, 1=Generator, 2=Substation
        self.node_types = np.zeros(self.num_nodes, dtype=int)
        
        # ====================================================================
        # STEP 1: ASSIGN NODE TYPES
        # ====================================================================
        # Force node 0 to be a large generator (the slack bus)
        self.node_types[0] = 1 
        
        # Choose other generators (20-25% of nodes)
        num_generators = max(int(self.num_nodes * 0.22) - 1, 5)  # At least 5 generators
        possible_gen_indices = [i for i in range(1, self.num_nodes)]
        gen_indices = np.random.choice(possible_gen_indices, num_generators, replace=False)
        self.node_types[gen_indices] = 1
        
        all_gen_indices = np.concatenate([[0], gen_indices])
        
        # Choose substations (10% of nodes)
        num_substations = int(self.num_nodes * 0.10)
        possible_sub_indices = [i for i in range(1, self.num_nodes) if i not in all_gen_indices]
        sub_indices = np.random.choice(
            possible_sub_indices,
            num_substations, replace=False
        )
        self.node_types[sub_indices] = 2
        
        # ====================================================================
        # STEP 2: CALCULATE EXPECTED LOAD (BEFORE ASSIGNING CAPACITY)
        # ====================================================================
        # This ensures we size generators appropriately
        self.base_load = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            if self.node_types[i] == 1:  # Generator
                self.base_load[i] = np.random.uniform(5, 20)
            elif self.node_types[i] == 2:  # Substation
                self.base_load[i] = np.random.uniform(50, 150)
            else:  # Load bus
                self.base_load[i] = np.random.uniform(30, 200)
        
        total_load = self.base_load.sum()
        
        # ====================================================================
        # STEP 3: SIZE GENERATORS FOR CONVERGENCE (150% of load)
        # ====================================================================
        # Target: 150% of total load (50% reserve margin)
        target_total_capacity = total_load * 1.50
        
        # Slack bus gets 30-40% of total capacity
        slack_capacity_ratio = np.random.uniform(0.30, 0.40)
        slack_capacity = target_total_capacity * slack_capacity_ratio
        
        # Remaining capacity distributed among other generators
        remaining_capacity = target_total_capacity - slack_capacity
        num_other_gens = len(gen_indices)
        
        self.gen_capacity = np.zeros(self.num_nodes)
        self.gen_capacity[0] = slack_capacity
        
        # Distribute remaining capacity with realistic variation
        # Use Dirichlet distribution for realistic capacity distribution
        if num_other_gens > 0:
            # Generate random weights that sum to 1
            alpha = np.ones(num_other_gens) * 2.0  # Concentration parameter
            weights = np.random.dirichlet(alpha)
            
            # Assign capacities based on weights
            for i, idx in enumerate(gen_indices):
                self.gen_capacity[idx] = remaining_capacity * weights[i]
        
        # ====================================================================
        # STEP 4: VERIFY CONVERGENCE CONDITIONS
        # ====================================================================
        total_capacity = self.gen_capacity.sum()
        reserve_margin = (total_capacity - total_load) / total_load * 100
        
        print(f"  Convergence-aware sizing:")
        print(f"    Total load: {total_load:.1f} MW")
        print(f"    Total gen capacity: {total_capacity:.1f} MW")
        print(f"    Reserve margin: {reserve_margin:.1f}%")
        print(f"    Slack bus capacity: {slack_capacity:.1f} MW ({slack_capacity/total_capacity*100:.1f}%)")
        print(f"    Number of generators: {len(all_gen_indices)}")
        
        # Sanity check
        if reserve_margin < 20:
            print(f"  [WARNING] Low reserve margin ({reserve_margin:.1f}%), increasing capacity...")
            self.gen_capacity *= 1.3
            reserve_margin = (self.gen_capacity.sum() - total_load) / total_load * 100
            print(f"    Adjusted reserve margin: {reserve_margin:.1f}%")
        
        # ====================================================================
        # STEP 5: FAILURE THRESHOLDS (unchanged)
        # ====================================================================
        # Loading threshold: node fails if loading > threshold
        self.loading_failure_threshold = np.random.uniform(1.05, 1.15, self.num_nodes)
        self.loading_damage_threshold = self.loading_failure_threshold - np.random.uniform(0.05, 0.1)
        
        # Voltage threshold: node fails if voltage < threshold
        self.voltage_failure_threshold = np.random.uniform(0.88, 0.92, self.num_nodes)
        self.voltage_damage_threshold = self.voltage_failure_threshold + np.random.uniform(0.03, 0.05)
        
        # Temperature threshold: node fails if temperature > threshold
        self.temperature_failure_threshold = np.random.uniform(85, 95, self.num_nodes)
        self.temperature_damage_threshold = self.temperature_failure_threshold - np.random.uniform(10, 15)
        
        # Frequency threshold: node fails if frequency < threshold
        self.frequency_failure_threshold = np.random.uniform(58.5, 59.2, self.num_nodes)
        self.frequency_damage_threshold = self.frequency_failure_threshold + np.random.uniform(0.3, 0.5)
        
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
        Initialize edge features and CASCADE PROPAGATION RULES with CONVERGENCE GUARANTEE.
        
        This method ensures thermal limits are sized appropriately for the
        expected power flows to avoid overloads in baseline conditions.
        
        CONVERGENCE STRATEGY:
        ---------------------
        1. Calculate expected power flow per line (based on load distribution)
        2. Size thermal limits to 150-200% of expected flow
        3. Ensure minimum thermal limits for short lines
        4. This guarantees no overloads in baseline operation
        
        ELECTRICAL PARAMETERS:
        ----------------------
        - line_reactance (X): Inductive reactance in p.u.
        - line_resistance (R): Resistive losses in p.u. (R/X ratio ~0.1)
        - line_susceptance (B): Shunt capacitance in p.u.
        - line_conductance (G): Shunt conductance (negligible)
        - thermal_limits: Maximum power flow in MVA
        
        CASCADE PROPAGATION:
        --------------------
        - Directed graph: Gen → Sub → Load (failures flow "downhill")
        - Propagation weights: 0.6-0.9 (how strongly failure spreads)
        """
        src, dst = self.edge_index
        distances = np.linalg.norm(
            self.positions[src] - self.positions[dst], axis=1
        )
        
        # ====================================================================
        # STEP 1: ELECTRICAL PARAMETERS (unchanged, already realistic)
        # ====================================================================
        # Reactance (X): Use realistic per-unit values
        self.line_reactance = np.random.uniform(0.0003, 0.0005, self.num_edges) * distances
        self.line_reactance = np.maximum(self.line_reactance, 1e-6)  # Minimum value
        
        # Resistance (R): R/X ratio ~0.1 for transmission lines
        self.line_resistance = self.line_reactance * 0.1
        self.line_resistance = np.maximum(self.line_resistance, 1e-7)  # Minimum value
        
        # Shunt susceptance (B): Very small for transmission lines
        self.line_susceptance = np.random.uniform(1e-6, 3e-6, self.num_edges) * distances
        
        # Conductance (G): Negligible for transmission
        self.line_conductance = np.zeros(self.num_edges)
        
        # ====================================================================
        # STEP 2: CONVERGENCE-AWARE THERMAL LIMITS
        # ====================================================================
        # Estimate expected power flow per line based on network structure
        # Simple heuristic: flow proportional to connected load
        
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
        
        # Ensure minimum thermal limits
        min_thermal_limit = total_load * 0.05  # At least 5% of total load
        self.thermal_limits = np.maximum(self.thermal_limits, min_thermal_limit)
        
        print(f"  Convergence-aware thermal limits:")
        print(f"    Average thermal limit: {self.thermal_limits.mean():.1f} MVA")
        print(f"    Min thermal limit: {self.thermal_limits.min():.1f} MVA")
        print(f"    Max thermal limit: {self.thermal_limits.max():.1f} MVA")
        print(f"    Total capacity: {self.thermal_limits.sum():.1f} MVA")
        print(f"    Capacity/Load ratio: {self.thermal_limits.sum() / total_load:.2f}x")
        
        # ====================================================================
        # STEP 3: CASCADE PROPAGATION (unchanged)
        # ====================================================================
        # When node A fails, how much does it affect connected node B?
        self.cascade_propagation_weight = np.random.uniform(0.6, 0.9, self.num_edges)
        
        # Build DIRECTED cascade propagation graph (Gen -> Sub -> Load)
        print("  Building DIRECTED cascade propagation graph (Gen -> Sub -> Load)")
        self.adjacency_list = [[] for _ in range(self.num_nodes)]
        node_types = self.node_types  # 0=Load, 1=Gen, 2=Sub
        
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
            elif s_type == 2 and d_type == 1: add_edge(d, s, i, weight)

            # Gen(1) -> Load(0)
            elif s_type == 1 and d_type == 0: add_edge(s, d, i, weight)
            elif s_type == 0 and d_type == 1: pass  # Load failure does not propagate to Gen

            # Sub(2) -> Load(0)
            elif s_type == 2 and d_type == 0: add_edge(s, d, i, weight)
            elif s_type == 0 and d_type == 2: pass  # Load failure does not propagate to Sub
        
        total_directed_paths = sum(len(paths) for paths in self.adjacency_list)
        print(f"    Total directed propagation paths: {total_directed_paths} (vs {self.num_edges * 2} if undirected)")
        print(f"  Cascade propagation weights: {self.cascade_propagation_weight.mean():.2f} ± {self.cascade_propagation_weight.std():.2f}")
    
    def _initialize_pypsa_network(self):
        """
        Initialize PyPSA network for accurate AC power flow calculations.

        This creates a PyPSA Network object that mirrors the grid topology
        and electrical parameters, enabling realistic power flow computation.
        """

        self.pypsa_network = pypsa.Network()

        # Add buses (nodes)
        for i in range(self.num_nodes):
            self.pypsa_network.add(
                "Bus",
                f"bus_{i}",
                v_nom=138.0,  # 138 kV transmission voltage
                x=self.positions[i, 0],
                y=self.positions[i, 1]
            )

        # Add generators
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            self.pypsa_network.add(
                "Generator",
                f"gen_{idx}",
                bus=f"bus_{idx}",
                p_nom=self.gen_capacity[idx],
                control="PQ",  # PQ control for non-slack generators
                p_set=0.0  # Will be set dynamically during simulation
            )

        # Set bus 0 as slack bus
        if 0 in gen_indices:
            self.pypsa_network.generators.loc[f"gen_0", "control"] = "Slack"

        # Add loads
        for i in range(self.num_nodes):
            self.pypsa_network.add(
                "Load",
                f"load_{i}",
                bus=f"bus_{i}",
                p_set=0.0  # Will be set dynamically during simulation
            )

        # Add transmission lines
        src, dst = self.edge_index
        for i in range(self.num_edges):
            s, d = int(src[i]), int(dst[i])
            self.pypsa_network.add(
                "Line",
                f"line_{i}",
                bus0=f"bus_{s}",
                bus1=f"bus_{d}",
                x=self.line_reactance[i],
                r=self.line_resistance[i],
                b=self.line_susceptance[i],
                g=self.line_conductance[i],
                s_nom=self.thermal_limits[i],
                length=1.0  # Normalized length
            )

        print(f"  Initialized PyPSA network: {len(self.pypsa_network.buses)} buses, "
              f"{len(self.pypsa_network.generators)} generators, "
              f"{len(self.pypsa_network.lines)} lines")
    
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
    # START: PYPSA AC POWER FLOW FUNCTION (UPDATED FOR Q)
    # ====================================================================
    def _compute_pypsa_power_flow(
        self,
        generation: np.ndarray,
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Returns:
            voltages, angles, line_flows (P), node_reactive (Q), line_reactive_flows (Q), is_stable
        """
        # 1. Calculate Reactive Load (Q) based on ~0.95 Power Factor
        q_load = load * 0.33  
        
        # 2. Update generator setpoints
        gen_indices = np.where(self.node_types == 1)[0]
        for idx in gen_indices:
            gen_name = f"gen_{idx}"
            if failed_nodes and idx in failed_nodes:
                self.pypsa_network.generators.loc[gen_name, "p_set"] = 0.0
            else:
                self.pypsa_network.generators.loc[gen_name, "p_set"] = generation[idx]
        
        # 3. Update load setpoints (Active AND Reactive)
        for i in range(self.num_nodes):
            load_name = f"load_{i}"
            if failed_nodes and i in failed_nodes:
                self.pypsa_network.loads.loc[load_name, "p_set"] = 0.0
                self.pypsa_network.loads.loc[load_name, "q_set"] = 0.0 
            else:
                self.pypsa_network.loads.loc[load_name, "p_set"] = load[i]
                self.pypsa_network.loads.loc[load_name, "q_set"] = q_load[i] 
        
        # 4. Handle failed lines (Temporary high impedance)
        original_x = {}
        original_r = {}
        if failed_lines:
            for line_idx in failed_lines:
                line_name = f"line_{line_idx}"
                original_x[line_name] = self.pypsa_network.lines.loc[line_name, "x"]
                original_r[line_name] = self.pypsa_network.lines.loc[line_name, "r"]
                self.pypsa_network.lines.loc[line_name, "x"] = 1e6
                self.pypsa_network.lines.loc[line_name, "r"] = 1e6
        
        try:
            status = self.pypsa_network.pf()
            is_stable = status.get("converged",{}).get("0",{}).get("now", False)
            
            if not is_stable:
                self._restore_lines(original_x, original_r)
                return (np.ones(self.num_nodes) * 0.95, np.zeros(self.num_nodes), 
                        np.zeros(self.num_edges), np.zeros(self.num_nodes), 
                        np.zeros(self.num_edges), False)
            
            # 5. Extract results from PyPSA
            voltages = np.zeros(self.num_nodes)
            angles = np.zeros(self.num_nodes)
            node_reactive = np.zeros(self.num_nodes) # Net Q at Bus
            
            for i in range(self.num_nodes):
                bus_name = f"bus_{i}"
                voltages[i] = self.pypsa_network.buses_t.v_mag_pu.loc["now", bus_name]
                angles[i] = np.radians(self.pypsa_network.buses_t.v_ang.loc["now", bus_name])
                
                # Net reactive power at bus (Balance of Gen - Load - Line Shunts)
                # Note: 'q' in buses_t represents the nodal balance
                node_reactive[i] = self.pypsa_network.buses_t.q.loc["now", bus_name]
            
            line_flows_p = np.zeros(self.num_edges)
            line_flows_q = np.zeros(self.num_edges)
            
            for i in range(self.num_edges):
                line_name = f"line_{i}"
                if failed_lines and i in failed_lines:
                    line_flows_p[i] = 0.0
                    line_flows_q[i] = 0.0
                else:
                    # p0 and q0 are the active/reactive flows leaving the source bus
                    line_flows_p[i] = self.pypsa_network.lines_t.p0.loc["now", line_name]
                    line_flows_q[i] = self.pypsa_network.lines_t.q0.loc["now", line_name]
            
            self._restore_lines(original_x, original_r)
            return voltages, angles, line_flows_p, node_reactive, line_flows_q, True
            
        except Exception as e:
            print(f"  [WARNING] PyPSA power flow exception: {e}")
            self._restore_lines(original_x, original_r)
            return (np.ones(self.num_nodes) * 0.95, np.zeros(self.num_nodes), 
                    np.zeros(self.num_edges), np.zeros(self.num_nodes), 
                    np.zeros(self.num_edges), False)

    def _restore_lines(self, original_x, original_r):
        """Helper to reset line impedances after a failure simulation."""
        for line_name, x_val in original_x.items():
            self.pypsa_network.lines.loc[line_name, "x"] = x_val
        for line_name, r_val in original_r.items():
            self.pypsa_network.lines.loc[line_name, "r"] = r_val
    # ====================================================================
    # END: PYPSA AC POWER FLOW FUNCTION
    # ====================================================================

    # ====================================================================
    # START: NEW SIMPLIFIED *NON-LINEAR* POWER FLOW FUNCTION (DEPRECATED)
    # ====================================================================
    def _compute_simplified_power_flow(
        self,
        generation: np.ndarray, 
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None,
        failed_nodes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        DEPRECATED: Use _compute_pypsa_power_flow instead.
        
        Compute simplified non-linear power flow for the grid.
        
        This function is deprecated in favor of PyPSA-based power flow
        which provides accurate AC power flow calculations based on proper
        physics equations.
        
        This is a SIMPLIFIED version of AC power flow that's fast and stable,
        but still captures the essential non-linear behavior that makes cascade
        prediction challenging.
        
        WHY SIMPLIFIED?
        ---------------
        Real AC power flow requires solving non-linear equations iteratively
        (Newton-Raphson method), which is slow and can fail to converge. For
        generating thousands of training scenarios, we need something faster
        and more stable.
        
        PHYSICS MODELS:
        ---------------
        1. VOLTAGE (Quadratic drop under load):
           V = 1.05 - 0.01×load - 0.04×load²
           - Light load (0.5): V ≈ 1.04 p.u. (normal)
           - Heavy load (1.0): V ≈ 1.00 p.u. (stressed)
           - Overload (1.2): V ≈ 0.94 p.u. (critical)
           Models accelerating voltage collapse under stress
        
        2. ANGLE (Cubic response to power):
           θ = 0.05×P + 0.01×P³
           - Generators (P > 0): Positive angle (leading)
           - Loads (P < 0): Negative angle (lagging)
           - Non-linear response captures instability
        
        3. LINE FLOW (Sine-based, like real AC):
           Flow = Susceptance × sin(θ_source - θ_dest) × 100
           - Approximates real AC power flow equation
           - Flow proportional to angle difference
           - Susceptance is line property (higher = more flow capacity)
        
        Parameters:
        -----------
        generation : np.ndarray, shape (num_nodes,)
            Power generation at each node in MW
            
        load : np.ndarray, shape (num_nodes,)
            Power consumption at each node in MW
            
        failed_lines : List[int], optional
            Indices of transmission lines that have failed (flow = 0)
            
        failed_nodes : List[int], optional
            Indices of nodes that have failed (generation = 0, load = 0)
        
        Returns:
        --------
        voltages : np.ndarray, shape (num_nodes,)
            Voltage magnitude at each node in per-unit (p.u.)
            Normal range: 0.95-1.05, Critical: <0.90
            
        angles : np.ndarray, shape (num_nodes,)
            Voltage angle at each node in radians
            Reference: Node 0 (slack bus) = 0.0
            
        line_flows : np.ndarray, shape (num_edges,)
            Active power flow on each line in MW
            Positive = source→destination, Negative = destination→source
            
        is_stable : bool
            Always True (this simplified model doesn't check stability)
            Real power flow can fail to converge (unstable system)
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
        Replaced by _compute_pypsa_power_flow for accurate AC power flow.
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
        Generate synthetic environmental data correlated with grid failures.
        
        PURPOSE:
        --------
        Creates realistic-looking satellite imagery, weather data, and threat
        indicators that are CORRELATED with the grid's physical state. This
        teaches the model to fuse multi-modal data.
        
        WHY SYNTHETIC?
        --------------
        Real satellite/weather data for power grids is:
        - Expensive to obtain
        - Rarely available during actual cascades
        - Hard to label (which weather caused which failure?)
        
        Synthetic data lets us create perfect correlations for training.
        
        CORRELATION STRATEGY:
        ---------------------
        1. Base patterns: Realistic but random (weather cycles, terrain)
        2. Stress indicators: Increase with grid stress level
        3. Precursor signals: Appear 10-15 timesteps before failures
        4. Failure signatures: Heat/smoke appear when equipment fails
        5. Spatial propagation: Threats spread to nearby nodes
        
        GENERATED DATA:
        ---------------
        1. SATELLITE IMAGERY (12 channels × 16×16 pixels per node):
           - Channels 0-3: Visible spectrum (RGB + Near-Infrared)
           - Channels 4-7: Vegetation indices (NDVI, etc.)
           - Channels 8-9: Water/moisture content
           - Channels 10-11: Thermal infrared (heat signatures)
           
           Correlation examples:
           - Failed nodes show heat signatures in thermal bands
           - Smoke reduces visibility in RGB bands
           - Storm patterns correlate with high stress
        
        2. WEATHER SEQUENCE (10 timesteps × 8 features per node):
           - Temperature (°C): Diurnal cycle + stress correlation
           - Humidity (%): Inverse correlation with temperature
           - Wind speed (m/s): Higher during stressed scenarios
           - Precipitation (mm/h): Exponential distribution
           - Pressure (hPa): Random walk around 1000 hPa
           - Solar radiation (W/m²): Follows sun angle
           - Cloud cover (%): Inverse of solar radiation
           - Visibility (km): Reduced by precipitation
        
        3. THREAT INDICATORS (6 types per node):
           - [0] Fire/heat threat: High near failed nodes
           - [1] Storm severity: Correlated with stress level
           - [2] Geohazard (landslide/earthquake): Random baseline
           - [3] Flood risk: Correlated with precipitation
           - [4] Ice/snow loading: Seasonal (not implemented)
           - [5] Equipment damage: High on failed lines
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed (used to add heat signatures)
            
        failed_lines : List[int]
            Lines that have failed (used to add damage indicators)
            
        timestep : int
            Current timestep (0 to sequence_length-1)
            Used for time-of-day effects (temperature, solar radiation)
            
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
            Used to add precursor signals before failures
            
        stress_level : float
            Overall grid stress (0.0 to 1.0)
            Higher stress → more threatening weather
        
        Returns:
        --------
        satellite_data : np.ndarray, shape (num_nodes, 12, 16, 16)
            Synthetic satellite imagery for each node
            dtype=float16 to save memory
            
        weather_sequence : np.ndarray, shape (num_nodes, 10, 8)
            Weather time series (last 10 timesteps, 8 features)
            dtype=float16 to save memory
            
        threat_indicators : np.ndarray, shape (num_nodes, 6)
            Threat levels for 6 hazard types (0.0 to 1.0)
            dtype=float16 to save memory
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
        precursor_duration = np.random.randint(8, 20)
        if timestep >= cascade_start - 15:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
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
        Generate synthetic robotic sensor data correlated with equipment condition.
        
        PURPOSE:
        --------
        Simulates what drones and robotic sensors would observe when inspecting
        power grid equipment. This data is correlated with equipment age, condition,
        temperature, and impending failures.
        
        WHY ROBOTIC DATA?
        -----------------
        In real grids, utilities are deploying:
        - Drones with visual/thermal cameras for line inspection
        - Acoustic sensors to detect corona discharge
        - Vibration sensors to detect mechanical stress
        - Oil quality sensors in transformers
        
        This data provides early warning signs that SCADA sensors miss.
        
        CORRELATION STRATEGY:
        ---------------------
        1. Equipment age/condition affects visual appearance (rust, corrosion)
        2. Temperature affects thermal camera readings
        3. Precursor signals (vibration, acoustic) appear before failures
        4. Partial discharge increases as equipment degrades
        
        GENERATED DATA:
        ---------------
        1. VISUAL FEED (3 channels RGB × 32×32 pixels per node):
           - Base: Gray image (0.5 ± 0.1) representing equipment
           - Degradation: Reddish tint (rust), dark spots (corrosion)
           - Correlation: Worse condition → more visual defects
           
           Example:
           - New equipment (condition=1.0): Clean gray
           - Aged equipment (condition=0.6): Rust spots, discoloration
           - Failing equipment: Visible damage, smoke
        
        2. THERMAL CAMERA (1 channel × 32×32 pixels per node):
           - Base: Equipment temperature (from thermal dynamics)
           - Hotspots: Random locations with +5 to +15°C
           - Precursor: Temperature rises 10 timesteps before failure
           
           Example:
           - Normal: 60-80°C, uniform
           - Stressed: 90-100°C, some hotspots
           - Failing: 110-120°C, multiple hotspots
        
        3. SENSOR ARRAY (12 features per node):
           [0-2] Vibration (3-axis accelerometer, m/s²):
                 - Base: 0.5 + equipment_age × 0.02
                 - Increases before failure (mechanical stress)
           
           [3-4] Acoustic (2 microphones, dB):
                 - Base: 0.3 (ambient noise)
                 - Spikes before failure (corona discharge, arcing)
           
           [5-7] Magnetic field (3-axis magnetometer, Tesla):
                 - Base: 1.0 (normal field)
                 - Random noise (not strongly correlated)
           
           [8] Oil quality (transformers only, 0-1):
               - Decreases with equipment condition
               - 0.95 = new oil, 0.70 = degraded oil
           
           [9] Oil moisture content (ppm):
               - Increases with equipment condition
               - High moisture = insulation breakdown risk
           
           [10] Oil acidity (mg KOH/g):
                - Increases with equipment condition
                - High acidity = oil degradation
           
           [11] Partial discharge (pC):
                - Increases with equipment condition
                - Spikes before failure (insulation breakdown)
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed (used to add failure signatures)
            
        failed_lines : List[int]
            Lines that have failed (not used currently)
            
        timestep : int
            Current timestep (0 to sequence_length-1)
            
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
            Used to add precursor signals 10 timesteps before
            
        equipment_temps : np.ndarray, shape (num_nodes,)
            Current equipment temperature at each node (°C)
            From thermal dynamics simulation
        
        Returns:
        --------
        visual_data : np.ndarray, shape (num_nodes, 3, 32, 32)
            RGB visual feed from drone cameras
            dtype=float16 to save memory
            
        thermal_data : np.ndarray, shape (num_nodes, 1, 32, 32)
            Thermal camera feed (temperature in °C)
            dtype=float16 to save memory
            
        sensor_data : np.ndarray, shape (num_nodes, 12)
            Multi-sensor array readings
            dtype=float16 to save memory
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
        
        precursor_duration = np.random.randint(8, 20)
        if timestep >= cascade_start - 10:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
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
    
    # ====================================================================
    # START: IMPROVEMENT 1 (Return initial_reason) - THIS FUNCTION IS NO LONGER USED
    # ====================================================================
    def _simulate_rule_based_cascade(
        self,
        stress_level: float,
        sequence_length: int = 60,
        target_failure_percentage: Optional[float] = None
    ) -> Tuple[List[int], List[float], List[str], int, str]: # <-- Added str for reason
        """
        DEPRECATED: This function uses a "forced" trigger.
        The new _generate_scenario_data is fully deterministic.
        """
        pass
    # ====================================================================
    # END: IMPROVEMENT 1
    # ====================================================================


    # ====================================================================
    # START: IMPROVEMENT 3 (Fix Timing Gaps)
    # ====================================================================
    def _propagate_cascade_controlled(
        self,
            initial_failed_nodes: List[int],
            current_loading: np.ndarray,
            current_voltage: np.ndarray,
            current_temperature: np.ndarray,
            current_frequency: float,
            target_num_failures: int,
            generation: np.ndarray,
            load: np.ndarray
        ) -> List[Tuple[int, float, str]]:
            """
            Propagate cascade failures through the power grid with physics-based power flow recomputation.

            After each failure, this method recomputes the AC power flow using PyPSA to get accurate
            voltages and line loadings for the remaining network. It then evaluates neighboring nodes
            for potential failures based on the actual physics, not simplified stress multipliers.

            Args:
                initial_failed_nodes: List of initially failed node IDs
                current_loading: Initial per-node loading ratios (not used after recomputation)
                current_voltage: Initial voltages (not used after recomputation)
                current_temperature: Node temperatures (Celsius)
                current_frequency: Grid frequency (Hz)
                target_num_failures: Maximum failures to simulate
                generation: Active power generation array (MW)
                load: Active power load array (MW)

            Returns:
                List of (node_id, failure_time_minutes, reason) tuples
            """
            failed_nodes = set(node[0] for node in initial_failed_nodes)
            failed_reasons = [node[1] for node in initial_failed_nodes]
            failure_sequence = [(fail_node, 0, fail_reason) for fail_node, fail_reason in zip(failed_nodes, failed_reasons)]
            queue = [(node[0], 0.0) for node in initial_failed_nodes]
            visited = set(initial_failed_nodes)

            # Recompute power flow after initial failures
            print(f"    [POWER FLOW] Recomputing after initial failures: {list(failed_nodes)}")
            voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable= \
                self._compute_pypsa_power_flow(generation, load, failed_lines=[], failed_nodes=list(failed_nodes))

            if not is_stable:
                print(f"    [WARNING] Power flow unstable after initial failures")

            # Calculate loading ratios
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            src, dst = self.edge_index
            node_loading = np.zeros(self.num_nodes)
            for i in range(self.num_edges):
                s, d = src[i].item(), dst[i].item()
                node_loading[s] = max(node_loading[s], loading_ratios[i])
                node_loading[d] = max(node_loading[d], loading_ratios[i])

            print(f"    [POWER FLOW] Voltage: {voltages.min():.3f}-{voltages.max():.3f} p.u., Max loading: {loading_ratios.max():.3f}")

            while queue and len(failed_nodes) < target_num_failures:
                current_node, current_time = queue.pop(0)

                for neighbor, edge_idx, propagation_weight in self.adjacency_list[current_node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    # Use ACTUAL power flow values
                    neighbor_loading = node_loading[neighbor]
                    neighbor_voltage = voltages[neighbor]
                    neighbor_temperature = current_temperature[neighbor]

                    failure_state, reason = self._check_node_state(
                        neighbor, neighbor_loading, neighbor_voltage, 
                        neighbor_temperature, current_frequency
                    )

                    if failure_state == 2:  # Full Failure
                        physical_delay = np.random.uniform(0.1, 0.5)
                        failure_time = current_time + physical_delay

                        failure_sequence.append((neighbor, failure_time, reason))
                        failed_nodes.add(neighbor)
                        queue.append((neighbor, failure_time))

                        print(f"    [CASCADE] Node {current_node} → {neighbor} FAILS: {reason}, t={failure_time:.2f}min, V={neighbor_voltage:.3f}, L={neighbor_loading:.3f}")

                        # Recompute power flow after this failure
                        print(f"    [POWER FLOW] Recomputing after node {neighbor} failure...")
                        voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable = \
                            self._compute_pypsa_power_flow(generation, load, failed_lines=[], failed_nodes=list(failed_nodes))

                        if not is_stable:
                            print(f"    [WARNING] Power flow unstable after {len(failed_nodes)} failures")

                        # Update loading ratios
                        loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
                        node_loading = np.zeros(self.num_nodes)
                        for i in range(self.num_edges):
                            s, d = src[i].item(), dst[i].item()
                            node_loading[s] = max(node_loading[s], loading_ratios[i])
                            node_loading[d] = max(node_loading[d], loading_ratios[i])

                        print(f"    [POWER FLOW] New voltage: {voltages.min():.3f}-{voltages.max():.3f} p.u., Max loading: {loading_ratios.max():.3f}")

                        if len(failed_nodes) >= target_num_failures:
                            break

                    elif failure_state == 1:  # Partial Failure
                        print(f"    [PARTIAL] Node {current_node} → {neighbor} DAMAGED: {reason} - Cascade stops")

                if len(failed_nodes) >= target_num_failures:
                    break

            return failure_sequence
    # ====================================================================
    # END: IMPROVEMENT 3
    # ====================================================================

    # ====================================================================
    # START: DETERMINISTIC (UN-CONFUSING) SCENARIO GENERATOR
    # ====================================================================
    def _generate_scenario_data(
        self,
        stress_level: float,
        sequence_length: int = 30
    ) -> Optional[Dict]:
        """
        Generate a complete power grid scenario with multi-modal data.
        
        This is the MAIN FUNCTION that orchestrates the entire simulation process.
        It generates one scenario (30 timesteps) with all three data modalities.
        
        PROCESS OVERVIEW:
        -----------------
        1. DETERMINE SCENARIO TYPE (based on stress_level):
           - stress 0.3-0.7: NORMAL (no failures)
           - stress 0.8-0.9: STRESSED (high load, no failures)
           - stress > 0.9: CASCADE (failures propagate)
        
        2. CHECK FOR INITIAL FAILURE:
           - Set load/generation based on stress level
           - Run power flow simulation
           - Check if any node exceeds failure thresholds
           - If yes → cascade scenario
           - If no → normal/stressed scenario
        
        3. PROPAGATE CASCADE (if failure detected):
           - Start from failed node (trigger)
           - Power reroutes to neighbors
           - Neighbors overload and fail
           - Repeat until cascade stops or 60% of grid fails
           - Record failure sequence and timing
        
        4. GENERATE TIME SERIES (30 timesteps):
           For each timestep:
           - Update load/generation (ramp up to stress level)
           - Run power flow (voltages, angles, line flows)
           - Update frequency dynamics (responds to imbalance)
           - Update thermal dynamics (equipment heats up)
           - Apply failures at correct timesteps
           - Generate correlated environmental data
           - Generate correlated robotic data
           - Record all measurements
        
        5. COMPUTE GROUND TRUTH LABELS:
           - Node failure labels (binary: 0=safe, 1=failed)
           - Cascade timing (minutes until failure, -1=never)
           - Risk vectors (7-dimensional assessment)
           - Causal path (sequence of failures)
        
        DETERMINISTIC BEHAVIOR:
        -----------------------
        Given the same stress_level and random seed, this function will
        generate the EXACT same scenario. This is crucial for:
        - Reproducibility (debugging, paper results)
        - Parallel generation (multiple machines, same topology)
        - Consistent train/val/test splits
        
        FAILURE THRESHOLDS:
        -------------------
        A node fails if ANY of these conditions are met:
        - Loading > 100% (overload)
        - Voltage < 0.85 p.u. (voltage collapse)
        - Temperature > 120°C (thermal failure)
        - Frequency < 58 Hz or > 62 Hz (frequency instability)
        
        Parameters:
        -----------
        stress_level : float, range [0.0, 1.0]
            Overall grid stress level that determines scenario type:
            - 0.3-0.7: Normal operation (no failures expected)
            - 0.8-0.9: Stressed operation (near limits, no failures)
            - 0.9-1.0: Critical stress (failures likely, cascade expected)
            
            Higher stress → higher load, lower generation margin
            
        sequence_length : int, default=30
            Number of timesteps to simulate (default: 30 minutes)
            Each timestep represents 1 minute of real time
        
        Returns:
        --------
        scenario : Dict or None
            Complete scenario data if successful, None if generation failed.
            
            Dictionary contains:
            - 'temporal_sequence': Infrastructure data [T, N, 13]
            - 'satellite_data': Environmental imagery [T, N, 12, 16, 16]
            - 'weather_sequence': Weather time series [T, N, 10, 8]
            - 'threat_indicators': Threat levels [T, N, 6]
            - 'visual_data': Drone visual feed [T, N, 3, 32, 32]
            - 'thermal_data': Drone thermal feed [T, N, 1, 32, 32]
            - 'sensor_data': Drone sensors [T, N, 12]
            - 'node_failure_labels': Binary labels [N]
            - 'cascade_timing': Time to failure [N]
            - 'ground_truth_risk': Risk vectors [N, 7]
            - 'is_cascade': Boolean (True if cascade occurred)
            - 'failed_nodes': List of failed node IDs
            - 'failure_times': List of failure times (minutes)
            - 'failure_reasons': List of failure causes
            - 'stress_level': Input stress level
            
        Notes:
        ------
        - Generation can fail if physics simulation becomes unstable
        - Failed generations return None and are retried
        - Typical success rate: >95% for stress < 0.95
        """
        
        print(f"  [INPUT] Generating scenario with stress_level: {stress_level:.3f}")
        
        # This is the base stress level for the scenario
        base_stress_level = stress_level
        
        # --- 1. Determine pre-cascade state ---
        # We run the physics *before* the cascade to see if it fails naturally
        
        cascade_start_time = int(sequence_length * np.random.uniform(0.65, 0.85))
        
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        
        # Ramp up to the target stress level (simulating the lead-up)
        ramp_factor = 1.0 # Use full stress for the check
        current_stress = base_stress_level * ramp_factor
        
        load_multiplier = 0.7 + current_stress * 0.4 
        load_noise = 0.05
        
        load_values = self.base_load * load_multiplier * (1 + np.random.normal(0, load_noise, self.num_nodes))
        
        total_load = load_values.sum()
        gen_indices = np.where(self.node_types == 1)[0]
        total_capacity = self.gen_capacity.sum()
        for idx in gen_indices:
            if total_capacity > 0:
                generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
            else:
                generation[idx] = 0
        
        voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable = self._compute_pypsa_power_flow(
            generation, load_values, failed_lines=[], failed_nodes=[]
        )
        
        loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
        
        ambient_temp_base = 25 + 10 * np.random.rand()
        
        # ====================================================================
        # START: INDEXERROR FIX
        # ====================================================================
        # We must use the full thermal simulation to get per-node temps
        # This call was missing, causing equipment_temps to be a scalar
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base) # Reset temps
        equipment_temps = self._update_thermal_dynamics(loading_ratios, ambient_temp_base, dt=1.0)
        
        # Calculate the per-node loading ratio array (also missing)
        node_loading = load_values / (self.base_load + 1e-6)
        # ====================================================================
        # END: INDEXERROR FIX
        # ====================================================================

        current_frequency = 60.0 - (node_loading.mean() - 0.9) * 5 # Use the new node_loading array
        current_frequency = np.clip(current_frequency, 58.0, 60.5)

        # --- 2. Check for failure ---
        initial_failed_nodes = []
        initial_reason = "none"
        
        failed_reason = []

        for n in range(self.num_nodes):
            # Check for a "damage" (state 1) or "failure" (state 2)
            state, reason = self._check_node_state(
                n,
                node_loading[n], # <-- Use the fixed array
                voltages[n],
                equipment_temps[n], # <-- Use the fixed array
                current_frequency
            )
            
            if state == 2: # State 2 is a full failure
                initial_failed_nodes.append(n)
                initial_reason = reason
                failed_reason.append(reason)
        
        if len(initial_failed_nodes) > 0:
            is_cascade = True
            # Pick just one trigger node to start the propagation
            trigger_node = np.random.choice(initial_failed_nodes)
            print(f"  [CASCADE] Nodes {initial_failed_nodes} FAILED deterministically. Reason: {failed_reason}")
            
            failure_sequence = self._propagate_cascade_controlled(
                list(zip(initial_failed_nodes, failed_reason)),
                node_loading, # <-- Pass the fixed array
                voltages,
                equipment_temps, # <-- Pass the fixed array
                current_frequency,
                target_num_failures=int(self.num_nodes * 0.6), # Target 60%
                generation=generation,  # Pass generation for power flow recomputation
                load=load_values        # Pass load for power flow recomputation
            )
            
            failed_nodes = [node for node, _, _ in failure_sequence]
            failure_times = [time for _, time, _ in failure_sequence]
            failure_reasons = [r for _, _, r in failure_sequence]
            
            risk_vec = np.zeros(7, dtype=np.float32)
            risk_vec[0] = stress_level # threat_severity
            if 'loading' in initial_reason or 'temperature' in initial_reason:
                risk_vec[1] = 0.8 # vulnerability
                risk_vec[2] = 0.7 # operational_impact
                risk_vec[6] = 0.7 # urgency
            elif 'voltage' in initial_reason or 'frequency' in initial_reason:
                risk_vec[1] = 0.7 # vulnerability
                risk_vec[2] = 0.9 # operational_impact
                risk_vec[6] = 0.9 # urgency
            risk_vec[3] = 0.5 + 0.5 * (len(failed_nodes) / self.num_nodes) # cascade_probability
            ground_truth_risk = risk_vec
            
        else:
            # NO failure was triggered
            is_cascade = False
            if stress_level > 0.8:
                print(f"  [STRESSED] No failure thresholds crossed. (e.g., Voltage: {voltages.min():.3f})")
            else:
                print(f"  [NORMAL] No failure thresholds crossed.")
            failed_nodes, failure_times, failure_reasons = [], [], []
            cascade_start_time = -1
            # Risk is based on stress level
            ground_truth_risk = np.array([stress_level, stress_level*0.7, stress_level*0.5, 0.1, 0.1, 0.1, stress_level], dtype=np.float32)
        
        
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
        # Reset equipment temps for the *actual* sequence generation
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base)
        
        generation = np.zeros(self.num_nodes)
        load_values = np.zeros(self.num_nodes)
        
        cumulative_failed_nodes = set()
        
        for t in range(sequence_length):
            
            current_stress = base_stress_level
            if is_cascade and t < cascade_start_time:
                ramp_factor = 0.6 + 0.4 * (t / max(1, cascade_start_time - 1))
                current_stress = base_stress_level * ramp_factor
            elif not is_cascade:
                current_stress = base_stress_level
            
            if is_cascade:
                load_multiplier = 0.7 + current_stress * 0.4 
                load_noise = 0.05
            else:
                load_multiplier = 0.5 + current_stress * 0.4
                load_noise = 0.02
            
            load_values = self.base_load * load_multiplier * (1 + np.random.normal(0, load_noise, self.num_nodes))
            
            total_load = load_values.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            total_capacity = self.gen_capacity.sum()
            for idx in gen_indices:
                if total_capacity > 0:
                    generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
                else:
                    generation[idx] = 0
            
            if t in timestep_to_failed_nodes:
                cumulative_failed_nodes.update(timestep_to_failed_nodes[t])
            
            failed_nodes_t = list(cumulative_failed_nodes)
            failed_lines_t = []
            
            voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable = self._compute_pypsa_power_flow(
                generation, load_values, failed_lines_t, failed_nodes_t
            )

            num_failed = len(failed_nodes_t)
            failure_ratio = num_failed / self.num_nodes
            
            if not is_stable:
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
                failed_nodes_t, failed_lines_t, t, cascade_start_time, current_stress
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
                    node_reactive,
                    load_values,
                    line_flows_q,
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
                'reactive_injection': (node_reactive).astype(np.float32),
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
            print(f"  [SUCCESS] Generated {len(sequence)} timesteps of quality {'cascade' if is_cascade else ('stressed' if stress_level > 0.8 else 'normal')} data")
        
        return {
            'sequence': sequence,
            'edge_index': self.edge_index,
            'metadata': {
                'cascade_start_time': cascade_start_time,
                'failed_nodes': failed_nodes,
                'failure_times': failure_times,
                'failure_reasons': failure_reasons,
                'ground_truth_risk': ground_truth_risk,
                'is_cascade': is_cascade,
                'stress_level': base_stress_level,
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'base_mva': 100.0,
            }
        }
    # ====================================================================
    # END: DETERMINISTIC SCENARIO GENERATOR
    # ====================================================================


def main():
    """
    Main entry point for dataset generation.
    
    WORKFLOW:
    ---------
    1. Parse command-line arguments
    2. Create output directories (train/val/test)
    3. Load or create grid topology
    4. Generate scenarios in batches:
       - Normal scenarios (low stress, no failures)
       - Stressed scenarios (high stress, no failures)
       - Cascade scenarios (critical stress, failures)
    5. Split scenarios into train/val/test sets
    6. Save batches as pickle files
    
    EXAMPLE USAGE:
    --------------
    # Generate 200 scenarios (100 normal, 80 cascade, 20 stressed)
    python multimodal_data_generator.py \\
        --normal 100 \\
        --cascade 80 \\
        --stressed 20 \\
        --output-dir data \\
        --topology-file data/grid_topology.pkl
    
    # For production training (10,000 scenarios)
    python multimodal_data_generator.py \\
        --normal 5000 \\
        --cascade 4000 \\
        --stressed 1000 \\
        --output-dir data \\
        --topology-file data/grid_topology.pkl
    
    # Parallel generation on multiple machines
    # Machine 1:
    python multimodal_data_generator.py --normal 2500 --cascade 2000 \\
        --output-dir data_p1 --start_batch 0 --topology-file shared/grid_topology.pkl
    
    # Machine 2:
    python multimodal_data_generator.py --normal 2500 --cascade 2000 \\
        --output-dir data_p2 --start_batch 4500 --topology-file shared/grid_topology.pkl
    
    # Then merge data_p1 and data_p2 folders
    
    OUTPUT STRUCTURE:
    -----------------
    data/
    ├── grid_topology.pkl          # Grid structure (shared by all scenarios)
    ├── train/
    │   ├── batch_0000.pkl        # Scenarios 0-9 (if batch_size=10)
    │   ├── batch_0001.pkl        # Scenarios 10-19
    │   └── ...
    ├── val/
    │   ├── batch_0000.pkl
    │   └── ...
    └── test/
        ├── batch_0000.pkl
        └── ...
    """
    parser = argparse.ArgumentParser(
        description='Generate multi-modal cascade failure dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (200 scenarios)
  python multimodal_data_generator.py --normal 100 --cascade 80 --stressed 20
  
  # Production dataset (10,000 scenarios)
  python multimodal_data_generator.py --normal 5000 --cascade 4000 --stressed 1000
  
  # Use existing topology
  python multimodal_data_generator.py --normal 100 --cascade 80 \\
      --topology-file data/grid_topology.pkl
        """
    )
    
    # ====================================================================
    # START: MODIFICATION (Add --stressed)
    # ====================================================================
    parser.add_argument('--normal', type=int, default=50, help='Number of normal (low-stress) scenarios ')
    parser.add_argument('--cascade', type=int, default=50, help='Number of cascade scenarios ')
    parser.add_argument('--stressed', type=int, default=50, help='Number of stressed (high-stress, non-failing) scenarios ')
    # ====================================================================
    # END: MODIFICATION
    # ====================================================================
    
    parser.add_argument('--grid-size', type=int, default=118, help='Number of nodes in grid')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length (timesteps)')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of scenarios to save in each .pkl file')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topology-file', type=str, default=None, help='Path to grid topology pickle file')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--start_batch', type=int, default=0, help='Starting batch number for output files (e.g., 5000)')

    args = parser.parse_args()
    
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"
    
    train_dir = Path(args.output_dir) / 'train'
    val_dir = Path(args.output_dir) / 'val'
    test_dir = Path(args.output_dir) / 'test'
    
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
            'edge_index': simulator.edge_index,
            'positions': simulator.positions
        }
        with open(os.path.join(args.output_dir, 'grid_topology.pkl'), 'wb') as f:
            pickle.dump(topology_data, f)
    
    # --- START: MODIFIED LOGIC FOR BATCHED GENERATION ---
    
    total_normal = args.normal
    total_cascade = args.cascade
    total_stressed = args.stressed
    total_scenarios = total_normal + total_cascade + total_stressed
    
    if total_scenarios == 0:
        print("No scenarios to generate (normal=0, cascade=0, stressed=0). Exiting.")
        return

    # Calculate precise counts for each split
    num_train_normal = int(total_normal * args.train_ratio)
    num_val_normal = int(total_normal * args.val_ratio)
    num_test_normal = total_normal - num_train_normal - num_val_normal
    
    num_train_cascade = int(total_cascade * args.train_ratio)
    num_val_cascade = int(total_cascade * args.val_ratio)
    num_test_cascade = total_cascade - num_train_cascade - num_val_cascade

    num_train_stressed = int(total_stressed * args.train_ratio)
    num_val_stressed = int(total_stressed * args.val_ratio)
    num_test_stressed = total_stressed - num_train_stressed - num_val_stressed

    num_train = num_train_normal + num_train_cascade + num_train_stressed
    num_val = num_val_normal + num_val_cascade + num_val_stressed
    num_test = num_test_normal + num_test_cascade + num_test_stressed
    
    print(f"\n{'='*80}")
    print(f"DATASET GENERATION PLAN")
    print(f"{'='*80}")
    print(f"  Total Scenarios: {total_scenarios}")
    print(f"    Normal (Stress 0.3-0.7):   {total_normal}")
    print(f"    Stressed (Stress 0.8-0.9): {total_stressed}")
    print(f"    Cascade (Stress > 0.9):  {total_cascade}")
    print(f"\n  TRAIN Set: {num_train} scenarios ({num_train/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_train_normal}")
    print(f"    Stressed: {num_train_stressed}")
    print(f"    Cascade:  {num_train_cascade}")
    print(f"  VAL Set:   {num_val} scenarios ({num_val/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_val_normal}")
    print(f"    Stressed: {num_val_stressed}")
    print(f"    Cascade:  {num_val_cascade}")
    print(f"  TEST Set:  {num_test} scenarios ({num_test/total_scenarios*100:.1f}%)")
    print(f"    Normal:   {num_test_normal}")
    print(f"    Stressed: {num_test_stressed}")
    print(f"    Cascade:  {num_test_cascade}")
    print(f"\n  Batch size: {args.batch_size} scenarios per file.")
    print(f"  Starting Batch Number: {args.start_batch}")

    # ====================================================================
    # START: MODIFICATION - Add retry logic
    # ====================================================================
    def generate_and_save_split_batched(
        num_normal: int, 
        num_cascade: int,
        num_stressed: int,
        output_dir: Path, 
        split_name: str,
        start_batch: int = 0
    ):
        """Generates scenarios for a split and saves them in batches."""
        print(f"\n{'='*80}")
        print(f"GENERATING {split_name} SET ({num_normal} Normal, {num_stressed} Stressed, {num_cascade} Cascade)")
        print(f"{'='*80}")

        total_to_generate = num_normal + num_cascade + num_stressed
        
        if total_to_generate == 0:
            print(f"  No scenarios to generate for {split_name} set. Skipping.")
            return

        types_to_gen = ['normal'] * num_normal + ['cascade'] * num_cascade + ['stressed'] * num_stressed
        np.random.shuffle(types_to_gen)
        
        current_batch = []
        batch_count = start_batch
        
        for i in range(total_to_generate):
            gen_type = types_to_gen[i]
            
            print(f"\n--- Generating {split_name} scenario {i+1}/{total_to_generate} (Target Type: {gen_type}) ---")
            
            scenario_data = None
            max_retries = 10 # Safeguard against an impossible-to-generate scenario
            retries = 0

            while scenario_data is None and retries < max_retries:
                if gen_type == 'cascade':
                    # Use a stress level high enough to *guarantee* a failure
                    stress_level = np.random.uniform(0.7, 1.0)
                elif gen_type == 'stressed':
                    # Use a stress level that is *high* but *just below* the failure threshold
                    stress_level = np.random.uniform(0.5, 0.62)
                else: # gen_type == 'normal'
                    stress_level = np.random.uniform(0.0, 0.5)
                
                # Generate a candidate scenario
                candidate_scenario = simulator._generate_scenario_data(
                    stress_level=stress_level,
                    sequence_length=args.sequence_length
                )

                if candidate_scenario is None:
                    # Generator rejected it (e.g., too short)
                    retries += 1
                    print(f"  [RETRY {retries}] Scenario rejected by generator (e.g., too short).")
                    continue

                # Check if the *output* type matches the *input* type
                is_cascade_output = candidate_scenario['metadata']['is_cascade']

                if gen_type == 'cascade' and not is_cascade_output:
                    # We *wanted* a cascade, but *got* a normal/stressed one. This is a FAILED ATTEMPT.
                    retries += 1
                    print(f"  [RETRY {retries}] Wanted 'cascade', but stress {stress_level:.3f} only produced 'normal'. Retrying with higher stress...")
                    continue # Loop again
                
                if (gen_type == 'stressed' or gen_type == 'normal') and is_cascade_output:
                    # We *wanted* a normal/stressed, but *got* a cascade. FAILED ATTEMPT.
                    retries += 1
                    print(f"  [RETRY {retries}] Wanted 'normal/stressed', but stress {stress_level:.3f} produced 'cascade'. Retrying with lower stress...")
                    continue # Loop again
                
                # If we get here, the scenario is valid AND matches the type we wanted.
                scenario_data = candidate_scenario
            
            # After the while loop, check if we succeeded
            if scenario_data is not None:
                current_batch.append(scenario_data)
            else:
                print(f"  [FAIL] Skipping {split_name} scenario {i+1}. Max retries ({max_retries}) exceeded.")
                # This will just skip this one scenario
            # ====================================================================
            # END: MODIFICATION
            # ====================================================================

            # Save batch if full or if it's the last item
            if (len(current_batch) == args.batch_size or (i == total_to_generate - 1)) and len(current_batch) > 0:
                batch_file = output_dir / f'scenarios_batch_{batch_count}.pkl'
                with open(batch_file, 'wb') as f:
                    pickle.dump(current_batch, f)
                
                print(f"\n  SAVED BATCH: {len(current_batch)} scenarios to {batch_file}")
                batch_count += 1
                current_batch = [] # Clear memory
                print(f"  Memory after saving batch: {MemoryMonitor.get_memory_usage():.1f} MB")
                gc.collect() # Force garbage collection

    # 3. Generate and save each split sequentially
    generate_and_save_split_batched(num_train_normal, num_train_cascade, num_train_stressed, train_dir, "TRAIN", args.start_batch)
    generate_and_save_split_batched(num_val_normal, num_val_cascade, num_val_stressed, val_dir, "VALIDATION", args.start_batch)
    generate_and_save_split_batched(num_test_normal, num_test_cascade, num_test_stressed, test_dir, "TEST", args.start_batch)
    
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
