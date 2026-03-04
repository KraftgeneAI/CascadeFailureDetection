"""
Topology Module
==============
Grid topology generation and initialization.

This module handles:
- Realistic grid topology generation
- Node property initialization
- Edge feature initialization
- Connectivity verification
"""

import numpy as np
import torch
from typing import Tuple, Dict


class GridTopologyGenerator:
    """
    Generates realistic power grid topologies.
    
    Creates meshed transmission networks with:
    - Multiple zones (regional areas)
    - Intra-zone connections (meshed)
    - Inter-zone tie lines (critical connections)
    - Guaranteed connectivity
    """
    
    def __init__(self, num_nodes: int = 118, seed: int = 42):
        """
        Initialize topology generator.
        
        Args:
            num_nodes: Number of nodes in the grid
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_topology(self) -> Dict:
        """
        Generate complete grid topology.
        
        Returns:
            Dictionary containing:
            - adjacency_matrix: Node connectivity
            - edge_index: Edge list format
            - positions: Geographic positions
            - num_edges: Number of edges
        """
        # Generate adjacency matrix
        adj = self._generate_realistic_topology()
        
        # Ensure connectivity
        adj = self._check_and_fix_connectivity(adj)
        
        # Convert to edge index
        edge_index = self._adjacency_to_edge_index(adj)
        
        # Generate positions
        positions = self._generate_geographic_positions()
        
        return {
            'adjacency_matrix': adj,
            'edge_index': edge_index,
            'positions': positions,
            'num_nodes': self.num_nodes,
            'num_edges': edge_index.shape[1]
        }
    
    def _generate_realistic_topology(self) -> np.ndarray:
        """
        Generate realistic meshed grid topology.
        
        Creates a multi-zone topology with:
        - 4 regional zones
        - Meshed connections within zones
        - Tie lines between zones
        
        Returns:
            Adjacency matrix
        """
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
                j = np.random.randint(
                    next_zone_start,
                    min(next_zone_start + nodes_per_zone, self.num_nodes)
                )
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _check_and_fix_connectivity(self, adj: np.ndarray) -> np.ndarray:
        """
        Ensure the graph is fully connected by adding tie-lines.
        
        This prevents power flow solver failures on islanded graphs.
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Connected adjacency matrix
        """
        num_nodes = adj.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        q = [0]  # Start BFS from slack bus 0
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
            return adj
        
        print(f"  [WARNING] Grid topology not connected. "
              f"Found {len(component)} nodes in main component.")
        print("  Adding tie lines to connect islands...")
        
        # Find all islands
        all_nodes = set(range(num_nodes))
        main_component_set = set(component)
        island_nodes = list(all_nodes - main_component_set)
        
        while island_nodes:
            # BFS from island node to find its component
            island_q = [island_nodes[0]]
            visited[island_nodes[0]] = True
            current_island_component = [island_nodes[0]]
            
            head = 0
            while head < len(island_q):
                u = island_q[head]
                head += 1
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        island_q.append(v)
                        current_island_component.append(v)
            
            # Connect island to main component
            island_node = current_island_component[0]
            main_node = component[np.random.randint(len(component))]
            
            adj[island_node, main_node] = 1
            adj[main_node, island_node] = 1
            print(f"    Added tie line: Node {island_node} (island) <-> "
                  f"Node {main_node} (main)")
            
            # Remove connected nodes from island list
            island_nodes = [
                n for n in island_nodes 
                if n not in current_island_component
            ]
        
        print("  Grid connectivity fixed.")
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        """
        Convert adjacency matrix to edge index format.
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _generate_geographic_positions(self) -> np.ndarray:
        """
        Generate realistic geographic positions.
        
        Positions are used for environmental data correlation.
        Nodes are clustered in zones to simulate regional areas.
        
        Returns:
            Position array [num_nodes, 2]
        """
        positions = []
        num_zones = 4
        nodes_per_zone = self.num_nodes // num_zones
        
        # Zone centers (geographic regions)
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


class NodePropertyInitializer:
    """
    Initializes node properties and failure thresholds.
    
    Handles:
    - Node type assignment (generator, load, substation)
    - Generation capacity sizing
    - Load distribution
    - Failure thresholds
    """
    
    def __init__(self, num_nodes: int, seed: int = 42):
        """
        Initialize property initializer.
        
        Args:
            num_nodes: Number of nodes
            seed: Random seed
        """
        self.num_nodes = num_nodes
        np.random.seed(seed)
    
    def initialize_properties(self) -> Dict:
        """
        Initialize all node properties.
        
        Returns:
            Dictionary with node properties
        """
        # Assign node types
        node_types, gen_indices = self._assign_node_types()
        
        # Calculate load
        base_load = self._calculate_base_load(node_types)
        
        # Size generators for convergence
        gen_capacity = self._size_generators(base_load, gen_indices)
        
        # Initialize failure thresholds
        thresholds = self._initialize_failure_thresholds()
        
        # Initialize thermal properties
        thermal = self._initialize_thermal_properties()
        
        print(f"  Defined failure thresholds (Damage / Failure):")
        print(f"    Loading: {thresholds['loading_damage_threshold'].mean():.2f} / {thresholds['loading_failure_threshold'].mean():.2f}")
        print(f"    Voltage: {thresholds['voltage_damage_threshold'].mean():.2f} / {thresholds['voltage_failure_threshold'].mean():.2f}")
        print(f"    Temperature: {thresholds['temperature_damage_threshold'].mean():.1f}°C / {thresholds['temperature_failure_threshold'].mean():.1f}°C")
        print(f"    Frequency: {thresholds['frequency_damage_threshold'].mean():.2f} Hz / {thresholds['frequency_failure_threshold'].mean():.2f} Hz")

        return {
            'node_types': node_types,
            'gen_capacity': gen_capacity,
            'base_load': base_load,
            **thresholds,
            **thermal
        }
    
    def _assign_node_types(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign node types: 0=Load, 1=Generator, 2=Substation.
        
        Returns:
            Tuple of (node_types, generator_indices)
        """
        node_types = np.zeros(self.num_nodes, dtype=int)
        
        # Node 0 is slack bus (large generator)
        node_types[0] = 1
        
        # Other generators (20-25% of nodes)
        num_generators = max(int(self.num_nodes * 0.22) - 1, 5)
        possible_gen_indices = list(range(1, self.num_nodes))
        gen_indices = np.random.choice(
            possible_gen_indices,
            num_generators,
            replace=False
        )
        node_types[gen_indices] = 1
        
        all_gen_indices = np.concatenate([[0], gen_indices])
        
        # Substations (10% of nodes)
        num_substations = int(self.num_nodes * 0.10)
        possible_sub_indices = [
            i for i in range(1, self.num_nodes) 
            if i not in all_gen_indices
        ]
        sub_indices = np.random.choice(
            possible_sub_indices,
            num_substations,
            replace=False
        )
        node_types[sub_indices] = 2
        
        return node_types, all_gen_indices
    
    def _calculate_base_load(self, node_types: np.ndarray) -> np.ndarray:
        """
        Calculate base load for each node.
        
        Args:
            node_types: Node type array
            
        Returns:
            Base load array
        """
        base_load = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            if node_types[i] == 1:  # Generator
                base_load[i] = np.random.uniform(5, 20)
            elif node_types[i] == 2:  # Substation
                base_load[i] = np.random.uniform(50, 150)
            else:  # Load bus
                base_load[i] = np.random.uniform(30, 200)
        
        return base_load
    
    def _size_generators(
        self,
        base_load: np.ndarray,
        gen_indices: np.ndarray
    ) -> np.ndarray:
        """
        Size generators for convergence (150% of load).
        
        Args:
            base_load: Base load array
            gen_indices: Generator node indices
            
        Returns:
            Generator capacity array
        """
        total_load = base_load.sum()
        target_total_capacity = total_load * 1.50
        
        # Slack bus gets 30-40% of total capacity
        slack_capacity_ratio = np.random.uniform(0.30, 0.40)
        slack_capacity = target_total_capacity * slack_capacity_ratio
        
        # Remaining capacity for other generators
        remaining_capacity = target_total_capacity - slack_capacity
        num_other_gens = len(gen_indices) - 1  # Exclude slack
        
        gen_capacity = np.zeros(self.num_nodes)
        gen_capacity[0] = slack_capacity
        
        if num_other_gens > 0:
            # Dirichlet distribution for realistic variation
            alpha = np.ones(num_other_gens) * 2.0
            weights = np.random.dirichlet(alpha)
            
            for i, idx in enumerate(gen_indices[1:]):
                gen_capacity[idx] = remaining_capacity * weights[i]
        
        total_capacity = gen_capacity.sum()

        # Verify reserve margin
        reserve_margin = (gen_capacity.sum() - total_load) / total_load * 100
        
        print(f"  Convergence-aware sizing:")
        print(f"    Total load: {total_load:.1f} MW")
        print(f"    Total gen capacity: {total_capacity:.1f} MW")
        print(f"    Reserve margin: {reserve_margin:.1f}%")
        print(f"    Slack bus capacity: {slack_capacity:.1f} MW ({slack_capacity/total_capacity*100:.1f}%)")
        print(f"    Number of generators: {len(gen_indices)}")

        if reserve_margin < 20:
            print(f"  [WARNING] Low reserve margin ({reserve_margin:.1f}%), "
                  f"increasing capacity...")
            gen_capacity *= 1.3
            reserve_margin = (total_capacity - total_load) / total_load * 100
            print(f"    Adjusted reserve margin: {reserve_margin:.1f}%")

        return gen_capacity
    
    def _initialize_failure_thresholds(self) -> Dict:
        """
        Initialize failure thresholds for all nodes.
        
        Returns:
            Dictionary with threshold arrays
        """
        loading_failure_threshold = np.random.uniform(1.05, 1.15, self.num_nodes)
        loading_damage_threshold = loading_failure_threshold - np.random.uniform(0.05, 0.1)
        
        # Voltage threshold: node fails if voltage < threshold
        voltage_failure_threshold = np.random.uniform(0.88, 0.92, self.num_nodes)
        voltage_damage_threshold = voltage_failure_threshold - np.random.uniform(0.03, 0.05)
        
        # Temperature threshold: node fails if temperature > threshold
        temperature_failure_threshold = np.random.uniform(85, 95, self.num_nodes)
        temperature_damage_threshold = temperature_failure_threshold - np.random.uniform(10, 15)
        
        # Frequency threshold: node fails if frequency < threshold
        frequency_failure_threshold = np.random.uniform(58.5, 59.2, self.num_nodes)
        frequency_damage_threshold = frequency_failure_threshold + np.random.uniform(0.3, 0.5)
        
        return {
            'loading_failure_threshold': loading_failure_threshold,
            'loading_damage_threshold': loading_damage_threshold,
            'voltage_failure_threshold': voltage_failure_threshold,
            'voltage_damage_threshold': voltage_damage_threshold,
            'temperature_failure_threshold': temperature_failure_threshold,
            'temperature_damage_threshold': temperature_damage_threshold,
            'frequency_failure_threshold': frequency_failure_threshold,
            'frequency_damage_threshold': frequency_damage_threshold,
        }
    
    def _initialize_thermal_properties(self) -> Dict:
        """
        Initialize thermal properties for equipment.
        
        Returns:
            Dictionary with thermal property arrays
        """
        equipment_age = np.random.uniform(0, 40, self.num_nodes)
        equipment_condition = np.clip(
            1.0 - 0.008 * equipment_age + np.random.normal(0, 0.05, self.num_nodes),
            0.6, 1.0
        )
        
        return {
            'equipment_age': equipment_age,
            'equipment_condition': equipment_condition,
            'thermal_time_constant': np.random.uniform(10, 30, self.num_nodes),
            'thermal_capacity': np.random.uniform(0.8, 1.2, self.num_nodes),
            'cooling_effectiveness': np.random.uniform(0.7, 1.0, self.num_nodes),
            'equipment_temperatures': np.full(self.num_nodes, 25.0)
        }
