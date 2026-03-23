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

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Settings


class GridTopologyGenerator:
    """
    Generates realistic power grid topologies.
    
    Creates meshed transmission networks with:
    - Multiple zones (regional areas)
    - Intra-zone connections (meshed)
    - Inter-zone tie lines (critical connections)
    - Guaranteed connectivity
    """
    
    def __init__(self, num_nodes: int = Settings.Topology.DEFAULT_NUM_NODES, seed: int = Settings.Scenario.DEFAULT_SEED):
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

    @classmethod
    def load_topology(
        cls,
        topology_file: str = Settings.Dataset.DEFAULT_TOPOLOGY_FILE,
    ) -> Optional[Dict]:
        """
        Load a previously saved grid topology from disk.

        Re-uses an existing topology pickle so that all training runs share
        the same grid structure.  Falls back to
        ``Settings.Dataset.DEFAULT_TOPOLOGY_FILE`` when no path is given.

        Args:
            topology_file: Path to the topology ``.pkl`` file.  Defaults to
                ``Settings.Dataset.DEFAULT_TOPOLOGY_FILE``
                (``"data/grid_topology.pkl"``).

        Returns:
            Topology dictionary with keys ``adjacency_matrix``, ``edge_index``,
            ``positions``, ``num_nodes``, and ``num_edges``, or ``None`` if the
            file does not exist.

        Example::

            # Load from the default location
            topo = GridTopologyGenerator.load_topology()

            # Load from a custom path
            topo = GridTopologyGenerator.load_topology("runs/exp1/topology.pkl")

            if topo is None:
                topo = GridTopologyGenerator(num_nodes=118).generate_topology()
        """
        path = Path(topology_file)
        if not path.exists():
            print(f"  [GridTopologyGenerator.load_topology] File not found: {path}")
            return None

        with open(path, "rb") as f:
            topology = pickle.load(f)

        # Derive num_nodes / num_edges when not stored explicitly so that
        # topology files saved before these keys existed still load cleanly.
        if "num_nodes" not in topology:
            topology["num_nodes"] = int(topology["adjacency_matrix"].shape[0])
        if "num_edges" not in topology:
            ei = topology["edge_index"]
            topology["num_edges"] = int(ei.shape[1] if hasattr(ei, "shape") else len(ei[0]))

        print(
            f"  Loaded topology: {topology['num_nodes']} nodes, "
            f"{topology['num_edges']} edges \u2190 {path}"
        )
        return topology

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
        
        # Create zones (like regional transmission areas)
        num_zones = Settings.Topology.NUM_ZONES
        nodes_per_zone = self.num_nodes // num_zones

        # Intra-zone connections (meshed within zone)
        for zone in range(num_zones):
            start = zone * nodes_per_zone
            end = start + nodes_per_zone if zone < num_zones - 1 else self.num_nodes

            for i in range(start, end):
                num_connections = np.random.randint(
                    Settings.Topology.INTRA_ZONE_CONN_MIN,
                    Settings.Topology.INTRA_ZONE_CONN_MAX,
                )
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
            for _ in range(np.random.randint(Settings.Topology.TIE_LINES_MIN, Settings.Topology.TIE_LINES_MAX)):
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
        num_zones = Settings.Topology.NUM_ZONES
        nodes_per_zone = self.num_nodes // num_zones

        for zone_idx, (cx, cy) in enumerate(Settings.Topology.ZONE_CENTERS):
            start = zone_idx * nodes_per_zone
            end = start + nodes_per_zone if zone_idx < num_zones - 1 else self.num_nodes
            num_in_zone = end - start

            # Nodes clustered around zone center
            zone_positions = np.random.randn(num_in_zone, 2) * Settings.Topology.ZONE_SPREAD_STD + np.array([cx, cy])
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
    
    def __init__(self, num_nodes: int, seed: int = Settings.Scenario.DEFAULT_SEED):
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
        
        # Other generators
        num_generators = max(int(self.num_nodes * Settings.Node.GENERATOR_FRACTION) - 1, Settings.Node.GENERATOR_MIN_COUNT)
        possible_gen_indices = list(range(1, self.num_nodes))
        gen_indices = np.random.choice(
            possible_gen_indices,
            num_generators,
            replace=False
        )
        node_types[gen_indices] = 1
        
        all_gen_indices = np.concatenate([[0], gen_indices])
        
        # Substations
        num_substations = int(self.num_nodes * Settings.Node.SUBSTATION_FRACTION)
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
                base_load[i] = np.random.uniform(Settings.Node.GEN_LOAD_MIN, Settings.Node.GEN_LOAD_MAX)
            elif node_types[i] == 2:  # Substation
                base_load[i] = np.random.uniform(Settings.Node.SUB_LOAD_MIN, Settings.Node.SUB_LOAD_MAX)
            else:  # Load bus
                base_load[i] = np.random.uniform(Settings.Node.LOAD_LOAD_MIN, Settings.Node.LOAD_LOAD_MAX)
        
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
        target_total_capacity = total_load * Settings.Node.TARGET_CAPACITY_FACTOR

        # Slack bus gets a fraction of total capacity
        slack_capacity_ratio = np.random.uniform(Settings.Node.SLACK_CAPACITY_MIN, Settings.Node.SLACK_CAPACITY_MAX)
        slack_capacity = target_total_capacity * slack_capacity_ratio
        
        # Remaining capacity for other generators
        remaining_capacity = target_total_capacity - slack_capacity
        num_other_gens = len(gen_indices) - 1  # Exclude slack
        
        gen_capacity = np.zeros(self.num_nodes)
        gen_capacity[0] = slack_capacity
        
        if num_other_gens > 0:
            # Dirichlet distribution for realistic variation
            alpha = np.ones(num_other_gens) * Settings.Node.DIRICHLET_ALPHA
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

        if reserve_margin < Settings.Node.MIN_RESERVE_MARGIN_PCT:
            print(f"  [WARNING] Low reserve margin ({reserve_margin:.1f}%), "
                  f"increasing capacity...")
            gen_capacity *= Settings.Node.CAPACITY_BOOST_FACTOR
            reserve_margin = (total_capacity - total_load) / total_load * 100
            print(f"    Adjusted reserve margin: {reserve_margin:.1f}%")

        return gen_capacity
    
    def _initialize_failure_thresholds(self) -> Dict:
        """
        Initialize failure thresholds for all nodes.
        
        Returns:
            Dictionary with threshold arrays
        """
        loading_failure_threshold = np.random.uniform(
            Settings.Node.LOADING_FAILURE_MIN, Settings.Node.LOADING_FAILURE_MAX, self.num_nodes)
        loading_damage_threshold = loading_failure_threshold - np.random.uniform(
            abs(Settings.Node.LOADING_DAMAGE_OFFSET_MIN), abs(Settings.Node.LOADING_DAMAGE_OFFSET_MAX))

        voltage_failure_threshold = np.random.uniform(
            Settings.Node.VOLTAGE_FAILURE_MIN, Settings.Node.VOLTAGE_FAILURE_MAX, self.num_nodes)
        voltage_damage_threshold = voltage_failure_threshold + np.random.uniform(
            Settings.Node.VOLTAGE_DAMAGE_OFFSET_MIN, Settings.Node.VOLTAGE_DAMAGE_OFFSET_MAX)

        temperature_failure_threshold = np.random.uniform(
            Settings.Node.TEMP_FAILURE_MIN_C, Settings.Node.TEMP_FAILURE_MAX_C, self.num_nodes)
        temperature_damage_threshold = temperature_failure_threshold - np.random.uniform(
            abs(Settings.Node.TEMP_DAMAGE_OFFSET_MIN_C), abs(Settings.Node.TEMP_DAMAGE_OFFSET_MAX_C))

        frequency_failure_threshold = np.random.uniform(
            Settings.Node.FREQ_FAILURE_MIN_HZ, Settings.Node.FREQ_FAILURE_MAX_HZ, self.num_nodes)
        frequency_damage_threshold = frequency_failure_threshold + np.random.uniform(
            Settings.Node.FREQ_DAMAGE_OFFSET_MIN, Settings.Node.FREQ_DAMAGE_OFFSET_MAX)
        
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
        equipment_age = np.random.uniform(
            Settings.Node.EQUIPMENT_AGE_MIN_YR, Settings.Node.EQUIPMENT_AGE_MAX_YR, self.num_nodes)
        equipment_condition = np.clip(
            1.0 - Settings.Node.CONDITION_AGE_COEFF * equipment_age
            + np.random.normal(0, Settings.Node.CONDITION_NOISE_STD, self.num_nodes),
            Settings.Node.CONDITION_MIN,
            Settings.Node.CONDITION_MAX,
        )

        return {
            'equipment_age': equipment_age,
            'equipment_condition': equipment_condition,
            'thermal_time_constant': np.random.uniform(
                Settings.Node.THERMAL_TAU_MIN, Settings.Node.THERMAL_TAU_MAX, self.num_nodes),
            'thermal_capacity': np.random.uniform(
                Settings.Node.THERMAL_CAP_MIN, Settings.Node.THERMAL_CAP_MAX, self.num_nodes),
            'cooling_effectiveness': np.random.uniform(
                Settings.Node.COOLING_EFF_MIN, Settings.Node.COOLING_EFF_MAX, self.num_nodes),
            'equipment_temperatures': np.full(self.num_nodes, Settings.Thermal.AMBIENT_TEMP_C),
        }
