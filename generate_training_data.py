"""
Cascade Failure Training Data Generator
Generates synthetic power grid scenarios with cascade failures for model training.

Author: Kraftgene AI Inc.
Date: October 2025
"""

import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime, timedelta
import random


class PowerGridSimulator:
    """Simulates power grid scenarios with cascade failures."""
    
    def __init__(self, num_nodes: int = 118, seed: int = 42):
        """
        Initialize power grid simulator.
        
        Args:
            num_nodes: Number of nodes in the grid (default: IEEE 118-bus system)
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate base grid topology
        self.adjacency_matrix = self._generate_grid_topology()
        self.edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
        
        # Grid parameters
        self.base_voltage = 345.0  # kV
        self.base_power = 100.0    # MVA
        
    def _generate_grid_topology(self) -> np.ndarray:
        """Generate realistic power grid topology."""
        # Create a sparse adjacency matrix (typical grid connectivity: 2-4 connections per node)
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        # Create backbone transmission lines (ring topology)
        for i in range(self.num_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        adj[self.num_nodes - 1, 0] = 1
        adj[0, self.num_nodes - 1] = 1
        
        # Add radial connections
        num_radial = int(self.num_nodes * 0.5)
        for _ in range(num_radial):
            i = np.random.randint(0, self.num_nodes)
            j = np.random.randint(0, self.num_nodes)
            if i != j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        """Convert adjacency matrix to edge_index format."""
        edges = np.where(adj > 0)
        edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
        return edge_index
    
    def generate_normal_scenario(self) -> Dict:
        """Generate a normal operating scenario (no cascade)."""
        # Node features: [voltage, angle, P_gen, Q_gen, P_load, Q_load, ...]
        node_features = np.zeros((self.num_nodes, 45))
        
        # Voltage magnitudes (0.95 - 1.05 pu)
        node_features[:, 0] = np.random.uniform(0.95, 1.05, self.num_nodes)
        
        # Voltage angles (-30 to 30 degrees)
        node_features[:, 1] = np.random.uniform(-30, 30, self.num_nodes)
        
        # Active power generation (0 - 500 MW)
        num_generators = int(self.num_nodes * 0.3)
        gen_indices = np.random.choice(self.num_nodes, num_generators, replace=False)
        node_features[gen_indices, 2] = np.random.uniform(50, 500, num_generators)
        
        # Reactive power generation
        node_features[gen_indices, 3] = np.random.uniform(-50, 200, num_generators)
        
        # Active power load (50 - 300 MW)
        node_features[:, 4] = np.random.uniform(50, 300, self.num_nodes)
        
        # Reactive power load
        node_features[:, 5] = np.random.uniform(10, 100, self.num_nodes)
        
        # Equipment age (0 - 40 years)
        node_features[:, 6] = np.random.uniform(0, 40, self.num_nodes)
        
        # Equipment condition (0.5 - 1.0, where 1.0 is perfect)
        node_features[:, 7] = np.random.uniform(0.7, 1.0, self.num_nodes)
        
        # Temperature (°C)
        node_features[:, 8] = np.random.uniform(15, 35, self.num_nodes)
        
        # Wind speed (m/s)
        node_features[:, 9] = np.random.uniform(0, 15, self.num_nodes)
        
        # N-1 violation indicator (0 or 1)
        node_features[:, 10] = np.random.binomial(1, 0.1, self.num_nodes)
        
        # Voltage stability index (0 - 1, where 1 is unstable)
        node_features[:, 11] = np.random.uniform(0.1, 0.5, self.num_nodes)
        
        # Line loading (0 - 0.9 for normal operation)
        node_features[:, 12] = np.random.uniform(0.3, 0.9, self.num_nodes)
        
        # Fill remaining features with noise
        node_features[:, 13:] = np.random.randn(self.num_nodes, 32) * 0.1
        
        # Edge features: [impedance, capacity, loading, temperature, ...]
        num_edges = self.edge_index.shape[1]
        edge_features = np.zeros((num_edges, 28))
        
        # Line impedance (0.01 - 0.5 pu)
        edge_features[:, 0] = np.random.uniform(0.01, 0.5, num_edges)
        
        # Line capacity (100 - 1000 MW)
        edge_features[:, 1] = np.random.uniform(100, 1000, num_edges)
        
        # Line loading (0.3 - 0.9)
        edge_features[:, 2] = np.random.uniform(0.3, 0.9, num_edges)
        
        # Line temperature (°C)
        edge_features[:, 3] = np.random.uniform(20, 60, num_edges)
        
        # Fill remaining features
        edge_features[:, 4:] = np.random.randn(num_edges, 24) * 0.1
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'cascade_label': 0,
            'failed_nodes': [],
            'failure_sequence': [],
            'time_to_cascade': -1
        }
    
    def generate_cascade_scenario(self) -> Dict:
        """Generate a cascade failure scenario."""
        # Start with normal scenario
        scenario = self.generate_normal_scenario()
        
        # Introduce stress conditions
        stress_level = np.random.uniform(0.5, 1.0)
        
        # Increase line loading
        scenario['node_features'][:, 12] *= (1 + stress_level * 0.3)
        scenario['edge_features'][:, 2] *= (1 + stress_level * 0.3)
        
        # Decrease voltage stability margin
        scenario['node_features'][:, 11] *= (1 + stress_level * 0.5)
        
        # Increase N-1 violations
        num_violations = int(self.num_nodes * stress_level * 0.3)
        violation_indices = np.random.choice(self.num_nodes, num_violations, replace=False)
        scenario['node_features'][violation_indices, 10] = 1
        
        # Simulate cascade progression
        num_failures = np.random.randint(4, 20)  # 4-20 component failures
        initial_failure = np.random.randint(0, self.num_nodes)
        
        failed_nodes = [initial_failure]
        failure_sequence = [initial_failure]
        
        # Propagate failures through network
        for _ in range(num_failures - 1):
            # Find neighbors of failed nodes
            candidates = []
            for failed_node in failed_nodes:
                neighbors = np.where(self.adjacency_matrix[failed_node] > 0)[0]
                candidates.extend([n for n in neighbors if n not in failed_nodes])
            
            if not candidates:
                break
            
            # Select next failure based on proximity and loading
            next_failure = random.choice(candidates)
            failed_nodes.append(next_failure)
            failure_sequence.append(next_failure)
        
        # Time to cascade (5-45 minutes)
        time_to_cascade = np.random.uniform(5, 45)
        
        scenario['cascade_label'] = 1
        scenario['failed_nodes'] = failed_nodes
        scenario['failure_sequence'] = failure_sequence
        scenario['time_to_cascade'] = time_to_cascade
        
        return scenario
    
    def generate_temporal_sequence(self, scenario: Dict, sequence_length: int = 60) -> List[Dict]:
        """Generate temporal sequence of grid states leading to cascade."""
        sequence = []
        
        for t in range(sequence_length):
            # Create time-varying scenario
            time_scenario = scenario.copy()
            time_scenario['node_features'] = scenario['node_features'].copy()
            time_scenario['edge_features'] = scenario['edge_features'].copy()
            
            # Add temporal variations
            time_factor = t / sequence_length
            
            # Gradually increase stress if cascade scenario
            if scenario['cascade_label'] == 1:
                stress_increase = time_factor * 0.2
                time_scenario['node_features'][:, 12] *= (1 + stress_increase)
                time_scenario['edge_features'][:, 2] *= (1 + stress_increase)
            
            # Add noise
            time_scenario['node_features'] += np.random.randn(*time_scenario['node_features'].shape) * 0.01
            time_scenario['edge_features'] += np.random.randn(*time_scenario['edge_features'].shape) * 0.01
            
            time_scenario['timestep'] = t
            sequence.append(time_scenario)
        
        return sequence


def generate_dataset(
    num_normal: int = 12000,
    num_cascade: int = 1200,
    num_nodes: int = 118,
    sequence_length: int = 60,
    output_dir: str = "data"
) -> None:
    """
    Generate complete training dataset.
    
    Args:
        num_normal: Number of normal scenarios
        num_cascade: Number of cascade scenarios
        num_nodes: Number of nodes in grid
        sequence_length: Length of temporal sequences
        output_dir: Output directory for data files
    """
    print("=" * 80)
    print("POWER GRID CASCADE FAILURE DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Normal scenarios: {num_normal}")
    print(f"  Cascade scenarios: {num_cascade}")
    print(f"  Grid size: {num_nodes} nodes")
    print(f"  Sequence length: {sequence_length} timesteps")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = PowerGridSimulator(num_nodes=num_nodes)
    
    # Generate datasets
    datasets = {
        'train': {'normal': int(num_normal * 0.7), 'cascade': int(num_cascade * 0.7)},
        'val': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)},
        'test': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)}
    }
    
    for split_name, split_config in datasets.items():
        print(f"\nGenerating {split_name} set...")
        print(f"  Normal: {split_config['normal']}, Cascade: {split_config['cascade']}")
        
        split_data = []
        
        # Generate normal scenarios
        for i in range(split_config['normal']):
            if (i + 1) % 1000 == 0:
                print(f"    Normal scenarios: {i + 1}/{split_config['normal']}")
            
            scenario = simulator.generate_normal_scenario()
            sequence = simulator.generate_temporal_sequence(scenario, sequence_length)
            split_data.append({
                'sequence': sequence,
                'metadata': {
                    'cascade': False,
                    'num_nodes': num_nodes,
                    'sequence_length': sequence_length
                }
            })
        
        # Generate cascade scenarios
        for i in range(split_config['cascade']):
            if (i + 1) % 100 == 0:
                print(f"    Cascade scenarios: {i + 1}/{split_config['cascade']}")
            
            scenario = simulator.generate_cascade_scenario()
            sequence = simulator.generate_temporal_sequence(scenario, sequence_length)
            split_data.append({
                'sequence': sequence,
                'metadata': {
                    'cascade': True,
                    'failed_nodes': scenario['failed_nodes'],
                    'failure_sequence': scenario['failure_sequence'],
                    'time_to_cascade': scenario['time_to_cascade'],
                    'num_nodes': num_nodes,
                    'sequence_length': sequence_length
                }
            })
        
        # Shuffle data
        random.shuffle(split_data)
        
        # Save to file
        output_file = output_path / f"{split_name}_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"  ✓ Saved to {output_file}")
        print(f"  Total samples: {len(split_data)}")
    
    # Save grid topology
    topology_file = output_path / "grid_topology.pkl"
    with open(topology_file, 'wb') as f:
        pickle.dump({
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index,
            'num_nodes': num_nodes
        }, f)
    print(f"\n✓ Saved grid topology to {topology_file}")
    
    # Save metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'num_normal': num_normal,
        'num_cascade': num_cascade,
        'num_nodes': num_nodes,
        'sequence_length': sequence_length,
        'splits': {
            'train': {'normal': datasets['train']['normal'], 'cascade': datasets['train']['cascade']},
            'val': {'normal': datasets['val']['normal'], 'cascade': datasets['val']['cascade']},
            'test': {'normal': datasets['test']['normal'], 'cascade': datasets['test']['cascade']}
        }
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nDataset summary:")
    print(f"  Total scenarios: {num_normal + num_cascade}")
    print(f"  Train: {datasets['train']['normal'] + datasets['train']['cascade']}")
    print(f"  Val: {datasets['val']['normal'] + datasets['val']['cascade']}")
    print(f"  Test: {datasets['test']['normal'] + datasets['test']['cascade']}")
    print(f"\nFiles created in '{output_dir}':")
    print(f"  - train_data.pkl")
    print(f"  - val_data.pkl")
    print(f"  - test_data.pkl")
    print(f"  - grid_topology.pkl")
    print(f"  - metadata.json")
    print()


if __name__ == "__main__":
    # Generate dataset with default parameters
    # For quick testing, use smaller numbers:
    # generate_dataset(num_normal=1000, num_cascade=100, sequence_length=30)
    
    # For full dataset (as in paper):
    generate_dataset(
        num_normal=1000,
        num_cascade=100,
        num_nodes=118,
        sequence_length=30,
        output_dir="data"
    )
