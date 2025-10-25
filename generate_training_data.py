"""
Optimized Cascade Failure Training Data Generator
==================================================
Memory-efficient generator with realistic physics-based cascade scenarios.

Key Improvements:
1. Memory-efficient batch processing with streaming to disk
2. Realistic DC power flow approximation
3. Physics-based cascade propagation
4. Correlated features and temporal dynamics
5. Realistic environmental factors

Author: Kraftgene AI Inc. (Optimized)
Date: October 2025
"""

import numpy as np
import torch
import pickle
import h5py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import gc
import psutil
import warnings


class MemoryMonitor:
    """Monitor and report memory usage."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_memory_threshold(threshold_mb=8000):
        """Check if memory usage exceeds threshold."""
        current = MemoryMonitor.get_memory_usage()
        if current > threshold_mb:
            warnings.warn(f"High memory usage: {current:.1f} MB")
            return True
        return False


class RealisticPowerGridSimulator:
    """
    Physics-based power grid simulator with realistic cascade dynamics.
    """
    
    def __init__(self, num_nodes: int = 118, seed: int = 42):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate realistic grid topology
        self.adjacency_matrix = self._generate_realistic_topology()
        self.edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
        self.num_edges = self.edge_index.shape[1]
        
        # Physical parameters
        self.base_voltage = 345.0  # kV
        self.base_power = 100.0    # MVA
        self.base_impedance = self.base_voltage ** 2 / self.base_power
        
        # Generate static grid properties
        self._initialize_grid_properties()
    
    def _generate_realistic_topology(self) -> np.ndarray:
        """Generate realistic power grid topology with zones."""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        
        # Create zones (regional interconnections)
        num_zones = 4
        nodes_per_zone = self.num_nodes // num_zones
        
        for zone in range(num_zones):
            start = zone * nodes_per_zone
            end = start + nodes_per_zone if zone < num_zones - 1 else self.num_nodes
            
            # Create meshed network within zone
            for i in range(start, end):
                # Connect to 2-4 neighbors within zone
                num_connections = np.random.randint(2, 5)
                neighbors = np.random.choice(
                    range(start, end), 
                    size=min(num_connections, end - start - 1), 
                    replace=False
                )
                for j in neighbors:
                    if i != j:
                        adj[i, j] = 1
                        adj[j, i] = 1
        
        # Inter-zone connections (tie lines)
        for zone in range(num_zones - 1):
            zone_end = (zone + 1) * nodes_per_zone
            next_zone_start = zone_end
            # 2-3 tie lines between zones
            for _ in range(np.random.randint(2, 4)):
                i = np.random.randint(zone * nodes_per_zone, zone_end)
                j = np.random.randint(next_zone_start, 
                                     min(next_zone_start + nodes_per_zone, self.num_nodes))
                adj[i, j] = 1
                adj[j, i] = 1
        
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> torch.Tensor:
        """Convert adjacency matrix to edge_index."""
        edges = np.where(adj > 0)
        return torch.tensor(np.vstack(edges), dtype=torch.long)
    
    def _initialize_grid_properties(self):
        """Initialize static grid properties."""
        # Node types: 0=load, 1=generator, 2=both
        self.node_types = np.zeros(self.num_nodes)
        num_generators = int(self.num_nodes * 0.25)
        gen_indices = np.random.choice(self.num_nodes, num_generators, replace=False)
        self.node_types[gen_indices] = 1
        
        # Generator capacities (50-500 MW)
        self.gen_capacity = np.zeros(self.num_nodes)
        self.gen_capacity[gen_indices] = np.random.uniform(50, 500, num_generators)
        
        # Base load (30-200 MW per node)
        self.base_load = np.random.uniform(30, 200, self.num_nodes)
        
        # Line parameters
        src, dst = self.edge_index
        self.line_length = np.random.uniform(10, 150, self.num_edges)  # km
        self.line_resistance = 0.05 * self.line_length / self.base_impedance  # pu
        self.line_reactance = 0.3 * self.line_length / self.base_impedance  # pu
        self.line_susceptance = 1.0 / (self.line_reactance + 1e-6)
        
        # Thermal limits (100-1000 MW)
        self.thermal_limits = np.random.uniform(100, 1000, self.num_edges)
        
        # Equipment age and condition
        self.equipment_age = np.random.uniform(0, 40, self.num_nodes)
        self.equipment_condition = np.clip(
            1.0 - 0.01 * self.equipment_age + np.random.normal(0, 0.1, self.num_nodes),
            0.5, 1.0
        )
    
    def _compute_dc_power_flow(
        self, 
        generation: np.ndarray, 
        load: np.ndarray,
        failed_lines: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DC power flow approximation.
        
        Returns:
            voltages, angles, line_flows
        """
        # Net injection
        P_net = generation - load
        
        # Build admittance matrix (excluding failed lines)
        B = np.zeros((self.num_nodes, self.num_nodes))
        src, dst = self.edge_index
        
        for i in range(self.num_edges):
            if failed_lines is not None and i in failed_lines:
                continue
            s, d = src[i].item(), dst[i].item()
            b = self.line_susceptance[i]
            B[s, s] += b
            B[d, d] += b
            B[s, d] -= b
            B[d, s] -= b
        
        # Solve for angles (DC power flow)
        # Set reference bus (bus 0)
        B_reduced = B[1:, 1:]
        P_reduced = P_net[1:]
        
        try:
            theta_reduced = np.linalg.solve(B_reduced, P_reduced)
            theta = np.zeros(self.num_nodes)
            theta[1:] = theta_reduced
        except np.linalg.LinAlgError:
            # Singular matrix - grid is islanded
            theta = np.zeros(self.num_nodes)
        
        # Compute line flows
        line_flows = np.zeros(self.num_edges)
        for i in range(self.num_edges):
            if failed_lines is not None and i in failed_lines:
                continue
            s, d = src[i].item(), dst[i].item()
            line_flows[i] = self.line_susceptance[i] * (theta[s] - theta[d])
        
        # Voltages (simplified - assume 1.0 pu with small variations)
        voltages = 1.0 + 0.05 * (P_net / self.base_load.max())
        voltages = np.clip(voltages, 0.9, 1.1)
        
        return voltages, theta, line_flows
    
    def _simulate_cascade_propagation(
        self,
        initial_failure: int,
        stress_level: float
    ) -> Tuple[List[int], List[float]]:
        """
        Simulate realistic cascade propagation based on physics.
        
        Returns:
            failed_nodes, failure_times
        """
        # Increase load to create stress
        load = self.base_load * (1.0 + stress_level * 0.3)
        
        # Dispatch generation to meet load
        total_load = load.sum()
        gen_indices = np.where(self.node_types == 1)[0]
        generation = np.zeros(self.num_nodes)
        
        # Economic dispatch (proportional to capacity)
        total_capacity = self.gen_capacity.sum()
        for idx in gen_indices:
            generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.05
        
        failed_lines = [initial_failure]
        failed_nodes = []
        failure_times = []
        current_time = 0.0
        
        # Propagate cascade
        for iteration in range(20):  # Max 20 iterations
            # Compute power flow with failed lines
            voltages, angles, line_flows = self._compute_dc_power_flow(
                generation, load, failed_lines
            )
            
            # Check for overloads
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            overloaded = np.where(loading_ratios > 1.0)[0]
            
            # Check for voltage violations
            voltage_violations = np.where((voltages < 0.95) | (voltages > 1.05))[0]
            
            if len(overloaded) == 0 and len(voltage_violations) == 0:
                break  # Cascade stopped
            
            # Select next failure (most overloaded line or worst voltage)
            if len(overloaded) > 0:
                next_failure = overloaded[np.argmax(loading_ratios[overloaded])]
                if next_failure not in failed_lines:
                    failed_lines.append(next_failure)
                    # Determine affected nodes
                    src, dst = self.edge_index
                    affected_nodes = [src[next_failure].item(), dst[next_failure].item()]
                    for node in affected_nodes:
                        if node not in failed_nodes:
                            failed_nodes.append(node)
                            # Time to failure (5-45 minutes)
                            current_time += np.random.uniform(2, 8)
                            failure_times.append(current_time)
            
            if len(voltage_violations) > 0:
                for node in voltage_violations[:2]:  # Limit to 2 per iteration
                    if node not in failed_nodes:
                        failed_nodes.append(node)
                        current_time += np.random.uniform(1, 5)
                        failure_times.append(current_time)
        
        return failed_nodes, failure_times
    
    def generate_scenario(
        self, 
        is_cascade: bool,
        sequence_length: int = 60
    ) -> Dict:
        """Generate a single scenario with temporal sequence."""
        # Determine stress level
        if is_cascade:
            stress_level = np.random.uniform(0.6, 1.0)
        else:
            stress_level = np.random.uniform(0.0, 0.4)
        
        # Base operating point
        load_profile = self._generate_load_profile(sequence_length)
        weather_profile = self._generate_weather_profile(sequence_length)
        
        # Initialize scenario
        failed_nodes = []
        failure_times = []
        cascade_start_time = -1
        
        if is_cascade:
            # Determine when cascade starts (70-90% through sequence)
            cascade_start_time = int(sequence_length * np.random.uniform(0.7, 0.9))
            initial_failure = np.random.randint(0, self.num_edges)
            failed_nodes, failure_times = self._simulate_cascade_propagation(
                initial_failure, stress_level
            )
        
        # Generate temporal sequence
        sequence = []
        for t in range(sequence_length):
            # Time-varying load
            load_factor = load_profile[t]
            load = self.base_load * load_factor * (1.0 + stress_level * 0.2)
            
            # Generation dispatch
            total_load = load.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            generation = np.zeros(self.num_nodes)
            total_capacity = self.gen_capacity.sum()
            for idx in gen_indices:
                generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.05
            
            # Determine failed lines at this timestep
            failed_lines_t = []
            if is_cascade and t >= cascade_start_time:
                # Lines fail progressively
                time_since_cascade = (t - cascade_start_time) * 2  # 2 minutes per timestep
                failed_lines_t = [i for i, ft in enumerate(failure_times) if ft <= time_since_cascade]
            
            # Compute power flow
            voltages, angles, line_flows = self._compute_dc_power_flow(
                generation, load, failed_lines_t if failed_lines_t else None
            )
            
            # Build node features [45 features]
            node_features = np.zeros((self.num_nodes, 45))
            node_features[:, 0] = voltages
            node_features[:, 1] = angles * 180 / np.pi  # Convert to degrees
            node_features[:, 2] = generation
            node_features[:, 3] = generation * 0.3  # Reactive power (approximation)
            node_features[:, 4] = load
            node_features[:, 5] = load * 0.2  # Reactive load
            node_features[:, 6] = self.equipment_age
            node_features[:, 7] = self.equipment_condition
            node_features[:, 8] = weather_profile[t, 0]  # Temperature
            node_features[:, 9] = weather_profile[t, 1]  # Wind speed
            
            # N-1 violations
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            src, dst = self.edge_index
            for i in range(self.num_edges):
                s, d = src[i].item(), dst[i].item()
                if loading_ratios[i] > 0.9:
                    node_features[s, 10] = 1.0
                    node_features[d, 10] = 1.0
            
            # Voltage stability index
            node_features[:, 11] = np.clip(1.0 - voltages, 0, 1)
            
            # Line loading (aggregate at nodes)
            for i in range(self.num_edges):
                s, d = src[i].item(), dst[i].item()
                node_features[s, 12] = max(node_features[s, 12], loading_ratios[i])
                node_features[d, 12] = max(node_features[d, 12], loading_ratios[i])
            
            # Add noise to remaining features
            node_features[:, 13:] = np.random.randn(self.num_nodes, 32) * 0.05
            
            # Build edge features [28 features]
            edge_features = np.zeros((self.num_edges, 28))
            edge_features[:, 0] = self.line_reactance
            edge_features[:, 1] = self.thermal_limits
            edge_features[:, 2] = loading_ratios
            edge_features[:, 3] = weather_profile[t, 0] + np.random.randn(self.num_edges) * 2  # Line temp
            edge_features[:, 4:] = np.random.randn(self.num_edges, 24) * 0.05
            
            sequence.append({
                'node_features': node_features,
                'edge_features': edge_features,
                'timestep': t
            })
        
        return {
            'sequence': sequence,
            'metadata': {
                'cascade': is_cascade,
                'failed_nodes': failed_nodes,
                'failure_times': failure_times,
                'time_to_cascade': failure_times[0] if failure_times else -1,
                'num_nodes': self.num_nodes,
                'sequence_length': sequence_length,
                'stress_level': stress_level
            }
        }
    
    def _generate_load_profile(self, length: int) -> np.ndarray:
        """Generate realistic daily load profile."""
        # Sinusoidal pattern with peak in afternoon
        t = np.linspace(0, 24, length)
        base = 0.7 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
        noise = np.random.randn(length) * 0.05
        return np.clip(base + noise, 0.5, 1.2)
    
    def _generate_weather_profile(self, length: int) -> np.ndarray:
        """Generate weather profile [temperature, wind_speed]."""
        # Temperature (15-35°C with daily variation)
        t = np.linspace(0, 24, length)
        temp = 25 + 8 * np.sin(2 * np.pi * (t - 6) / 24) + np.random.randn(length) * 2
        
        # Wind speed (0-15 m/s)
        wind = 5 + 3 * np.random.randn(length)
        wind = np.clip(wind, 0, 15)
        
        return np.column_stack([temp, wind])


def generate_dataset_streaming(
    num_normal: int = 500,
    num_cascade: int = 50,
    num_nodes: int = 118,
    sequence_length: int = 60,
    output_dir: str = "data",
    batch_size: int = 50
):
    """
    Generate dataset with streaming to disk (memory-efficient).
    Data is saved in batches and NOT combined to avoid memory issues.
    """
    print("=" * 80)
    print("OPTIMIZED CASCADE FAILURE DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Normal scenarios: {num_normal}")
    print(f"  Cascade scenarios: {num_cascade}")
    print(f"  Grid size: {num_nodes} nodes")
    print(f"  Sequence length: {sequence_length} timesteps")
    print(f"  Batch size: {batch_size} (memory-efficient)")
    print(f"  Output directory: {output_dir}")
    print(f"  Initial memory: {MemoryMonitor.get_memory_usage():.1f} MB\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    simulator = RealisticPowerGridSimulator(num_nodes=num_nodes)
    
    # Save topology
    topology_file = output_path / "grid_topology.pkl"
    with open(topology_file, 'wb') as f:
        pickle.dump({
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index,
            'num_nodes': num_nodes
        }, f)
    print(f"✓ Saved grid topology to {topology_file}")
    
    # Generate datasets in batches
    datasets = {
        'train': {'normal': int(num_normal * 0.7), 'cascade': int(num_cascade * 0.7)},
        'val': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)},
        'test': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)}
    }
    
    for split_name, split_config in datasets.items():
        print(f"\nGenerating {split_name} set...")
        print(f"  Normal: {split_config['normal']}, Cascade: {split_config['cascade']}")
        
        batch_dir = output_path / f"{split_name}_batches"
        batch_dir.mkdir(exist_ok=True)
        
        total_scenarios = split_config['normal'] + split_config['cascade']
        batch_count = 0
        
        for batch_start in range(0, total_scenarios, batch_size):
            batch_data = []
            batch_end = min(batch_start + batch_size, total_scenarios)
            
            for i in range(batch_start, batch_end):
                # Determine if cascade
                is_cascade = i >= split_config['normal']
                
                scenario = simulator.generate_scenario(is_cascade, sequence_length)
                batch_data.append(scenario)
                
                if (i + 1) % 10 == 0:
                    mem = MemoryMonitor.get_memory_usage()
                    print(f"    Generated {i + 1}/{total_scenarios} | Memory: {mem:.1f} MB")
            
            batch_file = batch_dir / f"batch_{batch_count:05d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            
            batch_count += 1
            del batch_data
            gc.collect()
            
            print(f"    ✓ Saved batch to {batch_file.name} | Memory: {MemoryMonitor.get_memory_usage():.1f} MB")
        
        metadata_file = batch_dir / "batch_info.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_batches': batch_count,
                'total_scenarios': total_scenarios,
                'batch_size': batch_size,
                'normal_scenarios': split_config['normal'],
                'cascade_scenarios': split_config['cascade']
            }, f, indent=2)
        
        print(f"  ✓ Saved {batch_count} batches to {batch_dir}")
        print(f"  Memory: {MemoryMonitor.get_memory_usage():.1f} MB")
    
    # Save metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'num_normal': num_normal,
        'num_cascade': num_cascade,
        'num_nodes': num_nodes,
        'sequence_length': sequence_length,
        'physics_based': True,
        'realistic_cascades': True,
        'splits': datasets,
        'note': 'Data is stored in batches for memory efficiency. Load batches individually during training.'
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Final memory usage: {MemoryMonitor.get_memory_usage():.1f} MB")
    print(f"\nImprovements:")
    print("  ✓ Memory-efficient batch processing")
    print("  ✓ No memory accumulation - batches stay on disk")
    print("  ✓ Realistic DC power flow physics")
    print("  ✓ Physics-based cascade propagation")
    print("  ✓ Correlated features and temporal dynamics")
    print("  ✓ Realistic load and weather profiles")
    print(f"\nData saved in batch directories:")
    print(f"  - {output_path}/train_batches/")
    print(f"  - {output_path}/val_batches/")
    print(f"  - {output_path}/test_batches/")


if __name__ == "__main__":
    # Generate optimized dataset
    generate_dataset_streaming(
        num_normal=3000,
        num_cascade=300,
        num_nodes=118,
        sequence_length=60,
        output_dir="data",
        batch_size=50  # Process in batches to manage memory
    )
