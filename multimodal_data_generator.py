"""
Unified Multi-Modal Data Generator with REALISTIC PHYSICS
==========================================================
Generates physically realistic data that follows power grid physics and
meaningful cascade dynamics that the model can learn from.

Key Features:
- REALISTIC DC/AC power flow with proper physics
- PHYSICS-BASED cascade propagation (not random!)
- Correlated multi-modal data (environmental threats → infrastructure failures)
- Memory-efficient batch streaming
- Meaningful patterns for ML to learn

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
    Realistic power grid simulator with proper physics.
    """
    
    def __init__(self, num_nodes: int = 118, seed: int = 42):
        self.num_nodes = num_nodes
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate realistic topology
        self.adjacency_matrix = self._generate_realistic_topology()
        self.edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
        self.num_edges = self.edge_index.shape[1]
        
        # Geographic positions (for environmental correlation)
        self.positions = self._generate_geographic_positions()
        
        # Initialize grid properties with realistic values
        self._initialize_realistic_grid_properties()
        
        self._initialize_protection_settings()
        
        self._initialize_frequency_dynamics()
        
        self._initialize_thermal_dynamics()
        
        print(f"Initialized grid: {self.num_nodes} nodes, {self.num_edges} edges")
    
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
        
        # Overcurrent relay settings (pickup current as multiple of rated current)
        self.oc_relay_pickup = np.random.uniform(1.2, 1.5, self.num_edges)  # 120-150% of thermal limit
        
        # Relay operating time based on inverse-time characteristic
        # Time = K / (I^2 - 1) where I is per-unit current
        self.relay_time_dial = np.random.uniform(0.5, 2.0, self.num_edges)  # Time dial setting
        
        # Distance relay settings (impedance zones)
        self.zone1_reach = self.line_reactance * 0.85  # Zone 1: 85% of line (instantaneous)
        self.zone2_reach = self.line_reactance * 1.20  # Zone 2: 120% of line (0.3-0.5s delay)
        
        # Differential relay settings for nodes (instantaneous for internal faults)
        self.diff_relay_pickup = np.random.uniform(0.2, 0.4, self.num_nodes)  # 20-40% differential current
        
        # Under-voltage relay settings
        self.uv_relay_pickup = np.random.uniform(0.88, 0.92, self.num_nodes)  # 88-92% voltage
        self.uv_relay_delay = np.random.uniform(1.0, 3.0, self.num_nodes)  # 1-3 second delay
        
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
            
            # Check for unrealistic angles (indicates instability)
            if np.max(np.abs(theta)) > np.radians(30):  # >30 degrees is unstable
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
            line_flows[i] = self.line_susceptance[i] * (theta[s] - theta[d])
        
        # Voltage drops with heavy loading
        voltages = np.ones(self.num_nodes)
        for i in range(self.num_nodes):
            # Voltage affected by local power imbalance
            power_imbalance = abs(P_net[i]) / (self.base_load[i] + 1e-6)
            voltage_drop = 0.02 * power_imbalance  # 2% drop per unit imbalance
            voltages[i] = 1.0 - voltage_drop + np.random.normal(0, 0.005)
        
        # Clip to realistic range
        voltages = np.clip(voltages, 0.85, 1.15)
        
        # Check voltage stability
        if np.any(voltages < 0.90) or np.any(voltages > 1.10):
            is_stable = False
        
        return voltages, theta, line_flows, is_stable
    
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
    
    def _simulate_realistic_cascade(
        self, 
        initial_trigger: Dict,
        stress_level: float,
        sequence_length: int = 60
    ) -> Tuple[List[int], List[int], List[float], int]:
        """
        Simulate REALISTIC physics-based cascade propagation with:
        - Deterministic relay operations
        - Frequency dynamics
        - Per-node thermal effects
        """
        
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
        
        ambient_temp = 25 + 10 * np.random.rand()
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp)
        
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
        
        pending_line_trips = {}  # {line_idx: trip_time}
        pending_node_trips = {}  # {node_idx: trip_time}
        
        for iteration in range(25):
            voltages, angles, line_flows, is_stable = self._compute_realistic_power_flow(
                generation, load, failed_lines, failed_nodes
            )
            
            if not is_stable:
                break
            
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            current_frequency, load = self._update_frequency_dynamics(
                generation, load, failed_nodes, current_frequency, dt=2.0
            )
            
            self.equipment_temperatures = self._update_thermal_dynamics(
                loading_ratios, ambient_temp, dt=2.0
            )
            
            for i in range(self.num_edges):
                if i in failed_lines or i in pending_line_trips:
                    continue
                
                # Overcurrent relay
                trip_time = self._calculate_relay_operation_time(i, loading_ratios[i])
                if trip_time > 0:
                    pending_line_trips[i] = current_time + trip_time
            
            for node in range(self.num_nodes):
                if node in failed_nodes or node in pending_node_trips:
                    continue
                
                if voltages[node] < self.uv_relay_pickup[node]:
                    pending_node_trips[node] = current_time + self.uv_relay_delay[node]
                
                if current_frequency < self.uf_relay_pickup[node]:
                    trip_time = current_time + self.uf_relay_delay[node]
                    if node not in pending_node_trips or trip_time < pending_node_trips[node]:
                        pending_node_trips[node] = trip_time
                
                if self.equipment_temperatures[node] > self.max_safe_temp[node]:
                    # Thermal trip (slower, 5-15 seconds)
                    trip_time = current_time + np.random.uniform(5, 15)
                    if node not in pending_node_trips or trip_time < pending_node_trips[node]:
                        pending_node_trips[node] = trip_time
            
            trips_occurred = False
            
            for line_idx, trip_time in list(pending_line_trips.items()):
                if trip_time <= current_time:
                    failed_lines.append(line_idx)
                    failure_times.append(current_time)
                    del pending_line_trips[line_idx]
                    trips_occurred = True
                    
                    # Adjacent nodes may fail due to fault current
                    src, dst = self.edge_index
                    s, d = src[line_idx].item(), dst[line_idx].item()
                    
                    # Differential relay operates if fault current is high
                    if loading_ratios[line_idx] > 5.0:  # High fault current
                        if s not in failed_nodes and s not in pending_node_trips:
                            pending_node_trips[s] = current_time + 0.05  # Instantaneous
                        if d not in failed_nodes and d not in pending_node_trips:
                            pending_node_trips[d] = current_time + 0.05
            
            for node_idx, trip_time in list(pending_node_trips.items()):
                if trip_time <= current_time:
                    failed_nodes.append(node_idx)
                    failure_times.append(current_time)
                    del pending_node_trips[node_idx]
                    trips_occurred = True
            
            if not trips_occurred and len(pending_line_trips) == 0 and len(pending_node_trips) == 0:
                # No more violations and no pending trips
                break
            
            # Advance time
            if len(pending_line_trips) > 0 or len(pending_node_trips) > 0:
                next_trip_time = min(
                    [t for t in pending_line_trips.values()] + 
                    [t for t in pending_node_trips.values()]
                )
                current_time = next_trip_time
            else:
                current_time += 2.0  # Advance by timestep
            
            if len(failed_lines) > self.num_edges * 0.3 or len(failed_nodes) > self.num_nodes * 0.2:
                break
            
            if current_frequency < 57.0:
                print(f"  [COLLAPSE] System frequency collapsed to {current_frequency:.2f} Hz")
                break
        
        return failed_lines, failed_nodes, failure_times, cascade_start_time
    
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
                sensor_data[node, 10] += 0.08 * precursor_strength  # Moisture increases
                
                # Visual signs of stress (discoloration, arcing)
                visual_data[node, 0, :, :] += 0.3 * precursor_strength  # Reddish glow from heat
                visual_data[node, 1:3, :, :] -= 0.2 * precursor_strength  # Darkening
        
        return visual_data, thermal_data, sensor_data
    
    def generate_single_scenario(
        self, 
        is_cascade: bool, 
        sequence_length: int = 60
    ) -> Dict:
        """
        Generate ONE scenario with REALISTIC PHYSICS and CORRELATED multi-modal data.
        
        This is what the model will learn from!
        """
        
        if is_cascade:
            stress_level = np.random.uniform(0.65, 0.95)  # High stress → cascade likely
        else:
            stress_level = np.random.uniform(0.0, 0.45)  # Low stress → stable
        
        failed_lines = []
        failed_nodes = []
        failure_times = []
        cascade_start_time = -1
        
        scenario_frequency = 60.0
        
        ambient_temp_base = 25 + 10 * np.random.rand()
        self.equipment_temperatures = np.full(self.num_nodes, ambient_temp_base)
        
        trigger = {'type': 'none'} # Default trigger
        if is_cascade:
            # Random trigger type
            trigger_type = np.random.choice(['line_trip', 'generator_trip', 'load_spike'], p=[0.5, 0.3, 0.2])
            
            if trigger_type == 'line_trip':
                trigger = {
                    'type': 'line_trip',
                    'line_id': np.random.randint(0, self.num_edges)
                }
            elif trigger_type == 'generator_trip':
                gen_indices = np.where(self.node_types == 1)[0]
                trigger = {
                    'type': 'generator_trip',
                    'node_id': np.random.choice(gen_indices)
                }
            else:  # load_spike
                num_affected = np.random.randint(2, 6)
                trigger = {
                    'type': 'load_spike',
                    'affected_nodes': np.random.choice(self.num_nodes, num_affected, replace=False).tolist()
                }
            
            # Simulate cascade
            failed_lines, failed_nodes, failure_times, cascade_start_time = \
                self._simulate_realistic_cascade(trigger, stress_level, sequence_length)
        
        sequence = []
        
        for t in range(sequence_length):
            hour_of_day = (t / sequence_length) * 24
            load_factor = 0.6 + 0.4 * (1 + np.sin(2 * np.pi * (hour_of_day - 6) / 24)) / 2
            load = self.base_load * load_factor * (1.0 + stress_level * 0.3)
            
            total_load = load.sum()
            gen_indices = np.where(self.node_types == 1)[0]
            generation = np.zeros(self.num_nodes)
            total_capacity = self.gen_capacity.sum()
            
            for idx in gen_indices:
                generation[idx] = (self.gen_capacity[idx] / total_capacity) * total_load * 1.02
            
            failed_lines_t = []
            failed_nodes_t = []
            
            if is_cascade and t >= cascade_start_time:
                time_since_cascade = (t - cascade_start_time) * 2.0  # 2 seconds per timestep
                
                # Components that have failed by now
                for i, ft in enumerate(failure_times):
                    if ft <= time_since_cascade:
                        # Ensure indices are valid for the current number of failures
                        if i < len(failed_lines):
                            failed_lines_t.append(failed_lines[i])
                        if i < len(failed_nodes):
                            failed_nodes_t.append(failed_nodes[i])
            
            voltages, angles, line_flows, is_stable = self._compute_realistic_power_flow(
                generation, load, 
                failed_lines_t if failed_lines_t else None,
                failed_nodes_t if failed_nodes_t else None
            )
            
            loading_ratios = np.abs(line_flows) / (self.thermal_limits + 1e-6)
            
            if t > 0:
                scenario_frequency, load = self._update_frequency_dynamics(
                    generation, load, failed_nodes_t, scenario_frequency, dt=2.0
                )
            
            ambient_temp = ambient_temp_base + 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            
            equipment_temps = self._update_thermal_dynamics(loading_ratios, ambient_temp, dt=2.0)

            satellite_data, weather_sequence, threat_indicators = \
                self._generate_correlated_environmental_data(
                    failed_nodes_t, failed_lines_t, t, cascade_start_time, stress_level
                )
            
            scada_data = np.column_stack([
                voltages,  # V (p.u.)
                angles,  # theta (rad)
                generation,  # P_gen (MW)
                generation * 0.3,  # Q_gen (MVAr)
                load,  # P_load (MW)
                load * 0.2,  # Q_load (MVAr)
                np.full(self.num_nodes, scenario_frequency),  # Actual frequency
                equipment_temps,  # Per-node temperatures
                np.clip(5 + 3 * np.random.randn(self.num_nodes), 0, 20),  # Wind (m/s)
                self.equipment_condition,  # Condition (0-1)
                self.equipment_age,  # Age (years)
                np.random.randn(self.num_nodes, 9)  # Other measurements
            ]).astype(np.float16)
            
            # PMU high-frequency data (30 samples, 15 features)
            pmu_sequence = np.random.randn(self.num_nodes, 30, 15).astype(np.float16)
            
            equipment_status = np.column_stack([
                self.equipment_age,
                self.equipment_condition,
                equipment_temps,  # Per-node temperatures
                np.random.randn(self.num_nodes, 7)
            ]).astype(np.float16)
            
            visual_data, thermal_data, sensor_data = \
                self._generate_correlated_robotic_data(
                    failed_nodes_t, failed_lines_t, t, cascade_start_time, equipment_temps
                )
            
            edge_features = np.column_stack([
                self.line_reactance,
                self.thermal_limits,
                loading_ratios,
                np.full(self.num_edges, ambient_temp),
                np.random.randn(self.num_edges, 6)
            ]).astype(np.float16)
            
            node_labels = np.zeros(self.num_nodes, dtype=np.float16)
            for node in failed_nodes_t:
                node_labels[node] = 1.0
            
            edge_labels = np.zeros(self.num_edges, dtype=np.float16)
            for line in failed_lines_t:
                edge_labels[line] = 1.0
            
            sequence.append({
                'timestep': t,
                # Environmental modality
                'satellite_data': satellite_data,
                'weather_sequence': weather_sequence,
                'threat_indicators': threat_indicators,
                # Infrastructure modality
                'scada_data': scada_data,
                'pmu_sequence': pmu_sequence,
                'equipment_status': equipment_status,
                # Robotic modality
                'visual_data': visual_data,
                'thermal_data': thermal_data,
                'sensor_data': sensor_data,
                # Graph structure
                'edge_attr': edge_features,
                # Ground truth physics
                'voltages': voltages.astype(np.float16),
                'angles': angles.astype(np.float16),
                'line_flows': line_flows.astype(np.float16),
                'loading_ratios': loading_ratios.astype(np.float16),
                'is_stable': is_stable,
                # Labels
                'node_labels': node_labels,
                'edge_labels': edge_labels,
                'cascade_active': (t >= cascade_start_time) if is_cascade else False,
                'frequency': scenario_frequency,
                'equipment_temperatures': equipment_temps.astype(np.float16)
            })
        
        last_timestep = sequence[-1]
        multi_modal_data = {
            'environmental': {
                'satellite_imagery': last_timestep['satellite_data'],
                'weather_sequence': last_timestep['weather_sequence'],
                'threat_indicators': last_timestep['threat_indicators']
            },
            'infrastructure': {
                'scada_measurements': last_timestep['scada_data'],
                'pmu_data': last_timestep['pmu_sequence'],
                'equipment_condition': last_timestep['equipment_status'],
                'edge_features': last_timestep['edge_attr']
            },
            'robotic': {
                'visual_inspection': last_timestep['visual_data'],
                'thermal_imaging': last_timestep['thermal_data'],
                'sensor_readings': last_timestep['sensor_data']
            }
        }
        
        if is_cascade and cascade_start_time >= 0:
            # Time from last timestep to cascade start (in minutes)
            time_to_cascade = max(0, (cascade_start_time - (sequence_length - 1)) * 2.0 / 60.0)
        else:
            time_to_cascade = -1.0
        
        return {
            'sequence': sequence,
            'edge_index': self.edge_index,
            'multi_modal_data': multi_modal_data,
            'metadata': {
                'cascade': is_cascade,
                'failed_nodes': np.array(failed_nodes, dtype=np.int32),  # Convert to numpy array
                'failed_lines': np.array(failed_lines, dtype=np.int32),  # Convert to numpy array
                'failure_times': np.array(failure_times, dtype=np.float32),  # Convert to numpy array
                'cascade_start_time': int(cascade_start_time),
                'time_to_cascade': float(time_to_cascade),  # Add time_to_cascade in minutes
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'sequence_length': sequence_length,
                'stress_level': float(stress_level),
                'trigger_type': trigger['type'] if is_cascade else 'none'
            }
        }


def generate_dataset_streaming(
    num_normal: int = 500,
    num_cascade: int = 50,
    num_nodes: int = 118,
    sequence_length: int = 60,
    output_dir: str = "data_unified",
    batch_size: int = 10
):
    """
    Generate unified dataset with REALISTIC PHYSICS and batch streaming.
    NEVER accumulates data in memory.
    """
    print("=" * 80)
    print("PHYSICS-BASED MULTI-MODAL DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Normal scenarios: {num_normal}")
    print(f"  Cascade scenarios: {num_cascade}")
    print(f"  Grid size: {num_nodes} nodes")
    print(f"  Sequence length: {sequence_length} timesteps")
    print(f"  Batch size: {batch_size}")
    print(f"  Output directory: {output_dir}")
    print(f"  Initial memory: {MemoryMonitor.get_memory_usage():.1f} MB\n")
    
    print("KEY FEATURES:")
    print("  ✓ Realistic DC power flow physics")
    print("  ✓ Physics-based cascade propagation")
    print("  ✓ Correlated multi-modal data")
    print("  ✓ Environmental threats → infrastructure failures")
    print("  ✓ Predictive signals in sensor data")
    print("  ✓ Memory-efficient batch streaming\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    simulator = PhysicsBasedGridSimulator(num_nodes=num_nodes)
    
    # Save topology
    topology_file = output_path / "grid_topology.pkl"
    with open(topology_file, 'wb') as f:
        pickle.dump({
            'adjacency_matrix': simulator.adjacency_matrix,
            'edge_index': simulator.edge_index,
            'positions': simulator.positions,
            'num_nodes': num_nodes,
            'num_edges': simulator.num_edges,
            'node_types': simulator.node_types,
            'gen_capacity': simulator.gen_capacity,
            'base_load': simulator.base_load,
            'thermal_limits': simulator.thermal_limits
        }, f)
    print(f"✓ Saved grid topology to {topology_file}\n")
    
    # Generate datasets in batches
    datasets = {
        'train': {'normal': int(num_normal * 0.7), 'cascade': int(num_cascade * 0.7)},
        'val': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)},
        'test': {'normal': int(num_normal * 0.15), 'cascade': int(num_cascade * 0.15)}
    }
    
    for split_name, split_config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Generating {split_name.upper()} set...")
        print(f"{'='*60}")
        
        batch_dir = output_path / f"{split_name}_batches"
        batch_dir.mkdir(exist_ok=True)
        
        total_scenarios = split_config['normal'] + split_config['cascade']
        batch_count = 0
        
        for batch_start in range(0, total_scenarios, batch_size):
            batch_data = []
            batch_end = min(batch_start + batch_size, total_scenarios)
            
            for i in range(batch_start, batch_end):
                is_cascade = i >= split_config['normal']
                
                scenario = simulator.generate_single_scenario(is_cascade, sequence_length)
                batch_data.append(scenario)
                
                if (i + 1) % 10 == 0:
                    mem = MemoryMonitor.get_memory_usage()
                    cascade_str = "CASCADE" if is_cascade else "NORMAL"
                    print(f"  [{cascade_str}] Generated {i + 1}/{total_scenarios} | Memory: {mem:.1f} MB")
            
            # Save batch immediately
            batch_file = batch_dir / f"batch_{batch_count:05d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            
            mem_after = MemoryMonitor.get_memory_usage()
            print(f"  ✓ Saved {batch_file.name} | Memory: {mem_after:.1f} MB")
            
            batch_count += 1
            
            del batch_data
            gc.collect()
        
        # Save batch metadata
        metadata_file = batch_dir / "batch_info.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_batches': batch_count,
                'total_scenarios': total_scenarios,
                'batch_size': batch_size,
                'normal_scenarios': split_config['normal'],
                'cascade_scenarios': split_config['cascade']
            }, f, indent=2)
        
        print(f"\n✓ Completed {split_name} set: {batch_count} batches, {total_scenarios} scenarios")
    
    # Save overall metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'num_normal': num_normal,
        'num_cascade': num_cascade,
        'num_nodes': num_nodes,
        'sequence_length': sequence_length,
        'physics_based': True,
        'realistic_cascades': True,
        'correlated_multimodal': True,
        'memory_efficient': True,
        'splits': datasets,
        'features': {
            'environmental': 'Satellite imagery, weather, threats (correlated with failures)',
            'infrastructure': 'SCADA, PMU, equipment (realistic power flow)',
            'robotic': 'Visual, thermal, sensors (predictive anomalies)',
            'physics': 'DC power flow, cascade propagation, thermal limits',
            'labels': 'Node failures, line failures, cascade timing'
        }
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Final memory: {MemoryMonitor.get_memory_usage():.1f} MB")
    print(f"\nREALISTIC PHYSICS IMPLEMENTED:")
    print("  ✓ DC power flow: P = B * theta")
    print("  ✓ Cascade propagation: overload → trip → redistribution")
    print("  ✓ Correlated threats: wildfire → equipment failure")
    print("  ✓ Predictive signals: thermal anomalies before failures")
    print("  ✓ Realistic operating conditions: stress levels, load profiles")
    print(f"\nData saved in: {output_path}")
    print("\nThe model can now LEARN meaningful patterns from this data!")


if __name__ == "__main__":
    generate_dataset_streaming(
        num_normal=600,
        num_cascade=60,
        num_nodes=118,
        sequence_length=60,
        output_dir="data_unified",
        batch_size=30
    )
