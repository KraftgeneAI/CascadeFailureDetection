"""
Tests for cascade_prediction/data/generator modules
====================================================
Tests topology, physics, cascade, environmental, robotic, simulator, and scenario generators.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

from cascade_prediction.data.generator.topology import (
    GridTopologyGenerator,
    NodePropertyInitializer
)
from cascade_prediction.data.generator.physics import (
    PowerFlowSimulator,
    FrequencyDynamicsSimulator,
    ThermalDynamicsSimulator
)
from cascade_prediction.data.generator.cascade import (
    CascadeSimulator,
    create_adjacency_list
)
from cascade_prediction.data.generator.environmental import (
    EnvironmentalDataGenerator
)
from cascade_prediction.data.generator.robotic import (
    RoboticDataGenerator
)
from cascade_prediction.data.generator.simulator import (
    PhysicsBasedGridSimulator
)
from cascade_prediction.data.generator.scenario import (
    ScenarioOrchestrator,
    generate_dataset_from_config
)
from cascade_prediction.data.generator.utils import (
    MemoryMonitor,
    save_scenarios,
    load_topology,
    save_topology,
    split_scenarios,
    validate_scenario,
    get_failed_lines_from_nodes
)


class TestTopologyGenerator:
    """Test suite for topology generation."""
    
    def test_topology_generation(self):
        """Test basic topology generation."""
        gen = GridTopologyGenerator(num_nodes=30, seed=42)
        topo = gen.generate_topology()
        
        assert 'adjacency_matrix' in topo
        assert 'edge_index' in topo
        assert 'positions' in topo
        assert topo['num_nodes'] == 30
        assert topo['num_edges'] > 0
    
    def test_topology_connectivity(self):
        """Test that generated topology is connected."""
        gen = GridTopologyGenerator(num_nodes=50, seed=42)
        topo = gen.generate_topology()
        
        adj = topo['adjacency_matrix']
        
        # BFS to check connectivity
        visited = np.zeros(50, dtype=bool)
        queue = [0]
        visited[0] = True
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(50):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        # All nodes should be reachable
        assert np.all(visited)
    
    def test_topology_symmetry(self):
        """Test that adjacency matrix is symmetric."""
        gen = GridTopologyGenerator(num_nodes=30, seed=42)
        topo = gen.generate_topology()
        
        adj = topo['adjacency_matrix']
        assert np.allclose(adj, adj.T)
    
    def test_edge_index_format(self):
        """Test edge index format."""
        gen = GridTopologyGenerator(num_nodes=30, seed=42)
        topo = gen.generate_topology()
        
        edge_index = topo['edge_index']
        
        # Should be 2 x num_edges
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == topo['num_edges']
        
        # All indices should be valid
        assert torch.all(edge_index >= 0)
        assert torch.all(edge_index < 30)
    
    def test_positions_generation(self):
        """Test geographic positions generation."""
        gen = GridTopologyGenerator(num_nodes=30, seed=42)
        topo = gen.generate_topology()
        
        positions = topo['positions']
        
        # Should be num_nodes x 2
        assert positions.shape == (30, 2)
        
        # Positions should be in reasonable range
        assert np.all(np.abs(positions) < 200)


class TestNodePropertyInitializer:
    """Test suite for node property initialization."""
    
    def test_property_initialization(self):
        """Test basic property initialization."""
        init = NodePropertyInitializer(num_nodes=30, seed=42)
        props = init.initialize_properties()
        
        assert 'node_types' in props
        assert 'gen_capacity' in props
        assert 'base_load' in props
        assert 'loading_failure_threshold' in props
    
    def test_node_type_distribution(self):
        """Test node type distribution."""
        init = NodePropertyInitializer(num_nodes=100, seed=42)
        props = init.initialize_properties()
        
        node_types = props['node_types']
        
        # Should have generators (type 1)
        assert np.sum(node_types == 1) > 0
        
        # Should have loads (type 0)
        assert np.sum(node_types == 0) > 0
        
        # Node 0 should be slack bus (generator)
        assert node_types[0] == 1
    
    def test_generator_capacity_sizing(self):
        """Test generator capacity sizing."""
        init = NodePropertyInitializer(num_nodes=50, seed=42)
        props = init.initialize_properties()
        
        gen_capacity = props['gen_capacity']
        base_load = props['base_load']
        
        # Total generation should exceed total load
        total_gen = gen_capacity.sum()
        total_load = base_load.sum()
        
        assert total_gen > total_load
    
    def test_failure_thresholds(self):
        """Test failure threshold initialization."""
        init = NodePropertyInitializer(num_nodes=30, seed=42)
        props = init.initialize_properties()
        
        # Check all threshold types exist
        assert 'loading_failure_threshold' in props
        assert 'voltage_failure_threshold' in props
        assert 'temperature_failure_threshold' in props
        assert 'frequency_failure_threshold' in props
        
        # Failure thresholds should be higher than damage thresholds
        assert np.all(
            props['loading_failure_threshold'] > props['loading_damage_threshold']
        )


class TestPowerFlowSimulator:
    """Test suite for power flow simulation."""
    
    @pytest.fixture
    def simple_grid(self):
        """Create a simple 3-node grid for testing."""
        num_nodes = 3
        edge_index = np.array([[0, 1], [1, 2]])
        positions = np.array([[0, 0], [1, 0], [2, 0]])
        node_types = np.array([1, 0, 0])  # Gen, Load, Load
        gen_capacity = np.array([200.0, 0.0, 0.0])
        
        line_reactance = np.array([0.01, 0.01])
        line_resistance = np.array([0.001, 0.001])
        line_susceptance = np.array([1e-5, 1e-5])
        line_conductance = np.array([0.0, 0.0])
        thermal_limits = np.array([100.0, 100.0])
        
        return {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'positions': positions,
            'node_types': node_types,
            'gen_capacity': gen_capacity,
            'line_reactance': line_reactance,
            'line_resistance': line_resistance,
            'line_susceptance': line_susceptance,
            'line_conductance': line_conductance,
            'thermal_limits': thermal_limits
        }
    
    def test_power_flow_initialization(self, simple_grid):
        """Test power flow simulator initialization."""
        sim = PowerFlowSimulator(**simple_grid)
        
        assert sim.num_nodes == 3
        assert sim.network is not None
    
    def test_power_flow_computation(self, simple_grid):
        """Test basic power flow computation."""
        sim = PowerFlowSimulator(**simple_grid)
        
        generation = np.array([100.0, 0.0, 0.0])
        load = np.array([0.0, 40.0, 40.0])
        
        voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable = \
            sim.compute_power_flow(generation, load)
        
        assert voltages.shape == (3,)
        assert angles.shape == (3,)
        assert line_flows_p.shape == (2,)
        assert isinstance(is_stable, bool)
    
    def test_power_flow_with_failures(self, simple_grid):
        """Test power flow with failed nodes."""
        sim = PowerFlowSimulator(**simple_grid)
        
        generation = np.array([100.0, 0.0, 0.0])
        load = np.array([0.0, 40.0, 40.0])
        failed_nodes = [2]
        
        voltages, angles, line_flows_p, node_reactive, line_flows_q, is_stable = \
            sim.compute_power_flow(generation, load, failed_nodes=failed_nodes)
        
        # Should still return results
        assert voltages.shape == (3,)


class TestFrequencyDynamicsSimulator:
    """Test suite for frequency dynamics."""
    
    def test_frequency_initialization(self):
        """Test frequency simulator initialization."""
        num_nodes = 30
        node_types = np.zeros(num_nodes)
        node_types[0] = 1  # One generator
        gen_capacity = np.zeros(num_nodes)
        gen_capacity[0] = 200.0
        
        sim = FrequencyDynamicsSimulator(num_nodes, node_types, gen_capacity)
        
        assert sim.num_nodes == 30
        assert sim.base_frequency == 60.0
    
    def test_frequency_update_balanced(self):
        """Test frequency update with balanced generation/load."""
        num_nodes = 30
        node_types = np.zeros(num_nodes)
        node_types[0] = 1
        gen_capacity = np.zeros(num_nodes)
        gen_capacity[0] = 200.0
        
        sim = FrequencyDynamicsSimulator(num_nodes, node_types, gen_capacity)
        
        generation = np.zeros(num_nodes)
        generation[0] = 100.0
        load = np.ones(num_nodes) * (100.0 / num_nodes)
        
        new_freq, adjusted_load = sim.update_frequency(generation, load, 60.0)
        
        # Frequency should remain near 60 Hz
        assert 59.0 < new_freq < 61.0
    
    def test_frequency_update_imbalance(self):
        """Test frequency update with generation deficit."""
        num_nodes = 30
        node_types = np.zeros(num_nodes)
        node_types[0] = 1
        gen_capacity = np.zeros(num_nodes)
        gen_capacity[0] = 200.0
        
        sim = FrequencyDynamicsSimulator(num_nodes, node_types, gen_capacity)
        
        generation = np.zeros(num_nodes)
        generation[0] = 50.0  # Low generation
        load = np.ones(num_nodes) * (100.0 / num_nodes)  # High load
        
        new_freq, adjusted_load = sim.update_frequency(generation, load, 60.0)
        
        # Frequency should drop
        assert new_freq < 60.0


class TestThermalDynamicsSimulator:
    """Test suite for thermal dynamics."""
    
    def test_thermal_initialization(self):
        """Test thermal simulator initialization."""
        num_nodes = 30
        thermal_time_constant = np.ones(num_nodes) * 20.0
        thermal_capacity = np.ones(num_nodes)
        cooling_effectiveness = np.ones(num_nodes) * 0.8
        
        sim = ThermalDynamicsSimulator(
            num_nodes, thermal_time_constant, thermal_capacity, cooling_effectiveness
        )
        
        assert sim.num_nodes == 30
        assert sim.ambient_temperature == 25.0
    
    def test_temperature_update(self):
        """Test temperature update."""
        num_nodes = 30
        thermal_time_constant = np.ones(num_nodes) * 20.0
        thermal_capacity = np.ones(num_nodes)
        cooling_effectiveness = np.ones(num_nodes) * 0.8
        
        sim = ThermalDynamicsSimulator(
            num_nodes, thermal_time_constant, thermal_capacity, cooling_effectiveness
        )
        
        # Apply heat
        heat_generation = np.ones(num_nodes) * 10.0
        temps = sim.update_temperatures(heat_generation)
        
        # Temperatures should increase
        assert np.all(temps >= sim.ambient_temperature)
    
    def test_temperature_reset(self):
        """Test temperature reset."""
        num_nodes = 30
        thermal_time_constant = np.ones(num_nodes) * 20.0
        thermal_capacity = np.ones(num_nodes)
        cooling_effectiveness = np.ones(num_nodes) * 0.8
        
        sim = ThermalDynamicsSimulator(
            num_nodes, thermal_time_constant, thermal_capacity, cooling_effectiveness
        )
        
        # Heat up
        heat_generation = np.ones(num_nodes) * 10.0
        sim.update_temperatures(heat_generation)
        
        # Reset
        sim.reset_temperatures()
        
        # Should be back to ambient
        assert np.all(sim.temperatures == sim.ambient_temperature)


class TestCascadeSimulator:
    """Test suite for cascade simulation."""
    
    @pytest.fixture
    def cascade_setup(self):
        """Create cascade simulator setup."""
        num_nodes = 10
        adjacency_list = [[] for _ in range(num_nodes)]
        
        # Simple chain: 0->1->2->3...
        for i in range(num_nodes - 1):
            adjacency_list[i].append((i+1, i, 0.8))
        
        thresholds = {
            'loading_failure': np.ones(num_nodes) * 1.1,
            'loading_damage': np.ones(num_nodes) * 1.0,
            'voltage_failure': np.ones(num_nodes) * 0.9,
            'voltage_damage': np.ones(num_nodes) * 0.92,
            'temperature_failure': np.ones(num_nodes) * 90.0,
            'temperature_damage': np.ones(num_nodes) * 80.0,
            'frequency_failure': np.ones(num_nodes) * 59.0,
            'frequency_damage': np.ones(num_nodes) * 59.5,
        }
        
        return num_nodes, adjacency_list, thresholds
    
    def test_cascade_initialization(self, cascade_setup):
        """Test cascade simulator initialization."""
        num_nodes, adjacency_list, thresholds = cascade_setup
        
        sim = CascadeSimulator(
            num_nodes, adjacency_list,
            thresholds['loading_failure'], thresholds['loading_damage'],
            thresholds['voltage_failure'], thresholds['voltage_damage'],
            thresholds['temperature_failure'], thresholds['temperature_damage'],
            thresholds['frequency_failure'], thresholds['frequency_damage']
        )
        
        assert sim.num_nodes == 10
    
    def test_check_node_state_ok(self, cascade_setup):
        """Test node state checking - OK state."""
        num_nodes, adjacency_list, thresholds = cascade_setup
        
        sim = CascadeSimulator(
            num_nodes, adjacency_list,
            thresholds['loading_failure'], thresholds['loading_damage'],
            thresholds['voltage_failure'], thresholds['voltage_damage'],
            thresholds['temperature_failure'], thresholds['temperature_damage'],
            thresholds['frequency_failure'], thresholds['frequency_damage']
        )
        
        state, reason = sim.check_node_state(0, 0.8, 1.0, 60.0, 60.0)
        
        assert state == 0  # OK
        assert reason == "none"
    
    def test_check_node_state_failure(self, cascade_setup):
        """Test node state checking - failure state."""
        num_nodes, adjacency_list, thresholds = cascade_setup
        
        sim = CascadeSimulator(
            num_nodes, adjacency_list,
            thresholds['loading_failure'], thresholds['loading_damage'],
            thresholds['voltage_failure'], thresholds['voltage_damage'],
            thresholds['temperature_failure'], thresholds['temperature_damage'],
            thresholds['frequency_failure'], thresholds['frequency_damage']
        )
        
        # Overload
        state, reason = sim.check_node_state(0, 1.2, 1.0, 60.0, 60.0)
        
        assert state == 2  # Failed
        assert "overload" in reason
    
    def test_create_adjacency_list(self):
        """Test adjacency list creation."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        node_types = np.array([1, 0, 2])  # Gen, Load, Sub
        
        adj_list = create_adjacency_list(edge_index, node_types)
        
        assert len(adj_list) == 3
        assert len(adj_list[0]) > 0  # Node 0 has neighbors


class TestEnvironmentalDataGenerator:
    """Test suite for environmental data generation."""
    
    @pytest.fixture
    def env_gen(self):
        """Create environmental data generator."""
        num_nodes = 30
        positions = np.random.randn(num_nodes, 2) * 50
        edge_index = np.random.randint(0, num_nodes, (2, 50))
        
        return EnvironmentalDataGenerator(num_nodes, positions, edge_index)
    
    def test_satellite_imagery_generation(self, env_gen):
        """Test satellite imagery generation."""
        failed_nodes = [0, 5]
        timestep = 10
        cascade_start = 5
        stress_level = 0.8
        
        sat_data = env_gen.generate_satellite_imagery(
            failed_nodes, timestep, cascade_start, stress_level
        )
        
        # Should be (num_nodes, 12, 16, 16)
        assert sat_data.shape == (30, 12, 16, 16)
        assert sat_data.dtype == np.float16
    
    def test_weather_sequence_generation(self, env_gen):
        """Test weather sequence generation."""
        timestep = 10
        stress_level = 0.7
        
        weather = env_gen.generate_weather_sequence(timestep, stress_level)
        
        # Should be (num_nodes, 10, 8)
        assert weather.shape == (30, 10, 8)
        assert weather.dtype == np.float16
    
    def test_threat_indicators_generation(self, env_gen):
        """Test threat indicators generation."""
        failed_nodes = [0]
        failed_lines = [1, 2]
        timestep = 10
        cascade_start = 5
        stress_level = 0.9
        
        threats = env_gen.generate_threat_indicators(
            failed_nodes, failed_lines, timestep, cascade_start, stress_level
        )
        
        # Should be (num_nodes, 6)
        assert threats.shape == (30, 6)
        assert threats.dtype == np.float16
        
        # Should be in [0, 1] range
        assert np.all(threats >= 0)
        assert np.all(threats <= 1)
    
    def test_correlated_environmental_data(self, env_gen):
        """Test complete environmental data generation."""
        failed_nodes = [0]
        failed_lines = [1]
        timestep = 10
        cascade_start = 5
        stress_level = 0.8
        
        sat_data, weather, threats = env_gen.generate_correlated_environmental_data(
            failed_nodes, failed_lines, timestep, cascade_start, stress_level
        )
        
        assert sat_data.shape == (30, 12, 16, 16)
        assert weather.shape == (30, 10, 8)
        assert threats.shape == (30, 6)


class TestRoboticDataGenerator:
    """Test suite for robotic data generation."""
    
    @pytest.fixture
    def robot_gen(self):
        """Create robotic data generator."""
        num_nodes = 30
        equipment_age = np.random.uniform(0, 40, num_nodes)
        equipment_condition = np.random.uniform(0.6, 1.0, num_nodes)
        
        return RoboticDataGenerator(num_nodes, equipment_age, equipment_condition)
    
    def test_visual_data_generation(self, robot_gen):
        """Test visual data generation."""
        failed_nodes = [0]
        timestep = 10
        cascade_start = 5
        
        visual = robot_gen.generate_visual_data(failed_nodes, timestep, cascade_start)
        
        # Should be (num_nodes, 3, 32, 32)
        assert visual.shape == (30, 3, 32, 32)
        assert visual.dtype == np.float16
    
    def test_thermal_data_generation(self, robot_gen):
        """Test thermal data generation."""
        equipment_temps = np.random.uniform(60, 80, 30)
        failed_nodes = [0]
        timestep = 10
        cascade_start = 5
        
        thermal = robot_gen.generate_thermal_data(
            equipment_temps, failed_nodes, timestep, cascade_start
        )
        
        # Should be (num_nodes, 1, 32, 32)
        assert thermal.shape == (30, 1, 32, 32)
        assert thermal.dtype == np.float16
    
    def test_sensor_data_generation(self, robot_gen):
        """Test sensor data generation."""
        failed_nodes = [0]
        timestep = 10
        cascade_start = 5
        
        sensors = robot_gen.generate_sensor_data(failed_nodes, timestep, cascade_start)
        
        # Should be (num_nodes, 12)
        assert sensors.shape == (30, 12)
        assert sensors.dtype == np.float16
    
    def test_correlated_robotic_data(self, robot_gen):
        """Test complete robotic data generation."""
        failed_nodes = [0]
        failed_lines = [1]
        timestep = 10
        cascade_start = 5
        equipment_temps = np.random.uniform(60, 80, 30)
        
        visual, thermal, sensors = robot_gen.generate_correlated_robotic_data(
            failed_nodes, failed_lines, timestep, cascade_start, equipment_temps
        )
        
        assert visual.shape == (30, 3, 32, 32)
        assert thermal.shape == (30, 1, 32, 32)
        assert sensors.shape == (30, 12)


class TestPhysicsBasedGridSimulator:
    """Test suite for complete grid simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        
        assert sim.num_nodes == 30
        assert sim.num_edges > 0
        assert sim.adjacency_matrix is not None
    
    def test_scenario_generation_normal(self):
        """Test normal scenario generation."""
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        
        scenario = sim.generate_scenario(stress_level=0.5, sequence_length=10)
        
        if scenario is not None:
            assert 'sequence' in scenario
            assert 'edge_index' in scenario
            assert 'metadata' in scenario
            assert len(scenario['sequence']) == 10
    
    def test_scenario_generation_cascade(self):
        """Test cascade scenario generation."""
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        
        # High stress to trigger cascade
        scenario = sim.generate_scenario(stress_level=0.95, sequence_length=10)
        
        if scenario is not None:
            assert 'sequence' in scenario
            assert 'metadata' in scenario


class TestScenarioOrchestrator:
    """Test suite for scenario orchestration."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_orchestrator_initialization(self, temp_output_dir):
        """Test orchestrator initialization."""
        sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
        
        orch = ScenarioOrchestrator(
            simulator=sim,
            output_dir=str(temp_output_dir),
            batch_size=5
        )
        
        assert orch.batch_size == 5
        assert orch.train_dir.exists()
        assert orch.val_dir.exists()
        assert orch.test_dir.exists()


class TestUtils:
    """Test suite for utility functions."""
    
    def test_memory_monitor(self):
        """Test memory monitoring."""
        usage = MemoryMonitor.get_memory_usage()
        
        assert usage > 0
        assert isinstance(usage, float)
    
    def test_get_failed_lines_from_nodes(self):
        """Test failed lines calculation."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        failed_nodes = {1}
        
        failed_lines = get_failed_lines_from_nodes(edge_index, failed_nodes)
        
        # Lines 0 and 1 connect to node 1
        assert 0 in failed_lines
        assert 1 in failed_lines
        assert 2 not in failed_lines
    
    def test_split_scenarios(self):
        """Test scenario splitting."""
        scenarios = [{'id': i} for i in range(100)]
        
        splits = split_scenarios(scenarios, train_split=0.7, val_split=0.15, test_split=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        assert len(splits['train']) == 70
        assert len(splits['val']) == 15
        assert len(splits['test']) == 15
    
    def test_validate_scenario_valid(self):
        """Test scenario validation with valid scenario."""
        scenario = {
            'sequence': [
                {
                    'scada_data': np.random.randn(30, 14).astype(np.float32),
                    'pmu_sequence': np.random.randn(30, 8).astype(np.float32),
                    'satellite_data': np.random.randn(30, 12, 16, 16).astype(np.float32),
                    'weather_sequence': np.random.randn(30, 10, 8).astype(np.float32),
                    'node_labels': np.zeros(30, dtype=np.float32),
                }
            ],
            'edge_index': np.random.randint(0, 30, (2, 50)),
            'metadata': {'is_cascade': False}
        }
        
        assert validate_scenario(scenario) == True
    
    def test_validate_scenario_invalid(self):
        """Test scenario validation with invalid scenario."""
        scenario = {
            'sequence': [],  # Empty sequence
            'edge_index': np.array([]),
            'metadata': {}
        }
        
        assert validate_scenario(scenario) == False
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_save_and_load_topology(self, temp_dir):
        """Test topology save and load."""
        adj = np.random.randint(0, 2, (30, 30))
        edge_index = np.random.randint(0, 30, (2, 50))
        positions = np.random.randn(30, 2)
        
        topo_file = temp_dir / "topology.pkl"
        
        save_topology(adj, edge_index, positions, str(topo_file))
        
        assert topo_file.exists()
        
        loaded = load_topology(str(topo_file))
        
        assert loaded is not None
        assert 'adjacency_matrix' in loaded
        assert np.array_equal(loaded['adjacency_matrix'], adj)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
