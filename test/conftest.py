"""
Shared pytest fixtures for cascade_prediction tests
====================================================
Provides common fixtures and utilities for all test modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import pickle
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_grid_topology():
    """Create a mock grid topology for testing."""
    num_nodes = 30
    num_edges = 50
    
    # Create adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    
    # Add some cross connections
    for _ in range(20):
        i, j = np.random.choice(num_nodes, 2, replace=False)
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Create edge index
    edges = np.where(adj > 0)
    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
    
    # Create positions
    positions = np.random.randn(num_nodes, 2) * 50
    
    return {
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'adjacency_matrix': adj,
        'edge_index': edge_index,
        'positions': positions
    }


@pytest.fixture
def mock_node_properties():
    """Create mock node properties for testing."""
    num_nodes = 30
    
    return {
        'node_types': np.random.randint(0, 3, num_nodes),
        'gen_capacity': np.random.uniform(0, 200, num_nodes),
        'base_load': np.random.uniform(10, 100, num_nodes),
        'equipment_age': np.random.uniform(0, 40, num_nodes),
        'equipment_condition': np.random.uniform(0.6, 1.0, num_nodes),
        'loading_failure_threshold': np.random.uniform(1.05, 1.15, num_nodes),
        'loading_damage_threshold': np.random.uniform(0.95, 1.05, num_nodes),
        'voltage_failure_threshold': np.random.uniform(0.88, 0.92, num_nodes),
        'voltage_damage_threshold': np.random.uniform(0.92, 0.95, num_nodes),
        'temperature_failure_threshold': np.random.uniform(85, 95, num_nodes),
        'temperature_damage_threshold': np.random.uniform(70, 80, num_nodes),
        'frequency_failure_threshold': np.random.uniform(58.5, 59.2, num_nodes),
        'frequency_damage_threshold': np.random.uniform(59.2, 59.7, num_nodes),
        'thermal_capacity': np.random.uniform(0.8, 1.2, num_nodes),
        'cooling_effectiveness': np.random.uniform(0.7, 1.0, num_nodes),
        'thermal_time_constant': np.random.uniform(10, 30, num_nodes),
    }


@pytest.fixture
def mock_timestep_data():
    """Create mock data for a single timestep."""
    num_nodes = 30
    num_edges = 50
    
    return {
        'scada_data': np.random.randn(num_nodes, 14).astype(np.float32),
        'pmu_sequence': np.random.randn(num_nodes, 8).astype(np.float32),
        'weather_sequence': np.random.randn(num_nodes, 10, 8).astype(np.float32),
        'satellite_data': np.random.randn(num_nodes, 12, 16, 16).astype(np.float16),
        'threat_indicators': np.random.randn(num_nodes, 6).astype(np.float16),
        'visual_data': np.random.randn(num_nodes, 3, 32, 32).astype(np.float16),
        'thermal_data': np.random.randn(num_nodes, 1, 32, 32).astype(np.float16),
        'sensor_data': np.random.randn(num_nodes, 12).astype(np.float16),
        'equipment_status': np.random.randn(num_nodes, 10).astype(np.float32),
        'edge_attr': np.random.randn(num_edges, 7).astype(np.float32),
        'node_labels': np.zeros(num_nodes, dtype=np.float32),
        'cascade_timing': np.full(num_nodes, -1, dtype=np.float32),
        'conductance': np.random.randn(num_edges).astype(np.float32),
        'susceptance': np.random.randn(num_edges).astype(np.float32),
        'thermal_limits': np.random.uniform(50, 150, num_edges).astype(np.float32),
        'power_injection': np.random.randn(num_nodes).astype(np.float32),
        'reactive_injection': np.random.randn(num_nodes).astype(np.float32),
    }


@pytest.fixture
def mock_cascade_scenario(mock_grid_topology, mock_timestep_data):
    """Create a complete mock cascade scenario."""
    sequence_length = 10
    cascade_start = 5
    
    sequence = []
    for t in range(sequence_length):
        timestep = mock_timestep_data.copy()
        
        # Add failures after cascade start
        if t >= cascade_start:
            timestep['node_labels'][0] = 1.0
            timestep['cascade_timing'][0] = float(t - cascade_start)
        
        sequence.append(timestep)
    
    return {
        'sequence': sequence,
        'edge_index': mock_grid_topology['edge_index'].numpy(),
        'metadata': {
            'is_cascade': True,
            'cascade_start_time': cascade_start,
            'num_nodes': mock_grid_topology['num_nodes'],
            'num_edges': mock_grid_topology['num_edges'],
            'ground_truth_risk': np.random.rand(7).astype(np.float32),
            'base_mva': 100.0,
            'failed_nodes': [0],
            'failure_times': [0.0],
            'failure_reasons': ['overload'],
            'stress_level': 0.9,
        }
    }


@pytest.fixture
def mock_normal_scenario(mock_grid_topology, mock_timestep_data):
    """Create a complete mock normal (non-cascade) scenario."""
    sequence_length = 10
    
    sequence = []
    for t in range(sequence_length):
        timestep = mock_timestep_data.copy()
        # No failures
        sequence.append(timestep)
    
    return {
        'sequence': sequence,
        'edge_index': mock_grid_topology['edge_index'].numpy(),
        'metadata': {
            'is_cascade': False,
            'cascade_start_time': -1,
            'num_nodes': mock_grid_topology['num_nodes'],
            'num_edges': mock_grid_topology['num_edges'],
            'ground_truth_risk': np.zeros(7, dtype=np.float32),
            'base_mva': 100.0,
            'failed_nodes': [],
            'failure_times': [],
            'failure_reasons': [],
            'stress_level': 0.5,
        }
    }


@pytest.fixture
def mock_scenario_file(temp_dir, mock_cascade_scenario):
    """Create a mock scenario pickle file."""
    scenario_file = temp_dir / "scenario_0.pkl"
    with open(scenario_file, 'wb') as f:
        pickle.dump(mock_cascade_scenario, f)
    return scenario_file


@pytest.fixture
def mock_data_dir(temp_dir, mock_cascade_scenario, mock_normal_scenario):
    """Create a mock data directory with multiple scenarios."""
    # Save cascade scenario
    cascade_file = temp_dir / "scenario_0.pkl"
    with open(cascade_file, 'wb') as f:
        pickle.dump(mock_cascade_scenario, f)
    
    # Save normal scenario
    normal_file = temp_dir / "scenario_1.pkl"
    with open(normal_file, 'wb') as f:
        pickle.dump(mock_normal_scenario, f)
    
    return temp_dir


@pytest.fixture
def simple_3node_grid():
    """Create a simple 3-node grid for basic testing."""
    return {
        'num_nodes': 3,
        'edge_index': np.array([[0, 1], [1, 2]]),
        'positions': np.array([[0, 0], [1, 0], [2, 0]]),
        'node_types': np.array([1, 0, 0]),  # Gen, Load, Load
        'gen_capacity': np.array([200.0, 0.0, 0.0]),
        'base_load': np.array([0.0, 50.0, 50.0]),
        'line_reactance': np.array([0.01, 0.01]),
        'line_resistance': np.array([0.001, 0.001]),
        'line_susceptance': np.array([1e-5, 1e-5]),
        'line_conductance': np.array([0.0, 0.0]),
        'thermal_limits': np.array([100.0, 100.0]),
    }


def assert_tensor_shape(tensor, expected_shape, name="tensor"):
    """Helper function to assert tensor shape."""
    assert tensor.shape == expected_shape, \
        f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"


def assert_tensor_dtype(tensor, expected_dtype, name="tensor"):
    """Helper function to assert tensor dtype."""
    assert tensor.dtype == expected_dtype, \
        f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"


def assert_tensor_range(tensor, min_val, max_val, name="tensor"):
    """Helper function to assert tensor value range."""
    assert torch.all(tensor >= min_val) and torch.all(tensor <= max_val), \
        f"{name} values out of range [{min_val}, {max_val}]: min={tensor.min()}, max={tensor.max()}"


def assert_numpy_shape(array, expected_shape, name="array"):
    """Helper function to assert numpy array shape."""
    assert array.shape == expected_shape, \
        f"{name} shape mismatch: expected {expected_shape}, got {array.shape}"


def assert_numpy_dtype(array, expected_dtype, name="array"):
    """Helper function to assert numpy array dtype."""
    assert array.dtype == expected_dtype, \
        f"{name} dtype mismatch: expected {expected_dtype}, got {array.dtype}"


def assert_numpy_range(array, min_val, max_val, name="array"):
    """Helper function to assert numpy array value range."""
    assert np.all(array >= min_val) and np.all(array <= max_val), \
        f"{name} values out of range [{min_val}, {max_val}]: min={array.min()}, max={array.max()}"


# Register custom markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
