"""
Data Generator Package
=====================
Modular components for generating synthetic cascade failure scenarios.

Components:
- topology: Grid topology generation and node/edge initialization
- physics: Physics-based simulation (power flow, frequency, thermal)
- cascade: Cascade failure propagation logic
- environmental: Environmental data generation (satellite, weather, threats)
- robotic: Robotic sensor data generation (visual, thermal, sensors)
- simulator: Physics-based grid simulation orchestrator
- scenario: Scenario orchestration and batch generation
- utils: Utility functions for data generation
"""

from .utils import MemoryMonitor, save_scenarios, load_topology, save_topology
from .topology import GridTopologyGenerator, NodePropertyInitializer
from .physics import PowerFlowSimulator, FrequencyDynamicsSimulator, ThermalDynamicsSimulator
from .cascade import CascadeSimulator, create_adjacency_list
from .environmental import EnvironmentalDataGenerator
from .robotic import RoboticDataGenerator
from .simulator import PhysicsBasedGridSimulator
from .scenario import ScenarioOrchestrator, generate_dataset_from_config

__all__ = [
    # Topology
    'GridTopologyGenerator',
    'NodePropertyInitializer',
    # Physics
    'PowerFlowSimulator',
    'FrequencyDynamicsSimulator',
    'ThermalDynamicsSimulator',
    # Cascade
    'CascadeSimulator',
    'create_adjacency_list',
    # Environmental
    'EnvironmentalDataGenerator',
    # Robotic
    'RoboticDataGenerator',
    # Simulator
    'PhysicsBasedGridSimulator',
    # Scenario
    'ScenarioOrchestrator',
    'generate_dataset_from_config',
    # Utils
    'MemoryMonitor',
    'save_scenarios',
    'load_topology',
    'save_topology',
]
