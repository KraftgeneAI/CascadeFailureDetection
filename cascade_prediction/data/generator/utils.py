"""
Generator Utilities
===================
Utility functions for data generation.
"""

import psutil
import warnings
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class MemoryMonitor:
    """
    Monitor memory usage to prevent Out-Of-Memory (OOM) errors.
    
    The data generator creates large arrays (satellite images, thermal maps, etc.)
    that can consume significant memory. This class helps track usage and warn
    before running out of memory.
    """
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_threshold(threshold_mb: float = 8000) -> bool:
        """
        Check if memory usage exceeds threshold.
        
        Args:
            threshold_mb: Memory threshold in MB
            
        Returns:
            True if threshold exceeded, False otherwise
        """
        current = MemoryMonitor.get_memory_usage()
        if current > threshold_mb:
            warnings.warn(f"High memory usage: {current:.1f} MB")
            return True
        return False


def save_scenarios(
    scenarios: List[Dict],
    output_path: str,
    batch_idx: int
):
    """
    Save scenarios to pickle file.
    
    Args:
        scenarios: List of scenario dictionaries
        output_path: Output directory path
        batch_idx: Batch index for filename
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"scenarios_batch_{batch_idx}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(scenarios, f)
    
    print(f"Saved batch {batch_idx}: {len(scenarios)} scenarios → {filename}")


def load_topology(topology_file: str) -> Optional[Dict]:
    """
    Load grid topology from file.
    
    Args:
        topology_file: Path to topology pickle file
        
    Returns:
        Topology dictionary or None if file doesn't exist
    """
    if not Path(topology_file).exists():
        return None
    
    with open(topology_file, 'rb') as f:
        topology = pickle.load(f)
    
    return topology


def save_topology(
    adjacency_matrix: np.ndarray,
    edge_index: np.ndarray,
    positions: np.ndarray,
    output_path: str
):
    """
    Save grid topology to file.
    
    Args:
        adjacency_matrix: Adjacency matrix
        edge_index: Edge index array
        positions: Node positions
        output_path: Output file path
    """
    topology = {
        'adjacency_matrix': adjacency_matrix,
        'edge_index': edge_index,
        'positions': positions,
        'num_nodes': adjacency_matrix.shape[0],
        'num_edges': edge_index.shape[1]
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(topology, f)
    
    print(f"Saved topology: {topology['num_nodes']} nodes, {topology['num_edges']} edges → {output_path}")


def split_scenarios(
    scenarios: List[Dict],
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Split scenarios into train/val/test sets.
    
    Args:
        scenarios: List of all scenarios
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_split + val_split + test_split - 1.0) < 0.01, \
        "Splits must sum to 1.0"
    
    # Shuffle scenarios
    np.random.seed(seed)
    indices = np.random.permutation(len(scenarios))
    
    # Calculate split points
    n_train = int(len(scenarios) * train_split)
    n_val = int(len(scenarios) * val_split)
    
    # Split
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': [scenarios[i] for i in train_indices],
        'val': [scenarios[i] for i in val_indices],
        'test': [scenarios[i] for i in test_indices]
    }


def validate_scenario(scenario: Dict) -> bool:
    """
    Validate scenario structure and data quality.
    
    Args:
        scenario: Scenario dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required keys
        required_keys = ['sequence', 'edge_index', 'metadata']
        for key in required_keys:
            if key not in scenario:
                print(f"Missing key: {key}")
                return False
        
        # Check sequence
        if not scenario['sequence']:
            print("Empty sequence")
            return False
        
        # Check first timestep
        timestep = scenario['sequence'][0]
        required_timestep_keys = [
            'scada_data', 'pmu_sequence', 'satellite_data',
            'weather_sequence', 'node_labels'
        ]
        for key in required_timestep_keys:
            if key not in timestep:
                print(f"Missing timestep key: {key}")
                return False
        
        # Check for NaN/Inf
        scada = timestep['scada_data']
        if np.any(np.isnan(scada)) or np.any(np.isinf(scada)):
            print("NaN or Inf in SCADA data")
            return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False
