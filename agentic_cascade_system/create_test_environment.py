"""
Create Test Environment for Agentic System
==========================================
Generates synthetic topology and data files needed to run the system.

This script creates:
1. Grid topology file (grid_topology.pkl)
2. Test model checkpoint (mock, for demo purposes)

Run this FIRST before running main.py!

Usage:
    python create_test_environment.py --data_dir ./data --num_nodes 50
"""

import pickle
import numpy as np
import torch
import argparse
from pathlib import Path
import sys


def create_grid_topology(num_nodes: int = 50, connectivity: float = 0.1) -> dict:
    """
    Create a synthetic power grid topology.
    
    Args:
        num_nodes: Number of nodes (substations/buses)
        connectivity: Edge probability (0.1 = ~10% of possible edges)
    
    Returns:
        Dictionary containing edge_index and node metadata
    """
    print(f"Creating grid topology with {num_nodes} nodes...")
    
    # Create random edges (bidirectional)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < connectivity:
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
    
    # Ensure connectivity - add edges to create spanning tree
    for i in range(1, num_nodes):
        parent = np.random.randint(0, i)
        if [parent, i] not in edges:
            edges.append([parent, i])
            edges.append([i, parent])
    
    edge_index = np.array(edges, dtype=np.int64).T  # Shape: (2, num_edges)
    num_edges = edge_index.shape[1]
    
    print(f"Created {num_edges} edges")
    
    # Node metadata
    node_types = np.random.choice(['substation', 'generator', 'load', 'transformer'], num_nodes)
    node_voltages = np.random.choice([115, 230, 345, 500], num_nodes)  # kV
    node_coords = np.random.randn(num_nodes, 2) * 100  # Geographic coordinates
    
    topology = {
        'edge_index': edge_index,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'node_types': node_types.tolist(),
        'node_voltages': node_voltages.tolist(),
        'node_coordinates': node_coords.tolist(),
        'metadata': {
            'created': str(np.datetime64('now')),
            'version': '1.0',
            'description': 'Synthetic test topology for agentic system'
        }
    }
    
    return topology


def create_mock_model_checkpoint(num_nodes: int, save_path: str):
    """
    Create a mock model checkpoint for testing.
    
    In production, this would be your trained UnifiedCascadePredictionModel.
    For testing the agent system, we create a minimal checkpoint.
    """
    print(f"Creating mock model checkpoint at {save_path}...")
    
    # Model configuration matching the real model architecture
    config = {
        'num_nodes': num_nodes,
        'scada_dim': 13,
        'pmu_dim': 7,
        'satellite_dim': 64,
        'weather_dim': 8,
        'threat_dim': 6,
        'equipment_dim': 10,
        'visual_dim': 64,
        'thermal_dim': 64,
        'sensor_dim': 8,
        'edge_dim': 4,
        'hidden_dim': 256,
        'embed_dim': 128,
        'num_gat_layers': 4,
        'num_heads': 8,
        'num_lstm_layers': 3,
        'dropout': 0.3
    }
    
    checkpoint = {
        'model_config': config,
        'model_state_dict': {},  # Empty - prediction agent will handle gracefully
        'epoch': 0,
        'is_mock': True,  # Flag to indicate this is for testing
        'metadata': {
            'description': 'Mock checkpoint for testing agentic system',
            'note': 'Replace with real trained model for production use'
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved mock checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Create test environment for agentic system")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--num_nodes", type=int, default=50, help="Number of grid nodes")
    parser.add_argument("--connectivity", type=float, default=0.1, help="Edge connectivity probability")
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints directory
    checkpoints_dir = data_dir.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CREATING TEST ENVIRONMENT")
    print("=" * 60)
    
    # Create topology
    topology = create_grid_topology(args.num_nodes, args.connectivity)
    topology_path = data_dir / "grid_topology.pkl"
    
    with open(topology_path, 'wb') as f:
        pickle.dump(topology, f)
    print(f"Saved topology to {topology_path}")
    
    # Create mock checkpoint
    checkpoint_path = checkpoints_dir / "test_model.pth"
    create_mock_model_checkpoint(args.num_nodes, str(checkpoint_path))
    
    print("\n" + "=" * 60)
    print("TEST ENVIRONMENT READY!")
    print("=" * 60)
    print(f"\nTopology: {topology_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"\nTo run the system:")
    print(f"  python main.py --model_path {checkpoint_path} --data_dir {data_dir} --duration 30")
    print("\nNote: The mock checkpoint will generate random predictions.")
    print("Replace with your trained model for real predictions.")


if __name__ == "__main__":
    main()
