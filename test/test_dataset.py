"""
Tests for cascade_prediction/data/dataset.py
=============================================
Tests the CascadeDataset class functionality.
"""

import pytest
import torch
import numpy as np
import pickle
import tempfile
import shutil
from pathlib import Path
import json

from cascade_prediction.data.dataset import CascadeDataset


class TestCascadeDataset:
    """Test suite for CascadeDataset class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with mock scenario files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_cascade_scenario(self):
        """Create a mock cascade scenario."""
        num_nodes = 30
        num_edges = 50
        sequence_length = 10
        
        # Create sequence
        sequence = []
        for t in range(sequence_length):
            timestep = {
                'scada_data': np.random.randn(num_nodes, 14).astype(np.float32),
                'pmu_sequence': np.random.randn(num_nodes, 8).astype(np.float32),
                'weather_sequence': np.random.randn(num_nodes, 10, 8).astype(np.float32),
                'satellite_data': np.random.randn(num_nodes, 12, 16, 16).astype(np.float32),
                'threat_indicators': np.random.randn(num_nodes, 6).astype(np.float32),
                'visual_data': np.random.randn(num_nodes, 3, 32, 32).astype(np.float32),
                'thermal_data': np.random.randn(num_nodes, 1, 32, 32).astype(np.float32),
                'sensor_data': np.random.randn(num_nodes, 12).astype(np.float32),
                'equipment_status': np.random.randn(num_nodes, 10).astype(np.float32),
                'edge_attr': np.random.randn(num_edges, 7).astype(np.float32),
                'node_labels': np.zeros(num_nodes, dtype=np.float32),
                'cascade_timing': np.full(num_nodes, -1, dtype=np.float32),
            }
            
            # Add failures after cascade start
            if t >= 5:
                timestep['node_labels'][0] = 1.0
                timestep['cascade_timing'][0] = float(t - 5)
            
            sequence.append(timestep)
        
        # Create scenario
        scenario = {
            'sequence': sequence,
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'metadata': {
                'is_cascade': True,
                'cascade_start_time': 5,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'ground_truth_risk': np.random.rand(7).astype(np.float32),
                'base_mva': 100.0,
            }
        }
        
        return scenario
    
    @pytest.fixture
    def mock_normal_scenario(self):
        """Create a mock normal (non-cascade) scenario."""
        num_nodes = 30
        num_edges = 50
        sequence_length = 10
        
        sequence = []
        for t in range(sequence_length):
            timestep = {
                'scada_data': np.random.randn(num_nodes, 14).astype(np.float32),
                'pmu_sequence': np.random.randn(num_nodes, 8).astype(np.float32),
                'weather_sequence': np.random.randn(num_nodes, 10, 8).astype(np.float32),
                'satellite_data': np.random.randn(num_nodes, 12, 16, 16).astype(np.float32),
                'threat_indicators': np.random.randn(num_nodes, 6).astype(np.float32),
                'visual_data': np.random.randn(num_nodes, 3, 32, 32).astype(np.float32),
                'thermal_data': np.random.randn(num_nodes, 1, 32, 32).astype(np.float32),
                'sensor_data': np.random.randn(num_nodes, 12).astype(np.float32),
                'equipment_status': np.random.randn(num_nodes, 10).astype(np.float32),
                'edge_attr': np.random.randn(num_edges, 7).astype(np.float32),
                'node_labels': np.zeros(num_nodes, dtype=np.float32),
                'cascade_timing': np.full(num_nodes, -1, dtype=np.float32),
            }
            sequence.append(timestep)
        
        scenario = {
            'sequence': sequence,
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)),
            'metadata': {
                'is_cascade': False,
                'cascade_start_time': -1,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'ground_truth_risk': np.zeros(7, dtype=np.float32),
                'base_mva': 100.0,
            }
        }
        
        return scenario
    
    def test_dataset_initialization(self, temp_data_dir, mock_cascade_scenario):
        """Test dataset initialization."""
        # Save a scenario file
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        # Initialize dataset
        dataset = CascadeDataset(
            data_dir=str(temp_data_dir),
            mode='full_sequence',
            base_mva=100.0,
            base_frequency=60.0
        )
        
        assert len(dataset) == 1
        assert dataset.cascade_labels[0] == True
    
    def test_dataset_length(self, temp_data_dir, mock_cascade_scenario, mock_normal_scenario):
        """Test dataset length calculation."""
        # Save multiple scenarios
        for i, scenario in enumerate([mock_cascade_scenario, mock_normal_scenario]):
            scenario_file = temp_data_dir / f"scenario_{i}.pkl"
            with open(scenario_file, 'wb') as f:
                pickle.dump(scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir))
        assert len(dataset) == 2
    
    def test_getitem_cascade(self, temp_data_dir, mock_cascade_scenario):
        """Test loading a cascade scenario."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir), mode='full_sequence')
        item = dataset[0]
        
        # Check required keys
        assert 'scada_data' in item
        assert 'edge_index' in item
        assert 'node_failure_labels' in item
        assert 'cascade_timing' in item
        assert 'graph_properties' in item
        
        # Check shapes
        assert item['scada_data'].dim() >= 2
        assert item['edge_index'].dim() == 2
        assert item['node_failure_labels'].dim() == 1
        
        # Check cascade label
        assert item['node_failure_labels'].sum() > 0
    
    def test_getitem_normal(self, temp_data_dir, mock_normal_scenario):
        """Test loading a normal scenario."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_normal_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir), mode='full_sequence')
        item = dataset[0]
        
        # Check no failures
        assert item['node_failure_labels'].sum() == 0
    
    def test_normalization(self, temp_data_dir, mock_cascade_scenario):
        """Test physics-based normalization."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(
            str(temp_data_dir),
            base_mva=100.0,
            base_frequency=60.0
        )
        item = dataset[0]
        
        # Check that power values are normalized (should be in reasonable range)
        scada_data = item['scada_data']
        if scada_data.shape[-1] >= 6:
            # Power values should be normalized (typically < 10 in p.u.)
            assert scada_data[..., 2:6].abs().max() < 100.0
    
    def test_edge_mask_creation(self, temp_data_dir, mock_cascade_scenario):
        """Test edge mask creation."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir), mode='full_sequence')
        item = dataset[0]
        
        assert 'edge_mask' in item
        assert item['edge_mask'].dim() >= 1
        # Edge mask should be binary
        assert torch.all((item['edge_mask'] == 0) | (item['edge_mask'] == 1))
    
    def test_graph_properties_extraction(self, temp_data_dir, mock_cascade_scenario):
        """Test graph properties extraction."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir))
        item = dataset[0]
        
        graph_props = item['graph_properties']
        
        # Check required properties
        assert 'thermal_limits' in graph_props
        assert 'conductance' in graph_props or 'susceptance' in graph_props
    
    def test_cache_creation(self, temp_data_dir, mock_cascade_scenario):
        """Test metadata cache creation."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        # First initialization creates cache
        dataset1 = CascadeDataset(str(temp_data_dir))
        
        cache_file = temp_data_dir / "metadata_cache.json"
        assert cache_file.exists()
        
        # Second initialization uses cache
        dataset2 = CascadeDataset(str(temp_data_dir))
        assert len(dataset2) == len(dataset1)
    
    def test_empty_directory(self, temp_data_dir):
        """Test handling of empty directory."""
        dataset = CascadeDataset(str(temp_data_dir))
        assert len(dataset) == 0
    
    def test_corrupted_file_handling(self, temp_data_dir):
        """Test handling of corrupted pickle files."""
        # Create a corrupted file
        corrupted_file = temp_data_dir / "scenario_0.pkl"
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid pickle file")
        
        # Should not crash
        dataset = CascadeDataset(str(temp_data_dir))
        assert len(dataset) == 1
        
        # Should return empty dict for corrupted file
        item = dataset[0]
        assert item == {} or len(item) == 0
    
    def test_get_cascade_label(self, temp_data_dir, mock_cascade_scenario, mock_normal_scenario):
        """Test get_cascade_label method."""
        for i, scenario in enumerate([mock_cascade_scenario, mock_normal_scenario]):
            scenario_file = temp_data_dir / f"scenario_{i}.pkl"
            with open(scenario_file, 'wb') as f:
                pickle.dump(scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir))
        
        assert dataset.get_cascade_label(0) == True
        assert dataset.get_cascade_label(1) == False
    
    def test_mode_full_sequence(self, temp_data_dir, mock_cascade_scenario):
        """Test full_sequence mode."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir), mode='full_sequence')
        item = dataset[0]
        
        # Should have temporal dimension
        assert 'temporal_sequence' in item
        assert item['temporal_sequence'].dim() >= 2
    
    def test_mode_last_timestep(self, temp_data_dir, mock_cascade_scenario):
        """Test last_timestep mode."""
        scenario_file = temp_data_dir / "scenario_0.pkl"
        with open(scenario_file, 'wb') as f:
            pickle.dump(mock_cascade_scenario, f)
        
        dataset = CascadeDataset(str(temp_data_dir), mode='last_timestep')
        item = dataset[0]
        
        # Should still work
        assert 'scada_data' in item
        assert 'node_failure_labels' in item


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
