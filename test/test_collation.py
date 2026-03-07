"""
Tests for cascade_prediction/data/collation.py
===============================================
Tests the batch collation functionality.
"""

import pytest
import torch
import numpy as np

from cascade_prediction.data.collation import collate_cascade_batch


class TestCollation:
    """Test suite for collation functions."""
    
    @pytest.fixture
    def sample_item(self):
        """Create a sample data item."""
        num_nodes = 30
        num_edges = 50
        seq_len = 10
        
        return {
            'scada_data': torch.randn(seq_len, num_nodes, 12),
            'pmu_sequence': torch.randn(seq_len, num_nodes, 8),
            'weather_sequence': torch.randn(seq_len, num_nodes, 80),
            'threat_indicators': torch.randn(seq_len, num_nodes, 6),
            'satellite_data': torch.randn(seq_len, num_nodes, 12, 16, 16),
            'visual_data': torch.randn(seq_len, num_nodes, 3, 32, 32),
            'thermal_data': torch.randn(seq_len, num_nodes, 1, 32, 32),
            'sensor_data': torch.randn(seq_len, num_nodes, 12),
            'equipment_status': torch.randn(seq_len, num_nodes, 10),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_attr': torch.randn(num_edges, 7),
            'edge_mask': torch.ones(seq_len, num_edges),
            'node_failure_labels': torch.zeros(num_nodes),
            'cascade_timing': torch.full((num_nodes,), -1.0),
            'ground_truth_risk': torch.randn(7),
            'temporal_sequence': torch.randn(seq_len, num_nodes, 12),
            'sequence_length': seq_len,
            'graph_properties': {
                'thermal_limits': torch.randn(num_edges),
                'conductance': torch.randn(num_edges),
                'susceptance': torch.randn(num_edges),
                'power_injection': torch.randn(num_nodes),
            }
        }
    
    def test_collate_single_item(self, sample_item):
        """Test collating a single item."""
        batch = [sample_item]
        collated = collate_cascade_batch(batch)
        
        assert 'scada_data' in collated
        assert 'edge_index' in collated
        assert 'node_failure_labels' in collated
        
        # Check batch dimension added
        assert collated['scada_data'].shape[0] == 1
    
    def test_collate_multiple_items(self, sample_item):
        """Test collating multiple items."""
        batch_size = 4
        batch = [sample_item for _ in range(batch_size)]
        collated = collate_cascade_batch(batch)
        
        # Check batch dimension
        assert collated['scada_data'].shape[0] == batch_size
        assert collated['node_failure_labels'].shape[0] == batch_size
    
    def test_collate_empty_batch(self):
        """Test collating empty batch."""
        batch = []
        collated = collate_cascade_batch(batch)
        
        assert len(collated) == 0
    
    def test_collate_with_empty_dicts(self, sample_item):
        """Test collating batch with empty dictionaries."""
        batch = [sample_item, {}, sample_item]
        collated = collate_cascade_batch(batch)
        
        # Should filter out empty dicts
        assert collated['scada_data'].shape[0] == 2
    
    def test_collate_variable_length_sequences(self):
        """Test collating sequences of different lengths."""
        num_nodes = 30
        num_edges = 50
        
        item1 = {
            'scada_data': torch.randn(5, num_nodes, 12),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_attr': torch.randn(num_edges, 7),
            'edge_mask': torch.ones(5, num_edges),
            'node_failure_labels': torch.zeros(num_nodes),
            'temporal_sequence': torch.randn(5, num_nodes, 12),
            'sequence_length': 5,
            'graph_properties': {}
        }
        
        item2 = {
            'scada_data': torch.randn(10, num_nodes, 12),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_attr': torch.randn(num_edges, 7),
            'edge_mask': torch.ones(10, num_edges),
            'node_failure_labels': torch.zeros(num_nodes),
            'temporal_sequence': torch.randn(10, num_nodes, 12),
            'sequence_length': 10,
            'graph_properties': {}
        }
        
        batch = [item1, item2]
        collated = collate_cascade_batch(batch)
        
        # Should pad to max length (10)
        assert collated['scada_data'].shape[1] == 10
        assert collated['temporal_sequence'].shape[1] == 10
        
        # Check sequence lengths preserved
        assert collated['sequence_length'][0] == 5
        assert collated['sequence_length'][1] == 10
    
    def test_collate_edge_index_shared(self, sample_item):
        """Test that edge_index is shared (not batched)."""
        batch = [sample_item, sample_item]
        collated = collate_cascade_batch(batch)
        
        # Edge index should not have batch dimension
        assert collated['edge_index'].dim() == 2
    
    def test_collate_graph_properties(self):
        """Test collating graph properties."""
        num_nodes = 30
        num_edges = 50
        
        item1 = {
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'node_failure_labels': torch.zeros(num_nodes),
            'graph_properties': {
                'thermal_limits': torch.randn(num_edges),
                'power_injection': torch.randn(num_nodes),
            }
        }
        
        item2 = {
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'node_failure_labels': torch.zeros(num_nodes),
            'graph_properties': {
                'thermal_limits': torch.randn(num_edges),
                'power_injection': torch.randn(num_nodes),
            }
        }
        
        batch = [item1, item2]
        collated = collate_cascade_batch(batch)
        
        assert 'graph_properties' in collated
        assert 'thermal_limits' in collated['graph_properties']
        assert 'power_injection' in collated['graph_properties']
        
        # Should be batched
        assert collated['graph_properties']['thermal_limits'].shape[0] == 2
    
    def test_collate_numpy_arrays(self):
        """Test collating numpy arrays."""
        num_nodes = 30
        
        item = {
            'edge_index': np.random.randint(0, num_nodes, (2, 50)),
            'node_failure_labels': np.zeros(num_nodes),
            'graph_properties': {}
        }
        
        batch = [item, item]
        collated = collate_cascade_batch(batch)
        
        # Should convert to tensors
        assert isinstance(collated['edge_index'], torch.Tensor)
        assert isinstance(collated['node_failure_labels'], torch.Tensor)
    
    def test_collate_preserves_dtypes(self, sample_item):
        """Test that collation preserves data types."""
        batch = [sample_item]
        collated = collate_cascade_batch(batch)
        
        # Float tensors should remain float
        assert collated['scada_data'].dtype == torch.float32
        
        # Edge index should be long
        assert collated['edge_index'].dtype == torch.int64
    
    def test_collate_4d_tensors(self):
        """Test collating 4D tensors (images)."""
        num_nodes = 30
        
        item = {
            'satellite_data': torch.randn(5, num_nodes, 12, 16, 16),
            'visual_data': torch.randn(5, num_nodes, 3, 32, 32),
            'edge_index': torch.randint(0, num_nodes, (2, 50)),
            'node_failure_labels': torch.zeros(num_nodes),
            'graph_properties': {}
        }
        
        batch = [item, item]
        collated = collate_cascade_batch(batch)
        
        # Should handle 4D tensors - adds batch dimension
        # Input: [T, N, C, H, W] -> Output: [B, T, N, C, H, W]
        assert collated['satellite_data'].dim() == 6  # [B, T, N, C, H, W]
        assert collated['visual_data'].dim() == 6
    
    def test_collate_edge_mask_padding(self):
        """Test edge mask padding for variable lengths."""
        num_nodes = 30
        num_edges = 50
        
        item1 = {
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_mask': torch.ones(5, num_edges),
            'node_failure_labels': torch.zeros(num_nodes),
            'graph_properties': {}
        }
        
        item2 = {
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_mask': torch.ones(10, num_edges),
            'node_failure_labels': torch.zeros(num_nodes),
            'graph_properties': {}
        }
        
        batch = [item1, item2]
        collated = collate_cascade_batch(batch)
        
        # Should pad to max length
        assert collated['edge_mask'].shape[1] == 10
        
        # Padded values should be zero
        assert torch.all(collated['edge_mask'][0, 5:] == 0)
    
    def test_collate_missing_keys(self):
        """Test collating when some items have missing keys."""
        item1 = {
            'edge_index': torch.randint(0, 30, (2, 50)),
            'node_failure_labels': torch.zeros(30),
            'scada_data': torch.randn(5, 30, 12),
            'graph_properties': {}
        }
        
        item2 = {
            'edge_index': torch.randint(0, 30, (2, 50)),
            'node_failure_labels': torch.zeros(30),
            # Missing scada_data
            'graph_properties': {}
        }
        
        batch = [item1, item2]
        
        # Should handle gracefully (may skip missing keys)
        collated = collate_cascade_batch(batch)
        
        # At minimum, common keys should be present
        assert 'edge_index' in collated
        assert 'node_failure_labels' in collated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
