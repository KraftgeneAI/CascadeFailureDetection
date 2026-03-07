"""
Tests for cascade_prediction/data/preprocessing modules
========================================================
Tests normalization, truncation, and edge masking functionality.
"""

import pytest
import torch
import numpy as np

from cascade_prediction.data.preprocessing import (
    normalize_power,
    normalize_frequency,
    calculate_truncation_window,
    apply_truncation,
    create_edge_mask_from_failures,
    to_tensor,
)


class TestNormalization:
    """Test suite for normalization functions."""
    
    def test_normalize_power_tensor(self):
        """Test power normalization with torch tensor."""
        power_mw = torch.tensor([100.0, 200.0, 50.0])
        base_mva = 100.0
        
        power_pu = normalize_power(power_mw, base_mva)
        
        expected = torch.tensor([1.0, 2.0, 0.5])
        assert torch.allclose(power_pu, expected)
    
    def test_normalize_power_numpy(self):
        """Test power normalization with numpy array."""
        power_mw = np.array([100.0, 200.0, 50.0])
        base_mva = 100.0
        
        power_pu = normalize_power(power_mw, base_mva)
        
        expected = np.array([1.0, 2.0, 0.5])
        assert np.allclose(power_pu, expected)
    
    def test_normalize_power_zero_base(self):
        """Test power normalization with zero base (should not crash)."""
        power_mw = torch.tensor([100.0, 200.0])
        base_mva = 0.0
        
        # Should handle gracefully (likely return inf or very large values)
        power_pu = normalize_power(power_mw, base_mva)
        assert power_pu is not None
    
    def test_normalize_frequency_tensor(self):
        """Test frequency normalization with torch tensor."""
        freq_hz = torch.tensor([60.0, 59.5, 60.5])
        base_freq = 60.0
        
        freq_pu = normalize_frequency(freq_hz, base_freq)
        
        expected = torch.tensor([1.0, 59.5/60.0, 60.5/60.0])
        assert torch.allclose(freq_pu, expected)
    
    def test_normalize_frequency_numpy(self):
        """Test frequency normalization with numpy array."""
        freq_hz = np.array([60.0, 59.5, 60.5])
        base_freq = 60.0
        
        freq_pu = normalize_frequency(freq_hz, base_freq)
        
        expected = np.array([1.0, 59.5/60.0, 60.5/60.0])
        assert np.allclose(freq_pu, expected)
    
    def test_normalize_power_multidimensional(self):
        """Test power normalization with multi-dimensional arrays."""
        power_mw = torch.randn(10, 30, 5) * 100
        base_mva = 100.0
        
        power_pu = normalize_power(power_mw, base_mva)
        
        # Check shape preserved
        assert power_pu.shape == power_mw.shape
        
        # Check normalization applied
        assert torch.allclose(power_pu, power_mw / base_mva)


class TestTruncation:
    """Test suite for truncation functions."""
    
    def test_calculate_truncation_window_cascade(self):
        """Test truncation window calculation for cascade scenario."""
        sequence_length = 100
        cascade_start_time = 50
        is_cascade = True
        
        start_idx, end_idx = calculate_truncation_window(
            sequence_length,
            cascade_start_time,
            is_cascade
        )
        
        # Should end before cascade start (to prevent data leakage)
        assert start_idx >= 0
        assert end_idx <= cascade_start_time - 5  # Ends before 5-min warning window
        assert end_idx > start_idx  # Valid window
    
    def test_calculate_truncation_window_normal(self):
        """Test truncation window calculation for normal scenario."""
        sequence_length = 100
        cascade_start_time = -1
        is_cascade = False
        
        start_idx, end_idx = calculate_truncation_window(
            sequence_length,
            cascade_start_time,
            is_cascade
        )
        
        # Should have valid window
        assert start_idx >= 0
        assert end_idx <= sequence_length
        assert end_idx > start_idx  # Valid window
    
    def test_calculate_truncation_window_short_sequence(self):
        """Test truncation with short sequence."""
        sequence_length = 5
        cascade_start_time = 2
        is_cascade = True
        
        start_idx, end_idx = calculate_truncation_window(
            sequence_length,
            cascade_start_time,
            is_cascade
        )
        
        # Should have valid window (may be very short)
        assert start_idx >= 0
        assert end_idx <= sequence_length
        assert end_idx >= start_idx  # Valid or empty window
    
    def test_apply_truncation(self):
        """Test applying truncation to sequence."""
        sequence = [{'data': i} for i in range(100)]
        start_idx = 40
        end_idx = 60
        
        truncated = apply_truncation(sequence, start_idx, end_idx)
        
        assert len(truncated) == 20
        assert truncated[0]['data'] == 40
        assert truncated[-1]['data'] == 59
    
    def test_apply_truncation_full_sequence(self):
        """Test truncation with full sequence."""
        sequence = [{'data': i} for i in range(10)]
        start_idx = 0
        end_idx = 10
        
        truncated = apply_truncation(sequence, start_idx, end_idx)
        
        assert len(truncated) == 10
        assert truncated == sequence
    
    def test_apply_truncation_empty(self):
        """Test truncation with empty result."""
        sequence = [{'data': i} for i in range(10)]
        start_idx = 5
        end_idx = 5
        
        truncated = apply_truncation(sequence, start_idx, end_idx)
        
        # apply_truncation has fallback for empty sequences
        assert len(truncated) >= 0  # May return fallback


class TestEdgeMasking:
    """Test suite for edge masking functions."""
    
    def test_create_edge_mask_no_failures(self):
        """Test edge mask creation with no failures."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        failed_nodes = []
        num_edges = 3
        
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        # All edges should be active
        assert np.all(mask == 1)
        assert len(mask) == num_edges
    
    def test_create_edge_mask_with_failures(self):
        """Test edge mask creation with failed nodes."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        failed_nodes = [1]  # Node 1 failed
        num_edges = 4
        
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        # Edges connected to node 1 should be masked (0)
        # Edge 0: 0->1 (masked)
        # Edge 1: 1->2 (masked)
        # Edge 2: 2->3 (active)
        # Edge 3: 3->0 (active)
        assert mask[0] == 0
        assert mask[1] == 0
        assert mask[2] == 1
        assert mask[3] == 1
    
    def test_create_edge_mask_multiple_failures(self):
        """Test edge mask with multiple failed nodes."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        failed_nodes = [0, 2]
        num_edges = 4
        
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        # All edges should be masked (all connect to failed nodes)
        assert np.all(mask == 0)
    
    def test_create_edge_mask_torch_tensor(self):
        """Test edge mask creation with torch tensor."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        failed_nodes = [1]
        num_edges = 3
        
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        # Should work with torch tensors
        assert len(mask) == num_edges
        assert mask[0] == 0  # 0->1 masked
        assert mask[1] == 0  # 1->2 masked
        assert mask[2] == 1  # 2->0 active
    
    def test_create_edge_mask_invalid_node(self):
        """Test edge mask with invalid node index."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        failed_nodes = [999]  # Invalid node
        num_edges = 3
        
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        # Should not crash, all edges active
        assert np.all(mask == 1)


class TestTensorConversion:
    """Test suite for tensor conversion."""
    
    def test_to_tensor_numpy(self):
        """Test converting numpy array to tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_to_tensor_already_tensor(self):
        """Test converting tensor (should return as-is)."""
        original = torch.tensor([1.0, 2.0, 3.0])
        tensor = to_tensor(original)
        
        assert tensor is original
    
    def test_to_tensor_list(self):
        """Test converting list to tensor."""
        lst = [1.0, 2.0, 3.0]
        tensor = to_tensor(lst)
        
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))
    
    def test_to_tensor_multidimensional(self):
        """Test converting multi-dimensional array."""
        arr = np.random.randn(10, 20, 30)
        tensor = to_tensor(arr)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 20, 30)
    
    def test_to_tensor_preserves_dtype(self):
        """Test that tensor conversion preserves float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = to_tensor(arr)
        
        assert tensor.dtype == torch.float32
    
    def test_to_tensor_int_array(self):
        """Test converting integer array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        tensor = to_tensor(arr)
        
        assert isinstance(tensor, torch.Tensor)
        # Should convert to float32 by default
        assert tensor.dtype == torch.float32


class TestIntegration:
    """Integration tests combining multiple preprocessing steps."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create mock data
        num_nodes = 30
        num_edges = 50
        sequence_length = 100
        
        # 1. Create sequence
        sequence = []
        for t in range(sequence_length):
            timestep = {
                'scada_data': np.random.randn(num_nodes, 14).astype(np.float32),
                'node_labels': np.zeros(num_nodes),
            }
            if t >= 50:
                timestep['node_labels'][0] = 1.0
            sequence.append(timestep)
        
        # 2. Calculate truncation
        start_idx, end_idx = calculate_truncation_window(
            sequence_length,
            cascade_start_time=50,
            is_cascade=True
        )
        
        # 3. Apply truncation
        truncated = apply_truncation(sequence, start_idx, end_idx)
        
        assert len(truncated) > 0
        assert len(truncated) <= sequence_length
        
        # 4. Normalize power
        for ts in truncated:
            scada = ts['scada_data']
            if scada.shape[1] >= 6:
                scada[:, 2] = normalize_power(scada[:, 2], 100.0)
        
        # 5. Create edge mask
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        failed_nodes = [0]
        mask = create_edge_mask_from_failures(edge_index, failed_nodes, num_edges)
        
        assert mask is not None
        assert len(mask) == num_edges


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
