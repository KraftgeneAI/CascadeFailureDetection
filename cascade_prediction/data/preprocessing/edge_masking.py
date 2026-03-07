"""
Edge Masking Module
===================
Dynamic topology masking for simulating line failures.

This module creates edge masks that zero out connections
from failed nodes, simulating the dynamic topology changes
during cascade propagation.
"""

import torch
import numpy as np
from typing import Union, List


def create_edge_mask_from_failures(
    edge_index: Union[torch.Tensor, np.ndarray],
    failed_node_indices: Union[List[int], np.ndarray],
    num_edges: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Create edge mask that zeros out edges connected to failed nodes.
    
    This implements dynamic topology masking by setting mask values to 0.0
    for edges where either the source or destination node has failed.
    
    Args:
        edge_index: Edge connectivity [2, num_edges] or tuple of (src, dst)
        failed_node_indices: Indices of nodes that have failed
        num_edges: Total number of edges
    
    Returns:
        Edge mask with shape [num_edges] where 1.0 = active, 0.0 = failed
    """
    # Determine if input is tensor
    is_tensor_input = isinstance(edge_index, torch.Tensor)
    
    # Convert everything to numpy for processing
    if is_tensor_input:
        src, dst = edge_index
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
    else:
        src, dst = edge_index
    
    # Create default mask (all edges active) as numpy
    edge_mask = np.ones(num_edges, dtype=np.float32)
    
    # If no failures, return all-active mask
    if len(failed_node_indices) == 0:
        if is_tensor_input:
            return torch.from_numpy(edge_mask).float()
        return edge_mask
    
    # Check if src OR dst is in failed_node_indices
    edge_failed_mask = np.isin(src, failed_node_indices) | np.isin(dst, failed_node_indices)
    edge_mask[edge_failed_mask] = 0.0
    
    # Convert back to tensor if input was tensor
    if is_tensor_input:
        return torch.from_numpy(edge_mask).float()
    
    return edge_mask


def create_edge_mask_sequence(
    sequence: list,
    edge_index: Union[torch.Tensor, np.ndarray],
    num_edges: int,
    start_idx: int = 0,
    use_previous_timestep: bool = True
) -> List[Union[torch.Tensor, np.ndarray]]:
    """
    Create edge mask sequence for temporal data.
    
    This function creates edge masks for each timestep in a sequence,
    using failures from the previous timestep (t-1) to prevent data leakage.
    
    Args:
        sequence: List of timestep dictionaries containing 'node_labels'
        edge_index: Edge connectivity [2, num_edges]
        num_edges: Total number of edges
        start_idx: Starting index in the original sequence (for sliding window)
        use_previous_timestep: Whether to use t-1 failures (default: True)
    
    Returns:
        List of edge masks, one per timestep
    """
    edge_mask_list = []
    
    for i in range(len(sequence)):
        # Calculate the global index to look up the PAST
        global_idx = start_idx + i
        
        prev_failed_node_indices = []
        
        # Only look for failures if we are past the first step
        if use_previous_timestep and global_idx > 0:
            # Get previous timestep from original sequence
            # Note: This requires access to the full sequence
            # In practice, this is handled by the dataset class
            prev_status = sequence[i - 1].get('node_labels', np.zeros(118))
            prev_failed_node_indices = np.where(prev_status > 0.5)[0]
        
        # Create mask for current timestep
        edge_mask = create_edge_mask_from_failures(
            edge_index,
            prev_failed_node_indices,
            num_edges
        )
        
        edge_mask_list.append(edge_mask)
    
    return edge_mask_list


def to_tensor(data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
    """
    Convert data to PyTorch tensor.
    
    Args:
        data: Input data (tensor, numpy array, or list)
    
    Returns:
        PyTorch tensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    else:
        return torch.tensor(data, dtype=torch.float32)
