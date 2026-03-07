"""
Collation Module
================
Batch collation functions for DataLoader.

This module provides functions to collate variable-length sequences
into batched tensors for efficient training.
"""

import torch
import numpy as np
from typing import List, Dict, Any


def collate_cascade_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with support for variable-length sequences.
    
    This function:
    - Filters out empty dictionaries
    - Handles variable-length temporal sequences with padding
    - Properly batches graph properties
    - Handles edge indices and masks
    
    Args:
        batch: List of sample dictionaries from dataset
    
    Returns:
        Batched dictionary with all samples collated
    """
    # Filter out empty dictionaries
    batch = [item for item in batch if item]
    
    batch_dict = {}
    if not batch:
        return batch_dict
    
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'edge_index':
            # Edge index is shared across batch
            edge_index = batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index
        
        elif key == 'sequence_length':
            # Stack sequence lengths
            batch_dict['sequence_length'] = torch.tensor(
                [item['sequence_length'] for item in batch],
                dtype=torch.long
            )
        
        elif key == 'temporal_sequence':
            # Pad temporal sequences to max length
            items = [item[key] for item in batch]
            max_len = max(item.shape[0] for item in items)
            
            padded_items = []
            for item in items:
                if item.shape[0] < max_len:
                    pad_size = max_len - item.shape[0]
                    padding = torch.zeros(
                        pad_size, *item.shape[1:],
                        dtype=item.dtype,
                        device=item.device
                    )
                    padded_item = torch.cat([item, padding], dim=0)
                else:
                    padded_item = item
                padded_items.append(padded_item)
            
            batch_dict['temporal_sequence'] = torch.stack(padded_items, dim=0)
        
        elif key == 'graph_properties':
            # Batch graph properties
            graph_props_batch = {}
            
            if batch[0]['graph_properties']:
                prop_keys = batch[0]['graph_properties'].keys()
                
                for prop_key in prop_keys:
                    props = [
                        item['graph_properties'][prop_key]
                        for item in batch
                        if prop_key in item['graph_properties']
                    ]
                    
                    if props:
                        if isinstance(props[0], torch.Tensor):
                            try:
                                graph_props_batch[prop_key] = torch.stack(props, dim=0)
                            except RuntimeError:
                                # If stacking fails, use first element
                                graph_props_batch[prop_key] = props[0]
                        else:
                            props_array = np.array(props)
                            graph_props_batch[prop_key] = torch.from_numpy(props_array).float()
            
            batch_dict[key] = graph_props_batch
        
        else:
            # Handle other keys - skip if not present in all items
            if not all(key in item for item in batch):
                continue
                
            items = [item[key] for item in batch]
            
            # Convert to tensors if needed
            if not isinstance(items[0], torch.Tensor):
                try:
                    if isinstance(items[0], np.ndarray):
                        items_array = np.array(items)
                        items = [
                            torch.from_numpy(items_array[i]).float()
                            for i in range(len(items))
                        ]
                    else:
                        items = [
                            torch.tensor(item, dtype=torch.float32)
                            if not isinstance(item, torch.Tensor) else item
                            for item in items
                        ]
                except Exception as e:
                    print(f"Error collating key {key}: {e}")
                    continue
            
            # Pad sequences if needed
            if (items[0].dim() >= 3 or key == 'edge_mask') and key in [
                'satellite_data', 'visual_data', 'thermal_data',
                'scada_data', 'weather_sequence', 'threat_indicators',
                'equipment_status', 'pmu_sequence', 'sensor_data', 'edge_mask'
            ]:
                first_dims = [item.shape[0] for item in items]
                max_len = max(first_dims)
                
                if len(set(first_dims)) > 1:
                    # Pad to max length
                    padded_items = []
                    for item in items:
                        if item.shape[0] < max_len:
                            pad_size = max_len - item.shape[0]
                            padding = torch.zeros(
                                pad_size, *item.shape[1:],
                                dtype=item.dtype,
                                device=item.device
                            )
                            padded_item = torch.cat([item, padding], dim=0)
                        else:
                            padded_item = item
                        padded_items.append(padded_item)
                    batch_dict[key] = torch.stack(padded_items, dim=0)
                else:
                    batch_dict[key] = torch.stack(items, dim=0)
            else:
                # Stack directly
                try:
                    batch_dict[key] = torch.stack(items, dim=0)
                except Exception as e:
                    try:
                        batch_dict[key] = torch.cat(items, dim=0)
                    except Exception as e2:
                        print(f"Error: Could not collate key {key}. Skipping. Error: {e2}")
    
    return batch_dict
