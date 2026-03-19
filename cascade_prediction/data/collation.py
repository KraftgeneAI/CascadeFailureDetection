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

# All keys with a temporal (T) first dimension.
# They must all be truncated to the same global min_len so the model
# never sees mismatched T across modalities (which causes a CUDA assert).
_TEMPORAL_KEYS = {
    'satellite_data', 'visual_data', 'thermal_data',
    'scada_data', 'weather_sequence', 'threat_indicators',
    'equipment_status', 'pmu_sequence', 'sensor_data', 'edge_mask',
    'temporal_sequence',
    'edge_attr',   # now [T, E, 7] — per-timestep line flows
}


def collate_cascade_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a batch by truncating all temporal sequences to the minimum
    sequence length present in the batch.

    Using a single global min_len for every temporal key guarantees that
    satellite_data, scada_data, pmu_sequence, etc. all share the same T,
    preventing out-of-bounds indexing in the model's temporal loop.

    Args:
        batch: List of sample dictionaries from dataset

    Returns:
        Batched dictionary with all samples collated
    """
    batch = [item for item in batch if item]

    batch_dict = {}
    if not batch:
        return batch_dict

    keys = batch[0].keys()

    # ------------------------------------------------------------------
    # Pass 1: compute a single global min_len across ALL temporal keys
    # ------------------------------------------------------------------
    global_min_len = None
    for key in _TEMPORAL_KEYS:
        if not all(key in item for item in batch):
            continue
        items = [item[key] for item in batch]
        if not isinstance(items[0], torch.Tensor) or items[0].dim() < 1:
            continue
        key_min = min(item.shape[0] for item in items)
        if global_min_len is None or key_min < global_min_len:
            global_min_len = key_min

    # ------------------------------------------------------------------
    # Pass 2: collate each key
    # ------------------------------------------------------------------
    for key in keys:
        # temporal_sequence is always an alias of scada_data — skip here,
        # added explicitly after the loop so it's always consistent.
        if key == 'temporal_sequence':
            continue

        if key == 'edge_index':
            edge_index = batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index

        elif key == 'sequence_length':
            batch_dict['sequence_length'] = torch.tensor(
                [item['sequence_length'] for item in batch],
                dtype=torch.long
            )

        elif key == 'graph_properties':
            graph_props_batch = {}
            if batch[0]['graph_properties']:
                for prop_key in batch[0]['graph_properties'].keys():
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
                                graph_props_batch[prop_key] = props[0]
                        else:
                            graph_props_batch[prop_key] = torch.from_numpy(
                                np.array(props)
                            ).float()
            batch_dict[key] = graph_props_batch

        else:
            if not all(key in item for item in batch):
                continue

            items = [item[key] for item in batch]

            # Convert to tensors if needed
            if not isinstance(items[0], torch.Tensor):
                try:
                    if isinstance(items[0], np.ndarray):
                        arr = np.array(items)
                        items = [torch.from_numpy(arr[i]).float() for i in range(len(items))]
                    else:
                        items = [
                            torch.tensor(item, dtype=torch.float32)
                            if not isinstance(item, torch.Tensor) else item
                            for item in items
                        ]
                except Exception as e:
                    print(f"Error collating key {key}: {e}")
                    continue

            # Temporal keys: truncate every item to global_min_len
            if key in _TEMPORAL_KEYS and global_min_len is not None and items[0].dim() >= 1:
                try:
                    batch_dict[key] = torch.stack(
                        [item[:global_min_len] for item in items], dim=0
                    )
                except Exception as e:
                    print(f"Error stacking temporal key {key}: {e}")
                continue

            # Non-temporal keys: stack directly
            try:
                batch_dict[key] = torch.stack(items, dim=0)
            except Exception as e:
                try:
                    batch_dict[key] = torch.cat(items, dim=0)
                except Exception as e2:
                    print(f"Error: Could not collate key {key}. Skipping. Error: {e2}")

    # ------------------------------------------------------------------
    # temporal_sequence is always identical to scada_data
    # ------------------------------------------------------------------
    if 'scada_data' in batch_dict:
        batch_dict['temporal_sequence'] = batch_dict['scada_data']

    return batch_dict
