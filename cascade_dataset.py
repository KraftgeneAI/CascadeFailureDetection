"""
Memory-efficient dataset loader for pre-generated cascade failure data.
Uses lazy loading to avoid loading all data into memory at once.
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import gc  # Added garbage collection import


class CascadeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for loading pre-generated cascade failure scenarios.
    
    Uses lazy loading - only loads batch files when needed, not all at once.
    Optional manual caching for recently accessed batches to balance memory vs speed.
    
    Works with existing batch files in format:
    - data_unified/train_batches/batch_00000.pkl
    - data_unified/val_batches/batch_00000.pkl
    
    Each batch file contains a list of scenarios with 'sequence' key.
    """
    
    def __init__(self, batch_dir: str, mode: str = 'last_timestep', cache_size: int = 2):  # Reduced default cache from 10 to 2
        """
        Initialize dataset from pre-generated batch files.
        
        Args:
            batch_dir: Directory containing batch_*.pkl files
            mode: 'last_timestep' (use final timestep) or 'full_sequence' (use all 60 timesteps)
            cache_size: Number of batch files to keep in memory. Set to 0 to disable caching.
                       REDUCED DEFAULT TO 2 FOR MEMORY EFFICIENCY
        """
        self.batch_dir = Path(batch_dir)
        self.mode = mode
        self.cache_size = cache_size
        
        self._batch_cache = {}  # {batch_idx: batch_data}
        self._cache_order = []  # Track access order for LRU eviction
        
        # Find all batch files
        self.batch_files = sorted(self.batch_dir.glob("batch_*.pkl"))
        
        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        print(f"Indexing scenarios from {len(self.batch_files)} batch files...")
        self.scenario_index = []  # List of (batch_file_idx, scenario_idx_in_batch)
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            # Load batch file temporarily just to count scenarios
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                
                # Count scenarios in this batch
                if isinstance(batch_data, list):
                    num_scenarios = len(batch_data)
                else:
                    num_scenarios = 1
                
                # Add entries to index
                for scenario_idx in range(num_scenarios):
                    self.scenario_index.append((batch_idx, scenario_idx))
        
        print(f"Indexed {len(self.scenario_index)} scenarios from {len(self.batch_files)} batch files")
        print(f"Memory-efficient mode: Loading batches on-demand (cache size: {cache_size})")
    
    def _load_batch_cached(self, batch_idx: int) -> List[Dict]:
        """
        Load a batch file with manual LRU caching (picklable for multiprocessing).
        
        Args:
            batch_idx: Index of the batch file to load
            
        Returns:
            List of scenarios in the batch
        """
        # Check if in cache
        if batch_idx in self._batch_cache:
            # Move to end of access order (most recently used)
            self._cache_order.remove(batch_idx)
            self._cache_order.append(batch_idx)
            return self._batch_cache[batch_idx]
        
        # Load from disk
        batch_file = self.batch_files[batch_idx]
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        # Ensure it's a list
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        
        # Add to cache if caching is enabled
        if self.cache_size > 0:
            # Evict oldest if cache is full
            if len(self._batch_cache) >= self.cache_size:
                oldest_idx = self._cache_order.pop(0)
                del self._batch_cache[oldest_idx]
                gc.collect()  # Force garbage collection after eviction
            
            # Add to cache
            self._batch_cache[batch_idx] = batch_data
            self._cache_order.append(batch_idx)
        else:
            gc.collect()
        
        return batch_data
    
    def __len__(self) -> int:
        return len(self.scenario_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single scenario (lazy loading - only loads the batch file containing this scenario).
        
        Returns dict with keys:
            - satellite_data: [N, C, H, W] or [T, N, C, H, W]
            - scada_data: [N, features] or [T, N, features]
            - weather_sequence: [N, features] or [T, N, features]
            - threat_indicators: [N, 7] or [T, N, 7]
            - visual_data: [N, C, H, W] or [T, N, C, H, W]
            - thermal_data: [N, C, H, W] or [T, N, C, H, W]
            - sensor_data: [N, features] or [T, N, features]
            - pmu_sequence: [N, features] or [T, N, features]
            - equipment_status: [N, features] or [T, N, features]
            - node_features: [N, node_dim]
            - edge_index: [2, E]
            - edge_attr: [E, edge_dim]
            - node_failure_labels: [N] or [T, N]
            - cascade_timing: [N] or [T, N]
            - graph_properties: dict with conductance, susceptance, thermal_limits
        """
        batch_idx, scenario_idx = self.scenario_index[idx]
        
        batch_scenarios = self._load_batch_cached(batch_idx)
        scenario = batch_scenarios[scenario_idx]
        
        # Extract sequence data
        sequence = scenario['sequence']
        edge_index = scenario['edge_index']
        metadata = scenario.get('metadata', {})
        
        if self.mode == 'last_timestep':
            # Use only the last timestep for prediction
            last_step = sequence[-1]
            
            return {
                'satellite_data': last_step['satellite_data'],
                'scada_data': last_step['scada_data'],
                'weather_sequence': last_step['weather_sequence'],
                'threat_indicators': last_step['threat_indicators'],
                'visual_data': last_step['visual_data'],
                'thermal_data': last_step['thermal_data'],
                'sensor_data': last_step['sensor_data'],
                'pmu_sequence': last_step['pmu_sequence'],
                'equipment_status': last_step['equipment_status'],
                'node_features': last_step['scada_data'],
                'edge_index': edge_index,
                'edge_attr': last_step['edge_attr'],
                'node_failure_labels': last_step['node_labels'],
                'cascade_timing': last_step.get('cascade_timing', torch.zeros(last_step['node_labels'].shape[0])),
                'graph_properties': self._extract_graph_properties(last_step, metadata)
            }
        
        elif self.mode == 'full_sequence':
            # Stack all timesteps for temporal modeling
            satellite_seq = torch.stack([step['satellite_data'] for step in sequence])
            scada_seq = torch.stack([step['scada_data'] for step in sequence])
            weather_seq = torch.stack([step['weather_sequence'] for step in sequence])
            threat_seq = torch.stack([step['threat_indicators'] for step in sequence])
            visual_seq = torch.stack([step['visual_data'] for step in sequence])
            thermal_seq = torch.stack([step['thermal_data'] for step in sequence])
            sensor_seq = torch.stack([step['sensor_data'] for step in sequence])
            pmu_seq = torch.stack([step['pmu_sequence'] for step in sequence])
            equipment_seq = torch.stack([step['equipment_status'] for step in sequence])
            node_feat_seq = scada_seq
            edge_feat_seq = torch.stack([step['edge_attr'] for step in sequence])
            label_seq = torch.stack([step['node_labels'] for step in sequence])
            
            return {
                'satellite_data': satellite_seq,
                'scada_data': scada_seq,
                'weather_sequence': weather_seq,
                'threat_indicators': threat_seq,
                'visual_data': visual_seq,
                'thermal_data': thermal_seq,
                'sensor_data': sensor_seq,
                'pmu_sequence': pmu_seq,
                'equipment_status': equipment_seq,
                'node_features': node_feat_seq,
                'edge_index': edge_index,
                'edge_attr': edge_feat_seq[-1],
                'node_failure_labels': label_seq[-1],
                'cascade_timing': sequence[-1].get('cascade_timing', torch.zeros(label_seq.shape[1])),
                'graph_properties': self._extract_graph_properties(sequence[-1], metadata)
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _extract_graph_properties(self, timestep_data: Dict, metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Extract graph properties for physics-informed loss.
        
        Extracts conductance, susceptance, thermal limits, and power injection
        from the scenario data.
        """
        graph_props = {}
        
        # Try to extract from SCADA data (columns might contain these)
        scada = timestep_data.get('scada_data', None)
        if scada is not None:
            num_nodes = scada.shape[0]
            
            # SCADA data format from generator:
            # [voltages, angles, generation, Q_gen, load, Q_load, frequency, equipment_temps, wind, condition, age, ...]
            # Extract what we can
            if scada.shape[1] >= 6:
                graph_props['power_injection'] = scada[:, 2]  # Real power generation
                graph_props['reactive_power'] = scada[:, 3]  # Reactive power generation
                graph_props['load'] = scada[:, 4]  # Real power load
        
        # Try to extract from edge features
        edge_attr = timestep_data.get('edge_attr', None)
        if edge_attr is not None:
            num_edges = edge_attr.shape[0]
            
            # Edge features format from generator:
            # [reactance, thermal_limits, loading_ratios, ambient_temp, ...]
            if edge_attr.shape[1] >= 3:
                reactance = edge_attr[:, 0] + 1e-6  # Avoid division by zero
                resistance = reactance * 0.1  # R/X ratio ~ 0.1 for transmission lines
                
                impedance_sq = resistance**2 + reactance**2
                graph_props['conductance'] = resistance / impedance_sq
                graph_props['susceptance'] = -reactance / impedance_sq
                graph_props['thermal_limits'] = edge_attr[:, 1]  # Thermal capacity
        
        # Add metadata if available
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def clear_cache(self):
        """Clear the manual cache to free memory."""
        self._batch_cache.clear()
        self._cache_order.clear()
        print("Cleared batch cache")


def collate_cascade_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to properly batch cascade scenarios.
    
    Handles variable-sized data and creates proper batch tensors.
    """
    # Initialize batch dict
    batch_dict = {}
    
    # Get keys from first item
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'edge_index':
            # Edge indices need special handling for batching graphs
            # Keep as single graph (not batched) - model handles batching internally
            # Just use the first item's edge_index since all graphs have same structure
            edge_index = batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index
        
        elif key == 'graph_properties':
            # Batch graph properties
            graph_props_batch = {}
            
            # Get all property keys from first item
            if batch[0]['graph_properties']:
                prop_keys = batch[0]['graph_properties'].keys()
                
                for prop_key in prop_keys:
                    # Stack properties from all items
                    props = [item['graph_properties'][prop_key] for item in batch if prop_key in item['graph_properties']]
                    if props:
                        if isinstance(props[0], torch.Tensor):
                            graph_props_batch[prop_key] = torch.stack(props, dim=0)
                        else:
                            props_array = np.array(props)
                            graph_props_batch[prop_key] = torch.from_numpy(props_array).float()
            
            batch_dict['graph_properties'] = graph_props_batch
        
        else:
            # Stack regular tensors
            items = [item[key] for item in batch]
            
            if not isinstance(items[0], torch.Tensor):
                # Check if items are numpy arrays
                if isinstance(items[0], np.ndarray):
                    items_array = np.array(items)
                    items = [torch.from_numpy(items_array[i]).float() for i in range(len(items))]
                else:
                    # Convert other types (lists, etc.) to tensors
                    items = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, torch.Tensor) else item 
                            for item in items]
            
            # Image modalities need [B, N, C, H, W] format
            if key in ['satellite_data', 'visual_data', 'thermal_data']:
                # Image data: Stack to [B, N, C, H, W]
                batch_dict[key] = torch.stack(items, dim=0)
            
            elif key in ['scada_data', 'weather_sequence', 'threat_indicators', 'equipment_status', 
                        'pmu_sequence', 'sensor_data', 'node_failure_labels']:
                # Stack to [B, N, features] or [B, N] for consistent dimensions
                batch_dict[key] = torch.stack(items, dim=0)
            
            elif key in ['edge_attr']:
                # Stack to [B, E, features] instead of concatenating
                batch_dict[key] = torch.stack(items, dim=0)
            
            # Node-level data for graph batching: Concatenate to [B*N, features]
            elif key in ['node_features', 'cascade_timing']:
                batch_dict[key] = torch.cat(items, dim=0)
            
            else:
                # Default stacking for other keys
                try:
                    batch_dict[key] = torch.stack(items, dim=0)
                except:
                    # If stacking fails, concatenate
                    batch_dict[key] = torch.cat(items, dim=0)
    
    return batch_dict
