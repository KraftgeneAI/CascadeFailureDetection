"""
Dataset loader for pre-generated cascade failure data.
Works with existing batch files without requiring data regeneration.
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class CascadeDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated cascade failure scenarios.
    
    Works with existing batch files in format:
    - data_unified/train_batches/batch_00000.pkl
    - data_unified/val_batches/batch_00000.pkl
    
    Each batch file contains a list of scenarios with 'sequence' key.
    """
    
    def __init__(self, batch_dir: str, mode: str = 'last_timestep'):
        """
        Initialize dataset from pre-generated batch files.
        
        Args:
            batch_dir: Directory containing batch_*.pkl files
            mode: 'last_timestep' (use final timestep) or 'full_sequence' (use all 60 timesteps)
        """
        self.batch_dir = Path(batch_dir)
        self.mode = mode
        
        # Load all batch files
        self.batch_files = sorted(self.batch_dir.glob("batch_*.pkl"))
        
        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        # Load all scenarios into memory (for faster training)
        print(f"Loading scenarios from {len(self.batch_files)} batch files...")
        self.scenarios = []
        
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                # Each batch file contains a list of scenarios
                if isinstance(batch_data, list):
                    self.scenarios.extend(batch_data)
                else:
                    self.scenarios.append(batch_data)
        
        print(f"Loaded {len(self.scenarios)} scenarios from {batch_dir}")
    
    def __len__(self) -> int:
        return len(self.scenarios)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single scenario.
        
        Returns dict with keys:
            - satellite_data: [N, C, H, W] or [T, N, C, H, W]
            - scada_data: [N, features] or [T, N, features]
            - weather_sequence: [N, features] or [T, N, features]
            - threat_indicators: [N, 7] or [T, N, 7]
            - robotic_data: [N, features] or [T, N, features]
            - node_features: [N, node_dim]
            - edge_index: [2, E]
            - edge_attr: [E, edge_dim]
            - node_failure_labels: [N] or [T, N]
            - cascade_timing: [N] or [T, N]
            - graph_properties: dict with conductance, susceptance, thermal_limits
        """
        scenario = self.scenarios[idx]
        
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
                'weather_sequence': last_step['weather_data'],
                'threat_indicators': last_step['threat_indicators'],
                'robotic_data': last_step['robotic_data'],
                'node_features': last_step['node_features'],
                'edge_index': edge_index,
                'edge_attr': last_step['edge_features'],
                'node_failure_labels': last_step['node_labels'],
                'cascade_timing': last_step.get('cascade_timing', torch.zeros(last_step['node_labels'].shape[0])),
                'graph_properties': self._extract_graph_properties(last_step, metadata)
            }
        
        elif self.mode == 'full_sequence':
            # Stack all timesteps for temporal modeling
            satellite_seq = torch.stack([step['satellite_data'] for step in sequence])
            scada_seq = torch.stack([step['scada_data'] for step in sequence])
            weather_seq = torch.stack([step['weather_data'] for step in sequence])
            threat_seq = torch.stack([step['threat_indicators'] for step in sequence])
            robotic_seq = torch.stack([step['robotic_data'] for step in sequence])
            node_feat_seq = torch.stack([step['node_features'] for step in sequence])
            edge_feat_seq = torch.stack([step['edge_features'] for step in sequence])
            label_seq = torch.stack([step['node_labels'] for step in sequence])
            
            return {
                'satellite_data': satellite_seq,  # [T, N, C, H, W]
                'scada_data': scada_seq,  # [T, N, features]
                'weather_sequence': weather_seq,  # [T, N, features]
                'threat_indicators': threat_seq,  # [T, N, 7]
                'robotic_data': robotic_seq,  # [T, N, features]
                'node_features': node_feat_seq,  # [T, N, node_dim]
                'edge_index': edge_index,  # [2, E]
                'edge_attr': edge_feat_seq[-1],  # Use last timestep edge features
                'node_failure_labels': label_seq[-1],  # Use last timestep labels
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
            
            # Assume SCADA data format: [voltage, current, power_real, power_reactive, frequency, ...]
            # Extract what we can
            if scada.shape[1] >= 4:
                graph_props['power_injection'] = scada[:, 2]  # Real power
                graph_props['reactive_power'] = scada[:, 3]  # Reactive power
        
        # Try to extract from edge features
        edge_attr = timestep_data.get('edge_features', None)
        if edge_attr is not None:
            num_edges = edge_attr.shape[0]
            
            # Assume edge features format: [resistance, reactance, capacity, flow, ...]
            if edge_attr.shape[1] >= 3:
                # Compute conductance and susceptance from resistance and reactance
                resistance = edge_attr[:, 0] + 1e-6  # Avoid division by zero
                reactance = edge_attr[:, 1] + 1e-6
                
                impedance_sq = resistance**2 + reactance**2
                graph_props['conductance'] = resistance / impedance_sq
                graph_props['susceptance'] = -reactance / impedance_sq
                graph_props['thermal_limits'] = edge_attr[:, 2]  # Capacity
        
        # Add metadata if available
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props


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
            # Offset node indices for each graph in the batch
            edge_indices = []
            node_offset = 0
            
            for item in batch:
                edge_index = item['edge_index']
                edge_indices.append(edge_index + node_offset)
                # Assume all graphs have same number of nodes
                num_nodes = item['node_features'].shape[0] if 'node_features' in item else item['scada_data'].shape[0]
                node_offset += num_nodes
            
            batch_dict['edge_index'] = torch.cat(edge_indices, dim=1)
        
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
                            graph_props_batch[prop_key] = torch.cat(props, dim=0)
                        else:
                            graph_props_batch[prop_key] = torch.tensor(props)
            
            batch_dict['graph_properties'] = graph_props_batch
        
        else:
            # Stack regular tensors
            items = [item[key] for item in batch]
            
            if isinstance(items[0], torch.Tensor):
                # Handle different tensor shapes
                if key in ['satellite_data', 'scada_data', 'weather_sequence', 'threat_indicators', 
                          'robotic_data', 'node_features', 'edge_attr', 'node_failure_labels', 'cascade_timing']:
                    # These should be stacked along batch dimension
                    # For node-level data: [N, ...] -> [B*N, ...]
                    batch_dict[key] = torch.cat(items, dim=0)
                else:
                    # Default stacking
                    batch_dict[key] = torch.stack(items, dim=0)
            else:
                batch_dict[key] = items
    
    return batch_dict
