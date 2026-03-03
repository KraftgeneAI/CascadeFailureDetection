"""
Inference Dataset Module
========================
Provides dataset class for inference on cascade scenarios.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List
import numpy as np


class ScenarioInferenceDataset(Dataset):
    """
    Dataset for inference on a single scenario with sliding window.
    
    Processes a scenario sequence and provides sliding windows for inference,
    replicating the validation methodology with teacher forcing.
    
    Args:
        scenario: Scenario dictionary containing sequence and metadata
        window_size: Maximum sequence length for sliding window
        base_mva: Base MVA for power normalization (default: 100.0)
        base_frequency: Base frequency for normalization (default: 60.0)
    """
    
    def __init__(
        self, 
        scenario: Dict, 
        window_size: int, 
        base_mva: float = 100.0, 
        base_frequency: float = 60.0
    ):
        self.window_size = window_size
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        self.sequence_original = scenario.get('sequence', [])
        self.edge_index = scenario['edge_index']
        
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.from_numpy(self.edge_index).long()
        
        self.total_steps = len(self.sequence_original)
        self.preprocessed_sequence = self._preprocess_full_sequence()
    
    def _normalize_power(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize power values by base MVA."""
        return tensor / self.base_mva
    
    def _normalize_frequency(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize frequency values by base frequency."""
        return tensor / self.base_frequency
    
    def _preprocess_full_sequence(self) -> Dict[str, torch.Tensor]:
        """
        Preprocess the full scenario sequence.
        
        Returns:
            Dictionary of preprocessed tensors for each modality
        """
        if self.total_steps == 0:
            return {}
        
        data_dicts = {
            'scada_data': [],
            'pmu_sequence': [],
            'satellite_data': [],
            'weather_sequence': [],
            'threat_indicators': [],
            'equipment_status': [],
            'visual_data': [],
            'thermal_data': [],
            'sensor_data': [],
            'edge_mask': []
        }
        
        num_edges = self.edge_index.shape[1]
        
        for i, ts in enumerate(self.sequence_original):
            # SCADA data preprocessing
            s = torch.tensor(ts.get('scada_data'), dtype=torch.float32)
            if s.shape[1] >= 13:
                s = s[:, :13]
            if s.shape[1] >= 6:
                s[:, 2] = self._normalize_power(s[:, 2])
                s[:, 3] = self._normalize_power(s[:, 3])
                s[:, 4] = self._normalize_power(s[:, 4])
                s[:, 5] = self._normalize_power(s[:, 5])
            data_dicts['scada_data'].append(s)
            
            # PMU data preprocessing
            p = torch.tensor(ts.get('pmu_sequence'), dtype=torch.float32)
            if p.shape[1] >= 6:
                p[:, 5] = self._normalize_frequency(p[:, 5])
            data_dicts['pmu_sequence'].append(p)
            
            # Other modalities
            data_dicts['satellite_data'].append(
                torch.tensor(ts.get('satellite_data'), dtype=torch.float32)
            )
            data_dicts['weather_sequence'].append(
                torch.tensor(ts.get('weather_sequence'), dtype=torch.float32).reshape(s.shape[0], -1)
            )
            data_dicts['threat_indicators'].append(
                torch.tensor(ts.get('threat_indicators'), dtype=torch.float32)
            )
            data_dicts['equipment_status'].append(
                torch.tensor(ts.get('equipment_status'), dtype=torch.float32)
            )
            data_dicts['visual_data'].append(
                torch.tensor(ts.get('visual_data'), dtype=torch.float32)
            )
            data_dicts['thermal_data'].append(
                torch.tensor(ts.get('thermal_data'), dtype=torch.float32)
            )
            data_dicts['sensor_data'].append(
                torch.tensor(ts.get('sensor_data'), dtype=torch.float32)
            )
            
            # Edge mask logic (teacher forcing)
            mask = torch.ones(num_edges, dtype=torch.float32)
            if i > 0:
                prev = self.sequence_original[i - 1]
                labels = prev.get('node_labels')
                if labels is not None:
                    failed = np.where(labels > 0.5)[0]
                    if len(failed) > 0:
                        src, dst = self.edge_index.numpy()
                        edge_failed = np.isin(src, failed) | np.isin(dst, failed)
                        mask[edge_failed] = 0.0
            data_dicts['edge_mask'].append(mask)
        
        # Stack all timesteps
        for k, v in data_dicts.items():
            data_dicts[k] = torch.stack(v)
        
        return data_dicts
    
    def __len__(self) -> int:
        """Return number of timesteps in the scenario."""
        return self.total_steps
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sliding window ending at the given timestep.
        
        Args:
            idx: Timestep index
            
        Returns:
            Dictionary containing windowed data
        """
        end_idx = idx + 1
        start_idx = max(0, end_idx - self.window_size)
        
        # Extract window from preprocessed sequence
        item = {
            k: v[start_idx:end_idx] 
            for k, v in self.preprocessed_sequence.items()
        }
        
        # Add edge attributes from the last timestep
        last_step = self.sequence_original[idx]
        edge_attr = torch.tensor(last_step.get('edge_attr'), dtype=torch.float32)
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])
        
        item['edge_attr'] = edge_attr
        item['edge_index'] = self.edge_index
        item['sequence_length'] = end_idx - start_idx
        item['temporal_sequence'] = item['scada_data']
        item['graph_properties'] = {}
        
        return item
