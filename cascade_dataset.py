"""
Memory-efficient dataset loader for pre-generated cascade failure/normal data.
=============================================================================
MODIFIED FOR 1-SCENARIO-PER-FILE FORMAT
This version reads individual scenario_*.pkl files, avoiding memory-intensive
batch files.
=============================================================================

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import gc
import glob # Added glob


class CascadeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for 1-scenario-per-file format.
    
    - No caching needed, as each file is tiny.
    - Loads one .pkl file per __getitem__ call.
    
    WITH PHYSICS-INFORMED NORMALIZATION for power, voltage, capacity, frequency.
    """
    
    def __init__(self, data_dir: str, mode: str = 'last_timestep',
                 base_mva: float = 100.0, base_frequency: float = 60.0):
        """
        Initialize dataset from a directory of individual scenario_*.pkl files.
        
        Args:
            data_dir: Directory containing scenario_*.pkl files
            mode: 'last_timestep' or 'full_sequence'
            base_mva: Base MVA for per-unit normalization (default: 100.0)
            base_frequency: Base frequency in Hz for per-unit normalization (default: 60.0)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        # --- ADDED: Normalization constants ---
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        # --- END ADDED ---
        
        
        # Find all individual scenario files
        print(f"Indexing scenarios from: {data_dir}")
        self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenario_*.pkl")))
        
        if len(self.scenario_files) == 0:
            raise ValueError(f"No 'scenario_*.pkl' files found in {data_dir}. "
                             "Did you run the rebatch_data.py script?")
        
        print(f"Physics normalization: base_mva={base_mva}, base_frequency={base_frequency}")
        
        # --- We must load the metadata from each file to get labels ---
        # This is memory-intensive but necessary for weighted sampling.
        # We will load, get label, and discard.
        print(f"Scanning {len(self.scenario_files)} files for cascade labels...")
        self.cascade_labels = []
        
        for scenario_file in self.scenario_files:
            try:
                with open(scenario_file, 'rb') as f:
                    # Load the single scenario dictionary
                    scenario = pickle.load(f)
                
                # Determine cascade label
                if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
                    has_cascade = scenario['metadata']['is_cascade']
                elif 'sequence' in scenario and len(scenario['sequence']) > 0:
                    last_step = scenario['sequence'][-1]
                    has_cascade = bool(np.max(last_step.get('node_labels', np.zeros(1))) > 0.5)
                else:
                    has_cascade = False # Default to False if undetectable
                
                self.cascade_labels.append(has_cascade)
                
            except (IOError, pickle.UnpicklingError) as e:
                print(f"Warning: Skipping corrupted or unreadable file: {scenario_file}. Error: {e}")
                self.cascade_labels.append(False) # Add a dummy label to keep index aligned
        
        print(f"Indexed {len(self.scenario_files)} scenarios.")
        
        if len(self.cascade_labels) == 0:
            print(f"  [WARNING] No valid scenarios found!")
        else:
            positive_count = sum(self.cascade_labels)
            print(f"  Cascade scenarios: {positive_count} ({positive_count/len(self.cascade_labels)*100:.1f}%)")
            print(f"  Normal scenarios: {len(self.cascade_labels) - positive_count} ({(len(self.cascade_labels) - positive_count)/len(self.cascade_labels)*100:.1f}%)")
        
        print(f"Ultra-memory-efficient mode: Loading 1 file per sample.")

    # --- ADDED: Normalization methods ---
    def _normalize_power(self, power_values: np.ndarray) -> np.ndarray:
        """Normalize power values to per-unit (divide by base_mva)."""
        return power_values / self.base_mva
    
    def _normalize_frequency(self, frequency_values: np.ndarray) -> np.ndarray:
        """Normalize frequency to per-unit (divide by base_frequency)."""
        return frequency_values / self.base_frequency
    # --- END ADDED ---
    
    # --- REMOVED: _load_batch_cached - no longer needed ---
    
    def __len__(self) -> int:
        return len(self.scenario_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single scenario by loading its individual file."""
        scenario_file = self.scenario_files[idx]
        
        try:
            with open(scenario_file, 'rb') as f:
                scenario = pickle.load(f)
        except Exception as e:
            print(f"Error loading {scenario_file}: {e}. Returning empty dict.")
            return {} # Dataloader collate_fn will skip this
        
        if 'sequence' in scenario and 'metadata' in scenario:
            sequence = scenario['sequence']
            
            if len(sequence) == 0 and 'failed_nodes' in scenario['metadata']:
                # Handle cases where generator failed but metadata is valid
                return self._process_metadata_format(scenario)
            elif len(sequence) > 0:
                return self._process_sequence_format(scenario)
            else:
                print(f"Warning: Scenario {idx} has empty sequence and no metadata. Skipping.")
                return {}
        
        else:
            print(f"Warning: Scenario {idx} has unknown format. Keys: {scenario.keys()}. Skipping.")
            return {}

    
    def _process_sequence_format(self, scenario: Dict) -> Dict[str, Any]:
        """Process NEW FORMAT (sequence of timestep dicts) WITH NORMALIZATION."""
        sequence = scenario['sequence']
        edge_index = scenario['edge_index']
        metadata = scenario.get('metadata', {})
        
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            else:
                return torch.tensor(data, dtype=torch.float32)
        
        # --- CRITICAL FIX: Apply normalization during data loading ---
        satellite_seq = []
        scada_seq = []
        weather_seq = []
        threat_seq = []
        visual_seq = []
        thermal_seq = []
        sensor_seq = []
        pmu_seq = []
        equipment_seq = []
        label_seq = []
        timing_seq = []
        
        # Get defaults from the last step if possible, or create zeros
        last_step = sequence[-1]
        
        # Helper to safely get data
        def safe_get(ts, key, default_val):
            data = ts.get(key)
            if data is None:
                return default_val
            # Handle shape mismatches by falling back to default
            if hasattr(default_val, 'shape') and hasattr(data, 'shape') and data.shape != default_val.shape:
                if data.size == default_val.size:
                    try:
                        return data.reshape(default_val.shape)
                    except ValueError:
                         return default_val
                return default_val
            return data

        # Get default shapes from last timestep
        num_nodes = last_step.get('scada_data', np.zeros((118,15))).shape[0]
        num_edges = edge_index.shape[1]
        
        sat_shape = last_step.get('satellite_data', np.zeros((num_nodes, 12, 16, 16))).shape
        weather_shape = last_step.get('weather_sequence', np.zeros((num_nodes, 10, 8))).shape
        threat_shape = last_step.get('threat_indicators', np.zeros((num_nodes, 6))).shape
        scada_shape = last_step.get('scada_data', np.zeros((num_nodes, 15))).shape
        pmu_shape = last_step.get('pmu_sequence', np.zeros((num_nodes, 8))).shape
        equip_shape = last_step.get('equipment_status', np.zeros((num_nodes, 10))).shape
        vis_shape = last_step.get('visual_data', np.zeros((num_nodes, 3, 32, 32))).shape
        therm_shape = last_step.get('thermal_data', np.zeros((num_nodes, 1, 32, 32))).shape
        sensor_shape = last_step.get('sensor_data', np.zeros((num_nodes, 12))).shape
        label_shape = last_step.get('node_labels', np.zeros(num_nodes)).shape
        timing_shape = last_step.get('cascade_timing', np.zeros(num_nodes)).shape

        for ts in sequence:
            # --- SCADA NORMALIZATION ---
            scada_data = safe_get(ts, 'scada_data', np.zeros(scada_shape)).astype(np.float32)
            if scada_data.shape[1] >= 6:
                scada_data[:, 2] = self._normalize_power(scada_data[:, 2]) # generation
                scada_data[:, 3] = self._normalize_power(scada_data[:, 3]) # reactive_generation
                scada_data[:, 4] = self._normalize_power(scada_data[:, 4]) # load_values
                scada_data[:, 5] = self._normalize_power(scada_data[:, 5]) # reactive_load
            scada_seq.append(to_tensor(scada_data))
            
            # --- PMU NORMALIZATION ---
            pmu_data = safe_get(ts, 'pmu_sequence', np.zeros(pmu_shape)).astype(np.float32)
            if pmu_data.shape[1] >= 6:
                pmu_data[:, 5] = self._normalize_frequency(pmu_data[:, 5]) # frequency
            pmu_seq.append(to_tensor(pmu_data))

            # --- Reshape weather from [N, 10, 8] to [N, 80] ---
            weather_data_raw = safe_get(ts, 'weather_sequence', np.zeros(weather_shape)).astype(np.float32)
            weather_data = weather_data_raw.reshape(num_nodes, -1) # Reshape to [N, 80]
            weather_seq.append(to_tensor(weather_data))

            # --- Other data (no normalization needed) ---
            satellite_seq.append(to_tensor(safe_get(ts, 'satellite_data', np.zeros(sat_shape))))
            threat_seq.append(to_tensor(safe_get(ts, 'threat_indicators', np.zeros(threat_shape))))
            visual_seq.append(to_tensor(safe_get(ts, 'visual_data', np.zeros(vis_shape))))
            thermal_seq.append(to_tensor(safe_get(ts, 'thermal_data', np.zeros(therm_shape))))
            sensor_seq.append(to_tensor(safe_get(ts, 'sensor_data', np.zeros(sensor_shape))))
            equipment_seq.append(to_tensor(safe_get(ts, 'equipment_status', np.zeros(equip_shape))))
            label_seq.append(to_tensor(safe_get(ts, 'node_labels', np.zeros(label_shape))))
            timing_seq.append(to_tensor(safe_get(ts, 'cascade_timing', np.zeros(timing_shape))))

        # --- EDGE ATTR NORMALIZATION ---
        edge_attr = safe_get(last_step, 'edge_attr', np.zeros((num_edges, 5))).astype(np.float32)
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1]) # thermal_limits
        edge_attr = to_tensor(edge_attr)
        
        # ====================================================================
        # START: IMPROVEMENT (Load ground_truth_risk)
        # ====================================================================
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))
        # ====================================================================
        # END: IMPROVEMENT
        # ====================================================================


        if self.mode == 'last_timestep':
            return {
                'satellite_data': satellite_seq[-1],
                'scada_data': scada_seq[-1],
                'weather_sequence': weather_seq[-1],
                'threat_indicators': threat_seq[-1],
                'visual_data': visual_seq[-1],
                'thermal_data': thermal_seq[-1],
                'sensor_data': sensor_seq[-1],
                'pmu_sequence': pmu_seq[-1],
                'equipment_status': equipment_seq[-1],
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_attr,
                'node_failure_labels': label_seq[-1],
                'cascade_timing': timing_seq[-1],
                'ground_truth_risk': to_tensor(ground_truth_risk), # <-- ADDED
                'graph_properties': self._extract_graph_properties(last_step, metadata, edge_attr)
            }
        
        elif self.mode == 'full_sequence':
            return {
                'satellite_data': torch.stack(satellite_seq),
                'scada_data': torch.stack(scada_seq),
                'weather_sequence': torch.stack(weather_seq),
                'threat_indicators': torch.stack(threat_seq),
                'visual_data': torch.stack(visual_seq),
                'thermal_data': torch.stack(thermal_seq),
                'sensor_data': torch.stack(sensor_seq),
                'pmu_sequence': torch.stack(pmu_seq),
                'equipment_status': torch.stack(equipment_seq),
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_attr, # Use last edge_attr
                'node_failure_labels': label_seq[-1], # Use last label
                'cascade_timing': timing_seq[-1], # Use last timing
                'ground_truth_risk': to_tensor(ground_truth_risk), # <-- ADDED
                'graph_properties': self._extract_graph_properties(last_step, metadata, edge_attr),
                'temporal_sequence': torch.stack(scada_seq), # Use SCADA as main temporal feature
                'sequence_length': len(sequence)
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _process_metadata_format(self, scenario: Dict) -> Dict[str, Any]:
        """Process NEW FORMAT with empty sequence but metadata containing failure information."""
        # This is a fallback for corrupted data.
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']
        
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            else:
                return torch.tensor(data, dtype=torch.float32)
        
        num_nodes = metadata.get('num_nodes', 118)
        num_edges = metadata.get('num_edges', edge_index.shape[1] if hasattr(edge_index, 'shape') else 186)
        
        node_failure_labels = np.zeros(num_nodes, dtype=np.float32)
        if 'failed_nodes' in metadata and len(metadata['failed_nodes']) > 0:
            failed_nodes = metadata['failed_nodes']
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        node_failure_labels[node_idx_int] = 1.0
                except (ValueError, TypeError):
                    continue
        
        # Create dummy data for one timestep
        T = 1
        scada_data = torch.randn(T, num_nodes, 15)
        weather_sequence = torch.randn(T, num_nodes, 80) # Model expects [N, 80]
        threat_indicators = torch.randn(T, num_nodes, 6)
        pmu_sequence = torch.randn(T, num_nodes, 8)
        equipment_status = torch.randn(T, num_nodes, 10)
        satellite_data = torch.randn(T, num_nodes, 12, 16, 16)
        visual_data = torch.randn(T, num_nodes, 3, 32, 32)
        thermal_data = torch.randn(T, num_nodes, 1, 32, 32)
        sensor_data = torch.randn(T, num_nodes, 12)
        edge_attr = torch.randn(num_edges, 5)

        # Normalize this dummy data
        edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])
        scada_data[..., 2:6] = self._normalize_power(scada_data[..., 2:6])
        pmu_sequence[..., 5] = self._normalize_frequency(pmu_sequence[..., 5])
            
        graph_props = self._extract_graph_properties_from_metadata(metadata, num_edges)
        
        # ====================================================================
        # START: IMPROVEMENT (Load ground_truth_risk)
        # ====================================================================
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))
        # ====================================================================
        # END: IMPROVEMENT
        # ====================================================================

        if self.mode == 'full_sequence':
            item = {
                'satellite_data': satellite_data,
                'scada_data': scada_data,
                'weather_sequence': weather_sequence,
                'threat_indicators': threat_indicators,
                'visual_data': visual_data,
                'thermal_data': thermal_data,
                'sensor_data': sensor_data,
                'pmu_sequence': pmu_sequence,
                'equipment_status': equipment_status,
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_attr,
                'node_failure_labels': to_tensor(node_failure_labels),
                'cascade_timing': torch.zeros(num_nodes),
                'ground_truth_risk': to_tensor(ground_truth_risk), # <-- ADDED
                'graph_properties': graph_props,
                'temporal_sequence': scada_data,
                'sequence_length': T
            }
        else: # last_timestep
            item = {
                'satellite_data': satellite_data[0],
                'scada_data': scada_data[0],
                'weather_sequence': weather_sequence[0],
                'threat_indicators': threat_indicators[0],
                'visual_data': visual_data[0],
                'thermal_data': thermal_data[0],
                'sensor_data': sensor_data[0],
                'pmu_sequence': pmu_sequence[0],
                'equipment_status': equipment_status[0],
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_attr,
                'node_failure_labels': to_tensor(node_failure_labels),
                'cascade_timing': torch.zeros(num_nodes),
                'ground_truth_risk': to_tensor(ground_truth_risk), # <-- ADDED
                'graph_properties': graph_props
            }
        return item
    
    
    def _extract_graph_properties(self, timestep_data: Dict, metadata: Dict, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract graph properties for physics-informed loss WITH NORMALIZATION."""
        graph_props = {}
        
        # Data from generator is raw (e.g., 150 MW)
        # We must normalize it for the loss function
        
        if 'conductance' in timestep_data:
            graph_props['conductance'] = torch.from_numpy(timestep_data['conductance']).float()
        else:
            graph_props['conductance'] = edge_attr[:, 4] # Col 4 = conductance
        
        if 'susceptance' in timestep_data:
            graph_props['susceptance'] = torch.from_numpy(timestep_data['susceptance']).float()
        else:
            graph_props['susceptance'] = edge_attr[:, 3] # Col 3 = susceptance
        
        # edge_attr is already normalized from _process_sequence_format
        graph_props['thermal_limits'] = edge_attr[:, 1] # Col 1 = thermal_limits
        
        if 'power_injection' in timestep_data:
            power_injection_raw = torch.from_numpy(timestep_data['power_injection']).float()
            graph_props['power_injection'] = self._normalize_power(power_injection_raw)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None and scada.shape[1] >= 6:
                # Data in scada is already normalized
                generation_pu = scada[:, 2]
                load_pu = scada[:, 4]
                graph_props['power_injection'] = generation_pu - load_pu
        
        if 'reactive_injection' in timestep_data:
             reactive_injection_raw = torch.from_numpy(timestep_data['reactive_injection']).float()
             graph_props['reactive_injection'] = self._normalize_power(reactive_injection_raw)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None and scada.shape[1] >= 6:
                # Data in scada is already normalized
                reac_gen_pu = scada[:, 3]
                reac_load_pu = scada[:, 5]
                graph_props['reactive_injection'] = reac_gen_pu - reac_load_pu

        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def _extract_graph_properties_from_metadata(self, metadata: Dict, num_edges: int) -> Dict[str, torch.Tensor]:
        """Extract graph properties from metadata when sequence is empty WITH NORMALIZATION."""
        graph_props = {}
        
        num_nodes = metadata.get('num_nodes', 118)
        
        thermal_limits_raw = torch.rand(num_edges) * 40.0 + 10.0  # [10, 50] MW
        graph_props['thermal_limits'] = self._normalize_power(thermal_limits_raw)
        
        reactance = torch.rand(num_edges) * 0.3 + 0.05
        resistance = reactance * 0.1
        impedance_sq = resistance**2 + reactance**2
        graph_props['conductance'] = resistance / impedance_sq
        graph_props['susceptance'] = -reactance / impedance_sq
        
        is_cascade = metadata.get('is_cascade', False)
        failed_nodes = metadata.get('failed_nodes', [])
        
        if is_cascade and len(failed_nodes) > 0:
            power_injection_raw = torch.randn(num_nodes) * 50.0
            
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        power_injection_raw[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            power_injection_raw = torch.randn(num_nodes) * 5.0
        
        graph_props['power_injection'] = self._normalize_power(power_injection_raw)
        
        if is_cascade and len(failed_nodes) > 0:
            reactive_injection_raw = torch.randn(num_nodes) * 30.0
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        reactive_injection_raw[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            reactive_injection_raw = torch.randn(num_nodes) * 3.0
        
        graph_props['reactive_injection'] = self._normalize_power(reactive_injection_raw)
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    
    def get_cascade_label(self, idx: int) -> bool:
        """Get cascade label without loading full data."""
        return self.cascade_labels[idx]


def collate_cascade_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with support for variable-length sequences.
    This function will now skip any empty dictionaries returned by __getitem__.
    """
    
    # Filter out empty dictionaries (from loading errors)
    batch = [item for item in batch if item]
    
    batch_dict = {}
    if not batch:
        return batch_dict
        
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'edge_index':
            # Edge index is shared, just take the first one
            edge_index = batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index
        
        elif key == 'sequence_length':
            batch_dict['sequence_length'] = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
        
        elif key == 'temporal_sequence':
            # This is a special key for full sequence models, pad it
            items = [item[key] for item in batch]
            max_len = max(item.shape[0] for item in items)
            
            padded_items = []
            for item in items:
                if item.shape[0] < max_len:
                    pad_size = max_len - item.shape[0]
                    padding = torch.zeros(pad_size, *item.shape[1:], dtype=item.dtype, device=item.device)
                    padded_item = torch.cat([item, padding], dim=0)
                else:
                    padded_item = item
                padded_items.append(padded_item)
            
            batch_dict['temporal_sequence'] = torch.stack(padded_items, dim=0)
        
        elif key == 'graph_properties':
            graph_props_batch = {}
            
            if batch[0]['graph_properties']:
                prop_keys = batch[0]['graph_properties'].keys()
                
                for prop_key in prop_keys:
                    props = [item['graph_properties'][prop_key] for item in batch if prop_key in item['graph_properties']]
                    if props:
                        if isinstance(props[0], torch.Tensor):
                            try:
                                graph_props_batch[prop_key] = torch.stack(props, dim=0)
                            except RuntimeError:
                                # Handle size mismatches, e.g., if one is [186] and another is [187]
                                # print(f"Warning: Size mismatch in graph_properties key {prop_key}, collating as list.")
                                # Just use the first one if they can't be stacked (e.g. edge features)
                                graph_props_batch[prop_key] = props[0]
                        else:
                            props_array = np.array(props)
                            graph_props_batch[prop_key] = torch.from_numpy(props_array).float()
            
            batch_dict[key] = graph_props_batch
        
        else:
            # Handle all other keys
            items = [item[key] for item in batch]
            
            if not isinstance(items[0], torch.Tensor):
                try:
                    if isinstance(items[0], np.ndarray):
                        items_array = np.array(items)
                        items = [torch.from_numpy(items_array[i]).float() for i in range(len(items))]
                    else:
                        items = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, torch.Tensor) else item 
                                for item in items]
                except Exception as e:
                    print(f"Error collating key {key}: {e}")
                    continue
            
            # Check for temporal sequences (dim=4 for [T,N,F] or dim=5 for [T,N,C,H,W])
            if items[0].dim() >= 4 and key in ['satellite_data', 'visual_data', 'thermal_data', 
                                                 'scada_data', 'weather_sequence', 'threat_indicators',
                                                 'equipment_status', 'pmu_sequence', 'sensor_data']:
                first_dims = [item.shape[0] for item in items]
                # If they are sequences of different lengths (e.g. 59 vs 60)
                if len(set(first_dims)) > 1:
                    max_len = max(first_dims)
                    padded_items = []
                    for item in items:
                        if item.shape[0] < max_len:
                            pad_size = max_len - item.shape[0]
                            # Create padding with the same shape as item[0] but with size=pad_size
                            padding = torch.zeros(pad_size, *item.shape[1:], dtype=item.dtype, device=item.device)
                            padded_item = torch.cat([item, padding], dim=0)
                        else:
                            padded_item = item
                        padded_items.append(padded_item)
                    batch_dict[key] = torch.stack(padded_items, dim=0)
                else:
                    # All sequences have same length
                    batch_dict[key] = torch.stack(items, dim=0)
            
            else:
                # Standard stacking for non-sequence or batch-1 sequences
                try:
                    batch_dict[key] = torch.stack(items, dim=0)
                except Exception as e:
                    # Fallback for items that can't be stacked (e.g., labels of diff size)
                    # print(f"Warning: Could not stack key {key}, concatenating. Error: {e}")
                    try:
                        batch_dict[key] = torch.cat(items, dim=0)
                    except Exception as e2:
                        print(f"Error: Could not collate key {key}. Skipping. Error: {e2}")
    
    return batch_dict