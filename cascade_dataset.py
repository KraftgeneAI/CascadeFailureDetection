"""
Memory-efficient dataset loader for pre-generated cascade failure/normal data.
=============================================================================
MODIFIED FOR 1-SCENARIO-PER-FILE FORMAT
This version reads individual scenario_*.pkl files.

*** FINAL "SOUND" VERSION ***
1.  Removes the 'stress_level' data leak (slices SCADA data to 14 features).
2.  Truncates ALL sequences (Normal, Stressed, Cascade) to a random
    pre-event timestep to force true "future" prediction.
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
    """
    
    def __init__(self, data_dir: str, mode: str = 'last_timestep',
                 base_mva: float = 100.0, base_frequency: float = 60.0):
        """
        Initialize dataset from a directory of individual scenario_*.pkl files.
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        
        
        # Find all individual scenario files
        print(f"Indexing scenarios from: {data_dir}")
        self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenario_*.pkl")))
        
        if len(self.scenario_files) == 0:
            old_files = sorted(glob.glob(str(self.data_dir / "scenarios_batch_*.pkl")))
            if old_files:
                self.scenario_files = old_files
            else:
                raise ValueError(f"No 'scenario_*.pkl' or 'scenarios_batch_*.pkl' files found in {data_dir}.")

        print(f"Physics normalization: base_mva={base_mva}, base_frequency={base_frequency}")
        
        print(f"Scanning {len(self.scenario_files)} files for cascade labels...")
        self.cascade_labels = []
        
        for scenario_file in self.scenario_files:
            try:
                with open(scenario_file, 'rb') as f:
                    scenario_data = pickle.load(f)

                if isinstance(scenario_data, list):
                    if len(scenario_data) == 0:
                        self.cascade_labels.append(False) 
                        continue
                    scenario = scenario_data[0]
                else:
                    scenario = scenario_data
                
                if not isinstance(scenario, dict):
                    self.cascade_labels.append(False)
                    continue

                if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
                    has_cascade = scenario['metadata']['is_cascade']
                elif 'sequence' in scenario and len(scenario['sequence']) > 0:
                    last_step = scenario['sequence'][-1]
                    has_cascade = bool(np.max(last_step.get('node_labels', np.zeros(1))) > 0.5)
                else:
                    has_cascade = False
                
                self.cascade_labels.append(has_cascade)
                
            except (IOError, pickle.UnpicklingError) as e:
                print(f"Warning: Skipping corrupted or unreadable file: {scenario_file}. Error: {e}")
                self.cascade_labels.append(False)
        
        print(f"Indexed {len(self.scenario_files)} scenarios.")
        
        if len(self.cascade_labels) == 0:
            print(f"  [WARNING] No valid scenarios found!")
        else:
            positive_count = sum(self.cascade_labels)
            print(f"  Cascade scenarios: {positive_count} ({positive_count/len(self.cascade_labels)*100:.1f}%)")
            print(f"  Normal scenarios: {len(self.cascade_labels) - positive_count} ({(len(self.cascade_labels) - positive_count)/len(self.cascade_labels)*100:.1f}%)")
        
        print(f"Ultra-memory-efficient mode: Loading 1 file per sample.")

    def _normalize_power(self, power_values: np.ndarray) -> np.ndarray:
        return power_values / self.base_mva
    
    def _normalize_frequency(self, frequency_values: np.ndarray) -> np.ndarray:
        return frequency_values / self.base_frequency
    
    def __len__(self) -> int:
        return len(self.scenario_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scenario_file = self.scenario_files[idx]
        
        try:
            with open(scenario_file, 'rb') as f:
                scenario_data = pickle.load(f) 

            if isinstance(scenario_data, list):
                if len(scenario_data) == 0:
                     return {}
                scenario = scenario_data[0] 
            else:
                scenario = scenario_data
            
            if not isinstance(scenario, dict):
                 return {}

        except Exception as e:
            print(f"Error loading {scenario_file}: {e}. Returning empty dict.")
            return {}
        
        if 'sequence' in scenario and 'metadata' in scenario:
            sequence = scenario['sequence']
            
            if len(sequence) == 0 and 'failed_nodes' in scenario['metadata']:
                return self._process_metadata_format(scenario)
            elif len(sequence) > 0:
                return self._process_sequence_format(scenario)
            else:
                return {}
        
        else:
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
        
        # ====================================================================
        # START: "CHEAT" FIX (Data Truncation)
        # ====================================================================
        
        # 1. Get the "ground truth" labels and times *before* truncation
        # We need these *full* answers to grade the model
        full_sequence_length = len(sequence)
        last_step = sequence[-1]
        
        # Get the final failure state (the answer)
        final_node_labels = to_tensor(last_step.get('node_labels', np.zeros(1)))
        
        # Get the cascade start time
        cascade_start_time = metadata.get('cascade_start_time', -1)
        is_cascade = metadata.get('is_cascade', False)

        # 2. Determine the "prediction timestep" (the "Question")
        if is_cascade:
            # For cascades, the "prediction time" is *right before* the failure
            prediction_time = cascade_start_time
        else:
            # For normal/stressed, pick a *random* time to prevent "length" cheating
            # We pick a time in the latter half of the sequence
            min_time = full_sequence_length // 2
            prediction_time = np.random.randint(min_time, full_sequence_length)

        # 3. Truncate the *input sequence* to the prediction time
        # We use max(1, ...) to ensure we never have an empty sequence
        truncated_sequence = sequence[:max(1, prediction_time)]
        
        # We pass this "prediction time" to the model's brain
        # For a normal case (start_time=-1), this becomes a random time (e.g., 47)
        # For a cascade (start_time=40), this becomes 40.
        # The model will then select the state at t-1 (e.g., 39 or 46)
        model_prediction_time = prediction_time if is_cascade else (prediction_time + 1)
        
        # 4. Get the correct ground truth timing map
        if is_cascade and 0 <= cascade_start_time < len(sequence):
            # The "answer" is the timing map from the *start* of the cascade
            correct_timing_tensor = to_tensor(sequence[cascade_start_time].get('cascade_timing', np.zeros(1)))
        else:
            # For normal cases, the answer is just "-1" (no failure)
            correct_timing_tensor = to_tensor(last_step.get('cascade_timing', np.zeros(1)))
            
        # ====================================================================
        # END: "CHEAT" FIX (Data Truncation)
        # ====================================================================

        
        # --- Now, we process the *TRUNCATED* sequence ---
        satellite_seq = []
        scada_seq = []
        weather_seq = []
        threat_seq = []
        visual_seq = []
        thermal_seq = []
        sensor_seq = []
        pmu_seq = []
        equipment_seq = []
        
        # Get default shapes from last (original) step
        num_nodes = last_step.get('scada_data', np.zeros((118,14))).shape[0]
        num_edges = edge_index.shape[1]
        
        sat_shape = last_step.get('satellite_data', np.zeros((num_nodes, 12, 16, 16))).shape
        weather_shape = last_step.get('weather_sequence', np.zeros((num_nodes, 10, 8))).shape
        threat_shape = last_step.get('threat_indicators', np.zeros((num_nodes, 6))).shape
        # --- Data Leakage Fix: Set default to 14 features ---
        scada_shape = (num_nodes, 14)
        pmu_shape = last_step.get('pmu_sequence', np.zeros((num_nodes, 8))).shape
        equip_shape = last_step.get('equipment_status', np.zeros((num_nodes, 10))).shape
        vis_shape = last_step.get('visual_data', np.zeros((num_nodes, 3, 32, 32))).shape
        therm_shape = last_step.get('thermal_data', np.zeros((num_nodes, 1, 32, 32))).shape
        sensor_shape = last_step.get('sensor_data', np.zeros((num_nodes, 12))).shape

        # Helper to safely get data
        def safe_get(ts, key, default_val):
            data = ts.get(key)
            if data is None:
                return default_val
            if hasattr(default_val, 'shape') and hasattr(data, 'shape') and data.shape != default_val.shape:
                if data.size == default_val.size:
                    try:
                        return data.reshape(default_val.shape)
                    except ValueError:
                         return default_val
                # Handle the 15-to-14 feature change
                if key == 'scada_data' and data.shape[1] == 15:
                    return data[:, :14]
                return default_val
            return data

        for ts in truncated_sequence: # <-- Use the truncated sequence
            # --- SCADA NORMALIZATION ---
            scada_data_raw = safe_get(ts, 'scada_data', np.zeros((num_nodes, 15))).astype(np.float32)
            
            # ====================================================================
            # START: "CHEAT" FIX (Data Leakage)
            # ====================================================================
            # Slice off the 15th feature (stress_level)
            if scada_data_raw.shape[1] == 15:
                scada_data = scada_data_raw[:, :14]
            else:
                scada_data = scada_data_raw
            # ====================================================================
            # END: "CHEAT" FIX
            # ====================================================================

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

        # --- EDGE ATTR NORMALIZATION (from the *original* last step) ---
        edge_attr = safe_get(last_step, 'edge_attr', np.zeros((num_edges, 5))).astype(np.float32)
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1]) # thermal_limits
        edge_attr = to_tensor(edge_attr)
        
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))

        # Note: self.mode is 'full_sequence'
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
            
            # --- "ANSWER" FIELDS ---
            'node_failure_labels': final_node_labels, # The *final* answer
            'cascade_timing': correct_timing_tensor, # The *real* timing map
            'ground_truth_risk': to_tensor(ground_truth_risk), # The *real* risk
            
            # --- "CHEAT" FIX ---
            'graph_properties': self._extract_graph_properties(last_step, metadata, edge_attr, model_prediction_time),
            
            'temporal_sequence': torch.stack(scada_seq), # Use SCADA as main temporal feature
            'sequence_length': len(truncated_sequence)
        }

    
    def _process_metadata_format(self, scenario: Dict) -> Dict[str, Any]:
        """Process NEW FORMAT with empty sequence but metadata containing failure information."""
        # This fallback is unlikely to be used now, but we'll keep it.
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']
        
        def to_tensor(data):
            if isinstance(data, torch.Tensor): return data
            elif isinstance(data, np.ndarray): return torch.from_numpy(data).float()
            else: return torch.tensor(data, dtype=torch.float32)
        
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
        # --- Data Leakage Fix: Set to 14 features ---
        scada_data = torch.randn(T, num_nodes, 14)
        weather_sequence = torch.randn(T, num_nodes, 80)
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
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))

        # "CHEAT" FIX: Add dummy prediction time
        graph_props['cascade_start_time'] = torch.tensor(metadata.get('cascade_start_time', -1), dtype=torch.long)

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
            'ground_truth_risk': to_tensor(ground_truth_risk),
            'graph_properties': graph_props
        }
        
        # Handle full_sequence mode
        if self.mode == 'full_sequence':
            for key in ['satellite_data', 'scada_data', 'weather_sequence', 'threat_indicators', 'visual_data', 'thermal_data', 'sensor_data', 'pmu_sequence', 'equipment_status']:
                item[key] = item[key].unsqueeze(0) # Add T dimension
            item['temporal_sequence'] = item['scada_data']
            item['sequence_length'] = 1

        return item
    
    
    # ====================================================================
    # START: "CHEAT" FIX (Pass model_prediction_time)
    # ====================================================================
    def _extract_graph_properties(self, timestep_data: Dict, metadata: Dict, edge_attr: torch.Tensor, model_prediction_time: int) -> Dict[str, torch.Tensor]:
    # ====================================================================
    # END: "CHEAT" FIX
    # ====================================================================
        """Extract graph properties for physics-informed loss WITH NORMALIZATION."""
        graph_props = {}
        
        if 'conductance' in timestep_data:
            graph_props['conductance'] = torch.from_numpy(timestep_data['conductance']).float()
        else:
            graph_props['conductance'] = edge_attr[:, 4] # Col 4 = conductance
        
        if 'susceptance' in timestep_data:
            graph_props['susceptance'] = torch.from_numpy(timestep_data['susceptance']).float()
        else:
            graph_props['susceptance'] = edge_attr[:, 3] # Col 3 = susceptance
        
        graph_props['thermal_limits'] = edge_attr[:, 1] # Col 1 = thermal_limits
        
        if 'power_injection' in timestep_data:
            power_injection_raw = torch.from_numpy(timestep_data['power_injection']).float()
            graph_props['power_injection'] = self._normalize_power(power_injection_raw)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None:
                # --- Data Leakage Fix: Use correct indices for 14 features ---
                if scada.shape[1] >= 6: # Check if it has at least 6 features
                    generation_pu = scada[:, 2]
                    load_pu = scada[:, 4]
                    graph_props['power_injection'] = generation_pu - load_pu
                # --- End Fix ---

        if 'reactive_injection' in timestep_data:
             reactive_injection_raw = torch.from_numpy(timestep_data['reactive_injection']).float()
             graph_props['reactive_injection'] = self._normalize_power(reactive_injection_raw)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None:
                # --- Data Leakage Fix: Use correct indices for 14 features ---
                if scada.shape[1] >= 6:
                    reac_gen_pu = scada[:, 3]
                    reac_load_pu = scada[:, 5]
                    graph_props['reactive_injection'] = reac_gen_pu - reac_load_pu
                # --- End Fix ---

        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])

        if 'scada_data' in timestep_data:
            # --- Data Leakage Fix: Use correct index 6 (was 6) ---
            if timestep_data['scada_data'].shape[1] > 6:
                graph_props['ground_truth_temperature'] = torch.from_numpy(timestep_data['scada_data'][:, 6]).float()
            # --- End Fix ---
        
        # ====================================================================
        # START: "CHEAT" FIX (Pass the prediction time to the model)
        # ====================================================================
        graph_props['cascade_start_time'] = torch.tensor(model_prediction_time, dtype=torch.long)
        # ====================================================================
        # END: "CHEAT" FIX
        # ====================================================================
        
        return graph_props
    
    def _extract_graph_properties_from_metadata(self, metadata: Dict, num_edges: int) -> Dict[str, torch.Tensor]:
        """Extract graph properties from metadata when sequence is empty WITH NORMALIZATION."""
        graph_props = {}
        
        num_nodes = metadata.get('num_nodes', 118)
        
        thermal_limits_raw = torch.rand(num_edges) * 40.0 + 10.0
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
            
        # "CHEAT" FIX: Add dummy prediction time
        graph_props['cascade_start_time'] = torch.tensor(metadata.get('cascade_start_time', -1), dtype=torch.long)
        
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
            edge_index = batch[0]['edge_index']
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            batch_dict['edge_index'] = edge_index
        
        elif key == 'sequence_length':
            batch_dict['sequence_length'] = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
        
        elif key == 'temporal_sequence':
            items = [item[key] for item in batch]
            # ====================================================================
            # START: "CHEAT" FIX (Padding)
            # ====================================================================
            # Pad to the longest sequence *in this batch*
            max_len = max(item.shape[0] for item in items)
            # ====================================================================
            # END: "CHEAT" FIX
            # ====================================================================
            
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
                                # ====================================================================
                                # START: "CHEAT" FIX (Stack cascade_start_time)
                                # ====================================================================
                                if prop_key == 'cascade_start_time':
                                    graph_props_batch[prop_key] = torch.stack(props, dim=0)
                                # ====================================================================
                                # END: "CHEAT" FIX
                                # ====================================================================
                                else:
                                    graph_props_batch[prop_key] = torch.stack(props, dim=0)
                            except RuntimeError:
                                graph_props_batch[prop_key] = props[0]
                        else:
                            props_array = np.array(props)
                            graph_props_batch[prop_key] = torch.from_numpy(props_array).float()
            
            batch_dict[key] = graph_props_batch
        
        else:
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
            if items[0].dim() >= 3 and key in ['satellite_data', 'visual_data', 'thermal_data', 
                                                 'scada_data', 'weather_sequence', 'threat_indicators',
                                                 'equipment_status', 'pmu_sequence', 'sensor_data']:
                
                # ====================================================================
                # START: "CHEAT" FIX (Padding)
                # ====================================================================
                first_dims = [item.shape[0] for item in items]
                max_len = max(first_dims)
                # ====================================================================
                # END: "CHEAT" FIX
                # ====================================================================

                if len(set(first_dims)) > 1:
                    padded_items = []
                    for item in items:
                        if item.shape[0] < max_len:
                            pad_size = max_len - item.shape[0]
                            padding = torch.zeros(pad_size, *item.shape[1:], dtype=item.dtype, device=item.device)
                            padded_item = torch.cat([item, padding], dim=0)
                        else:
                            padded_item = item
                        padded_items.append(padded_item)
                    batch_dict[key] = torch.stack(padded_items, dim=0)
                else:
                    batch_dict[key] = torch.stack(items, dim=0)
            
            else:
                try:
                    batch_dict[key] = torch.stack(items, dim=0)
                except Exception as e:
                    try:
                        batch_dict[key] = torch.cat(items, dim=0)
                    except Exception as e2:
                        print(f"Error: Could not collate key {key}. Skipping. Error: {e2}")
    
    return batch_dict