"""
Cascade Dataset Module
======================
Memory-efficient dataset loader for pre-generated cascade failure data.

This module provides the main CascadeDataset class that:
- Loads scenarios from individual pickle files
- Applies physics-based normalization
- Implements sliding window truncation
- Creates dynamic topology masks
- Handles both cascade and normal scenarios
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import glob
from tqdm import tqdm
import json

from .generator.config import Settings

from .preprocessing import (
    normalize_power,
    normalize_frequency,
    calculate_truncation_window,
    apply_truncation,
    create_edge_mask_from_failures,
    to_tensor,
)


class CascadeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for cascade failure scenarios.
    
    Features:
    - One scenario per file loading
    - Physics-based normalization
    - Sliding window truncation
    - Dynamic topology masking
    - Metadata caching for fast initialization
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'last_timestep',
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_frequency: float = Settings.Dataset.BASE_FREQUENCY
    ):
        """
        Initialize dataset from a directory of scenario files.
        
        Args:
            data_dir: Directory containing scenario_*.pkl files
            mode: Dataset mode ('last_timestep' or 'full_sequence')
            base_mva: Base MVA for power normalization
            base_frequency: Base frequency for normalization (Hz)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        
        # Find all scenario files
        print(f"Indexing scenarios from: {data_dir}")
        self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenario_*.pkl")))
        if not self.scenario_files:
            self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenarios_batch_*.pkl")))
        
        # Define cache path
        cache_file = self.data_dir / "metadata_cache.json"
        self.cascade_labels = []
        
        # Load from cache if available
        if os.path.exists(cache_file):
            print(f"Loading labels from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                self.cascade_labels = json.load(f)
            
            # Safety check
            if len(self.cascade_labels) != len(self.scenario_files):
                print("Warning: Cache length mismatch. Re-scanning...")
                self.cascade_labels = []
        
        # Scan files if no cache
        if not self.cascade_labels:
            print(f"Scanning {len(self.scenario_files)} files for metadata (First Run)...")
            
            for scenario_file in tqdm(self.scenario_files):
                try:
                    with open(scenario_file, 'rb') as f:
                        scenario_data = pickle.load(f)
                    
                    # Handle list vs dict wrapper
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
                    
                    # Extract label
                    if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
                        has_cascade = scenario['metadata']['is_cascade']
                    elif 'sequence' in scenario and len(scenario['sequence']) > 0:
                        last_step = scenario['sequence'][-1]
                        has_cascade = bool(np.max(last_step.get('node_labels', np.zeros(1))) > Settings.Dataset.CASCADE_LABEL_THRESHOLD)
                    else:
                        has_cascade = False
                    
                    self.cascade_labels.append(has_cascade)
                
                except (IOError, pickle.UnpicklingError, EOFError) as e:
                    print(f"Warning: Skipping corrupted file: {scenario_file}")
                    self.cascade_labels.append(False)
            
            # Save cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.cascade_labels, f)
                print(f"Saved metadata cache to {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
        
        # Print statistics
        print(f"Physics normalization: base_mva={base_mva}, base_frequency={base_frequency}")
        print(f"Indexed {len(self.scenario_files)} scenarios.")
        
        if len(self.cascade_labels) == 0:
            print(f"  [WARNING] No valid scenarios found!")
        else:
            positive_count = sum(self.cascade_labels)
            total = len(self.cascade_labels)
            print(f"  Cascade scenarios: {positive_count} ({positive_count/total*100:.1f}%)")
            print(f"  Normal scenarios: {total - positive_count} ({(total - positive_count)/total*100:.1f}%)")
        
        print(f"Ultra-memory-efficient mode: Loading 1 file per sample.")
    
    def __len__(self) -> int:
        return len(self.scenario_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single scenario.
        
        Args:
            idx: Scenario index
        
        Returns:
            Dictionary containing all processed data for the scenario
        """
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
        
        # Process based on format
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
        """
        Process sequence format with normalization and truncation.
        
        This method:
        1. Applies sliding window truncation
        2. Normalizes power and frequency values
        3. Creates dynamic edge masks
        4. Extracts graph properties
        
        Args:
            scenario: Scenario dictionary with 'sequence' and 'metadata'
        
        Returns:
            Processed data dictionary
        """
        sequence_original = scenario['sequence']
        edge_index = scenario['edge_index']
        metadata = scenario.get('metadata', {})
        
        # Apply truncation
        cascade_start_time = metadata.get('cascade_start_time', -1)
        is_cascade = metadata.get('is_cascade', False)
        
        start_idx, end_idx = calculate_truncation_window(
            len(sequence_original),
            cascade_start_time,
            is_cascade
        )
        
        sequence = apply_truncation(sequence_original, start_idx, end_idx)
        
        # Initialize data containers
        data_arrays = {
            'satellite_data': [],
            'scada_data': [],
            'weather_sequence': [],
            'threat_indicators': [],
            'visual_data': [],
            'thermal_data': [],
            'sensor_data': [],
            'pmu_sequence': [],
            'equipment_status': [],
            'edge_mask': []
        }
        
        last_step = sequence[-1]
        
        num_nodes = last_step.get('scada_data', np.zeros((Settings.Dataset.DEFAULT_NUM_NODES, 12))).shape[0]
        num_edges = edge_index.shape[1]
        timing_shape = last_step.get('cascade_timing', np.zeros(num_nodes)).shape

        # Process each timestep
        for i, ts in enumerate(sequence):
            # SCADA data with normalization (now 18 features with failure proximity ratios)
            scada_data_raw = ts.get('scada_data', np.zeros((num_nodes, 18))).astype(np.float32)
            
            if scada_data_raw.shape[1] >= 6:
                scada_data_raw[:, 2] = normalize_power(scada_data_raw[:, 2], self.base_mva)
                scada_data_raw[:, 3] = normalize_power(scada_data_raw[:, 3], self.base_mva)
                scada_data_raw[:, 4] = normalize_power(scada_data_raw[:, 4], self.base_mva)
            
            # Keep all 18 features, update time_ratio to be relative to window
            scada_data = scada_data_raw.copy()
            if scada_data.shape[1] > 12:
                # Feature 12 is time_ratio. Change it to delta_t (progress within this specific window)
                # i is the current step in the truncated sequence, len(sequence) is total steps
                scada_data[:, 12] = i / max(1, len(sequence)) 
                # scada_data[:, 13:18] = 0
                # 13: stress_level - CRITICAL for prediction!
                # 14: voltage_ratio (voltage / voltage_failure_threshold)
                # 15: temp_ratio (temp / temp_failure_threshold)
                # 16: freq_ratio (freq / freq_failure_threshold)
                # 17: loading_ratio (loading / loading_failure_threshold)
                
            data_arrays['scada_data'].append(to_tensor(scada_data))
            
            # PMU data with normalization
            pmu_data = ts.get('pmu_sequence', np.zeros((num_nodes, 8))).astype(np.float32)
            if pmu_data.shape[1] >= 6:
                pmu_data[:, 5] = normalize_frequency(pmu_data[:, 5], self.base_frequency)
            data_arrays['pmu_sequence'].append(to_tensor(pmu_data))
            
            # Other modalities
            weather_data = ts.get('weather_sequence', np.zeros((num_nodes, 10, 8))).astype(np.float32)
            data_arrays['weather_sequence'].append(to_tensor(weather_data.reshape(num_nodes, -1)))
            
            data_arrays['satellite_data'].append(to_tensor(ts.get('satellite_data', np.zeros((num_nodes, 12, 16, 16)))))
            data_arrays['threat_indicators'].append(to_tensor(ts.get('threat_indicators', np.zeros((num_nodes, 6)))))
            data_arrays['visual_data'].append(to_tensor(ts.get('visual_data', np.zeros((num_nodes, 3, 32, 32)))))
            data_arrays['thermal_data'].append(to_tensor(ts.get('thermal_data', np.zeros((num_nodes, 1, 32, 32)))))
            data_arrays['sensor_data'].append(to_tensor(ts.get('sensor_data', np.zeros((num_nodes, 12)))))
            data_arrays['equipment_status'].append(to_tensor(ts.get('equipment_status', np.zeros((num_nodes, 10)))))
            
            # Create edge mask using t-1 failures
            global_idx = start_idx + i
            prev_failed_node_indices = []
            
            if global_idx > 0:
                prev_ts = sequence_original[global_idx - 1]
                prev_status = prev_ts.get('node_labels', np.zeros(num_nodes))
                prev_failed_node_indices = np.where(prev_status > 0.5)[0]
            
            edge_mask = create_edge_mask_from_failures(
                edge_index,
                prev_failed_node_indices,
                num_edges
            )
            # Ensure edge_mask is a tensor
            if not isinstance(edge_mask, torch.Tensor):
                edge_mask = to_tensor(edge_mask)
            data_arrays['edge_mask'].append(edge_mask)
        
        # Process edge attributes
        edge_attr = last_step.get('edge_attr', np.zeros((num_edges, 7))).astype(np.float32)
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = normalize_power(edge_attr[:, 1], self.base_mva)
            edge_attr[:, 5] = normalize_power(edge_attr[:, 5], self.base_mva)
            edge_attr[:, 6] = normalize_power(edge_attr[:, 6], self.base_mva)
        edge_attr = to_tensor(edge_attr)
        
        # Ground truth labels and timing
        final_labels = to_tensor(sequence_original[-1].get('node_labels', np.zeros(num_nodes)))
        
        # Timing with shift correction
        # CRITICAL: cascade_timing values are RELATIVE times (minutes until failure),
        # NOT absolute timesteps. Do NOT subtract start_idx from them!
        original_cascade_start_time = metadata.get('cascade_start_time', -1)
        if is_cascade and 0 <= original_cascade_start_time < len(sequence_original):
            correct_timing_tensor = torch.full(timing_shape, -1.0, dtype=torch.float32)
            
           # ====================================================================
            # START: TIMING SHIFT FIX (Crucial for Sliding Window)
            # ====================================================================
            failed_nodes_list = metadata.get('failed_nodes', [])
            failure_times_list = metadata.get('failure_times', [])

            # Build boolean mask over nodes
            mask_failure = torch.zeros(num_nodes, dtype=torch.bool)
            if failed_nodes_list:
                mask_failure[failed_nodes_list] = True

            # Assign shifted timing for failing nodes
            if failed_nodes_list:
                shifted = torch.tensor(failure_times_list, dtype=torch.float32) - start_idx
                correct_timing_tensor[mask_failure] = shifted

            # Normalize if max_time_horizon is set
            if hasattr(self, 'max_time_horizon') and self.max_time_horizon > 0:
                correct_timing_tensor[mask_failure] = correct_timing_tensor[mask_failure] / self.max_time_horizon
            # ====================================================================
            # END: TIMING SHIFT FIX
            # ====================================================================
        else:
            correct_timing_tensor = to_tensor(np.full(num_nodes, -1, dtype=np.float32))
        
        # Data augmentation (training only)
        is_training = 'train' in str(self.data_dir)
        scada_tensor = torch.stack(data_arrays['scada_data'])
        
        if is_training:
            noise = torch.randn_like(scada_tensor) * Settings.Dataset.AUGMENTATION_NOISE_STD
            scada_tensor = scada_tensor + noise
        
        return {
            # Masked modalities (zeroed)
            'satellite_data': torch.zeros_like(torch.stack(data_arrays['satellite_data'])),
            'weather_sequence': torch.zeros_like(torch.stack(data_arrays['weather_sequence'])),
            'threat_indicators': torch.zeros_like(torch.stack(data_arrays['threat_indicators'])),
            'visual_data': torch.zeros_like(torch.stack(data_arrays['visual_data'])),
            'thermal_data': torch.zeros_like(torch.stack(data_arrays['thermal_data'])),
            'sensor_data': torch.zeros_like(torch.stack(data_arrays['sensor_data'])),
            # Active modalities
            'scada_data': scada_tensor,
            'pmu_sequence': torch.stack(data_arrays['pmu_sequence']),
            'equipment_status': torch.stack(data_arrays['equipment_status']),
            'edge_index': to_tensor(edge_index).long(),
            'edge_attr': edge_attr,
            'edge_mask': torch.stack(data_arrays['edge_mask']),
            'node_failure_labels': final_labels,
            'cascade_timing': correct_timing_tensor,
            'ground_truth_risk': to_tensor(metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))),
            'graph_properties': self._extract_graph_properties(last_step, metadata, edge_attr),
            'temporal_sequence': scada_tensor,
            'sequence_length': len(sequence)
        }
    
    def _process_metadata_format(self, scenario: Dict) -> Dict[str, Any]:
        """
        Process metadata-only format (fallback for empty sequences).
        
        Args:
            scenario: Scenario dictionary with metadata but no sequence
        
        Returns:
            Processed data dictionary with synthetic data
        """
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']
        
        num_nodes = metadata.get('num_nodes', Settings.Dataset.DEFAULT_NUM_NODES)
        num_edges = metadata.get('num_edges', edge_index.shape[1] if hasattr(edge_index, 'shape') else 186)
        
        # Create node failure labels
        node_failure_labels = np.zeros(num_nodes, dtype=np.float32)
        if 'failed_nodes' in metadata and len(metadata['failed_nodes']) > 0:
            for node_idx in metadata['failed_nodes']:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        node_failure_labels[node_idx_int] = 1.0
                except (ValueError, TypeError):
                    continue
        
        # Create synthetic data
        T = 1
        scada_data = torch.randn(T, num_nodes, 13)
        weather_sequence = torch.randn(T, num_nodes, 80)
        threat_indicators = torch.randn(T, num_nodes, 6)
        pmu_sequence = torch.randn(T, num_nodes, 8)
        equipment_status = torch.randn(T, num_nodes, 10)
        satellite_data = torch.randn(T, num_nodes, 12, 16, 16)
        visual_data = torch.randn(T, num_nodes, 3, 32, 32)
        thermal_data = torch.randn(T, num_nodes, 1, 32, 32)
        sensor_data = torch.randn(T, num_nodes, 12)
        edge_attr = torch.randn(num_edges, 7)
        edge_mask = torch.ones(T, num_edges)
        
        # Apply normalization
        edge_attr[:, 1] = normalize_power(edge_attr[:, 1], self.base_mva)
        scada_data[..., 2:6] = normalize_power(scada_data[..., 2:6], self.base_mva)
        pmu_sequence[..., 5] = normalize_frequency(pmu_sequence[..., 5], self.base_frequency)
        
        graph_props = self._extract_graph_properties_from_metadata(metadata, num_edges)
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))
        
        item = {
            # Masked modalities
            'satellite_data': torch.zeros_like(satellite_data[0]),
            'weather_sequence': torch.zeros_like(weather_sequence[0]),
            'threat_indicators': torch.zeros_like(threat_indicators[0]),
            'visual_data': torch.zeros_like(visual_data[0]),
            'thermal_data': torch.zeros_like(thermal_data[0]),
            'sensor_data': torch.zeros_like(sensor_data[0]),
            # Active modalities
            'scada_data': scada_data[0],
            'pmu_sequence': pmu_sequence[0],
            'equipment_status': equipment_status[0],
            'edge_index': to_tensor(edge_index).long(),
            'edge_attr': edge_attr,
            'edge_mask': edge_mask,
            'node_failure_labels': to_tensor(node_failure_labels),
            'cascade_timing': torch.zeros(num_nodes),
            'ground_truth_risk': to_tensor(ground_truth_risk),
            'graph_properties': graph_props
        }
        
        if self.mode == 'full_sequence':
            for key in ['satellite_data', 'scada_data', 'weather_sequence', 'threat_indicators',
                       'visual_data', 'thermal_data', 'sensor_data', 'pmu_sequence',
                       'equipment_status', 'edge_mask']:
                item[key] = item[key].unsqueeze(0)
            item['temporal_sequence'] = item['scada_data']
            item['sequence_length'] = 1
        
        return item
    
    def _extract_graph_properties(
        self,
        timestep_data: Dict,
        metadata: Dict,
        edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract graph properties from timestep data."""
        graph_props = {}
        
        # Conductance and susceptance
        if 'conductance' in timestep_data:
            graph_props['conductance'] = torch.from_numpy(timestep_data['conductance']).float()
        else:
            graph_props['conductance'] = edge_attr[:, 4]
        
        if 'susceptance' in timestep_data:
            graph_props['susceptance'] = torch.from_numpy(timestep_data['susceptance']).float()
        else:
            graph_props['susceptance'] = edge_attr[:, 3]
        
        graph_props['thermal_limits'] = edge_attr[:, 1]
        
        # Power injection
        if 'power_injection' in timestep_data:
            power_injection_raw = torch.from_numpy(timestep_data['power_injection']).float()
            graph_props['power_injection'] = normalize_power(power_injection_raw, self.base_mva)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None and scada.shape[1] >= 6:
                generation_pu = scada[:, 2]
                load_pu = scada[:, 4]
                graph_props['power_injection'] = generation_pu - load_pu
        
        # Reactive injection
        if 'reactive_injection' in timestep_data:
            reactive_injection_raw = torch.from_numpy(timestep_data['reactive_injection']).float()
            graph_props['reactive_injection'] = normalize_power(reactive_injection_raw, self.base_mva)
        else:
            scada = timestep_data.get('scada_data', None)
            if scada is not None and scada.shape[1] >= 6:
                reac_gen_pu = scada[:, 3]
                reac_load_pu = scada[:, 5]
                graph_props['reactive_injection'] = reac_gen_pu - reac_load_pu
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        # Ground truth temperature (index 5 in scada_data)
        # CRITICAL: scada_data[:, 5] is equipment_temps, NOT frequency!
        # scada_data[:, 6] is frequency (around 60 Hz) - DO NOT USE for temperature!
        if 'scada_data' in timestep_data:
            scada_data = timestep_data['scada_data']
            if scada_data.shape[1] > 5:
                # Extract equipment_temps from index 5 (range: 25-150°C)
                graph_props['ground_truth_temperature'] = torch.from_numpy(scada_data[:, 5]).float()
            else:
                graph_props['ground_truth_temperature'] = torch.zeros(scada_data.shape[0])
        
        return graph_props
    
    def _extract_graph_properties_from_metadata(
        self,
        metadata: Dict,
        num_edges: int
    ) -> Dict[str, torch.Tensor]:
        """Extract graph properties from metadata (synthetic)."""
        graph_props = {}
        num_nodes = metadata.get('num_nodes', Settings.Dataset.DEFAULT_NUM_NODES)
        
        # Thermal limits
        thermal_limits_raw = torch.rand(num_edges) * 40.0 + 10.0
        graph_props['thermal_limits'] = normalize_power(thermal_limits_raw, self.base_mva)
        
        # Conductance and susceptance
        reactance = torch.rand(num_edges) * 0.3 + 0.05
        resistance = reactance * 0.1
        impedance_sq = resistance**2 + reactance**2
        graph_props['conductance'] = resistance / impedance_sq
        graph_props['susceptance'] = -reactance / impedance_sq
        
        # Power injection
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
        
        graph_props['power_injection'] = normalize_power(power_injection_raw, self.base_mva)
        
        # Reactive injection
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
        
        graph_props['reactive_injection'] = normalize_power(reactive_injection_raw, self.base_mva)
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def get_cascade_label(self, idx: int) -> bool:
        """Get cascade label without loading full data."""
        return self.cascade_labels[idx]
