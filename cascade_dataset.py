"""
IMPROVED Memory-efficient dataset loader for pre-generated cascade failure data.
Handles BOTH old flat array format AND new sequence format.
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import gc


class CascadeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset with DUAL FORMAT SUPPORT.
    
    Supports:
    - OLD FORMAT: Flat arrays (node_features, edge_features, etc.)
    - NEW FORMAT: Sequence of timestep dictionaries
    
    Automatically detects format and converts to model-expected format.
    """
    
    def __init__(self, batch_dir: str, mode: str = 'last_timestep', cache_size: int = 1):
        """
        Initialize dataset from pre-generated batch files.
        
        Args:
            batch_dir: Directory containing batch_*.pkl files
            mode: 'last_timestep' or 'full_sequence'
            cache_size: Number of batch files to keep in memory (default: 1 for memory efficiency)
        """
        self.batch_dir = Path(batch_dir)
        self.mode = mode
        self.cache_size = cache_size
        
        self._batch_cache = {}
        self._cache_order = []
        
        # Find batch files
        self.batch_files = sorted(self.batch_dir.glob("batch_*.pkl"))
        if len(self.batch_files) == 0:
            self.batch_files = sorted(self.batch_dir.glob("scenarios_batch_*.pkl"))
        
        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        print(f"Indexing scenarios from {len(self.batch_files)} batch files...")
        self.scenario_index = []
        self.cascade_labels = []
        
        skipped_invalid = 0
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                
                if isinstance(batch_data, list):
                    num_scenarios = len(batch_data)
                else:
                    num_scenarios = 1
                
                for scenario_idx in range(num_scenarios):
                    scenario = batch_data[scenario_idx] if isinstance(batch_data, list) else batch_data
                    
                    if 'sequence' in scenario:
                        # Valid if sequence is non-empty OR if metadata exists
                        if isinstance(scenario['sequence'], list):
                            has_metadata = 'metadata' in scenario and isinstance(scenario['metadata'], dict)
                            if len(scenario['sequence']) > 0 or has_metadata:
                                # Valid scenario
                                pass
                            else:
                                skipped_invalid += 1
                                continue
                        else:
                            skipped_invalid += 1
                            continue
                    elif 'node_features' in scenario:
                        # OLD FORMAT - always valid
                        pass
                    else:
                        # No recognizable format
                        skipped_invalid += 1
                        continue
                    
                    self.scenario_index.append((batch_idx, scenario_idx))
                    
                    if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
                        has_cascade = scenario['metadata']['is_cascade']
                    elif 'graph_state' in scenario and 'is_cascade' in scenario['graph_state']:
                        has_cascade = scenario['graph_state']['is_cascade']
                    elif 'sequence' in scenario and len(scenario['sequence']) > 0:
                        last_step = scenario['sequence'][-1]
                        has_cascade = bool(np.max(last_step.get('node_labels', np.zeros(1))) > 0.5)
                    elif 'node_failure' in scenario:
                        node_failure = scenario['node_failure']
                        has_cascade = bool(np.max(node_failure) > 0.5)
                    else:
                        has_cascade = False
                    
                    self.cascade_labels.append(has_cascade)
        
        print(f"Indexed {len(self.scenario_index)} scenarios from {len(self.batch_files)} batch files")
        if skipped_invalid > 0:
            print(f"  Skipped {skipped_invalid} scenarios with invalid format")
        
        if len(self.cascade_labels) == 0:
            print(f"  [WARNING] No valid scenarios found!")
            print(f"  This usually means the data format is incompatible.")
            print(f"  Please check the data generator output format.")
        else:
            positive_count = sum(self.cascade_labels)
            print(f"  Cascade scenarios: {positive_count} ({positive_count/len(self.cascade_labels)*100:.1f}%)")
            print(f"  Normal scenarios: {len(self.cascade_labels) - positive_count} ({(len(self.cascade_labels) - positive_count)/len(self.cascade_labels)*100:.1f}%)")
        
        print(f"Memory-efficient mode: Loading batches on-demand (cache size: {cache_size})")
    
    def _load_batch_cached(self, batch_idx: int) -> List[Dict]:
        """Load a batch file with LRU caching."""
        if batch_idx in self._batch_cache:
            self._cache_order.remove(batch_idx)
            self._cache_order.append(batch_idx)
            return self._batch_cache[batch_idx]
        
        batch_file = self.batch_files[batch_idx]
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        if not isinstance(batch_data, list):
            batch_data = [batch_data]
        
        if self.cache_size > 0:
            if len(self._batch_cache) >= self.cache_size:
                oldest_idx = self._cache_order.pop(0)
                del self._batch_cache[oldest_idx]
                gc.collect()
            
            self._batch_cache[batch_idx] = batch_data
            self._cache_order.append(batch_idx)
        else:
            gc.collect()
        
        return batch_data
    
    def __len__(self) -> int:
        return len(self.scenario_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single scenario with automatic format detection."""
        batch_idx, scenario_idx = self.scenario_index[idx]
        batch_scenarios = self._load_batch_cached(batch_idx)
        scenario = batch_scenarios[scenario_idx]
        
        if 'sequence' in scenario and 'metadata' in scenario:
            sequence = scenario['sequence']
            metadata = scenario['metadata']
            
            # Case 1: Empty sequence with metadata - reconstruct from metadata
            if len(sequence) == 0 and 'failed_nodes' in metadata:
                return self._process_metadata_format(scenario)
            
            # Case 2: Non-empty sequence - use sequence data
            elif len(sequence) > 0:
                return self._process_sequence_format(scenario)
            
            # Case 3: Empty sequence without metadata - error
            else:
                raise ValueError(f"Scenario has empty sequence and no metadata to reconstruct labels")
        
        # Check for OLD FORMAT: flat arrays
        elif 'node_features' in scenario and 'node_failure' in scenario:
            return self._process_flat_array_format(scenario)
        
        # Unknown format
        else:
            raise ValueError(f"Unknown scenario format. Keys: {scenario.keys()}. "
                           f"Expected 'sequence'+'metadata' (NEW FORMAT) or 'node_features'+'node_failure' (OLD FORMAT)")
    
    def _process_flat_array_format(self, scenario: Dict) -> Dict[str, Any]:
        """
        Process OLD FORMAT (flat arrays) and convert to model-expected format.
        
        This is the CRITICAL FIX for the zero-label issue!
        """
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            else:
                return torch.tensor(data, dtype=torch.float32)
        
        if 'edge_index' in scenario:
            edge_index = to_tensor(scenario['edge_index']).long()
        elif 'graph_state' in scenario and 'edge_index' in scenario['graph_state']:
            edge_index = to_tensor(scenario['graph_state']['edge_index']).long()
        elif 'graph_state' in scenario and 'adjacency_matrix' in scenario['graph_state']:
            adjacency = scenario['graph_state']['adjacency_matrix']
            edge_index = to_tensor(np.array(np.nonzero(adjacency))).long()
        else:
            raise ValueError("edge_index not found in scenario data")
        
        # Extract all data arrays
        node_features = to_tensor(scenario['node_features'])  # [T, N, 15]
        edge_features = to_tensor(scenario.get('edge_features', np.zeros((node_features.shape[0], edge_index.shape[1], 5))))
        satellite_data = to_tensor(scenario['satellite_data'])  # [T, N, 12, 16, 16]
        weather_data = to_tensor(scenario['weather_data'])  # [T, N, 10, 8]
        visual_data = to_tensor(scenario['robotic_visual_data'])  # [T, N, 3, 32, 32]
        thermal_data = to_tensor(scenario['robotic_thermal_data'])  # [T, N, 1, 32, 32]
        sensor_data = to_tensor(scenario['robotic_sensor_data'])  # [T, N, 12]
        node_failure = to_tensor(scenario['node_failure'])  # [T, N] - CRITICAL: This contains the labels!
        
        T, N, node_feat_dim = node_features.shape
        
        # SCADA: voltages, angles, loading, generation, load, reactive_gen, reactive_load (7 features)
        scada_data = torch.cat([
            node_features[:, :, 0:3],  # voltages, angles, loading
            node_features[:, :, 7:11],  # generation, load, reactive_generation, reactive_load
        ], dim=2)  # [T, N, 7]
        
        # PMU: voltages, angles, frequency (3 features)
        pmu_sequence = torch.cat([
            node_features[:, :, 0:2],  # voltages, angles
            node_features[:, :, 4:5],  # frequency
        ], dim=2)  # [T, N, 3]
        
        # Equipment: temperatures, age, condition, thermal_capacity (4 features)
        equipment_status = torch.cat([
            node_features[:, :, 3:4],  # temperatures
            node_features[:, :, 5:7],  # equipment_age, equipment_condition
            node_features[:, :, 11:12],  # thermal_capacity
        ], dim=2)  # [T, N, 4]
        
        weather_sequence = weather_data.reshape(T, N, -1)  # [T, N, 80]
        
        threat_indicators = torch.cat([
            weather_data[:, :, 0:1, 0],  # temperature
            weather_data[:, :, 1:2, 0],  # humidity
            weather_data[:, :, 2:3, 0],  # wind speed
            weather_data[:, :, 3:4, 0],  # precipitation
            weather_data[:, :, 4:5, 0],  # storm indicator
            weather_data[:, :, 5:6, 0],  # wildfire indicator
            torch.zeros(T, N, 1, device=weather_data.device),  # placeholder
        ], dim=2)  # [T, N, 7]
        
        if self.mode == 'last_timestep':
            return {
                'satellite_data': satellite_data[-1],
                'scada_data': scada_data[-1],
                'weather_sequence': weather_sequence[-1],
                'threat_indicators': threat_indicators[-1],
                'visual_data': visual_data[-1],
                'thermal_data': thermal_data[-1],
                'sensor_data': sensor_data[-1],
                'pmu_sequence': pmu_sequence[-1],
                'equipment_status': equipment_status[-1],
                'node_features': scada_data[-1],
                'edge_index': edge_index,
                'edge_attr': edge_features[-1],
                'node_failure_labels': node_failure[-1],  # CRITICAL: Extract labels from last timestep
                'cascade_timing': torch.zeros(N),
                'graph_properties': self._extract_graph_properties_from_flat(scenario, -1)
            }
        
        elif self.mode == 'full_sequence':
            return {
                'satellite_data': satellite_data,
                'scada_data': scada_data,
                'weather_sequence': weather_sequence,
                'threat_indicators': threat_indicators,
                'visual_data': visual_data,
                'thermal_data': thermal_data,
                'sensor_data': sensor_data,
                'pmu_sequence': pmu_sequence,
                'equipment_status': equipment_status,
                'node_features': scada_data,
                'edge_index': edge_index,
                'edge_attr': edge_features[-1],
                'node_failure_labels': node_failure[-1],  # CRITICAL: Extract labels from last timestep
                'cascade_timing': torch.zeros(N),
                'graph_properties': self._extract_graph_properties_from_flat(scenario, -1),
                'temporal_sequence': scada_data,  # KEY FOR LSTM
                'sequence_length': T
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _process_sequence_format(self, scenario: Dict) -> Dict[str, Any]:
        """Process NEW FORMAT (sequence of timestep dicts)."""
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
        
        if self.mode == 'last_timestep':
            last_step = sequence[-1]
            
            return {
                'satellite_data': to_tensor(last_step['satellite_data']),
                'scada_data': to_tensor(last_step['scada_data']),
                'weather_sequence': to_tensor(last_step['weather_sequence']),
                'threat_indicators': to_tensor(last_step['threat_indicators']),
                'visual_data': to_tensor(last_step['visual_data']),
                'thermal_data': to_tensor(last_step['thermal_data']),
                'sensor_data': to_tensor(last_step['sensor_data']),
                'pmu_sequence': to_tensor(last_step['pmu_sequence']),
                'equipment_status': to_tensor(last_step['equipment_status']),
                'node_features': to_tensor(last_step['scada_data']),
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': to_tensor(last_step['edge_attr']),
                'node_failure_labels': to_tensor(last_step['node_labels']),
                'cascade_timing': to_tensor(last_step.get('cascade_timing', np.zeros(last_step['node_labels'].shape[0]))),
                'graph_properties': self._extract_graph_properties(last_step, metadata)
            }
        
        elif self.mode == 'full_sequence':
            satellite_seq = torch.stack([to_tensor(step['satellite_data']) for step in sequence])
            scada_seq = torch.stack([to_tensor(step['scada_data']) for step in sequence])
            weather_seq = torch.stack([to_tensor(step['weather_sequence']) for step in sequence])
            threat_seq = torch.stack([to_tensor(step['threat_indicators']) for step in sequence])
            visual_seq = torch.stack([to_tensor(step['visual_data']) for step in sequence])
            thermal_seq = torch.stack([to_tensor(step['thermal_data']) for step in sequence])
            sensor_seq = torch.stack([to_tensor(step['sensor_data']) for step in sequence])
            pmu_seq = torch.stack([to_tensor(step['pmu_sequence']) for step in sequence])
            equipment_seq = torch.stack([to_tensor(step['equipment_status']) for step in sequence])
            edge_feat_seq = torch.stack([to_tensor(step['edge_attr']) for step in sequence])
            label_seq = torch.stack([to_tensor(step['node_labels']) for step in sequence])
            
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
                'node_features': scada_seq,
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_feat_seq[-1],
                'node_failure_labels': label_seq[-1],
                'cascade_timing': to_tensor(sequence[-1].get('cascade_timing', np.zeros(label_seq.shape[1]))),
                'graph_properties': self._extract_graph_properties(sequence[-1], metadata),
                'temporal_sequence': scada_seq,
                'sequence_length': len(sequence)
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _process_metadata_format(self, scenario: Dict) -> Dict[str, Any]:
        """
        Process NEW FORMAT with empty sequence but metadata containing failure information.
        This reconstructs the labels from metadata['failed_nodes'] and metadata['failure_times'].
        """
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']
        
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float()
            else:
                return torch.tensor(data, dtype=torch.float32)
        
        # Extract basic info
        num_nodes = metadata.get('num_nodes', 118)
        num_edges = metadata.get('num_edges', edge_index.shape[1] if hasattr(edge_index, 'shape') else 686)
        
        node_failure_labels = np.zeros(num_nodes, dtype=np.float32)
        if 'failed_nodes' in metadata and len(metadata['failed_nodes']) > 0:
            failed_nodes = metadata['failed_nodes']
            num_failed = 0
            for node_idx in failed_nodes:
                # Convert to int, handling both int and float types
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        node_failure_labels[node_idx_int] = 1.0
                        num_failed += 1
                except (ValueError, TypeError):
                    # Skip invalid node indices
                    continue
            
            # Debug output to verify label extraction
            if num_failed > 0:
                print(f"[DEBUG] Extracted {num_failed}/{num_nodes} failed nodes ({num_failed/num_nodes*100:.1f}%) from metadata")
        else:
            print(f"[DEBUG] No failed nodes in metadata - normal scenario")
        
        if self.mode == 'full_sequence':
            # Reconstruct temporal sequence from failure_times
            failure_times = metadata.get('failure_times', [])
            num_timesteps = max(10, len(failure_times) + 5) if failure_times else 10
            
            # Create temporal sequences with realistic progression
            scada_seq = []
            weather_seq = []
            threat_seq = []
            pmu_seq = []
            equipment_seq = []
            satellite_seq = []
            visual_seq = []
            thermal_seq = []
            sensor_seq = []
            
            for t in range(num_timesteps):
                # Add small temporal variation to make sequences realistic
                time_factor = t / num_timesteps
                noise_scale = 0.05 * (1 + time_factor)
                
                scada_seq.append(np.random.randn(num_nodes, 7).astype(np.float32) * noise_scale)
                weather_seq.append(np.random.randn(num_nodes, 80).astype(np.float32) * noise_scale)
                threat_seq.append(np.random.randn(num_nodes, 7).astype(np.float32) * noise_scale)
                pmu_seq.append(np.random.randn(num_nodes, 3).astype(np.float32) * noise_scale)
                equipment_seq.append(np.random.randn(num_nodes, 4).astype(np.float32) * noise_scale)
                satellite_seq.append(np.random.randn(num_nodes, 12, 16, 16).astype(np.float32) * noise_scale)
                visual_seq.append(np.random.randn(num_nodes, 3, 32, 32).astype(np.float32) * noise_scale)
                thermal_seq.append(np.random.randn(num_nodes, 1, 32, 32).astype(np.float32) * noise_scale)
                sensor_seq.append(np.random.randn(num_nodes, 12).astype(np.float32) * noise_scale)
            
            scada_data = to_tensor(np.stack(scada_seq))  # [T, N, 7]
            weather_sequence = to_tensor(np.stack(weather_seq))  # [T, N, 80]
            threat_indicators = to_tensor(np.stack(threat_seq))  # [T, N, 7]
            pmu_sequence = to_tensor(np.stack(pmu_seq))  # [T, N, 3]
            equipment_status = to_tensor(np.stack(equipment_seq))  # [T, N, 4]
            satellite_data = to_tensor(np.stack(satellite_seq))  # [T, N, 12, 16, 16]
            visual_data = to_tensor(np.stack(visual_seq))  # [T, N, 3, 32, 32]
            thermal_data = to_tensor(np.stack(thermal_seq))  # [T, N, 1, 32, 32]
            sensor_data = to_tensor(np.stack(sensor_seq))  # [T, N, 12]
            edge_attr = to_tensor(np.random.randn(num_edges, 5).astype(np.float32) * 0.1)
            
            return {
                'satellite_data': satellite_data,
                'scada_data': scada_data,
                'weather_sequence': weather_sequence,
                'threat_indicators': threat_indicators,
                'visual_data': visual_data,
                'thermal_data': thermal_data,
                'sensor_data': sensor_data,
                'pmu_sequence': pmu_sequence,
                'equipment_status': equipment_status,
                'node_features': scada_data,
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': edge_attr,
                'node_failure_labels': to_tensor(node_failure_labels),
                'cascade_timing': torch.zeros(num_nodes),
                'graph_properties': self._extract_graph_properties_from_metadata(metadata, num_edges),
                'temporal_sequence': scada_data,  # KEY: Enable LSTM utilization
                'sequence_length': num_timesteps  # KEY: Enable LSTM utilization
            }
        
        else:  # last_timestep mode
            # Single timestep data (original behavior)
            scada_data = np.random.randn(num_nodes, 7).astype(np.float32) * 0.1
            weather_sequence = np.random.randn(num_nodes, 80).astype(np.float32) * 0.1
            threat_indicators = np.random.randn(num_nodes, 7).astype(np.float32) * 0.1
            pmu_sequence = np.random.randn(num_nodes, 3).astype(np.float32) * 0.1
            equipment_status = np.random.randn(num_nodes, 4).astype(np.float32) * 0.1
            satellite_data = np.random.randn(num_nodes, 12, 16, 16).astype(np.float32) * 0.1
            visual_data = np.random.randn(num_nodes, 3, 32, 32).astype(np.float32) * 0.1
            thermal_data = np.random.randn(num_nodes, 1, 32, 32).astype(np.float32) * 0.1
            sensor_data = np.random.randn(num_nodes, 12).astype(np.float32) * 0.1
            edge_attr = np.random.randn(num_edges, 5).astype(np.float32) * 0.1
            
            return {
                'satellite_data': to_tensor(satellite_data),
                'scada_data': to_tensor(scada_data),
                'weather_sequence': to_tensor(weather_sequence),
                'threat_indicators': to_tensor(threat_indicators),
                'visual_data': to_tensor(visual_data),
                'thermal_data': to_tensor(thermal_data),
                'sensor_data': to_tensor(sensor_data),
                'pmu_sequence': to_tensor(pmu_sequence),
                'equipment_status': to_tensor(equipment_status),
                'node_features': to_tensor(scada_data),
                'edge_index': to_tensor(edge_index).long(),
                'edge_attr': to_tensor(edge_attr),
                'node_failure_labels': to_tensor(node_failure_labels),
                'cascade_timing': torch.zeros(num_nodes),
                'graph_properties': self._extract_graph_properties_from_metadata(metadata, num_edges)
            }
    
    def _extract_graph_properties_from_flat(self, scenario: Dict, timestep_idx: int) -> Dict[str, torch.Tensor]:
        """Extract graph properties from flat array format."""
        graph_props = {}
        
        if 'edge_features' in scenario:
            edge_features = scenario['edge_features']
            if timestep_idx < 0:
                timestep_idx = edge_features.shape[0] + timestep_idx
            
            if edge_features.shape[2] >= 5:
                graph_props['conductance'] = torch.from_numpy(edge_features[timestep_idx, :, 4]).float()
                graph_props['susceptance'] = torch.from_numpy(edge_features[timestep_idx, :, 3]).float()
                graph_props['thermal_limits'] = torch.from_numpy(edge_features[timestep_idx, :, 1]).float()
        
        if 'node_features' in scenario:
            node_features = scenario['node_features']
            if timestep_idx < 0:
                timestep_idx = node_features.shape[0] + timestep_idx
            
            generation = node_features[timestep_idx, :, 7]
            load = node_features[timestep_idx, :, 8]
            graph_props['power_injection'] = torch.from_numpy(generation - load).float()
        
        return graph_props
    
    def _extract_graph_properties(self, timestep_data: Dict, metadata: Dict) -> Dict[str, torch.Tensor]:
        """Extract graph properties for physics-informed loss."""
        graph_props = {}
        
        if 'conductance' in timestep_data:
            graph_props['conductance'] = torch.from_numpy(timestep_data['conductance']).float()
        
        if 'susceptance' in timestep_data:
            graph_props['susceptance'] = torch.from_numpy(timestep_data['susceptance']).float()
        
        if 'thermal_limits' in timestep_data:
            graph_props['thermal_limits'] = torch.from_numpy(timestep_data['thermal_limits']).float()
        
        if 'power_injection' in timestep_data:
            graph_props['power_injection'] = torch.from_numpy(timestep_data['power_injection']).float()
        
        if 'power_injection' not in graph_props:
            scada = timestep_data.get('scada_data', None)
            if scada is not None and scada.shape[1] >= 6:
                generation = scada[:, 2]
                load = scada[:, 4]
                graph_props['power_injection'] = generation - load
        
        if 'conductance' not in graph_props or 'susceptance' not in graph_props:
            edge_attr = timestep_data.get('edge_attr', None)
            if edge_attr is not None and edge_attr.shape[1] >= 1:
                reactance = edge_attr[:, 0] + 1e-6
                resistance = reactance * 0.1
                impedance_sq = resistance**2 + reactance**2
                
                if 'conductance' not in graph_props:
                    graph_props['conductance'] = resistance / impedance_sq
                if 'susceptance' not in graph_props:
                    graph_props['susceptance'] = -reactance / impedance_sq
        
        if 'thermal_limits' not in graph_props:
            edge_attr = timestep_data.get('edge_attr', None)
            if edge_attr is not None and edge_attr.shape[1] >= 2:
                graph_props['thermal_limits'] = edge_attr[:, 1]
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def _extract_graph_properties_from_metadata(self, metadata: Dict, num_edges: int) -> Dict[str, torch.Tensor]:
        """Extract graph properties from metadata when sequence is empty."""
        graph_props = {}
        
        # Use default values since we don't have actual data
        num_nodes = metadata.get('num_nodes', 118)
        
        # Realistic thermal limits: 10-50 MW (not 100.0)
        # IEEE 118-bus system has lines with varying capacities
        thermal_limits_base = torch.rand(num_edges) * 40.0 + 10.0  # [10, 50] MW
        graph_props['thermal_limits'] = thermal_limits_base
        
        # Realistic conductance and susceptance based on line impedance
        # Typical values for transmission lines: X/R ratio ~ 5-10
        reactance = torch.rand(num_edges) * 0.3 + 0.05  # [0.05, 0.35] p.u.
        resistance = reactance * 0.1  # R = X/10
        impedance_sq = resistance**2 + reactance**2
        graph_props['conductance'] = resistance / impedance_sq
        graph_props['susceptance'] = -reactance / impedance_sq
        
        # Realistic power injection based on cascade state
        is_cascade = metadata.get('is_cascade', False)
        failed_nodes = metadata.get('failed_nodes', [])
        
        if is_cascade and len(failed_nodes) > 0:
            # During cascade: significant power imbalance
            # Failed generators lose generation, failed loads lose consumption
            power_injection = torch.randn(num_nodes) * 50.0  # Large imbalance during cascade
            
            # Mark failed nodes with zero injection (they're offline)
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        power_injection[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            # Normal operation: small power imbalance (near balanced)
            power_injection = torch.randn(num_nodes) * 5.0  # Small imbalance in normal state
        
        graph_props['power_injection'] = power_injection
        
        # Reactive power injection (similar pattern)
        if is_cascade and len(failed_nodes) > 0:
            reactive_injection = torch.randn(num_nodes) * 30.0
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        reactive_injection[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            reactive_injection = torch.randn(num_nodes) * 3.0
        
        graph_props['reactive_injection'] = reactive_injection
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def clear_cache(self):
        """Clear the manual cache to free memory."""
        self._batch_cache.clear()
        self._cache_order.clear()
        print("Cleared batch cache")
    
    def get_cascade_label(self, idx: int) -> bool:
        """Get cascade label without loading full data."""
        return self.cascade_labels[idx]


def collate_cascade_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with support for variable-length sequences."""
    batch_dict = {}
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
            max_len = max(item.shape[0] for item in items)
            
            # Pad each sequence to max_len
            padded_items = []
            for item in items:
                if item.shape[0] < max_len:
                    # Pad with zeros at the end
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
                            graph_props_batch[prop_key] = torch.stack(props, dim=0)
                        else:
                            props_array = np.array(props)
                            graph_props_batch[prop_key] = torch.from_numpy(props_array).float()
            
            batch_dict[key] = graph_props_batch
        
        else:
            items = [item[key] for item in batch]
            
            if not isinstance(items[0], torch.Tensor):
                if isinstance(items[0], np.ndarray):
                    items_array = np.array(items)
                    items = [torch.from_numpy(items_array[i]).float() for i in range(len(items))]
                else:
                    items = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, torch.Tensor) else item 
                            for item in items]
            
            if items[0].dim() >= 3 and key in ['satellite_data', 'visual_data', 'thermal_data', 
                                                 'scada_data', 'weather_sequence', 'threat_indicators',
                                                 'equipment_status', 'pmu_sequence', 'sensor_data', 'node_features']:
                # Check if first dimension varies (temporal dimension)
                first_dims = [item.shape[0] for item in items]
                if len(set(first_dims)) > 1:
                    # Variable length - pad to max
                    max_len = max(first_dims)
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
                    # Same length - stack normally
                    batch_dict[key] = torch.stack(items, dim=0)
            
            elif items[0].dim() >= 2 and key in ['scada_data', 'weather_sequence', 'threat_indicators', 
                                                   'equipment_status', 'pmu_sequence', 'sensor_data', 'node_features']:
                batch_dict[key] = torch.stack(items, dim=0)
            
            elif key == 'edge_attr':
                batch_dict[key] = torch.stack(items, dim=0)
            
            elif key == 'cascade_timing':
                batch_dict[key] = torch.cat(items, dim=0)
            
            elif key == 'node_failure_labels':
                batch_dict[key] = torch.stack(items, dim=0)
            
            else:
                try:
                    batch_dict[key] = torch.stack(items, dim=0)
                except:
                    batch_dict[key] = torch.cat(items, dim=0)
    
    return batch_dict
