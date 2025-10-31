"""
Cascade Failure Prediction Model Inference Script 
============================================================

Author: Kraftgene AI Inc.
Date: October 2025
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
import glob
import time  # Added for timing
import sys

# Ensure the model class is importable
try:
    from multimodal_cascade_model import UnifiedCascadePredictionModel
except ImportError:
    print("Error: Could not import UnifiedCascadePredictionModel.")
    print("Please ensure multimodal_cascade_model.py is in your Python path.")
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class CascadePredictor:
    """Inference engine for cascade prediction with physics-informed normalization."""
    
    def __init__(self, model_path: str, topology_path: str, device: str = "cpu", 
                 base_mva: float = 100.0,
                 base_frequency: float = 60.0):
        """
        Initialize predictor with normalization parameters.
        
        Args:
            model_path: Path to trained model checkpoint
            topology_path: Path to grid topology file
            device: Device to run inference on
            base_mva: Base MVA for power normalization (default: 100.0)
            base_frequency: Base frequency in Hz for normalization (default: 60.0)
        """
        self.device = torch.device(device)
        
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        
        # Load topology
        print(f"Loading grid topology from {topology_path}...")
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
            # --- FIX 1 (AttributeError): Convert numpy array to torch tensor ---
            if not isinstance(topology['edge_index'], torch.Tensor):
                edge_index_numpy = topology['edge_index']
                self.edge_index = torch.from_numpy(edge_index_numpy).long().to(self.device)
            else:
                self.edge_index = topology['edge_index'].to(self.device)
            # --- END FIX 1 ---

            # --- FIX 2 (KeyError): Infer num_nodes from adjacency_matrix shape ---
            if 'num_nodes' in topology:
                self.num_nodes = topology['num_nodes']
            elif 'adjacency_matrix' in topology:
                self.num_nodes = topology['adjacency_matrix'].shape[0]
            else:
                raise KeyError("Could not determine num_nodes from topology file. Missing 'num_nodes' and 'adjacency_matrix' keys.")
            # --- END FIX 2 ---
            
            self.topology = topology
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # --- IMPROVEMENT: Load thresholds from checkpoint ---
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.5)
        self.node_threshold = checkpoint.get('node_threshold', 0.5)
        # --- END IMPROVEMENT ---
        
        # Initialize model architecture
        self.model = UnifiedCascadePredictionModel(
            embedding_dim=128,
            hidden_dim=128,
            num_gnn_layers=3,
            heads=4,
            dropout=0.1
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Best validation loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Cascade threshold: {self.cascade_threshold:.4f} (Loaded from checkpoint)")
        print(f"  Node threshold: {self.node_threshold:.4f} (Loaded from checkpoint)")
        print(f"  Normalization: base_mva={self.base_mva}, base_freq={self.base_frequency}")
    
    
    def _normalize_power(self, power_values: np.ndarray) -> np.ndarray:
        """Normalize power values to per-unit (divide by base_mva)."""
        return power_values / self.base_mva
    
    def _normalize_frequency(self, frequency_values: np.ndarray) -> np.ndarray:
        """Normalize frequency to per-unit (divide by base_frequency)."""
        return frequency_values / self.base_frequency
    
    def _denormalize_power(self, power_pu: np.ndarray) -> np.ndarray:
        """Convert per-unit power back to MW."""
        return power_pu * self.base_mva
    
    def _denormalize_frequency(self, frequency_pu: np.ndarray) -> np.ndarray:
        """Convert per-unit frequency back to Hz."""
        return frequency_pu * self.base_frequency
    
    def load_data(self, data_path: str) -> List[Dict]:
        """Load data from file or batch directory."""
        data_path = Path(data_path)
        
        if data_path.is_file():
            print(f"Loading data from file: {data_path}")
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        elif data_path.is_dir():
            print(f"Loading data from batch directory: {data_path}")
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            if not batch_files:
                batch_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
            if not batch_files:
                raise ValueError(f"No batch files found in {data_path}")
            
            print(f"Found {len(batch_files)} batch files")
            all_data = []
            for batch_file in batch_files:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    all_data.extend(batch_data)
            return all_data
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    def load_scenarios_streaming(self, data_path: str):
        """Generator that yields scenarios one at a time from batch files."""
        data_path = Path(data_path)
        
        if data_path.is_file():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                for scenario in data:
                    yield scenario
        elif data_path.is_dir():
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            if not batch_files:
                batch_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
            if not batch_files:
                raise ValueError(f"No batch files found in {data_path}")
            
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'rb') as f:
                        batch_data = pickle.load(f)
                        for scenario in batch_data:
                            yield scenario
                except (IOError, pickle.UnpicklingError) as e:
                    print(f"Warning: Could not read batch file {batch_file}: {e}")
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    def count_scenarios(self, data_path: str) -> int:
        """Count total scenarios without loading all data into memory."""
        data_path = Path(data_path)
        
        if data_path.is_file():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                return len(data)
        elif data_path.is_dir():
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            if not batch_files:
                batch_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
            
            total = 0
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'rb') as f:
                        batch_data = pickle.load(f)
                        total += len(batch_data)
                except (IOError, pickle.UnpicklingError) as e:
                    print(f"Warning: Could not read batch file {batch_file}: {e}")
            return total
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    def extract_temporal_sequences(self, scenario: Dict) -> Dict:
        """
        Extract full temporal sequences with proper normalization.
        
        Args:
            scenario: Scenario dictionary with 'sequence' key
            
        Returns:
            Dictionary with normalized temporal sequences for all modalities
        """
        if 'sequence' not in scenario:
            raise ValueError("Scenario missing 'sequence' key")
        
        sequence = scenario['sequence']
        if not isinstance(sequence, list) or len(sequence) == 0:
            print(f"Warning: Scenario has empty or invalid sequence (type: {type(sequence)}).")
            # Create a single dummy timestep to avoid crashing
            sequence = [{}]

        # --- Helper to safely get data, handling missing keys ---
        def safe_get_or_default(ts, key, default_shape, dtype=np.float32):
            data = ts.get(key)
            if data is None:
                # print(f"Warning: missing {key} in timestep, using default.")
                return np.zeros(default_shape, dtype=dtype)
            
            # Ensure correct shape if possible, simple cases
            if hasattr(data, 'shape') and data.shape != default_shape:
                # Try to reshape if numel matches, otherwise use default
                if data.size == np.prod(default_shape):
                    try:
                        return data.reshape(default_shape).astype(dtype)
                    except ValueError:
                        return np.zeros(default_shape, dtype=dtype) 
                else:
                    # print(f"Warning: wrong shape for {key} (got {data.shape}, expected {default_shape}). Using default.")
                    return np.zeros(default_shape, dtype=dtype)
            
            if not hasattr(data, 'shape'):
                 return np.zeros(default_shape, dtype=dtype)

            return data.astype(dtype)
        # --- End Helper ---

        # Define expected default shapes (based on model and generator)
        sat_shape = (self.num_nodes, 12, 16, 16)
        weather_shape_raw = (self.num_nodes, 10, 8)
        
        threat_shape = (self.num_nodes, 6)
        scada_shape = (self.num_nodes, 15)
        pmu_shape = (self.num_nodes, 8)
        equip_shape = (self.num_nodes, 10)
        
        vis_shape = (self.num_nodes, 3, 32, 32)
        therm_shape = (self.num_nodes, 1, 32, 32)
        sensor_shape = (self.num_nodes, 12)
        
        # Get edge_attr from last step or scenario root
        edge_attr_data = sequence[-1].get('edge_attr')
        if edge_attr_data is None:
            edge_attr_data = scenario.get('edge_attr')
        
        num_edges = self.edge_index.shape[1]
        edge_shape = (num_edges, 5)
        
        if edge_attr_data is None:
            edge_attr = np.zeros(edge_shape, dtype=np.float32)
        elif edge_attr_data.shape != edge_shape:
             edge_attr = np.zeros(edge_shape, dtype=np.float32)
        else:
            edge_attr = edge_attr_data.astype(np.float32)

        satellite_sequence = np.stack([safe_get_or_default(ts, 'satellite_data', sat_shape) 
                                       for ts in sequence])
        
        # Handle weather sequence shape mismatch
        weather_sequence_raw = np.stack([safe_get_or_default(ts, 'weather_sequence', weather_shape_raw)
                                         for ts in sequence])
        weather_sequence = weather_sequence_raw.reshape(
            weather_sequence_raw.shape[0], self.num_nodes, -1
        )
        
        threat_sequence = np.stack([safe_get_or_default(ts, 'threat_indicators', threat_shape) 
                                    for ts in sequence])
        
        # SCADA data - normalize power components
        scada_sequence = []
        for ts in sequence:
            scada_data = safe_get_or_default(ts, 'scada_data', scada_shape)
            if scada_data.shape[1] >= 6:
                scada_data[:, 2] = self._normalize_power(scada_data[:, 2]) # generation
                scada_data[:, 3] = self._normalize_power(scada_data[:, 3]) # reactive_generation
                scada_data[:, 4] = self._normalize_power(scada_data[:, 4]) # load_values
                scada_data[:, 5] = self._normalize_power(scada_data[:, 5]) # reactive_load
            scada_sequence.append(scada_data)
        scada_sequence = np.stack(scada_sequence)
        
        # PMU data - normalize frequency
        pmu_sequence = []
        for ts in sequence:
            pmu_data = safe_get_or_default(ts, 'pmu_sequence', pmu_shape)
            if pmu_data.shape[1] >= 6:
                pmu_data[:, 5] = self._normalize_frequency(pmu_data[:, 5])
            pmu_sequence.append(pmu_data)
        pmu_sequence = np.stack(pmu_sequence)
        
        equipment_sequence = np.stack([safe_get_or_default(ts, 'equipment_status', equip_shape) 
                                       for ts in sequence])
        visual_sequence = np.stack([safe_get_or_default(ts, 'visual_data', vis_shape) 
                                    for ts in sequence])
        thermal_sequence = np.stack([safe_get_or_default(ts, 'thermal_data', therm_shape) 
                                     for ts in sequence])
        sensor_sequence = np.stack([safe_get_or_default(ts, 'sensor_data', sensor_shape) 
                                    for ts in sequence])
        
        # Normalize thermal limits in edge attributes (column 1 in generator)
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])
        
        return {
            'environmental': {
                'satellite_sequence': satellite_sequence,
                'weather_sequence': weather_sequence,
                'threat_sequence': threat_sequence
            },
            'infrastructure': {
                'scada_sequence': scada_sequence,
                'pmu_sequence': pmu_sequence,
                'equipment_sequence': equipment_sequence,
                'edge_features': edge_attr
            },
            'robotic': {
                'visual_sequence': visual_sequence,
                'thermal_sequence': thermal_sequence,
                'sensor_sequence': sensor_sequence
            }
        }
    
    def predict(self, temporal_data: Dict, use_temporal: bool = True) -> Dict:
        """
        Make prediction with physics-informed constraints.
        
        Args:
            temporal_data: Dictionary containing temporal sequences
            use_temporal: If True, use full temporal sequences
        
        Returns:
            Dictionary containing comprehensive predictions with denormalized outputs
        """
        with torch.no_grad():
            def safe_tensor_convert(data, name="data"):
                """Safely convert data to tensor."""
                try:
                    if isinstance(data, torch.Tensor):
                        return data
                    if isinstance(data, (list, tuple)):
                        data = np.array(data, dtype=np.float32)
                    elif isinstance(data, np.ndarray):
                        data = data.astype(np.float32)
                    else:
                        data = np.array(data, dtype=np.float32)
                    return torch.from_numpy(data)
                except Exception as e:
                    print(f"Error converting {name}: {e}")
                    raise
            
            # --- CRITICAL IMPROVEMENT: Handle temporal vs. non-temporal ---
            
            batch = {'edge_index': self.edge_index}
            
            env = temporal_data['environmental']
            infra = temporal_data['infrastructure']
            robot = temporal_data['robotic']

            if use_temporal:
                # Pass the full sequence with Batch dimension: [B, T, ...]
                batch['satellite_data'] = safe_tensor_convert(env['satellite_sequence'], 'satellite_data').unsqueeze(0).to(self.device)
                batch['weather_sequence'] = safe_tensor_convert(env['weather_sequence'], 'weather_sequence').unsqueeze(0).to(self.device)
                batch['threat_indicators'] = safe_tensor_convert(env['threat_sequence'], 'threat_indicators').unsqueeze(0).to(self.device)
                batch['scada_data'] = safe_tensor_convert(infra['scada_sequence'], 'scada_data').unsqueeze(0).to(self.device)
                batch['pmu_sequence'] = safe_tensor_convert(infra['pmu_sequence'], 'pmu_sequence').unsqueeze(0).to(self.device)
                batch['equipment_status'] = safe_tensor_convert(infra['equipment_sequence'], 'equipment_status').unsqueeze(0).to(self.device)
                batch['visual_data'] = safe_tensor_convert(robot['visual_sequence'], 'visual_data').unsqueeze(0).to(self.device)
                batch['thermal_data'] = safe_tensor_convert(robot['thermal_sequence'], 'thermal_data').unsqueeze(0).to(self.device)
                batch['sensor_data'] = safe_tensor_convert(robot['sensor_sequence'], 'sensor_data').unsqueeze(0).to(self.device)
                batch['edge_attr'] = safe_tensor_convert(infra['edge_features'], 'edge_attr').unsqueeze(0).to(self.device)
            
            else:
                # Pass only the last timestep: [B, N, ...] or [B, C, H, W] etc.
                batch['satellite_data'] = safe_tensor_convert(env['satellite_sequence'][-1], 'satellite_data').unsqueeze(0).to(self.device)
                batch['weather_sequence'] = safe_tensor_convert(env['weather_sequence'][-1], 'weather_sequence').unsqueeze(0).to(self.device)
                batch['threat_indicators'] = safe_tensor_convert(env['threat_sequence'][-1], 'threat_indicators').unsqueeze(0).to(self.device)
                batch['scada_data'] = safe_tensor_convert(infra['scada_sequence'][-1], 'scada_data').unsqueeze(0).to(self.device)
                batch['pmu_sequence'] = safe_tensor_convert(infra['pmu_sequence'][-1], 'pmu_sequence').unsqueeze(0).to(self.device)
                batch['equipment_status'] = safe_tensor_convert(infra['equipment_sequence'][-1], 'equipment_status').unsqueeze(0).to(self.device)
                batch['visual_data'] = safe_tensor_convert(robot['visual_sequence'][-1], 'visual_data').unsqueeze(0).to(self.device)
                batch['thermal_data'] = safe_tensor_convert(robot['thermal_sequence'][-1], 'thermal_data').unsqueeze(0).to(self.device)
                batch['sensor_data'] = safe_tensor_convert(robot['sensor_sequence'][-1], 'sensor_data').unsqueeze(0).to(self.device)
                batch['edge_attr'] = safe_tensor_convert(infra['edge_features'], 'edge_attr').unsqueeze(0).to(self.device)
            
            # --- END CRITICAL IMPROVEMENT ---
            
            outputs = self.model(batch) 
            
            node_failure_prob = outputs['failure_probability'].squeeze(0).squeeze(-1).cpu().numpy()
            
            failure_timing_raw = outputs.get('failure_timing')
            if failure_timing_raw is None:
                 failure_timing_raw = outputs.get('cascade_timing', torch.zeros(1, self.num_nodes, 1))

            if failure_timing_raw.shape[1] == self.edge_index.shape[1]:
                src, dst = self.edge_index.cpu().numpy()
                failure_timing_edge = failure_timing_raw.squeeze(0).squeeze(-1).cpu().numpy()
                failure_timing_node = np.full(self.num_nodes, -1.0)
                for i in range(self.edge_index.shape[1]):
                    s, d = src[i], dst[i]
                    t = failure_timing_edge[i]
                    if failure_timing_node[s] == -1.0 or t < failure_timing_node[s]:
                        failure_timing_node[s] = t
                    if failure_timing_node[d] == -1.0 or t < failure_timing_node[d]:
                        failure_timing_node[d] = t
            else:
                failure_timing_node = failure_timing_raw.squeeze(0).squeeze(-1).cpu().numpy()

            
            risk_scores = outputs['risk_scores'].squeeze(0).cpu().numpy()
            
            voltages_pu = outputs['voltages'].squeeze(0).squeeze(-1).cpu().numpy()
            angles_rad = outputs['angles'].squeeze(0).squeeze(-1).cpu().numpy()
            line_flows_pu = outputs['line_flows'].squeeze(0).squeeze(-1).cpu().numpy()
            frequency_pu = outputs['frequency'].squeeze(0).squeeze(-1).cpu().numpy()
            
            # Convert to physical units for interpretability
            line_flows_mw = self._denormalize_power(line_flows_pu)
            frequency_hz = self._denormalize_frequency(frequency_pu)
            
            relay_outputs = {}
            if 'relay_outputs' in outputs:
                relay_outputs = {
                    'time_dial': outputs['relay_outputs']['time_dial'].squeeze(0).squeeze(-1).cpu().numpy(),
                    'pickup_current': outputs['relay_outputs']['pickup_current'].squeeze(0).squeeze(-1).cpu().numpy(),
                    'operating_time': outputs['relay_outputs']['operating_time'].squeeze(0).squeeze(-1).cpu().numpy(),
                    'will_operate': (outputs['relay_outputs']['will_operate'].squeeze(0).squeeze(-1).cpu().numpy() > 0.5).astype(bool)
                }
            
            # Cascade detection
            cascade_prob = float(np.max(node_failure_prob))
            cascade_detected = bool(cascade_prob > self.cascade_threshold)
            
            high_risk_nodes_mask = node_failure_prob > self.node_threshold
            
            if cascade_detected and np.any(high_risk_nodes_mask):
                failure_times = failure_timing_node[high_risk_nodes_mask]
                valid_failure_times = failure_times[failure_times >= 0]
                if valid_failure_times.size > 0:
                    time_to_cascade_value = float(np.min(valid_failure_times))
                else:
                    time_to_cascade_value = -1.0
            else:
                time_to_cascade_value = -1.0
            
            high_risk_nodes = np.where(high_risk_nodes_mask)[0].tolist()
            
            node_risks = [(i, float(node_failure_prob[i])) for i in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            top_risk_nodes = node_risks[:10]
            
            aggregated_risk_scores = np.mean(risk_scores, axis=0)
            
            per_node_risks = [
                {
                    'node_id': int(i),
                    'threat_severity': float(risk_scores[i, 0]),
                    'vulnerability': float(risk_scores[i, 1]),
                    'operational_impact': float(risk_scores[i, 2]),
                    'cascade_probability': float(risk_scores[i, 3]),
                    'response_complexity': float(risk_scores[i, 4]),
                    'public_safety': float(risk_scores[i, 5]),
                    'urgency': float(risk_scores[i, 6])
                }
                for i in range(self.num_nodes)
            ]
            
            cascade_path = []
            if cascade_detected:
                failure_sequence = [
                    (i, failure_timing_node[i]) 
                    for i in range(self.num_nodes) 
                    if high_risk_nodes_mask[i] and failure_timing_node[i] >= 0
                ]
                failure_sequence.sort(key=lambda x: x[1])
                cascade_path = [
                    {'node_id': int(node), 'time_minutes': float(time)} 
                    for node, time in failure_sequence
                ]
            
            return {
                'cascade_probability': cascade_prob,
                'cascade_detected': cascade_detected,
                'time_to_cascade_minutes': time_to_cascade_value,
                'high_risk_nodes': high_risk_nodes,
                'top_10_risk_nodes': [
                    {'node_id': int(node_id), 'failure_probability': float(prob)}
                    for node_id, prob in top_risk_nodes
                ],
                'risk_assessment': {
                    'aggregated': {
                        'threat_severity': float(aggregated_risk_scores[0]),
                        'vulnerability': float(aggregated_risk_scores[1]),
                        'operational_impact': float(aggregated_risk_scores[2]),
                        'cascade_probability': float(aggregated_risk_scores[3]),
                        'response_complexity': float(aggregated_risk_scores[4]),
                        'public_safety': float(aggregated_risk_scores[5]),
                        'urgency': float(aggregated_risk_scores[6])
                    },
                    'per_node': per_node_risks # Still include for JSON, just won't print
                },
                'system_state': {
                    'voltages_pu': voltages_pu.tolist(),
                    'angles_rad': angles_rad.tolist(),
                    'line_flows_mw': line_flows_mw.tolist(),
                    'line_flows_pu': line_flows_pu.tolist(),
                    'frequency_hz': float(frequency_hz.item()) if frequency_hz.size == 1 else float(frequency_hz[0]),
                    'frequency_pu': float(frequency_pu.item()) if frequency_pu.size == 1 else float(frequency_pu[0])
                },
                'relay_operations': relay_outputs,
                'cascade_path': cascade_path,
                'total_nodes_at_risk': len(high_risk_nodes),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_from_file(self, data_path: str, scenario_idx: int = 0, 
                         use_temporal: bool = True) -> Dict:
        """Make prediction from data file with proper normalization."""
        print(f"Loading scenario {scenario_idx} from {data_path}...")
        
        current_idx = 0
        for scenario in self.load_scenarios_streaming(data_path):
            if current_idx == scenario_idx:
                
                # Extract data and make prediction
                temporal_data = self.extract_temporal_sequences(scenario)
                prediction = self.predict(temporal_data, use_temporal=use_temporal)
                
                # Add ground truth if available
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    # Extract time from metadata
                    cascade_start_time = metadata.get('cascade_start_time', -1)
                    failure_times = metadata.get('failure_times', [])
                    actual_time_to_cascade = -1.0
                    if cascade_start_time != -1 and failure_times:
                        actual_time_to_cascade = float(np.min(failure_times))
                    
                    prediction['ground_truth'] = {
                        'is_cascade': bool(metadata.get('is_cascade', False)),
                        'failed_nodes': [int(x) for x in failed_nodes],
                        'time_to_cascade': actual_time_to_cascade
                    }
                
                return prediction
            current_idx += 1
        
        raise ValueError(f"Scenario index {scenario_idx} out of range")
    
    def batch_predict(self, data_path: str, max_scenarios: int = None,
                     use_temporal: bool = True) -> List[Dict]:
        """Make predictions on multiple scenarios with proper normalization."""
        print(f"Streaming data from {data_path}...")
        
        total_scenarios = self.count_scenarios(data_path)
        num_scenarios = total_scenarios if max_scenarios is None else min(max_scenarios, total_scenarios)
        print(f"Processing {num_scenarios} scenarios (total available: {total_scenarios})...")
        print(f"Temporal processing: {'ENABLED (Full Sequence)' if use_temporal else 'DISABLED (Last Timestep)'}")
        print(f"Physics-informed normalization: ENABLED (base_mva={self.base_mva}, base_freq={self.base_frequency})")
        
        predictions = []
        processed = 0
        
        for scenario in self.load_scenarios_streaming(data_path):
            if processed >= num_scenarios:
                break
            
            if (processed + 1) % 100 == 0:
                print(f"  Processed {processed + 1}/{num_scenarios}")
            
            try:
                temporal_data = self.extract_temporal_sequences(scenario)
                pred = self.predict(temporal_data, use_temporal=use_temporal)
                
                # Add ground truth
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    cascade_start_time = metadata.get('cascade_start_time', -1)
                    failure_times = metadata.get('failure_times', [])
                    actual_time_to_cascade = -1.0
                    if cascade_start_time != -1 and failure_times:
                        actual_time_to_cascade = float(np.min(failure_times))
                    
                    pred['ground_truth'] = {
                        'is_cascade': bool(metadata.get('is_cascade', False)),
                        'failed_nodes': [int(x) for x in failed_nodes],
                        'time_to_cascade': actual_time_to_cascade,
                        'scenario_id': processed # Add scenario ID for tracking
                    }
                
                predictions.append(pred)
                processed += 1
                
            except Exception as e:
                import traceback
                print(f"  Error processing scenario {processed}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                processed += 1 # Increment even if error to avoid infinite loop
        
        print(f"✓ Completed processing {len(predictions)} scenarios")
        return predictions
    
    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        """Evaluate prediction performance with comprehensive metrics."""
        if not predictions or 'ground_truth' not in predictions[0]:
            print("No ground truth available for evaluation")
            return {}
        
        cascade_correct = 0
        cascade_total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        time_errors = []
        
        for pred in predictions:
            gt = pred['ground_truth']
            
            predicted_cascade = pred['cascade_detected']
            actual_cascade = gt['is_cascade']
            
            if predicted_cascade == actual_cascade:
                cascade_correct += 1
            
            if predicted_cascade and actual_cascade:
                true_positives += 1
                if pred['time_to_cascade_minutes'] > 0 and gt['time_to_cascade'] > 0:
                    time_errors.append(abs(pred['time_to_cascade_minutes'] - gt['time_to_cascade']))
            elif predicted_cascade and not actual_cascade:
                false_positives += 1
            elif not predicted_cascade and not actual_cascade:
                true_negatives += 1
            elif not predicted_cascade and actual_cascade:
                false_negatives += 1
            
            cascade_total += 1
        
        accuracy = cascade_correct / cascade_total if cascade_total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        mean_time_error = np.mean(time_errors) if time_errors else 0
        median_time_error = np.median(time_errors) if time_errors else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_scenarios': cascade_total,
            'time_to_cascade_mae': mean_time_error,
            'time_to_cascade_median_error': median_time_error
        }
        
        return metrics

# ============================================================================
# NEW: User-Friendly Report Functions
# ============================================================================

def print_single_prediction_report(prediction: Dict, inference_time: float, cascade_threshold: float, node_threshold: float):
    """Prints a clear, user-friendly report for a single prediction."""
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS (Scenario Analysis)")
    print("=" * 80)
    
    gt = prediction.get('ground_truth', {})
    actual_cascade = gt.get('is_cascade', False)
    predicted_cascade = prediction['cascade_detected']
    
    # --- 1. Overall Verdict ---
    print(f"Inference Time: {inference_time:.4f} seconds\n")
    print("--- 1. Overall Verdict ---")
    if predicted_cascade and actual_cascade:
        print("✅ Correctly detected a cascade.")
    elif not predicted_cascade and not actual_cascade:
        print("✅ Correctly identified a normal scenario.")
    elif not predicted_cascade and actual_cascade:
        print("❌ FALSE NEGATIVE (Missed Cascade)")
    elif predicted_cascade and not actual_cascade:
        print("⚠️ FALSE POSITIVE (False Alarm)")

    print(f"Prediction: {predicted_cascade} (Prob: {prediction['cascade_probability']:.3f} / Thresh: {cascade_threshold:.3f})")
    print(f"Ground Truth: {actual_cascade}")

    # --- 2. Node-Level Analysis ---
    if actual_cascade or predicted_cascade:
        print("\n--- 2. Node-Level Analysis ---")
        predicted_nodes = set(prediction['high_risk_nodes'])
        actual_nodes = set(gt.get('failed_nodes', []))
        
        tp_nodes = len(predicted_nodes.intersection(actual_nodes))
        fp_nodes = len(predicted_nodes.difference(actual_nodes))
        fn_nodes = len(actual_nodes.difference(predicted_nodes))
        
        print(f"Predicted Nodes at Risk: {len(predicted_nodes)} (Thresh: {node_threshold:.3f})")
        print(f"Actual Failed Nodes:     {len(actual_nodes)}")
        print(f"  - Correctly Identified (TP): {tp_nodes}")
        print(f"  - Missed Nodes (FN):         {fn_nodes}")
        print(f"  - False Alarms (FP):         {fp_nodes}")

    # --- 3. Timing Analysis ---
    if actual_cascade or predicted_cascade:
        print("\n--- 3. Time-to-Cascade Analysis ---")
        pred_time = prediction['time_to_cascade_minutes']
        actual_time = gt.get('time_to_cascade', -1.0)
        
        if pred_time == -1 and actual_time == -1:
            print("  - No timing information available.")
        else:
            print(f"  - Predicted Lead Time: {pred_time:.2f} minutes")
            print(f"  - Actual Lead Time:    {actual_time:.2f} minutes")

    # --- 4. Critical Information ---
    print("\n--- 4. Critical Information ---")
    print(f"System Frequency: {prediction['system_state']['frequency_hz']:.2f} Hz")
    print(f"Voltage Range:    [{np.min(prediction['system_state']['voltages_pu']):.3f}, "
          f"{np.max(prediction['system_state']['voltages_pu']):.3f}] p.u.")
    
    print("\nTop 5 High-Risk Nodes:")
    if not prediction['top_10_risk_nodes']:
        print("  - None")
    for node_info in prediction['top_10_risk_nodes'][:5]:
        print(f"  - Node {node_info['node_id']}: {node_info['failure_probability']:.4f}")
    
    print("\nAggregated Risk Assessment (7-Dimensions):")
    risk = prediction['risk_assessment']['aggregated']
    print(f"  - Threat: {risk['threat_severity']:.3f} | Vulnerability: {risk['vulnerability']:.3f} | Impact: {risk['operational_impact']:.3f}")
    print(f"  - Cascade Prob: {risk['cascade_probability']:.3f} | Response: {risk['response_complexity']:.3f} | Safety: {risk['public_safety']:.3f}")
    
    print("=" * 80 + "\n")


def print_batch_report(predictions: List[Dict], metrics: Dict, total_time: float):
    """Prints a clear, user-friendly report for a batch prediction run."""
    
    print("\n" + "=" * 80)
    print("BATCH PREDICTION REPORT")
    print("=" * 80)

    # --- 1. Performance Summary ---
    print("--- 1. Performance Summary ---")
    print(f"Total Scenarios: {metrics.get('total_scenarios', 0)}")
    print(f"Accuracy:        {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:       {metrics.get('precision', 0):.4f}")
    print(f"Recall:          {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:        {metrics.get('f1_score', 0):.4f}")
    print(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")

    # --- 2. Timing Summary ---
    print("\n--- 2. Timing Summary ---")
    avg_time = total_time / len(predictions) if predictions else 0
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Avg. Time per Scenario: {avg_time:.4f} seconds")
    print(f"Time-to-Cascade MAE: {metrics.get('time_to_cascade_mae', 0):.2f} minutes")
    
    # --- 3. Critical Events Summary ---
    print("\n--- 3. Critical Events Summary ---")
    true_positives = []
    false_positives = []
    false_negatives = []
    
    for i, pred in enumerate(predictions):
        # Use the scenario_id from ground_truth if available, otherwise just use index
        scenario_id = pred.get('ground_truth', {}).get('scenario_id', i)
        gt_cascade = pred.get('ground_truth', {}).get('is_cascade', False)
        pred_cascade = pred['cascade_detected']
        
        if pred_cascade and gt_cascade:
            true_positives.append(scenario_id)
        elif pred_cascade and not gt_cascade:
            false_positives.append(scenario_id)
        elif not pred_cascade and gt_cascade:
            false_negatives.append(scenario_id)
    
    # False Negatives (Most Important)
    if not false_negatives:
        print("✅ False Negatives (Missed Cascades): 0")
    else:
        print(f"❌ False Negatives (Missed Cascades): {len(false_negatives)}")
        print(f"   - Scenario IDs: {false_negatives}")

    # False Positives
    if not false_positives:
        print("✅ False Positives (False Alarms): 0")
    else:
        print(f"⚠️ False Positives (False Alarms): {len(false_positives)}")
        print(f"   - Scenario IDs: {false_positives}")
    
    # True Positives
    print(f"✅ True Positives (Detected Cascades): {len(true_positives)}")
    if true_positives:
        print(f"   - Scenario IDs (first 20): {true_positives[:20]}{'...' if len(true_positives) > 20 else ''}")
        
    print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference (Physics-Informed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data/grid_topology.pkl", 
                       help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data/test", 
                       help="Path to test data file or directory (default: data/test)")
    parser.add_argument("--scenario_idx", type=int, default=0, 
                       help="Scenario index for single prediction")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction on all scenarios in data_path")
    parser.add_argument("--max_scenarios", type=int, default=None, 
                       help="Max scenarios for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.json", 
                       help="Output file for predictions")
    parser.add_argument("--device", type=str, 
                       default=None, 
                       help="Device (e.g., 'cuda', 'cpu'). Autodetects if not set.")
    parser.add_argument("--base_mva", type=float, default=100.0,
                       help="Base MVA for power normalization (default: 100.0)")
    parser.add_argument("--base_frequency", type=float, default=60.0,
                       help="Base frequency in Hz (default: 60.0)")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Disable temporal sequence processing (uses last timestep only)")
    
    args = parser.parse_args()
    
    # --- Device Selection ---
    if args.device:
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available() and DEVICE.type == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU.")
        DEVICE = torch.device("cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Initialize predictor
    global predictor # Make predictor global so helper functions can see thresholds
    predictor = CascadePredictor(
        model_path=args.model_path,
        topology_path=args.topology_path,
        device=DEVICE,
        base_mva=args.base_mva,
        base_frequency=args.base_frequency
    )
    
    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - PHYSICS-INFORMED INFERENCE")
    print("=" * 80 + "\n")
    
    use_temporal = not args.no_temporal
    
    if args.batch:
        # --- Batch Prediction Mode ---
        start_time = time.time()
        predictions = predictor.batch_predict(
            args.data_path, 
            args.max_scenarios,
            use_temporal=use_temporal
        )
        total_time = time.time() - start_time
        
        if not predictions:
            print("No predictions were generated. Exiting.")
            return

        metrics = predictor.evaluate_predictions(predictions)
        
        # Print the new user-friendly batch report
        print_batch_report(predictions, metrics, total_time)
        
        # Save the full data to JSON for deep analysis
        results = {
            'metrics': metrics,
            'predictions': predictions
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Full batch results saved to {args.output}")
    
    else:
        # --- Single Prediction Mode ---
        try:
            start_time = time.time()
            prediction = predictor.predict_from_file(
                args.data_path, 
                args.scenario_idx,
                use_temporal=use_temporal
            )
            inference_time = time.time() - start_time
            
            # Print the new user-friendly single report
            print_single_prediction_report(
                prediction, 
                inference_time,
                predictor.cascade_threshold,
                predictor.node_threshold
            )

            # Save the full data to JSON
            with open(args.output, 'w') as f:
                json.dump(prediction, f, indent=2, cls=NumpyEncoder)
            print(f"Full prediction details saved to {args.output}")
            
        except ValueError as e:
            print(f"\nError: {e}")
            print(f"Could not load scenario index {args.scenario_idx}.")
            print("Please ensure the index is valid and the data path is correct.")


if __name__ == "__main__":
    main()
