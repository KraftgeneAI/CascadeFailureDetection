"""
Cascade Failure Prediction Model Inference Script 
============================================================
Improved version with proper normalization, temporal processing, 
and alignment with the trained model's capabilities.

Key improvements:
- [FIX] Correctly loads edge_index from topology file (converts numpy to tensor).
- [FIX] Correctly infers num_nodes from topology file's adjacency_matrix shape.
- [CRITICAL] Implements full temporal sequence processing, enabling the model's LSTM 
  architecture as described in the paper [cite: 327, 331-335].
- [CRITICAL] Loads the dynamic 'cascade_threshold' and 'node_threshold' from the
  saved model checkpoint instead of using hardcoded values.
- [FIX] Uses the correct 'node_threshold' for identifying high-risk nodes.
- [CLEANUP] Removed redundant 'graph_properties' logic not used by the model during inference.
- Applies same normalization as dataset loader (power, frequency, capacity)
- Denormalizes outputs to physical units for interpretability
- Full alignment with paper requirements

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

# Ensure the model class is importable
import sys
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
            edge_index_numpy = topology['edge_index']
            self.edge_index = torch.from_numpy(edge_index_numpy).long().to(self.device)
            # --- END FIX 1 ---

            # --- FIX 2 (KeyError): Infer num_nodes from adjacency_matrix shape ---
            self.num_nodes = topology['adjacency_matrix'].shape[0]
            # --- END FIX 2 ---
            
            self.topology = topology
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # --- IMPROVEMENT: Load thresholds from checkpoint ---
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.5) # Use 0.5 as a safer default
        self.node_threshold = checkpoint.get('node_threshold', 0.5) # Use 0.5 as a safer default
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
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    for scenario in batch_data:
                        yield scenario
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
        # Note: These shapes are *per-node* or *per-item* as generator stacks them
        sat_shape = (self.num_nodes, 12, 16, 16) # From generator _generate_correlated_environmental_data
        
        # Weather shape: generator saves as (N, 10, 8) but data loader expects (N, 80)
        # The multimodal_cascade_model.py's EnvironmentalEmbedding expects (N, 80)
        # Let's adjust to match the model's expectation
        weather_shape_raw = (self.num_nodes, 10, 8)
        weather_shape_model = (self.num_nodes, 80) # 10*8 = 80
        
        threat_shape = (self.num_nodes, 6) # From generator
        
        # From generator _generate_scenario_data (15 features)
        # But model's InfrastructureEmbedding expects 15
        scada_shape = (self.num_nodes, 15) 
        
        # From generator _generate_scenario_data (8 features)
        # Model's InfrastructureEmbedding expects 8
        pmu_shape = (self.num_nodes, 8)
        
        # From generator _generate_scenario_data (10 features)
        # Model's InfrastructureEmbedding expects 10
        equip_shape = (self.num_nodes, 10)
        
        vis_shape = (self.num_nodes, 3, 32, 32) # From generator _generate_correlated_robotic_data
        therm_shape = (self.num_nodes, 1, 32, 32) # From generator
        sensor_shape = (self.num_nodes, 12) # From generator
        
        # Get edge_attr from last step or scenario root
        edge_attr_data = sequence[-1].get('edge_attr')
        if edge_attr_data is None:
            edge_attr_data = scenario.get('edge_attr')
        
        num_edges = self.edge_index.shape[1]
        edge_shape = (num_edges, 5) # From generator _generate_scenario_data
        
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
        # Reshape from [T, N, 10, 8] -> [T, N, 80] to match model
        weather_sequence = weather_sequence_raw.reshape(
            weather_sequence_raw.shape[0], self.num_nodes, -1
        )
        
        threat_sequence = np.stack([safe_get_or_default(ts, 'threat_indicators', threat_shape) 
                                    for ts in sequence])
        
        # SCADA data - normalize power components
        scada_sequence = []
        for ts in sequence:
            scada_data = safe_get_or_default(ts, 'scada_data', scada_shape)
            # Normalize power columns (2=gen, 3=reac_gen, 4=load, 5=reac_load)
            # Based on multimodal_data_generator.py
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
            # Normalize frequency (column 5 in generator's PMU data)
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
                # .unsqueeze(0) adds the batch dimension (B=1)
                batch['satellite_data'] = safe_tensor_convert(env['satellite_sequence'], 'satellite_data').unsqueeze(0).to(self.device)
                batch['weather_sequence'] = safe_tensor_convert(env['weather_sequence'], 'weather_sequence').unsqueeze(0).to(self.device)
                batch['threat_indicators'] = safe_tensor_convert(env['threat_sequence'], 'threat_indicators').unsqueeze(0).to(self.device)
                batch['scada_data'] = safe_tensor_convert(infra['scada_sequence'], 'scada_data').unsqueeze(0).to(self.device)
                batch['pmu_sequence'] = safe_tensor_convert(infra['pmu_sequence'], 'pmu_sequence').unsqueeze(0).to(self.device)
                batch['equipment_status'] = safe_tensor_convert(infra['equipment_sequence'], 'equipment_status').unsqueeze(0).to(self.device)
                batch['visual_data'] = safe_tensor_convert(robot['visual_sequence'], 'visual_data').unsqueeze(0).to(self.device)
                batch['thermal_data'] = safe_tensor_convert(robot['thermal_sequence'], 'thermal_data').unsqueeze(0).to(self.device)
                batch['sensor_data'] = safe_tensor_convert(robot['sensor_sequence'], 'sensor_data').unsqueeze(0).to(self.device)
                # Edge attributes are [B, E, F], so unsqueeze(0) is correct
                batch['edge_attr'] = safe_tensor_convert(infra['edge_features'], 'edge_attr').unsqueeze(0).to(self.device)
            
            else:
                # Pass only the last timestep: [B, N, ...] or [B, C, H, W] etc.
                # .unsqueeze(0) adds the batch dimension (B=1)
                batch['satellite_data'] = safe_tensor_convert(env['satellite_sequence'][-1], 'satellite_data').unsqueeze(0).to(self.device)
                batch['weather_sequence'] = safe_tensor_convert(env['weather_sequence'][-1], 'weather_sequence').unsqueeze(0).to(self.device)
                batch['threat_indicators'] = safe_tensor_convert(env['threat_sequence'][-1], 'threat_indicators').unsqueeze(0).to(self.device)
                batch['scada_data'] = safe_tensor_convert(infra['scada_sequence'][-1], 'scada_data').unsqueeze(0).to(self.device)
                batch['pmu_sequence'] = safe_tensor_convert(infra['pmu_sequence'][-1], 'pmu_sequence').unsqueeze(0).to(self.device)
                batch['equipment_status'] = safe_tensor_convert(infra['equipment_sequence'][-1], 'equipment_status').unsqueeze(0).to(self.device)
                batch['visual_data'] = safe_tensor_convert(robot['visual_sequence'][-1], 'visual_data').unsqueeze(0).to(self.device)
                batch['thermal_data'] = safe_tensor_convert(robot['thermal_sequence'][-1], 'thermal_data').unsqueeze(0).to(self.device)
                batch['sensor_data'] = safe_tensor_convert(robot['sensor_sequence'][-1], 'sensor_data').unsqueeze(0).to(self.device)
                # Edge attributes are [B, E, F], so unsqueeze(0) is correct
                batch['edge_attr'] = safe_tensor_convert(infra['edge_features'], 'edge_attr').unsqueeze(0).to(self.device)
            
            # --- END CRITICAL IMPROVEMENT ---
            
            # The model's forward pass does not use return_sequence
            outputs = self.model(batch) 
            
            # Extract predictions
            # Model outputs logits if trained with PhysicsInformedLoss(use_logits=True)
            # But the provided train_model.py sets MODEL_OUTPUTS_LOGITS = False
            # And the model's failure_prob_head ends in nn.Sigmoid()
            # Therefore, outputs['failure_probability'] is already a probability.
            node_failure_prob = outputs['failure_probability'].squeeze(0).squeeze(-1).cpu().numpy()
            
            # Squeeze all outputs to remove batch dim
            failure_timing_raw = outputs.get('failure_timing') # This might be per-edge
            if failure_timing_raw is None:
                 failure_timing_raw = outputs.get('cascade_timing', torch.zeros(1, self.num_nodes, 1))

            # Handle edge-based vs node-based timing
            if failure_timing_raw.shape[1] == self.edge_index.shape[1]:
                # Edge-based timing (from relay_model)
                src, dst = self.edge_index.cpu().numpy()
                failure_timing_edge = failure_timing_raw.squeeze(0).squeeze(-1).cpu().numpy()
                failure_timing_node = np.full(self.num_nodes, -1.0)
                # Assign earliest edge failure time to each node
                for i in range(self.edge_index.shape[1]):
                    s, d = src[i], dst[i]
                    t = failure_timing_edge[i]
                    if failure_timing_node[s] == -1.0 or t < failure_timing_node[s]:
                        failure_timing_node[s] = t
                    if failure_timing_node[d] == -1.0 or t < failure_timing_node[d]:
                        failure_timing_node[d] = t
            else:
                # Node-based timing
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
            
            # --- IMPROVEMENT: Use self.node_threshold ---
            high_risk_nodes_mask = node_failure_prob > self.node_threshold
            
            if cascade_detected and np.any(high_risk_nodes_mask):
                failure_times = failure_timing_node[high_risk_nodes_mask]
                # Filter out any unassigned times (-1)
                valid_failure_times = failure_times[failure_times >= 0]
                if valid_failure_times.size > 0:
                    time_to_cascade_value = float(np.min(valid_failure_times))
                else:
                    time_to_cascade_value = -1.0 # No valid timing found
            else:
                time_to_cascade_value = -1.0
            
            high_risk_nodes = np.where(high_risk_nodes_mask)[0].tolist()
            # --- END IMPROVEMENT ---
            
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
                    'per_node': per_node_risks
                },
                'system_state': {
                    'voltages_pu': voltages_pu.tolist(),
                    'angles_rad': angles_rad.tolist(),
                    'line_flows_mw': line_flows_mw.tolist(),  # Denormalized to MW
                    'line_flows_pu': line_flows_pu.tolist(),  # Also keep per-unit
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
                # --- REMOVED graph_properties ---
                temporal_data = self.extract_temporal_sequences(scenario)
                
                # Make prediction
                prediction = self.predict(temporal_data, use_temporal=use_temporal)
                # --- END REMOVED ---
                
                # Add ground truth if available
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    time_to_cascade = metadata.get('time_to_cascade', -1)
                    if isinstance(time_to_cascade, np.ndarray):
                        time_to_cascade = float(time_to_cascade.flatten()[0]) if time_to_cascade.size > 0 else -1.0
                    elif isinstance(time_to_cascade, (np.floating, np.integer)):
                        time_to_cascade = float(time_to_cascade)
                    elif time_to_cascade is None:
                        time_to_cascade = -1.0
                    
                    prediction['ground_truth'] = {
                        'is_cascade': bool(metadata.get('is_cascade', False)),
                        'failed_nodes': [int(x) for x in failed_nodes],
                        'time_to_cascade': float(time_to_cascade)
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
                # --- REMOVED graph_properties ---
                temporal_data = self.extract_temporal_sequences(scenario)
                pred = self.predict(temporal_data, use_temporal=use_temporal)
                # --- END REMOVED ---
                
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    time_to_cascade = metadata.get('time_to_cascade', -1)
                    if isinstance(time_to_cascade, np.ndarray):
                        time_to_cascade = float(time_to_cascade.flatten()[0]) if time_to_cascade.size > 0 else -1.0
                    elif isinstance(time_to_cascade, (np.floating, np.integer)):
                        time_to_cascade = float(time_to_cascade)
                    elif time_to_cascade is None:
                        time_to_cascade = -1.0
                    
                    pred['ground_truth'] = {
                        'is_cascade': bool(metadata.get('is_cascade', False)),
                        'failed_nodes': [int(x) for x in failed_nodes],
                        'time_to_cascade': float(time_to_cascade)
                    }
                
                predictions.append(pred)
                processed += 1
                
            except Exception as e:
                import traceback
                print(f"  Error processing scenario {processed}: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                processed += 1
        
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


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference (Physics-Informed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data/grid_topology.pkl", 
                       help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data/test_batches", 
                       help="Path to test data file or directory (default: data/test_batches)")
    parser.add_argument("--scenario_idx", type=int, default=0, 
                       help="Scenario index for single prediction")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction")
    parser.add_argument("--max_scenarios", type=int, default=None, 
                       help="Max scenarios for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.json", 
                       help="Output file for predictions")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device")
    # Removed cascade_threshold argument, as it's now loaded from the model
    parser.add_argument("--base_mva", type=float, default=100.0,
                       help="Base MVA for power normalization (default: 100.0)")
    parser.add_argument("--base_frequency", type=float, default=60.0,
                       help="Base frequency in Hz (default: 60.0)")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Disable temporal sequence processing (uses last timestep only)")
    
    args = parser.parse_args()
    
    # Initialize predictor with normalization parameters
    predictor = CascadePredictor(
        model_path=args.model_path,
        topology_path=args.topology_path,
        device=args.device,
        # cascade_threshold is now loaded internally
        base_mva=args.base_mva,
        base_frequency=args.base_frequency
    )
    
    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - PHYSICS-INFORMED INFERENCE")
    print("=" * 80 + "\n")
    
    use_temporal = not args.no_temporal
    
    if args.batch:
        predictions = predictor.batch_predict(
            args.data_path, 
            args.max_scenarios,
            use_temporal=use_temporal
        )
        
        metrics = predictor.evaluate_predictions(predictions)
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total scenarios: {metrics.get('total_scenarios', 0)}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall: {metrics.get('recall', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}")
        print(f"Time-to-Cascade MAE: {metrics.get('time_to_cascade_mae', 0):.2f} minutes")
        print(f"Time-to-Cascade Median Error: {metrics.get('time_to_cascade_median_error', 0):.2f} minutes")
        print("=" * 80 + "\n")
        
        results = {
            'predictions': predictions,
            'metrics': metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Results saved to {args.output}")
    
    else:
        prediction = predictor.predict_from_file(
            args.data_path, 
            args.scenario_idx,
            use_temporal=use_temporal
        )
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"Temporal Processing: {'ENABLED (Full Sequence)' if use_temporal else 'DISABLED (Last Timestep)'}")
        print(f"Cascade Detected: {prediction['cascade_detected']}")
        print(f"Cascade Probability: {prediction['cascade_probability']:.4f} (Threshold: {predictor.cascade_threshold:.4f})")
        print(f"Time to Cascade: {prediction['time_to_cascade_minutes']:.2f} minutes")
        print(f"Nodes at Risk: {prediction['total_nodes_at_risk']} (Threshold: {predictor.node_threshold:.4f})")
        
        print("\nTop 10 High-Risk Nodes:")
        for node_info in prediction['top_10_risk_nodes']:
            print(f"  Node {node_info['node_id']}: {node_info['failure_probability']:.4f}")
        
        print("\nSystem State:")
        print(f"  Frequency: {prediction['system_state']['frequency_hz']:.2f} Hz")
        print(f"  Voltage Range: [{np.min(prediction['system_state']['voltages_pu']):.3f}, "
              f"{np.max(prediction['system_state']['voltages_pu']):.3f}] p.u.")
        print(f"  Line Flow Range: [{np.min(prediction['system_state']['line_flows_mw']):.1f}, "
              f"{np.max(prediction['system_state']['line_flows_mw']):.1f}] MW")
        
        print("\nCascade Propagation Path (Predicted):")
        if not prediction['cascade_path']:
            print("  No propagation path predicted.")
        for step in prediction['cascade_path'][:5]:
            print(f"  Node {step['node_id']} fails at {step['time_minutes']:.2f} minutes")
        
        if 'ground_truth' in prediction:
            gt = prediction['ground_truth']
            print("\nGround Truth:")
            print(f"  Actual Cascade: {gt['is_cascade']}")
            print(f"  Actual Failed Nodes: {len(gt.get('failed_nodes', []))}")
            print(f"  Actual Time: {gt.get('time_to_cascade', -1):.2f} minutes")
        
        print("=" * 80 + "\n")
        
        with open(args.output, 'w') as f:
            json.dump(prediction, f, indent=2, cls=NumpyEncoder)
        print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    main()