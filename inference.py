"""
Cascade Failure Prediction Model Inference Script 
============================================================
(MODIFIED for "New, Sound Training Methodology")
- Feeds ONLY pre-cascade data (or random trunks) to the model.
- Removes "stress_level" data leakage.
- Fixes "Urgency" reporting bug.
- IMPROVED: "Timing Analysis" now shows Ranking Score Spread.
- IMPROVED: "Cascade Path" uses Anchor-Based Grouping for clear steps.
============================================================

Author: Kraftgene AI Inc.
Date: October 2025
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import argparse
from datetime import datetime
import glob
import time
import sys
from tqdm import tqdm

# Ensure the model class is importable
try:
    from multimodal_cascade_model import UnifiedCascadePredictionModel
    from cascade_dataset import collate_cascade_batch
except ImportError:
    print("Error: Could not import UnifiedCascadePredictionModel or collate_cascade_batch.")
    print("Please ensure multimodal_cascade_model.py and cascade_dataset.py are in your Python path.")
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
        self.device = torch.device(device)
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        
        # Load topology
        print(f"Loading grid topology from {topology_path}...")
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
            # --- START IMPROVEMENT ---
            if not isinstance(topology['edge_index'], torch.Tensor):
                self.edge_index_np = topology['edge_index'] # Cache numpy version for fast masking
                self.edge_index = torch.from_numpy(self.edge_index_np).long()
            else:
                self.edge_index = topology['edge_index']
                self.edge_index_np = self.edge_index.cpu().numpy() # Cache numpy version
            # --- END IMPROVEMENT ---

            if 'num_nodes' in topology:
                self.num_nodes = topology['num_nodes']
            elif 'adjacency_matrix' in topology:
                self.num_nodes = topology['adjacency_matrix'].shape[0]
            else:
                raise KeyError("Could not determine num_nodes from topology file.")
            
            self.topology = topology
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.5)
        self.node_threshold = checkpoint.get('node_threshold', 0.5)
        
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
        return power_values / self.base_mva
    
    def _normalize_frequency(self, frequency_values: np.ndarray) -> np.ndarray:
        return frequency_values / self.base_frequency
    
    def _denormalize_power(self, power_pu: np.ndarray) -> np.ndarray:
        return power_pu * self.base_mva
    
    def _denormalize_frequency(self, frequency_pu: np.ndarray) -> np.ndarray:
        return frequency_pu * self.base_frequency
    
    
    def load_scenarios_streaming(self, data_path: str):
        data_path = Path(data_path)
        if not data_path.is_dir():
             raise ValueError(f"Data path {data_path} is not a directory.")

        scenario_files = sorted(glob.glob(str(data_path / "scenario_*.pkl")))
        if not scenario_files:
            scenario_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
            if not scenario_files:
                raise ValueError(f"No scenario files found in {data_path}.")
            print(f"Warning: Found 'scenarios_batch_*.pkl' files. Loading in compatibility mode.")
        
        print(f"Streaming from {len(scenario_files)} scenario files in {data_path}...")
        
        for scenario_file in scenario_files:
            try:
                with open(scenario_file, 'rb') as f:
                    scenario_data = pickle.load(f)

                if isinstance(scenario_data, list):
                    if len(scenario_data) == 0: continue
                    scenario = scenario_data[0]
                else:
                    scenario = scenario_data
                
                if not isinstance(scenario, dict): continue

                scenario['edge_index'] = self.edge_index
                yield scenario
            except (IOError, pickle.UnpicklingError) as e:
                print(f"Warning: Could not read scenario file {scenario_file}: {e}")
    
    def count_scenarios(self, data_path: str) -> int:
        data_path = Path(data_path)
        scenario_files = sorted(glob.glob(str(data_path / "scenario_*.pkl")))
        if not scenario_files:
            scenario_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
        return len(scenario_files)
    
    def extract_and_preprocess_scenario(self, scenario: Dict) -> Optional[Dict[str, Any]]:
        try:
            if 'sequence' not in scenario: raise ValueError("Scenario missing 'sequence' key")
            sequence = scenario['sequence']
            if not isinstance(sequence, list) or len(sequence) == 0: return None

            def safe_get_or_default(ts, key, default_shape, dtype=np.float32):
                data = ts.get(key)
                if data is None: return np.zeros(default_shape, dtype=dtype)
                if hasattr(data, 'shape') and data.shape != default_shape:
                    if data.size == np.prod(default_shape):
                        try: return data.reshape(default_shape).astype(dtype)
                        except ValueError: return np.zeros(default_shape, dtype=dtype) 
                    else: return np.zeros(default_shape, dtype=dtype)
                if not hasattr(data, 'shape'): return np.zeros(default_shape, dtype=dtype)
                return data.astype(dtype)

            # Define shapes
            scada_shape = (self.num_nodes, 13)
            sat_shape = (self.num_nodes, 12, 16, 16)
            weather_shape_raw = (self.num_nodes, 10, 8)
            threat_shape = (self.num_nodes, 6)
            pmu_shape = (self.num_nodes, 8)
            equip_shape = (self.num_nodes, 10)
            vis_shape = (self.num_nodes, 3, 32, 32)
            therm_shape = (self.num_nodes, 1, 32, 32)
            sensor_shape = (self.num_nodes, 12)
            
            # Edge attributes
            edge_attr_data = sequence[-1].get('edge_attr')
            if edge_attr_data is None: edge_attr_data = scenario.get('edge_attr')
            
            num_edges = self.edge_index.shape[1]
            edge_shape = (num_edges, 5)
            
            if edge_attr_data is None: edge_attr = np.zeros(edge_shape, dtype=np.float32)
            elif edge_attr_data.shape != edge_shape: edge_attr = np.zeros(edge_shape, dtype=np.float32)
            else: edge_attr = edge_attr_data.astype(np.float32)

            # Initialize Lists
            satellite_sequence = [] # Changed from np.stack for consistency
            weather_sequence_raw_list = []
            threat_sequence = []
            scada_sequence = []
            pmu_sequence = []
            
            #### NEW: Initialize Mask List ####
            edge_mask_sequence = [] 
            ###################################

            for ts in sequence:
                # --- SCADA ---
                scada_data_raw = safe_get_or_default(ts, 'scada_data', (self.num_nodes, 15)) 
                if scada_data_raw.shape[1] >= 13: scada_data = scada_data_raw[:, :13] 
                else: scada_data = np.zeros(scada_shape, dtype=np.float32)
                
                if scada_data.shape[1] >= 6:
                    scada_data[:, 2] = self._normalize_power(scada_data[:, 2])
                    scada_data[:, 3] = self._normalize_power(scada_data[:, 3])
                    scada_data[:, 4] = self._normalize_power(scada_data[:, 4])
                    scada_data[:, 5] = self._normalize_power(scada_data[:, 5])
                scada_sequence.append(scada_data)
                
                # --- PMU ---
                pmu_data = safe_get_or_default(ts, 'pmu_sequence', pmu_shape)
                if pmu_data.shape[1] >= 6: pmu_data[:, 5] = self._normalize_frequency(pmu_data[:, 5])
                pmu_sequence.append(pmu_data)

                # --- Others ---
                satellite_sequence.append(safe_get_or_default(ts, 'satellite_data', sat_shape))
                weather_sequence_raw_list.append(safe_get_or_default(ts, 'weather_sequence', weather_shape_raw))
                threat_sequence.append(safe_get_or_default(ts, 'threat_indicators', threat_shape))

                #### NEW: DYNAMIC TOPOLOGY MASKING LOGIC ####
                # Create default mask (all 1.0 = active)
                current_edge_mask = np.ones(num_edges, dtype=np.float32)
                
                # Check history for failed nodes
                node_status = ts.get('node_labels')
                if node_status is not None:
                    # Identify failed nodes (value > 0.5)
                    failed_node_indices = np.where(node_status > 0.5)[0]
                    
                    if len(failed_node_indices) > 0:
                        # Use cached numpy edge index
                        src, dst = self.edge_index.cpu().numpy()
                        # If src OR dst is failed, the line is effectively dead
                        edge_failed_mask = np.isin(src, failed_node_indices) | np.isin(dst, failed_node_indices)
                        current_edge_mask[edge_failed_mask] = 0.0
                
                edge_mask_sequence.append(current_edge_mask)
                #############################################

            # Stack all sequences
            scada_sequence = np.stack(scada_sequence)
            pmu_sequence = np.stack(pmu_sequence)
            satellite_sequence = np.stack(satellite_sequence)
            threat_sequence = np.stack(threat_sequence)
            
            weather_sequence_raw = np.stack(weather_sequence_raw_list)
            weather_sequence = weather_sequence_raw.reshape(weather_sequence_raw.shape[0], self.num_nodes, -1)
            
            equipment_sequence = np.stack([safe_get_or_default(ts, 'equipment_status', equip_shape) for ts in sequence])
            visual_sequence = np.stack([safe_get_or_default(ts, 'visual_data', vis_shape) for ts in sequence])
            thermal_sequence = np.stack([safe_get_or_default(ts, 'thermal_data', therm_shape) for ts in sequence])
            sensor_sequence = np.stack([safe_get_or_default(ts, 'sensor_data', sensor_shape) for ts in sequence])
            
            #### NEW: Stack Mask ####
            edge_mask_sequence = np.stack(edge_mask_sequence)
            #########################
            
            if edge_attr.shape[1] >= 2: edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])

            # ... (Ground Truth extraction logic remains identical) ...
            ground_truth = {}
            if 'metadata' in scenario:
                metadata = scenario['metadata']
                failed_nodes = metadata.get('failed_nodes', [])
                if isinstance(failed_nodes, np.ndarray): failed_nodes = failed_nodes.flatten().tolist()
                elif not isinstance(failed_nodes, list): failed_nodes = [failed_nodes] if failed_nodes is not None else []
                
                cascade_start_time = metadata.get('cascade_start_time', -1)
                actual_time_to_cascade = float(cascade_start_time)
                
                failure_times = metadata.get('failure_times', [])
                failure_reasons = metadata.get('failure_reasons', [])
                actual_cascade_path = []
                if failed_nodes and failure_times and failure_reasons:
                    try:
                        actual_cascade_path = [
                            {'node_id': int(node), 'time_minutes': float(time), 'reason': str(reason)}
                            for node, time, reason in zip(failed_nodes, failure_times, failure_reasons)
                        ]
                        actual_cascade_path.sort(key=lambda x: x['time_minutes'])
                    except Exception: actual_cascade_path = [] 
                
                ground_truth = {
                    'is_cascade': bool(metadata.get('is_cascade', False)),
                    'failed_nodes': [int(x) for x in failed_nodes],
                    'time_to_cascade': actual_time_to_cascade,
                    'cascade_path': actual_cascade_path,
                    'ground_truth_risk': metadata.get('ground_truth_risk', [])
                }

            def to_tensor(data): return torch.from_numpy(data.astype(np.float32))

            return {
                'satellite_data': to_tensor(satellite_sequence),
                'weather_sequence': to_tensor(weather_sequence),
                'threat_indicators': to_tensor(threat_sequence),
                'scada_data': to_tensor(scada_sequence),
                'pmu_sequence': to_tensor(pmu_sequence),
                'equipment_status': to_tensor(equipment_sequence),
                'visual_data': to_tensor(visual_sequence),
                'thermal_data': to_tensor(thermal_sequence),
                'sensor_data': to_tensor(sensor_sequence),
                'edge_attr': to_tensor(edge_attr),
                #### NEW: Add mask to return dict ####
                'edge_mask': to_tensor(edge_mask_sequence),
                #####################################
                'edge_index': scenario['edge_index'].long(),
                'ground_truth': ground_truth,
                'sequence_length': len(sequence),
                'node_failure_labels': torch.zeros(self.num_nodes, dtype=torch.float32),
                'cascade_timing': torch.zeros(self.num_nodes, dtype=torch.float32),
                'graph_properties': {}
            }
        except Exception as e:
            import traceback
            print(f"Error extracting scenario: {e}\n{traceback.format_exc()}")
            return None
            
    def predict_batch(self, batch_of_scenarios: List[Dict]) -> List[Dict]:
        if not batch_of_scenarios: return []
        ground_truths = [s.pop('ground_truth', {}) for s in batch_of_scenarios]
        batch = collate_cascade_batch(batch_of_scenarios)
        
        batch_device = {}
        for k, v in batch.items():
            if k == 'graph_properties': continue
            elif isinstance(v, torch.Tensor): batch_device[k] = v.to(self.device)
            else: batch_device[k] = v 
        
        with torch.no_grad():
            outputs = self.model(batch_device)
        
        return self._unpack_batch_predictions(outputs, ground_truths)


    def _unpack_batch_predictions(self, outputs: Dict[str, torch.Tensor], 
                                  ground_truths: List[Dict]) -> List[Dict]:
        batch_size = len(ground_truths)
        predictions = []
        
        node_failure_prob = outputs['failure_probability'].squeeze(-1).cpu().numpy()
        risk_scores = outputs['risk_scores'].cpu().numpy()
        voltages_pu = outputs['voltages'].squeeze(-1).cpu().numpy()
        angles_rad = outputs['angles'].squeeze(-1).cpu().numpy()
        line_flows_pu = outputs['line_flows'].squeeze(-1).cpu().numpy()
        frequency_hz = outputs['frequency'].squeeze(-1).squeeze(-1).cpu().numpy()
        
        relay_outputs_list = [{} for _ in range(batch_size)]
        if 'relay_outputs' in outputs:
            for i in range(batch_size):
                relay_outputs_list[i] = {
                    'time_dial': outputs['relay_outputs']['time_dial'][i].squeeze(-1).cpu().numpy(),
                    'pickup_current': outputs['relay_outputs']['pickup_current'][i].squeeze(-1).cpu().numpy(),
                    'operating_time': outputs['relay_outputs']['operating_time'][i].squeeze(-1).cpu().numpy(),
                    'will_operate': (outputs['relay_outputs']['will_operate'][i].squeeze(-1).cpu().numpy() > 0.5).astype(bool)
                }

        failure_timing_raw = outputs.get('failure_timing', outputs.get('cascade_timing'))
        if failure_timing_raw is None:
             failure_timing_raw = torch.zeros(batch_size, self.num_nodes, 1, device=self.device)

        if failure_timing_raw.shape[1] == self.edge_index.shape[1]: 
            src, dst = self.edge_index.cpu().numpy()
            failure_timing_edge_batch = failure_timing_raw.squeeze(-1).cpu().numpy() 
            failure_timing_node_batch = np.full((batch_size, self.num_nodes), np.inf, dtype=np.float32)
            for i in range(batch_size):
                np.minimum.at(failure_timing_node_batch[i], src, failure_timing_edge_batch[i])
                np.minimum.at(failure_timing_node_batch[i], dst, failure_timing_edge_batch[i])
            failure_timing_node_batch[failure_timing_node_batch == np.inf] = -1.0
        else: 
            failure_timing_node_batch = failure_timing_raw.squeeze(-1).cpu().numpy()

        for i in range(batch_size):
            node_prob_i = node_failure_prob[i]
            risk_scores_i = risk_scores[i]
            voltages_pu_i = voltages_pu[i]
            angles_rad_i = angles_rad[i]
            line_flows_pu_i = line_flows_pu[i]
            frequency_hz_i = frequency_hz[i]
            failure_timing_node_i = failure_timing_node_batch[i]
            relay_outputs_i = relay_outputs_list[i]
            ground_truth = ground_truths[i]

            frequency_pu_i = frequency_hz_i / self.base_frequency
            line_flows_mw_i = self._denormalize_power(line_flows_pu_i)
            
            cascade_prob = float(np.max(node_prob_i))
            cascade_detected = bool(cascade_prob > self.cascade_threshold)
            
            high_risk_nodes_mask = node_prob_i > self.node_threshold
            high_risk_nodes = np.where(high_risk_nodes_mask)[0].tolist()
            
            if cascade_detected and np.any(high_risk_nodes_mask):
                failure_times = failure_timing_node_i[high_risk_nodes_mask]
                valid_failure_times = failure_times[failure_times >= 0]
                time_to_cascade_value = float(np.min(valid_failure_times)) if valid_failure_times.size > 0 else -1.0
            else:
                time_to_cascade_value = -1.0
            
            node_risks = [(j, float(node_prob_i[j])) for j in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            top_risk_nodes = node_risks[:10]
            
            aggregated_risk_scores = np.mean(risk_scores_i, axis=0)
            
            cascade_path = []
            if cascade_detected:
                failure_sequence = [
                    (j, failure_timing_node_i[j]) 
                    for j in range(self.num_nodes) 
                    if high_risk_nodes_mask[j] and failure_timing_node_i[j] >= 0
                ]
                failure_sequence.sort(key=lambda x: x[1])
                
                cascade_path = [
                    {'order': idx + 1, 'node_id': int(node), 'ranking_score': float(score)} 
                    for idx, (node, score) in enumerate(failure_sequence)
                ]

            prediction = {
                'cascade_probability': cascade_prob,
                'cascade_detected': cascade_detected,
                'time_to_cascade_minutes': time_to_cascade_value,
                'high_risk_nodes': high_risk_nodes,
                'top_10_risk_nodes': [{'node_id': int(n), 'failure_probability': float(p)} for n, p in top_risk_nodes],
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
                },
                'system_state': {
                    'voltages_pu': voltages_pu_i.tolist(),
                    'angles_rad': angles_rad_i.tolist(),
                    'line_flows_mw': line_flows_mw_i.tolist(),
                    'line_flows_pu': line_flows_pu_i.tolist(),
                    'frequency_hz': float(frequency_hz_i),
                    'frequency_pu': float(frequency_pu_i)
                },
                'relay_operations': relay_outputs_i,
                'cascade_path': cascade_path,
                'total_nodes_at_risk': len(high_risk_nodes),
                'timestamp': datetime.now().isoformat(),
                'ground_truth': ground_truth
            }
            predictions.append(prediction)
            
        return predictions

    
    def predict_from_file(self, data_path: str, scenario_idx: int = 0, 
                         use_temporal: bool = True) -> Dict:
        print(f"Loading scenario {scenario_idx} from {data_path}...")
        
        data_path = Path(data_path)
        scenario_files = sorted(glob.glob(str(data_path / "scenario_*.pkl")))
        if not scenario_files:
            scenario_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
            if not scenario_files: raise ValueError(f"No scenario files found.")
            print(f"Warning: Found 'scenarios_batch_*.pkl' files.")
        
        if scenario_idx < 0 or scenario_idx >= len(scenario_files):
            raise ValueError(f"Scenario index {scenario_idx} out of range.")

        target_file = scenario_files[scenario_idx]
        print(f"Directly loading file: {target_file}")

        try:
            with open(target_file, 'rb') as f:
                scenario_data = pickle.load(f)
            
            if isinstance(scenario_data, list):
                if len(scenario_data) == 0: raise ValueError(f"Empty list in file.")
                scenario_full = scenario_data[0]
            else: scenario_full = scenario_data
            
            if not isinstance(scenario_full, dict): raise ValueError(f"Data not a dict.")

        except (IOError, pickle.UnpicklingError) as e:
            raise ValueError(f"Could not read file: {e}")

        scenario_full['edge_index'] = self.edge_index
        preprocessed_full = self.extract_and_preprocess_scenario(scenario_full)
        if preprocessed_full is None: raise ValueError(f"Failed to preprocess full scenario.")
        ground_truth_full = preprocessed_full['ground_truth']

        scenario_truncated = pickle.loads(pickle.dumps(scenario_full))
        
        metadata = scenario_truncated.get('metadata', {})
        cascade_start_time = metadata.get('cascade_start_time', -1)
        is_cascade = metadata.get('is_cascade', False)
        sequence_original = scenario_truncated['sequence']
        
        if is_cascade:
            prediction_timestep = int(cascade_start_time)
            if prediction_timestep <= 0 or prediction_timestep >= len(sequence_original):
                prediction_timestep = int(len(sequence_original) * 0.7) 
            
            print(f"  [Inference] Cascade detected at t={cascade_start_time}.")
            print(f"  [Inference] Truncating sequence to pre-cascade (0 to {prediction_timestep - 1}).")
            scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
        else:
            min_len = int(len(sequence_original) * 0.6)
            max_len = int(len(sequence_original) * 0.95) 
            prediction_timestep = np.random.randint(min_len, max_len)
            print(f"  [Inference] Normal scenario. Random truncation (0 to {prediction_timestep - 1}).")
            scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
        
        if len(scenario_truncated['sequence']) == 0:
            raise ValueError(f"Scenario {scenario_idx} has no pre-cascade data.")

        preprocessed_scenario = self.extract_and_preprocess_scenario(scenario_truncated)
        if preprocessed_scenario is None: raise ValueError(f"Failed to preprocess truncated.")
        
        prediction_list = self.predict_batch([preprocessed_scenario])
        prediction_list[0]['ground_truth'] = ground_truth_full
        
        return prediction_list[0]
    
    def batch_predict(self, data_path: str, batch_size: int = 16, 
                      max_scenarios: int = None,
                      use_temporal: bool = True) -> List[Dict]:
        
        total_scenarios = self.count_scenarios(data_path)
        if total_scenarios == 0: return []
            
        num_scenarios = total_scenarios if max_scenarios is None else min(max_scenarios, total_scenarios)
        
        print(f"Processing {num_scenarios} scenarios...")
        
        all_predictions = []
        scenario_batch_buffer = []
        processed_count = 0
        
        pbar = tqdm(self.load_scenarios_streaming(data_path), total=num_scenarios, desc="Batch Predicting")
        
        for i, scenario_full in enumerate(pbar):
            if processed_count >= num_scenarios: break
            
            try:
                scenario_truncated = pickle.loads(pickle.dumps(scenario_full))
                
                metadata = scenario_truncated.get('metadata', {})
                cascade_start_time = metadata.get('cascade_start_time', -1)
                is_cascade = metadata.get('is_cascade', False)
                sequence_original = scenario_truncated['sequence']

                if is_cascade:
                    prediction_timestep = int(cascade_start_time)
                    if prediction_timestep <= 0 or prediction_timestep >= len(sequence_original):
                        prediction_timestep = int(len(sequence_original) * 0.7) 
                    scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
                else:
                    min_len = int(len(sequence_original) * 0.6)
                    max_len = int(len(sequence_original) * 0.95)
                    np.random.seed(i)
                    prediction_timestep = np.random.randint(min_len, max_len)
                    scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
                
                if len(scenario_truncated['sequence']) == 0:
                    processed_count += 1
                    continue
            
                preprocessed_scenario = self.extract_and_preprocess_scenario(scenario_truncated)
                if preprocessed_scenario is None:
                    processed_count += 1
                    continue
                
                full_gt = self.extract_and_preprocess_scenario(scenario_full)['ground_truth']
                preprocessed_scenario['ground_truth'] = full_gt
                
                scenario_batch_buffer.append(preprocessed_scenario)
                
                is_last_scenario = (processed_count == num_scenarios - 1)
                if len(scenario_batch_buffer) == batch_size or (is_last_scenario and scenario_batch_buffer):
                    batch_predictions = self.predict_batch(scenario_batch_buffer)
                    all_predictions.extend(batch_predictions)
                    scenario_batch_buffer = []
                
                processed_count += 1
                
            except Exception: processed_count += 1 
        
        pbar.close()
        return all_predictions
    
    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        if not predictions or 'ground_truth' not in predictions[0]: return {}
        
        cascade_correct, cascade_total = 0, 0
        tp, fp, tn, fn = 0, 0, 0, 0
        time_errors = []
        
        for pred in predictions:
            gt = pred['ground_truth']
            predicted_cascade = pred['cascade_detected']
            actual_cascade = gt['is_cascade']
            
            if predicted_cascade == actual_cascade: cascade_correct += 1
            
            if predicted_cascade and actual_cascade:
                tp += 1
                pred_time = pred['time_to_cascade_minutes']
                actual_path = gt.get('cascade_path', [])
                actual_time = actual_path[0].get('time_minutes', 0.0) if actual_path else 0.0
                if pred_time >= 0: time_errors.append(abs(pred_time - actual_time))
            elif predicted_cascade and not actual_cascade: fp += 1
            elif not predicted_cascade and not actual_cascade: tn += 1
            elif not predicted_cascade and actual_cascade: fn += 1
            
            cascade_total += 1
        
        accuracy = cascade_correct / cascade_total if cascade_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        mean_time_error = np.mean(time_errors) if time_errors else 0
        median_time_error = np.median(time_errors) if time_errors else 0
        
        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'false_positive_rate': fpr, 'true_positives': tp, 'false_positives': fp,
            'true_negatives': tn, 'false_negatives': fn, 'total_scenarios': cascade_total,
            'time_to_cascade_mae': mean_time_error, 'time_to_cascade_median_error': median_time_error
        }

def print_single_prediction_report(prediction: Dict, inference_time: float, cascade_threshold: float, node_threshold: float):
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS (Scenario Analysis)")
    print("=" * 80)
    
    gt = prediction.get('ground_truth', {})
    actual_cascade = gt.get('is_cascade', False)
    predicted_cascade = prediction['cascade_detected']
    
    print(f"Inference Time: {inference_time:.4f} seconds\n")
    print("--- 1. Overall Verdict ---")
    if predicted_cascade and actual_cascade: print("✅ Correctly detected a cascade.")
    elif not predicted_cascade and not actual_cascade: print("✅ Correctly identified a normal scenario.")
    elif not predicted_cascade and actual_cascade: print("❌ FALSE NEGATIVE (Missed Cascade)")
    elif predicted_cascade and not actual_cascade: print("⚠️ FALSE POSITIVE (False Alarm)")

    print(f"Prediction: {predicted_cascade} (Prob: {prediction['cascade_probability']:.3f} / Thresh: {cascade_threshold:.3f})")
    print(f"Ground Truth: {actual_cascade}")

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

    # --- 3. IMPROVED TIMING ANALYSIS ---
    if actual_cascade or predicted_cascade:
        print("\n--- 3. Timing Analysis ---")
        
        pred_path = prediction.get('cascade_path', [])
        actual_path = gt.get('cascade_path', [])
        
        has_pred_time = pred_path and 'time_minutes' in pred_path[0]
        actual_duration = 0.0
        if actual_path:
            t_min = min(x['time_minutes'] for x in actual_path)
            t_max = max(x['time_minutes'] for x in actual_path)
            actual_duration = t_max - t_min

        print(f"  Metric                      | Predicted       | Ground Truth")
        print(f"  ----------------------------|-----------------|-----------------")
        
        if has_pred_time:
            # Old Time-based model
            p_times = [p['time_minutes'] for p in pred_path]
            p_dur = max(p_times) - min(p_times) if p_times else 0
            print(f"  Prediction Mode             | Time (Minutes)  | Absolute Time")
            print(f"  First Failure               | {min(p_times):.2f} min      | 0.00 min")
            print(f"  Total Duration              | {p_dur:.2f} min      | {actual_duration:.2f} min")
        else:
            # New Ranking-based model
            if pred_path:
                min_s = min(p['ranking_score'] for p in pred_path)
                max_s = max(p['ranking_score'] for p in pred_path)
                spread = max_s - min_s
                print(f"  Prediction Mode             | Relative Rank   | Absolute Time")
                print(f"  Range (Start -> End)        | {min_s:.3f} -> {max_s:.3f} | 0.00 -> {actual_duration:.2f} min")
                print(f"  Sequence Spread             | {spread:.3f} (Score)  | {actual_duration:.2f} minutes")
            else:
                print("  (No predicted path data available)")

    print("\n--- 4. Critical Information ---")
    print(f"System Frequency: {prediction['system_state']['frequency_hz']:.2f} Hz")
    v_min, v_max = np.min(prediction['system_state']['voltages_pu']), np.max(prediction['system_state']['voltages_pu'])
    print(f"Voltage Range:    [{v_min:.3f}, {v_max:.3f}] p.u.")
    
    if predicted_cascade:
        print("\nTop 5 High-Risk Nodes:")
        actual_nodes = set(gt.get('failed_nodes', []))
        for node_info in prediction['top_10_risk_nodes'][:5]:
            node_id = int(node_info['node_id'])
            prob = float(node_info['failure_probability'])
            status = "✓ (Actual)" if node_id in actual_nodes else "✗ (Not Actual)"
            if 'is_cascade' not in gt: status = "" 
            print(f"  - Node {node_id:<3}: {prob:.4f} {status}")
    
    def get_risk_label(score):
        if score < 0.3: return "(Low)"
        if score < 0.6: return "(Medium)"
        if score < 0.8: return "(Severe)"
        return "(Critical)"
        
    print("\nAggregated Risk Assessment (7-Dimensions):")
    risk = prediction['risk_assessment']['aggregated']
    print(f"  - Threat: {risk['threat_severity']:.3f} {get_risk_label(risk['threat_severity']):<10} | Vulnerability: {risk['vulnerability']:.3f} {get_risk_label(risk['vulnerability']):<10} | Impact: {risk['operational_impact']:.3f} {get_risk_label(risk['operational_impact']):<10}")
    print(f"  - Cascade Prob: {risk['cascade_probability']:.3f} {get_risk_label(risk['cascade_probability']):<10} | Response: {risk['response_complexity']:.3f} {get_risk_label(risk['response_complexity']):<10} | Safety: {risk['public_safety']:.3f} {get_risk_label(risk['public_safety']):<10}")
    print(f"  - Urgency: {risk['urgency']:.3f} {get_risk_label(risk['urgency']):<10}")

    gt_risk_scores = gt.get('ground_truth_risk', [])
    if len(gt_risk_scores) == 7:
        print("\n  Ground Truth Risk Assessment:")
        print(f"  - Threat: {gt_risk_scores[0]:.3f} {get_risk_label(gt_risk_scores[0]):<10} | Vulnerability: {gt_risk_scores[1]:.3f} {get_risk_label(gt_risk_scores[1]):<10} | Impact: {gt_risk_scores[2]:.3f} {get_risk_label(gt_risk_scores[2]):<10}")
        print(f"  - Cascade Prob: {gt_risk_scores[3]:.3f} {get_risk_label(gt_risk_scores[3]):<10} | Response: {gt_risk_scores[4]:.3f} {get_risk_label(gt_risk_scores[4]):<10} | Safety: {gt_risk_scores[5]:.3f} {get_risk_label(gt_risk_scores[5]):<10}")
        print(f"  - Urgency: {gt_risk_scores[6]:.3f} {get_risk_label(gt_risk_scores[6]):<10}")

    # --- 5. IMPROVED CASCADE PATH GROUPING ---
    print("\n--- 5. Cascade Path Analysis (Sequence Order) ---")
    pred_path = prediction.get('cascade_path', [])
    actual_path = gt.get('cascade_path', [])
    
    if not pred_path and not actual_path:
        print("  - No cascade path information available.")
    else:
        print(f"  {'Seq #':<6} | {'Predicted Node':<15} | {'Score':<8} | {'Actual Seq #':<15} | {'Actual Node':<15} | {'Delta T (min)':<15}")
        print(f"  {'-'*6} | {'-'*15} | {'-'*8} | {'-'*15} | {'-'*15} | {'-'*15}")
        
        max_len = max(len(pred_path), len(actual_path))
        
        # Grouping helpers
        current_act_rank = 0
        last_act_time_anchor = -999.0
        
        current_pred_rank = 0
        last_pred_score_anchor = -999.0
        
        for i in range(max_len):
            # Prediction Column (Score Grouping)
            p_seq, p_node, p_score = "", "", ""
            if i < len(pred_path):
                s = pred_path[i]['ranking_score']
                p_node = f"Node {pred_path[i]['node_id']}"
                p_score = f"{s:.3f}"
                
                # New Group if score > anchor + 0.05
                if abs(s - last_pred_score_anchor) > 0.05: 
                    current_pred_rank += 1
                    last_pred_score_anchor = s
                
                p_seq = str(current_pred_rank)

            # Actual Column (Time Grouping)
            a_seq, a_node, a_time = "", "", ""
            if i < len(actual_path):
                t = actual_path[i]['time_minutes']
                a_node = f"Node {actual_path[i]['node_id']}"
                a_time = f"{t:.2f}"
                
                # New Group if time > anchor + 0.1 min (6 sec)
                if t > last_act_time_anchor + 0.1:
                    current_act_rank += 1
                    last_act_time_anchor = t
                
                a_seq = str(current_act_rank)
            
            print(f"  {p_seq:<6} | {p_node:<15} | {p_score:<8} | {a_seq:<15} | {a_node:<15} | {a_time:<15}")

    print("=" * 80 + "\n")


def print_batch_report(predictions: List[Dict], metrics: Dict, total_time: float):
    print("\n" + "=" * 80)
    print("BATCH PREDICTION REPORT")
    print("=" * 80)
    print(f"Total Scenarios: {metrics.get('total_scenarios', 0)}")
    print(f"Accuracy:        {metrics.get('accuracy', 0):.4f}")
    print(f"Precision:       {metrics.get('precision', 0):.4f}")
    print(f"Recall:          {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:        {metrics.get('f1_score', 0):.4f}")
    print(f"Time-to-Cascade MAE: {metrics.get('time_to_cascade_mae', 0):.2f} minutes")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print("=" * 80 + "\n")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference (Physics-Informed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data/grid_topology.pkl", 
                       help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data/test", 
                       help="Path to test data directory (e.g., data/test)")
    parser.add_argument("--scenario_idx", type=int, default=0, 
                       help="Scenario index for single prediction")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction on all scenarios in data_path")
    
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for batch prediction")
    
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
    
    if args.device:
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available() and DEVICE.type == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU.")
        DEVICE = torch.device("cpu")
    
    print(f"Using device: {DEVICE}")
    
    global predictor
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
            batch_size=args.batch_size,
            max_scenarios=args.max_scenarios,
            use_temporal=use_temporal
        )
        total_time = time.time() - start_time
        
        if not predictions:
            print("No predictions were generated. Exiting.")
            return

        for i, pred in enumerate(predictions):
            if 'ground_truth' in pred:
                pred['ground_truth']['scenario_id'] = i

        metrics = predictor.evaluate_predictions(predictions)
        
        print_batch_report(predictions, metrics, total_time)
        
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
            
            print_single_prediction_report(
                prediction, 
                inference_time,
                predictor.cascade_threshold,
                predictor.node_threshold
            )

            with open(args.output, 'w') as f:
                json.dump(prediction, f, indent=2, cls=NumpyEncoder)
            print(f"Full prediction details saved to {args.output}")
            
        except ValueError as e:
            print(f"\nError: {e}")
            print(f"Could not load scenario index {args.scenario_idx}.")
            print("Please ensure the index is valid and the data path is correct.")
        except NameError as e:
            if "NipEncoder" in str(e):
                print("\n[Code Error] Fixing typo: NipEncoder -> NumpyEncoder")
                with open(args.output, 'w') as f:
                    json.dump(prediction, f, indent=2, cls=NumpyEncoder)
                print(f"Full prediction details saved to {args.output}")
            else:
                raise e


if __name__ == "__main__":
    main()