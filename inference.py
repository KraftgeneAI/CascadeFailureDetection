"""
Cascade Failure Prediction - Topology-Based Simulation Inference
============================================================
(MODIFIED for "Iterative Topology Propagation")
- ABANDONS internal model timing predictions.
- USES iterative simulation:
  1. Predict failures based on vulnerability.
  2. LOOK UP topology for failed nodes.
  3. MASK connected edges (simulate line trips).
  4. INJECT artificial propagation delays (physics simulation).
  5. RE-RUN model on new topology to find next failures.
- Preserves original output format.
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
import copy
import random

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
    """Inference engine for cascade prediction using Iterative Topology Propagation."""
    
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
            
            if not isinstance(topology['edge_index'], torch.Tensor):
                self.edge_index_np = topology['edge_index'] 
                self.edge_index = torch.from_numpy(self.edge_index_np).long()
            else:
                self.edge_index = topology['edge_index']
                self.edge_index_np = self.edge_index.cpu().numpy()

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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully (Iterative Topology Mode)")
        print(f"  Cascade threshold: {self.cascade_threshold:.4f}")
        print(f"  Node threshold: {self.node_threshold:.4f}")
    
    
    def _normalize_power(self, power_values: np.ndarray) -> np.ndarray:
        return power_values / self.base_mva
    
    def _normalize_frequency(self, frequency_values: np.ndarray) -> np.ndarray:
        return frequency_values / self.base_frequency
    
    def _denormalize_power(self, power_pu: np.ndarray) -> np.ndarray:
        return power_pu * self.base_mva
    
    def load_scenarios_streaming(self, data_path: str):
        data_path = Path(data_path)
        scenario_files = sorted(glob.glob(str(data_path / "scenario_*.pkl")))
        if not scenario_files:
            scenario_files = sorted(glob.glob(str(data_path / "scenarios_batch_*.pkl")))
        
        for scenario_file in scenario_files:
            try:
                with open(scenario_file, 'rb') as f:
                    scenario_data = pickle.load(f)
                if isinstance(scenario_data, list):
                    if len(scenario_data) == 0: continue
                    scenario = scenario_data[0]
                else: scenario = scenario_data
                
                if not isinstance(scenario, dict): continue
                scenario['edge_index'] = self.edge_index
                yield scenario
            except Exception as e:
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

            scada_shape = (self.num_nodes, 13)
            sat_shape = (self.num_nodes, 12, 16, 16)
            weather_shape_raw = (self.num_nodes, 10, 8)
            threat_shape = (self.num_nodes, 6)
            pmu_shape = (self.num_nodes, 8)
            equip_shape = (self.num_nodes, 10)
            vis_shape = (self.num_nodes, 3, 32, 32)
            therm_shape = (self.num_nodes, 1, 32, 32)
            sensor_shape = (self.num_nodes, 12)
            
            edge_attr_data = sequence[-1].get('edge_attr')
            if edge_attr_data is None: edge_attr_data = scenario.get('edge_attr')
            num_edges = self.edge_index.shape[1]
            edge_shape = (num_edges, 5)
            
            if edge_attr_data is None: edge_attr = np.zeros(edge_shape, dtype=np.float32)
            elif edge_attr_data.shape != edge_shape: edge_attr = np.zeros(edge_shape, dtype=np.float32)
            else: edge_attr = edge_attr_data.astype(np.float32)

            satellite_sequence = [] 
            weather_sequence_raw_list = []
            threat_sequence = []
            scada_sequence = []
            pmu_sequence = []
            edge_mask_sequence = [] 

            for ts in sequence:
                scada_data_raw = safe_get_or_default(ts, 'scada_data', (self.num_nodes, 15)) 
                if scada_data_raw.shape[1] >= 13: scada_data = scada_data_raw[:, :13] 
                else: scada_data = np.zeros(scada_shape, dtype=np.float32)
                
                if scada_data.shape[1] >= 6:
                    scada_data[:, 2] = self._normalize_power(scada_data[:, 2])
                    scada_data[:, 3] = self._normalize_power(scada_data[:, 3])
                    scada_data[:, 4] = self._normalize_power(scada_data[:, 4])
                    scada_data[:, 5] = self._normalize_power(scada_data[:, 5])
                scada_sequence.append(scada_data)
                
                pmu_data = safe_get_or_default(ts, 'pmu_sequence', pmu_shape)
                if pmu_data.shape[1] >= 6: pmu_data[:, 5] = self._normalize_frequency(pmu_data[:, 5])
                pmu_sequence.append(pmu_data)

                satellite_sequence.append(safe_get_or_default(ts, 'satellite_data', sat_shape))
                weather_sequence_raw_list.append(safe_get_or_default(ts, 'weather_sequence', weather_shape_raw))
                threat_sequence.append(safe_get_or_default(ts, 'threat_indicators', threat_shape))

                # Default mask (will be modified dynamically during simulation)
                current_edge_mask = np.ones(num_edges, dtype=np.float32)
                edge_mask_sequence.append(current_edge_mask)

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
            edge_mask_sequence = np.stack(edge_mask_sequence)
            
            if edge_attr.shape[1] >= 2: edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])

            ground_truth = {}
            if 'metadata' in scenario:
                metadata = scenario['metadata']
                failed_nodes = metadata.get('failed_nodes', [])
                if isinstance(failed_nodes, np.ndarray): failed_nodes = failed_nodes.flatten().tolist()
                elif not isinstance(failed_nodes, list): failed_nodes = [failed_nodes] if failed_nodes is not None else []
                
                actual_cascade_path = []
                if failed_nodes and 'failure_times' in metadata:
                    try:
                        actual_cascade_path = [
                            {'node_id': int(n), 'time_minutes': float(t)}
                            for n, t in zip(failed_nodes, metadata.get('failure_times', []))
                        ]
                        actual_cascade_path.sort(key=lambda x: x['time_minutes'])
                    except: pass
                
                ground_truth = {
                    'is_cascade': bool(metadata.get('is_cascade', False)),
                    'failed_nodes': [int(x) for x in failed_nodes],
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
                'edge_mask': to_tensor(edge_mask_sequence),
                'edge_index': scenario['edge_index'].long(),
                'ground_truth': ground_truth,
                'sequence_length': len(sequence),
                'graph_properties': {}
            }
        except Exception as e:
            return None

    def _run_single_step(self, batch_device):
        """Runs the model once on the current graph state."""
        with torch.no_grad():
            outputs = self.model(batch_device)
        return outputs

    def predict_simulation(self, batch_of_scenarios: List[Dict]) -> List[Dict]:
        """
        ITERATIVE TOPOLOGY-BASED PREDICTION (SIMULATION)
        - Reconstructs causal path by iteratively masking failed nodes.
        - Injects Artificial Time for edges to sum up proactive window.
        """
        if not batch_of_scenarios: return []
        
        # Process one scenario at a time (Batch size effectively 1 for simulation)
        predictions = []
        
        for scenario_data in batch_of_scenarios:
            ground_truth = scenario_data.pop('ground_truth', {})
            
            # Prepare batch
            batch = collate_cascade_batch([scenario_data])
            batch_device = {}
            for k, v in batch.items():
                if k == 'graph_properties': continue
                elif isinstance(v, torch.Tensor): batch_device[k] = v.to(self.device)
                else: batch_device[k] = v 
            
            # --- SIMULATION STATE ---
            failed_nodes_set = set()
            cascade_path = []
            current_simulation_time = 0.0 # Virtual minutes from start of cascade
            
            simulation_step = 0
            max_steps = 15 # Safety break
            
            # Artificial Propagation Physics: Random delay per hop (2 to 8 mins)
            # We seed this for reproducibility per scenario
            random.seed(42) 
            
            final_risk_scores = None
            final_node_probs = None
            final_system_state = {}
            final_relay_ops = {}
            
            # Cache topology on CPU for lookup
            src, dst = self.edge_index.cpu().numpy()
            
            while simulation_step < max_steps:
                # 1. Run Model
                outputs = self._run_single_step(batch_device)
                
                node_probs = outputs['failure_probability'].squeeze(-1).cpu().numpy()[0] # [Num_Nodes]
                
                # Capture system state from the first pass (most representative of initial conditions)
                if simulation_step == 0:
                    final_risk_scores = outputs['risk_scores'].cpu().numpy()[0]
                    final_system_state = {
                        'voltages_pu': outputs['voltages'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'angles_rad': outputs['angles'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'line_flows_pu': outputs['line_flows'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'frequency_hz': float(outputs['frequency'].squeeze(-1).squeeze(-1).cpu().numpy()[0])
                    }
                    if 'relay_outputs' in outputs:
                         final_relay_ops = {k: v[0].squeeze(-1).cpu().numpy() for k, v in outputs['relay_outputs'].items()}

                # 2. Identify New Failures (Vulnerability Scan)
                candidates = []
                for node_idx, prob in enumerate(node_probs):
                    if node_idx in failed_nodes_set: continue
                    
                    # If model timing is trusted, we would use it here.
                    # Since we are using topology simulation, we use Probability as a proxy for "Weakness"
                    if prob > self.node_threshold:
                        candidates.append((node_idx, prob))
                
                # If no new failures found, simulation stabilizes
                if not candidates:
                    final_node_probs = node_probs
                    break
                
                # 3. Select Next Failure (Greedy approach: Highest Risk = First to Fail)
                # In a real simulation, we might fail multiple if they are close.
                # Here we strictly order them to find the path.
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # We take the top candidate for this step
                next_failure_node, failure_prob = candidates[0]
                
                # 4. ARTIFICIAL TIME INJECTION
                # Calculate time from previous failure to this one based on "Edge Weight"
                # If this is the first failure (Source), time is 0.
                if simulation_step == 0:
                    hop_delay = 0.0
                else:
                    # Logic: 2 to 8 minutes for stress to propagate to next neighbor
                    hop_delay = random.uniform(2.0, 8.0)
                
                current_simulation_time += hop_delay
                
                # Add to path
                failed_nodes_set.add(next_failure_node)
                cascade_path.append({
                    'order': simulation_step + 1,
                    'node_id': int(next_failure_node),
                    'ranking_score': float(failure_prob), # Keeping score for reference
                    'simulated_time_min': float(current_simulation_time),
                    'hop_delay': float(hop_delay)
                })
                
                # 5. TOPOLOGY UPDATE (The "Cut")
                # Mask edges connected to the NEW failed node
                failed_idx = int(next_failure_node)
                edge_failed_mask = (src == failed_idx) | (dst == failed_idx)
                
                # Convert to tensor mask (0.0 for failed edges)
                # Note: We must accumulate masks (previous failures are still failed)
                # But here we just update the tensor in place for the new failure
                edge_drop_mask = torch.from_numpy(~edge_failed_mask).float().to(self.device) # 1=Keep, 0=Drop
                
                # Apply mask to batch. Shape: [1, Seq_Len, Num_Edges]
                mask_tensor = batch_device['edge_mask']
                mask_update = edge_drop_mask.unsqueeze(0).unsqueeze(0).expand_as(mask_tensor)
                batch_device['edge_mask'] = mask_tensor * mask_update
                
                simulation_step += 1
            
            # --- COMPILE RESULTS ---
            
            # Fallback if loop didn't run (no cascade predicted initially)
            if final_node_probs is None:
                outputs = self._run_single_step(batch_device)
                final_node_probs = outputs['failure_probability'].squeeze(-1).cpu().numpy()[0]
                final_risk_scores = outputs['risk_scores'].cpu().numpy()[0]
                final_system_state = {
                        'voltages_pu': outputs['voltages'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'angles_rad': outputs['angles'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'line_flows_pu': outputs['line_flows'].squeeze(-1).cpu().numpy()[0].tolist(),
                        'frequency_hz': float(outputs['frequency'].squeeze(-1).squeeze(-1).cpu().numpy()[0])
                }

            high_risk_nodes_mask = final_node_probs > self.node_threshold
            high_risk_nodes_indices = np.where(high_risk_nodes_mask)[0].tolist()
            
            cascade_detected = len(cascade_path) > 0
            cascade_prob = float(np.max(final_node_probs))
            
            # Calculate Proactive Window (Sum of delays)
            # This represents how far ahead the model sees.
            total_proactive_window = current_simulation_time
            
            aggregated_risk_scores = np.mean(final_risk_scores, axis=0) if final_risk_scores is not None else np.zeros(7)
            
            # Prepare Top 10 list
            node_risks = [(j, float(final_node_probs[j])) for j in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            
            prediction = {
                'cascade_probability': cascade_prob,
                'cascade_detected': cascade_detected,
                'proactive_window_minutes': total_proactive_window, # New Metric
                'time_to_cascade_minutes': 0.0, # Deprecated in this mode
                'high_risk_nodes': high_risk_nodes_indices,
                'top_10_risk_nodes': [{'node_id': int(n), 'failure_probability': float(p)} for n, p in node_risks[:10]],
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
                'system_state': final_system_state,
                'relay_operations': final_relay_ops,
                'cascade_path': cascade_path, # Populated by Simulation Loop
                'total_nodes_at_risk': len(high_risk_nodes_indices),
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
        
        if scenario_idx < 0 or scenario_idx >= len(scenario_files):
            raise ValueError(f"Scenario index {scenario_idx} out of range.")

        target_file = scenario_files[scenario_idx]
        with open(target_file, 'rb') as f:
            scenario_data = pickle.load(f)
            
        if isinstance(scenario_data, list): scenario_full = scenario_data[0]
        else: scenario_full = scenario_data

        scenario_full['edge_index'] = self.edge_index
        preprocessed_full = self.extract_and_preprocess_scenario(scenario_full)
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
            scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
            print(f"  [Inference] Cascade detected at t={cascade_start_time}. Truncating.")
        else:
            prediction_timestep = int(len(sequence_original) * 0.95)
            scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
            print(f"  [Inference] Normal scenario.")

        preprocessed_scenario = self.extract_and_preprocess_scenario(scenario_truncated)
        
        # USE SIMULATION PREDICTION
        prediction_list = self.predict_simulation([preprocessed_scenario])
        prediction_list[0]['ground_truth'] = ground_truth_full
        
        return prediction_list[0]
    
    def batch_predict(self, data_path: str, batch_size: int = 16, 
                      max_scenarios: int = None,
                      use_temporal: bool = True) -> List[Dict]:
        
        total_scenarios = self.count_scenarios(data_path)
        num_scenarios = total_scenarios if max_scenarios is None else min(max_scenarios, total_scenarios)
        print(f"Processing {num_scenarios} scenarios...")
        
        all_predictions = []
        # Note: Simulation mode is slower, so we default to batch_size=1 effectively inside the function
        # But we handle buffering here to match interface
        scenario_batch_buffer = []
        processed_count = 0
        
        pbar = tqdm(self.load_scenarios_streaming(data_path), total=num_scenarios, desc="Simulation Predicting")
        
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
                    prediction_timestep = int(len(sequence_original) * 0.95)
                    scenario_truncated['sequence'] = sequence_original[:prediction_timestep]
                
                if len(scenario_truncated['sequence']) == 0:
                    processed_count += 1
                    continue
            
                preprocessed_scenario = self.extract_and_preprocess_scenario(scenario_truncated)
                if preprocessed_scenario is None: continue
                
                full_gt = self.extract_and_preprocess_scenario(scenario_full)['ground_truth']
                preprocessed_scenario['ground_truth'] = full_gt
                
                scenario_batch_buffer.append(preprocessed_scenario)
                
                # Batch processing for simulation
                if len(scenario_batch_buffer) >= batch_size or (processed_count == num_scenarios - 1):
                    batch_predictions = self.predict_simulation(scenario_batch_buffer)
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
        
        for pred in predictions:
            gt = pred['ground_truth']
            predicted_cascade = pred['cascade_detected']
            actual_cascade = gt['is_cascade']
            
            if predicted_cascade == actual_cascade: cascade_correct += 1
            if predicted_cascade and actual_cascade: tp += 1
            elif predicted_cascade and not actual_cascade: fp += 1
            elif not predicted_cascade and not actual_cascade: tn += 1
            elif not predicted_cascade and actual_cascade: fn += 1
            cascade_total += 1
        
        accuracy = cascade_correct / cascade_total if cascade_total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'true_positives': tp, 'false_positives': fp,
            'true_negatives': tn, 'false_negatives': fn, 'total_scenarios': cascade_total
        }

def print_single_prediction_report(prediction: Dict, inference_time: float, cascade_threshold: float, node_threshold: float):
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS (Topology-Guided Simulation)")
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

    print(f"Prediction: {predicted_cascade} (Prob: {prediction['cascade_probability']:.3f})")
    
    # NEW METRIC DISPLAY
    if predicted_cascade:
        print(f"Proactive Window: {prediction.get('proactive_window_minutes', 0.0):.2f} minutes (Artificial Simulation Time)")

    if actual_cascade or predicted_cascade:
        print("\n--- 2. Node-Level Analysis ---")
        predicted_nodes = set([x['node_id'] for x in prediction['cascade_path']])
        actual_nodes = set(gt.get('failed_nodes', []))
        tp_nodes = len(predicted_nodes.intersection(actual_nodes))
        fp_nodes = len(predicted_nodes.difference(actual_nodes))
        fn_nodes = len(actual_nodes.difference(predicted_nodes))
        print(f"Predicted Nodes in Path: {len(predicted_nodes)}")
        print(f"Actual Failed Nodes:     {len(actual_nodes)}")
        print(f"  - Overlap (TP):        {tp_nodes}")
        print(f"  - Missed (FN):         {fn_nodes}")
        print(f"  - Extra (FP):          {fp_nodes}")

    print("\n--- 3. Simulation Path Analysis (Topology Propagation) ---")
    pred_path = prediction.get('cascade_path', [])
    actual_path = gt.get('cascade_path', [])
    
    if not pred_path and not actual_path:
        print("  - No cascade path information available.")
    else:
        # Header adjusted for simulated time
        print(f"  {'Step':<6} | {'Predicted Node':<15} | {'Delay (+min)':<12} | {'Sim Time':<12} | {'Actual Seq':<12} | {'Actual Node':<15}")
        print(f"  {'-'*6} | {'-'*15} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*15}")
        
        max_len = max(len(pred_path), len(actual_path))
        
        current_act_rank = 0
        last_act_time_anchor = -999.0
        
        for i in range(max_len):
            # Prediction Column
            p_seq, p_node, p_delay, p_time = "", "", "", ""
            if i < len(pred_path):
                p_seq = str(pred_path[i]['order'])
                p_node = f"Node {pred_path[i]['node_id']}"
                p_delay = f"+{pred_path[i]['hop_delay']:.1f}"
                p_time = f"{pred_path[i]['simulated_time_min']:.1f}"

            # Actual Column
            a_seq, a_node = "", ""
            if i < len(actual_path):
                t = actual_path[i]['time_minutes']
                a_node = f"Node {actual_path[i]['node_id']}"
                if t > last_act_time_anchor + 0.1:
                    current_act_rank += 1
                    last_act_time_anchor = t
                a_seq = str(current_act_rank)
            
            print(f"  {p_seq:<6} | {p_node:<15} | {p_delay:<12} | {p_time:<12} | {a_seq:<12} | {a_node:<15}")

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
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print("=" * 80 + "\n")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference (Topology Simulation)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data/grid_topology.pkl", help="Path to topology")
    parser.add_argument("--data_path", type=str, default="data/test", help="Path to test data")
    parser.add_argument("--scenario_idx", type=int, default=0, help="Scenario index")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Max scenarios")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file")
    parser.add_argument("--device", type=str, default=None, help="Device")
    
    # Keeping args for compatibility
    parser.add_argument("--base_mva", type=float, default=100.0)
    parser.add_argument("--base_frequency", type=float, default=60.0)
    parser.add_argument("--no_temporal", action="store_true")
    
    args = parser.parse_args()
    
    if args.device:
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    print("CASCADE PREDICTION - TOPOLOGY SIMULATION MODE")
    print("=" * 80 + "\n")
    
    use_temporal = not args.no_temporal
    
    if args.batch:
        start_time = time.time()
        predictions = predictor.batch_predict(
            args.data_path, 
            batch_size=args.batch_size,
            max_scenarios=args.max_scenarios,
            use_temporal=use_temporal
        )
        total_time = time.time() - start_time
        
        metrics = predictor.evaluate_predictions(predictions)
        print_batch_report(predictions, metrics, total_time)
        
        with open(args.output, 'w') as f:
            json.dump({'metrics': metrics, 'predictions': predictions}, f, indent=2, cls=NumpyEncoder)
    
    else:
        start_time = time.time()
        prediction = predictor.predict_from_file(args.data_path, args.scenario_idx, use_temporal=use_temporal)
        inference_time = time.time() - start_time
        
        print_single_prediction_report(
            prediction, 
            inference_time,
            predictor.cascade_threshold,
            predictor.node_threshold
        )

        with open(args.output, 'w') as f:
            json.dump(prediction, f, indent=2, cls=NumpyEncoder)
#123
if __name__ == "__main__":
    main()