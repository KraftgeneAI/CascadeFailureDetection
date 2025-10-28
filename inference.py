"""
Cascade Failure Prediction Model Inference Script (IMPROVED)
============================================================
Load trained model and make predictions on new data.
Updated to fully utilize the model's temporal and multi-task capabilities.

Key Improvements:
- Processes full 60-timestep sequences (LSTM temporal modeling)
- Returns all 8 prediction heads (voltages, angles, flows, frequency, relays)
- Correct time-to-cascade calculation (first failure, not average)
- Per-node risk scores in addition to aggregated scores
- Configurable cascade threshold
- Cascade propagation path tracking

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

from multimodal_cascade_model import UnifiedCascadePredictionModel

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
    """Inference engine for cascade prediction with full model utilization."""
    
    def __init__(self, model_path: str, topology_path: str, device: str = "cpu", 
                 cascade_threshold: float = 0.3):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            topology_path: Path to grid topology file
            device: Device to run inference on
            cascade_threshold: Threshold for cascade detection (default: 0.3)
        """
        self.device = torch.device(device)
        self.cascade_threshold = cascade_threshold
        
        # Load topology
        print(f"Loading grid topology from {topology_path}...")
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index'].to(self.device)
            self.num_nodes = topology['num_nodes']
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Cascade threshold: {self.cascade_threshold}")
    
    def load_data(self, data_path: str) -> List[Dict]:
        """
        Load data from file or batch directory.
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            List of scenarios
        """
        data_path = Path(data_path)
        
        if data_path.is_file():
            print(f"Loading data from file: {data_path}")
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        elif data_path.is_dir():
            print(f"Loading data from batch directory: {data_path}")
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
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
        """
        Generator that yields scenarios one at a time from batch files.
        Memory-efficient for large datasets.
        
        Args:
            data_path: Path to data file or directory
            
        Yields:
            Individual scenarios
        """
        data_path = Path(data_path)
        
        if data_path.is_file():
            print(f"Loading data from file: {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                for scenario in data:
                    yield scenario
        elif data_path.is_dir():
            print(f"Streaming data from batch directory: {data_path}")
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            if not batch_files:
                raise ValueError(f"No batch files found in {data_path}")
            
            print(f"Found {len(batch_files)} batch files")
            for batch_file in batch_files:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    for scenario in batch_data:
                        yield scenario
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    def count_scenarios(self, data_path: str) -> int:
        """
        Count total scenarios without loading all data into memory.
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            Total number of scenarios
        """
        data_path = Path(data_path)
        
        if data_path.is_file():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                return len(data)
        elif data_path.is_dir():
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            total = 0
            for batch_file in batch_files:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    total += len(batch_data)
            return total
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
    
    def extract_temporal_sequences(self, scenario: Dict) -> Dict:
        """
        Extract full temporal sequences (60 timesteps) from scenario.
        This is CRITICAL for utilizing the LSTM temporal modeling.
        
        Args:
            scenario: Scenario dictionary with 'sequence' key
            
        Returns:
            Dictionary with temporal sequences for all modalities
        """
        if 'sequence' not in scenario:
            raise ValueError("Scenario missing 'sequence' key")
        
        sequence = scenario['sequence']
        if len(sequence) == 0:
            raise ValueError("Scenario sequence is empty")
        
        # Stack all timesteps for each modality
        # Shape: [T, ...] where T is sequence length (typically 60)
        satellite_sequence = np.stack([ts.get('satellite_data', np.zeros((3, 64, 64))) 
                                       for ts in sequence])
        weather_sequence = np.stack([ts.get('weather_data', np.zeros((24, 10))) 
                                     for ts in sequence])
        threat_sequence = np.stack([ts.get('threat_data', np.zeros(15)) 
                                    for ts in sequence])
        scada_sequence = np.stack([ts.get('scada_data', np.zeros(self.num_nodes * 5)) 
                                   for ts in sequence])
        pmu_sequence = np.stack([ts.get('pmu_data', np.zeros((self.num_nodes, 10, 6))) 
                                 for ts in sequence])
        equipment_sequence = np.stack([ts.get('equipment_health', np.zeros(self.num_nodes * 8)) 
                                       for ts in sequence])
        visual_sequence = np.stack([ts.get('visual_data', np.zeros((3, 128, 128))) 
                                    for ts in sequence])
        thermal_sequence = np.stack([ts.get('thermal_data', np.zeros((1, 64, 64))) 
                                     for ts in sequence])
        sensor_sequence = np.stack([ts.get('sensor_data', np.zeros(self.num_nodes * 12)) 
                                    for ts in sequence])
        
        # Extract edge features from scenario level
        edge_attr = scenario.get('edge_attr', np.zeros((scenario['edge_index'].shape[1], 4)))
        
        return {
            'environmental': {
                'satellite_sequence': satellite_sequence,  # [T, 3, 64, 64]
                'weather_sequence': weather_sequence,      # [T, 24, 10]
                'threat_sequence': threat_sequence         # [T, 15]
            },
            'infrastructure': {
                'scada_sequence': scada_sequence,          # [T, N*5]
                'pmu_sequence': pmu_sequence,              # [T, N, 10, 6]
                'equipment_sequence': equipment_sequence,  # [T, N*8]
                'edge_features': edge_attr                 # [E, 4]
            },
            'robotic': {
                'visual_sequence': visual_sequence,        # [T, 3, 128, 128]
                'thermal_sequence': thermal_sequence,      # [T, 1, 64, 64]
                'sensor_sequence': sensor_sequence         # [T, N*12]
            }
        }
    
    def predict(self, temporal_data: Dict, use_temporal: bool = True) -> Dict:
        """
        Make prediction on grid scenario with full temporal processing.
        
        Args:
            temporal_data: Dictionary containing temporal sequences for all modalities
            use_temporal: If True, use full temporal sequences (recommended)
        
        Returns:
            Dictionary containing comprehensive predictions
        """
        with torch.no_grad():
            def safe_tensor_convert(data, name="data"):
                """Safely convert data to tensor with detailed error reporting."""
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
                    print(f"  Type: {type(data)}")
                    print(f"  Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                    raise
            
            if use_temporal:
                # Use full sequences for temporal modeling
                env = temporal_data['environmental']
                infra = temporal_data['infrastructure']
                robot = temporal_data['robotic']
                
                # Get last timestep for single-timestep inputs
                satellite_data = safe_tensor_convert(env['satellite_sequence'][-1], 'satellite_data')
                weather_sequence = safe_tensor_convert(env['weather_sequence'][-1], 'weather_sequence')
                threat_indicators = safe_tensor_convert(env['threat_sequence'][-1], 'threat_indicators')
                scada_data = safe_tensor_convert(infra['scada_sequence'][-1], 'scada_data')
                pmu_sequence = safe_tensor_convert(infra['pmu_sequence'][-1], 'pmu_sequence')
                equipment_status = safe_tensor_convert(infra['equipment_sequence'][-1], 'equipment_status')
                visual_data = safe_tensor_convert(robot['visual_sequence'][-1], 'visual_data')
                thermal_data = safe_tensor_convert(robot['thermal_sequence'][-1], 'thermal_data')
                sensor_data = safe_tensor_convert(robot['sensor_sequence'][-1], 'sensor_data')
                
                # Create temporal sequence tensor for LSTM processing
                # Stack all timesteps: [T, N, features]
                temporal_sequence = []
                for t in range(len(env['satellite_sequence'])):
                    # Combine all modalities at timestep t
                    # This is a simplified version - in practice, you'd want proper embedding
                    timestep_features = np.concatenate([
                        env['satellite_sequence'][t].flatten(),
                        env['weather_sequence'][t].flatten(),
                        env['threat_sequence'][t].flatten(),
                        infra['scada_sequence'][t].flatten(),
                        infra['pmu_sequence'][t].flatten(),
                        infra['equipment_sequence'][t].flatten(),
                        robot['visual_sequence'][t].flatten(),
                        robot['thermal_sequence'][t].flatten(),
                        robot['sensor_sequence'][t].flatten()
                    ])
                    temporal_sequence.append(timestep_features)
                
                temporal_sequence = safe_tensor_convert(np.array(temporal_sequence), 'temporal_sequence')
            else:
                # Fallback to last timestep only (not recommended)
                env = temporal_data['environmental']
                infra = temporal_data['infrastructure']
                robot = temporal_data['robotic']
                
                satellite_data = safe_tensor_convert(env['satellite_sequence'][-1], 'satellite_data')
                weather_sequence = safe_tensor_convert(env['weather_sequence'][-1], 'weather_sequence')
                threat_indicators = safe_tensor_convert(env['threat_sequence'][-1], 'threat_indicators')
                scada_data = safe_tensor_convert(infra['scada_sequence'][-1], 'scada_data')
                pmu_sequence = safe_tensor_convert(infra['pmu_sequence'][-1], 'pmu_sequence')
                equipment_status = safe_tensor_convert(infra['equipment_sequence'][-1], 'equipment_status')
                visual_data = safe_tensor_convert(robot['visual_sequence'][-1], 'visual_data')
                thermal_data = safe_tensor_convert(robot['thermal_sequence'][-1], 'thermal_data')
                sensor_data = safe_tensor_convert(robot['sensor_sequence'][-1], 'sensor_data')
                temporal_sequence = None
            
            batch = {
                'satellite_data': satellite_data.unsqueeze(0).to(self.device),
                'weather_sequence': weather_sequence.unsqueeze(0).to(self.device),
                'threat_indicators': threat_indicators.unsqueeze(0).to(self.device),
                'scada_data': scada_data.unsqueeze(0).to(self.device),
                'pmu_sequence': pmu_sequence.unsqueeze(0).to(self.device),
                'equipment_status': equipment_status.unsqueeze(0).to(self.device),
                'visual_data': visual_data.unsqueeze(0).to(self.device),
                'thermal_data': thermal_data.unsqueeze(0).to(self.device),
                'sensor_data': sensor_data.unsqueeze(0).to(self.device),
                'edge_index': self.edge_index,
                'edge_attr': safe_tensor_convert(
                    temporal_data['infrastructure']['edge_features'], 
                    'edge_attr'
                ).unsqueeze(0).to(self.device)
            }
            
            if temporal_sequence is not None:
                batch['temporal_sequence'] = temporal_sequence.unsqueeze(0).unsqueeze(1).to(self.device)
            
            outputs = self.model(batch, return_sequence=(temporal_sequence is not None))
            
            # Extract all prediction heads
            node_failure_prob = outputs['failure_probability'].squeeze(0).squeeze(-1).cpu().numpy()
            failure_timing = outputs['failure_timing'].squeeze(0).squeeze(-1).cpu().numpy()
            risk_scores = outputs['risk_scores'].squeeze(0).cpu().numpy()  # [num_nodes, 7]
            
            voltages = outputs['voltages'].squeeze(0).cpu().numpy()
            angles = outputs['angles'].squeeze(0).cpu().numpy()
            line_flows = outputs['line_flows'].squeeze(0).cpu().numpy()
            frequency = outputs['frequency'].squeeze(0).cpu().numpy()
            
            relay_outputs = {
                'time_dial': outputs['relay_outputs']['time_dial'].squeeze(0).cpu().numpy(),
                'pickup_current': outputs['relay_outputs']['pickup_current'].squeeze(0).cpu().numpy(),
                'operating_time': outputs['relay_outputs']['operating_time'].squeeze(0).cpu().numpy(),
                'will_operate': (outputs['relay_outputs']['will_operate'].squeeze(0).cpu().numpy() > 0.5).astype(bool)
            }
            
            # Cascade detection
            if node_failure_prob.size == 1:
                cascade_prob = float(node_failure_prob.item())
            else:
                cascade_prob = float(np.max(node_failure_prob))
            
            cascade_detected = bool(cascade_prob > self.cascade_threshold)
            
            high_risk_nodes_mask = node_failure_prob > self.cascade_threshold
            if cascade_detected and np.any(high_risk_nodes_mask):
                failure_times = failure_timing[high_risk_nodes_mask]
                time_to_cascade_value = float(np.min(failure_times))  # First failure
            else:
                time_to_cascade_value = -1.0
            
            # Identify high-risk nodes
            high_risk_threshold = self.cascade_threshold + 0.05
            high_risk_nodes = np.where(node_failure_prob > high_risk_threshold)[0].tolist()
            
            # Sort nodes by failure probability
            node_risks = [(i, float(node_failure_prob[i])) for i in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            top_risk_nodes = node_risks[:10]
            
            aggregated_risk_scores = np.mean(risk_scores, axis=0)  # [7]
            
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
                    (i, failure_timing[i]) 
                    for i in range(self.num_nodes) 
                    if node_failure_prob[i] > self.cascade_threshold
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
                    'voltages': voltages.tolist(),
                    'angles': angles.tolist(),
                    'line_flows': line_flows.tolist(),
                    'frequency_hz': float(frequency.item()) if frequency.size == 1 else float(frequency[0])
                },
                'relay_operations': relay_outputs,
                'cascade_path': cascade_path,
                'total_nodes_at_risk': len(high_risk_nodes),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_from_file(self, data_path: str, scenario_idx: int = 0, 
                         use_temporal: bool = True) -> Dict:
        """
        Make prediction from data file or batch directory.
        
        Args:
            data_path: Path to data file or directory
            scenario_idx: Index of scenario to predict
            use_temporal: If True, use full temporal sequences
        
        Returns:
            Dictionary containing predictions
        """
        print(f"Loading scenario {scenario_idx} from {data_path}...")
        
        current_idx = 0
        for scenario in self.load_scenarios_streaming(data_path):
            if current_idx == scenario_idx:
                try:
                    temporal_data = self.extract_temporal_sequences(scenario)
                except Exception as e:
                    print(f"Error extracting temporal sequences: {e}")
                    print(f"Scenario keys: {scenario.keys()}")
                    raise
                
                # Make prediction
                prediction = self.predict(temporal_data, use_temporal=use_temporal)
                
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
                        'is_cascade': bool(metadata.get('cascade', False)),
                        'failed_nodes': [int(x) for x in failed_nodes],
                        'time_to_cascade': float(time_to_cascade)
                    }
                
                return prediction
            current_idx += 1
        
        raise ValueError(f"Scenario index {scenario_idx} out of range")
    
    def batch_predict(self, data_path: str, max_scenarios: int = None,
                     use_temporal: bool = True) -> List[Dict]:
        """
        Make predictions on multiple scenarios.
        Memory-efficient streaming approach.
        
        Args:
            data_path: Path to data file or directory
            max_scenarios: Maximum number of scenarios to process
            use_temporal: If True, use full temporal sequences
        
        Returns:
            List of prediction dictionaries
        """
        print(f"Streaming data from {data_path}...")
        
        total_scenarios = self.count_scenarios(data_path)
        num_scenarios = total_scenarios if max_scenarios is None else min(max_scenarios, total_scenarios)
        print(f"Processing {num_scenarios} scenarios (total available: {total_scenarios})...")
        print(f"Temporal processing: {'ENABLED' if use_temporal else 'DISABLED'}")
        
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
                        'is_cascade': bool(metadata.get('cascade', False)),
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
        """
        Evaluate prediction performance with comprehensive metrics.
        
        Args:
            predictions: List of predictions with ground truth
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not predictions or 'ground_truth' not in predictions[0]:
            print("No ground truth available for evaluation")
            return {}
        
        # Calculate metrics
        cascade_correct = 0
        cascade_total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        time_errors = []
        
        for pred in predictions:
            gt = pred['ground_truth']
            
            # Cascade detection
            predicted_cascade = pred['cascade_detected']
            actual_cascade = gt['is_cascade']
            
            if predicted_cascade == actual_cascade:
                cascade_correct += 1
            
            if predicted_cascade and actual_cascade:
                true_positives += 1
                # Time-to-cascade error
                if pred['time_to_cascade_minutes'] > 0 and gt['time_to_cascade'] > 0:
                    time_errors.append(abs(pred['time_to_cascade_minutes'] - gt['time_to_cascade']))
            elif predicted_cascade and not actual_cascade:
                false_positives += 1
            elif not predicted_cascade and not actual_cascade:
                true_negatives += 1
            elif not predicted_cascade and actual_cascade:
                false_negatives += 1
            
            cascade_total += 1
        
        # Calculate metrics
        accuracy = cascade_correct / cascade_total if cascade_total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Time-to-cascade metrics
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
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference (Improved)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data_unified/grid_topology.pkl", 
                       help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data_unified/test_data.pkl", 
                       help="Path to test data file or directory")
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
    parser.add_argument("--cascade_threshold", type=float, default=0.3,
                       help="Cascade detection threshold (default: 0.3)")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Disable temporal sequence processing (not recommended)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CascadePredictor(
        model_path=args.model_path,
        topology_path=args.topology_path,
        device=args.device,
        cascade_threshold=args.cascade_threshold
    )
    
    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - IMPROVED INFERENCE")
    print("=" * 80 + "\n")
    
    use_temporal = not args.no_temporal
    
    if args.batch:
        # Batch prediction
        predictions = predictor.batch_predict(
            args.data_path, 
            args.max_scenarios,
            use_temporal=use_temporal
        )
        
        # Evaluate
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
        # Single prediction
        prediction = predictor.predict_from_file(
            args.data_path, 
            args.scenario_idx,
            use_temporal=use_temporal
        )
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        print(f"Cascade Detected: {prediction['cascade_detected']}")
        print(f"Cascade Probability: {prediction['cascade_probability']:.4f}")
        print(f"Time to Cascade: {prediction['time_to_cascade_minutes']:.2f} minutes")
        print(f"Nodes at Risk: {prediction['total_nodes_at_risk']}")
        
        print("\nTop 10 High-Risk Nodes:")
        for node_info in prediction['top_10_risk_nodes']:
            print(f"  Node {node_info['node_id']}: {node_info['failure_probability']:.4f}")
        
        print("\nSystem State:")
        print(f"  Frequency: {prediction['system_state']['frequency_hz']:.2f} Hz")
        print(f"  Voltage Range: [{np.min(prediction['system_state']['voltages']):.3f}, "
              f"{np.max(prediction['system_state']['voltages']):.3f}] p.u.")
        
        print("\nCascade Propagation Path:")
        for step in prediction['cascade_path'][:5]:  # Show first 5 failures
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
