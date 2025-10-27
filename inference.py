"""
Cascade Failure Prediction Model Inference Script
Load trained model and make predictions on new data.
Updated to work with unified multi-modal model and batch-based data generation.

Author: Kraftgene AI Inc.
Date: October 2025
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
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
    """Inference engine for cascade prediction."""
    
    def __init__(self, model_path: str, topology_path: str, device: str = "cpu"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            topology_path: Path to grid topology file
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
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
            # Single file
            print(f"Loading data from file: {data_path}")
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        elif data_path.is_dir():
            # Batch directory
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
                # Batch data goes out of scope and gets garbage collected
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
    
    def predict(self, multi_modal_data: Dict) -> Dict:
        """
        Make prediction on grid scenario.
        
        Args:
            multi_modal_data: Dictionary containing all multi-modal inputs
        
        Returns:
            Dictionary containing predictions
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
            
            batch = {
                'satellite_data': safe_tensor_convert(multi_modal_data['environmental']['satellite_imagery'], 'satellite_data').unsqueeze(0).to(self.device),
                'weather_sequence': safe_tensor_convert(multi_modal_data['environmental']['weather_sequence'], 'weather_sequence').unsqueeze(0).to(self.device),
                'threat_indicators': safe_tensor_convert(multi_modal_data['environmental']['threat_indicators'], 'threat_indicators').unsqueeze(0).to(self.device),
                'scada_data': safe_tensor_convert(multi_modal_data['infrastructure']['scada_measurements'], 'scada_data').unsqueeze(0).to(self.device),
                'pmu_sequence': safe_tensor_convert(multi_modal_data['infrastructure']['pmu_data'], 'pmu_sequence').unsqueeze(0).to(self.device),
                'equipment_status': safe_tensor_convert(multi_modal_data['infrastructure']['equipment_condition'], 'equipment_status').unsqueeze(0).to(self.device),
                'visual_data': safe_tensor_convert(multi_modal_data['robotic']['visual_inspection'], 'visual_data').unsqueeze(0).to(self.device),
                'thermal_data': safe_tensor_convert(multi_modal_data['robotic']['thermal_imaging'], 'thermal_data').unsqueeze(0).to(self.device),
                'sensor_data': safe_tensor_convert(multi_modal_data['robotic']['sensor_readings'], 'sensor_data').unsqueeze(0).to(self.device),
                'edge_index': self.edge_index,
                'edge_attr': safe_tensor_convert(multi_modal_data['infrastructure']['edge_features'], 'edge_attr').unsqueeze(0).to(self.device)
            }
            
            outputs = self.model(batch)
            
            node_failure_prob = outputs['failure_probability'].squeeze(0).squeeze(-1).cpu().numpy()
            
            # Convert to float properly - handle both scalar and array cases
            if node_failure_prob.size == 1:
                cascade_prob = float(node_failure_prob.item())
            else:
                cascade_prob = float(np.max(node_failure_prob))
            
            failure_timing = outputs['failure_timing'].squeeze(0).squeeze(-1).cpu().numpy()
            risk_scores = outputs['risk_scores'].squeeze(0).cpu().numpy()  # Shape: [num_nodes, 7]
            
            cascade_threshold = 0.3
            cascade_detected = bool(cascade_prob > cascade_threshold)
            
            # Use cascade_threshold + 0.05 to identify nodes slightly above the cascade detection level
            high_risk_threshold = cascade_threshold + 0.05
            high_risk_nodes = np.where(node_failure_prob > high_risk_threshold)[0].tolist()
            
            # Sort nodes by failure probability
            node_risks = [(i, float(node_failure_prob[i])) for i in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            top_risk_nodes = node_risks[:10]
            
            # Handle failure timing conversion properly
            if failure_timing.size == 1:
                time_to_cascade_value = float(failure_timing.item()) if cascade_detected else -1.0
            else:
                time_to_cascade_value = float(np.mean(failure_timing)) if cascade_detected else -1.0
            
            # risk_scores has shape [num_nodes, 7], aggregate across nodes first
            aggregated_risk_scores = np.mean(risk_scores, axis=0)  # Shape: [7]
            
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
                    'threat_severity': float(aggregated_risk_scores[0]),
                    'vulnerability': float(aggregated_risk_scores[1]),
                    'operational_impact': float(aggregated_risk_scores[2]),
                    'cascade_probability': float(aggregated_risk_scores[3]),
                    'response_complexity': float(aggregated_risk_scores[4]),
                    'public_safety': float(aggregated_risk_scores[5]),
                    'urgency': float(aggregated_risk_scores[6])
                },
                'total_nodes_at_risk': len(high_risk_nodes),
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_multi_modal_data(self, scenario: Dict) -> Dict:
        """
        Extract multi_modal_data from scenario, handling both old and new formats.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            multi_modal_data dictionary
        """
        # Check if multi_modal_data already exists (new format)
        if 'multi_modal_data' in scenario:
            return scenario['multi_modal_data']
        
        # Extract from sequence (old format)
        if 'sequence' not in scenario:
            raise ValueError("Scenario missing both 'multi_modal_data' and 'sequence' keys")
        
        sequence = scenario['sequence']
        if len(sequence) == 0:
            raise ValueError("Scenario sequence is empty")
        
        # Get last timestep
        last_timestep = sequence[-1]
        
        # Extract edge features from scenario level
        edge_attr = scenario.get('edge_attr', np.zeros((scenario['edge_index'].shape[1], 4)))
        
        # Build multi_modal_data structure
        multi_modal_data = {
            'environmental': {
                'satellite_imagery': last_timestep.get('satellite_data', np.zeros((3, 64, 64))),
                'weather_sequence': last_timestep.get('weather_data', np.zeros((24, 10))),
                'threat_indicators': last_timestep.get('threat_data', np.zeros(15))
            },
            'infrastructure': {
                'scada_measurements': last_timestep.get('scada_data', np.zeros(self.num_nodes * 5)),
                'pmu_data': last_timestep.get('pmu_data', np.zeros((self.num_nodes, 10, 6))),
                'equipment_condition': last_timestep.get('equipment_health', np.zeros(self.num_nodes * 8)),
                'edge_features': edge_attr
            },
            'robotic': {
                'visual_inspection': last_timestep.get('visual_data', np.zeros((3, 128, 128))),
                'thermal_imaging': last_timestep.get('thermal_data', np.zeros((1, 64, 64))),
                'sensor_readings': last_timestep.get('sensor_data', np.zeros(self.num_nodes * 12))
            }
        }
        
        return multi_modal_data
    
    def predict_from_file(self, data_path: str, scenario_idx: int = 0) -> Dict:
        """
        Make prediction from data file or batch directory.
        
        Args:
            data_path: Path to data file or directory
            scenario_idx: Index of scenario to predict
        
        Returns:
            Dictionary containing predictions
        """
        print(f"Loading scenario {scenario_idx} from {data_path}...")
        
        current_idx = 0
        for scenario in self.load_scenarios_streaming(data_path):
            if current_idx == scenario_idx:
                try:
                    multi_modal_data = self.extract_multi_modal_data(scenario)
                except Exception as e:
                    print(f"Error extracting multi_modal_data: {e}")
                    print(f"Scenario keys: {scenario.keys()}")
                    raise
                
                # Make prediction
                prediction = self.predict(multi_modal_data)
                
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    # Handle failed_nodes
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    # Handle time_to_cascade
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
    
    def batch_predict(self, data_path: str, max_scenarios: int = None) -> List[Dict]:
        """
        Make predictions on multiple scenarios.
        Memory-efficient streaming approach.
        
        Args:
            data_path: Path to data file or directory
            max_scenarios: Maximum number of scenarios to process
        
        Returns:
            List of prediction dictionaries
        """
        print(f"Streaming data from {data_path}...")
        
        total_scenarios = self.count_scenarios(data_path)
        num_scenarios = total_scenarios if max_scenarios is None else min(max_scenarios, total_scenarios)
        print(f"Processing {num_scenarios} scenarios (total available: {total_scenarios})...")
        
        predictions = []
        processed = 0
        
        for scenario in self.load_scenarios_streaming(data_path):
            if processed >= num_scenarios:
                break
            
            if (processed + 1) % 100 == 0:
                print(f"  Processed {processed + 1}/{num_scenarios}")
            
            try:
                multi_modal_data = self.extract_multi_modal_data(scenario)
                
                pred = self.predict(multi_modal_data)
                
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    
                    # Handle failed_nodes
                    failed_nodes = metadata.get('failed_nodes', [])
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.flatten().tolist()
                    elif not isinstance(failed_nodes, list):
                        failed_nodes = [failed_nodes] if failed_nodes is not None else []
                    
                    # Handle time_to_cascade
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
        Evaluate prediction performance.
        
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
        
        for pred in predictions:
            gt = pred['ground_truth']
            
            # Cascade detection
            predicted_cascade = pred['cascade_detected']
            actual_cascade = gt['is_cascade']
            
            if predicted_cascade == actual_cascade:
                cascade_correct += 1
            
            if predicted_cascade and actual_cascade:
                true_positives += 1
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
            'total_scenarios': cascade_total
        }
        
        return metrics


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Cascade Prediction Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--topology_path", type=str, default="data_unified/grid_topology.pkl", help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data_unified/test_data.pkl", help="Path to test data file or directory")
    parser.add_argument("--scenario_idx", type=int, default=0, help="Scenario index for single prediction")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Max scenarios for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file for predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CascadePredictor(
        model_path=args.model_path,
        topology_path=args.topology_path,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - INFERENCE")
    print("=" * 80 + "\n")
    
    if args.batch:
        # Batch prediction
        predictions = predictor.batch_predict(args.data_path, args.max_scenarios)
        
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
        prediction = predictor.predict_from_file(args.data_path, args.scenario_idx)
        
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
