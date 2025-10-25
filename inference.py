"""
Cascade Failure Prediction Model Inference Script
Load trained model and make predictions on new data.
Updated to work with batch-based data generation.

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

from cascade_prediction_model import CompleteCascadePredictionModel

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
        self.model = CompleteCascadePredictionModel(
            node_features=45,
            edge_features=28,
            hidden_dim=128,
            num_gnn_layers=4,
            num_heads=4,
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
    
    def predict(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> Dict:
        """
        Make prediction on grid scenario.
        
        Args:
            node_features: Node features [T, N, F_node]
            edge_features: Edge features [T, E, F_edge]
        
        Returns:
            Dictionary containing predictions
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if node_features.dim() == 3:
                node_features = node_features.unsqueeze(0)
                edge_features = edge_features.unsqueeze(0)
            
            # Move to device
            node_features = node_features.to(self.device)
            edge_features = edge_features.to(self.device)
            
            batch_size, seq_len, num_nodes, node_feat_dim = node_features.shape
            x_sequence = [node_features[:, t, :, :] for t in range(seq_len)]
            
            graph_properties = {
                'thermal_limits': edge_features[:, :, 1:2].mean(dim=1),
                'susceptance': edge_features[:, :, 1:2].mean(dim=1),
                'line_flows': edge_features[:, :, 2:3].mean(dim=1)
            }
            
            outputs = self.model(
                x_sequence=x_sequence,
                edge_index=self.edge_index,
                edge_attr=edge_features[:, -1, :, :],  # Use last timestep edge features
                graph_properties=graph_properties
            )
            
            node_failure_prob = outputs['failure_probability']  # [B, N, 1]
            cascade_prob, _ = torch.max(node_failure_prob, dim=1)  # [B, 1]
            time_to_cascade = outputs['failure_timing']  # [B, N, 1]
            time_to_cascade_mean = time_to_cascade.mean(dim=1)  # [B, 1]
            
            # Convert to numpy
            cascade_prob = cascade_prob.cpu().numpy()
            node_failure_prob = node_failure_prob.squeeze(-1).cpu().numpy()  # [B, N]
            time_to_cascade_mean = time_to_cascade_mean.cpu().numpy()
            
            cascade_threshold = 0.3
            cascade_detected = bool(cascade_prob[0, 0] > cascade_threshold)
            
            # Identify high-risk nodes
            high_risk_nodes = np.where(node_failure_prob[0] > 0.5)[0].tolist()
            
            # Sort nodes by failure probability
            node_risks = [(i, node_failure_prob[0, i]) for i in range(self.num_nodes)]
            node_risks.sort(key=lambda x: x[1], reverse=True)
            top_risk_nodes = node_risks[:10]
            
            time_to_cascade_value = float(time_to_cascade_mean[0, 0]) if cascade_detected else -1
            
            return {
                'cascade_probability': float(cascade_prob[0, 0]),
                'cascade_detected': cascade_detected,
                'time_to_cascade_minutes': time_to_cascade_value,
                'high_risk_nodes': high_risk_nodes,
                'top_10_risk_nodes': [
                    {'node_id': int(node_id), 'failure_probability': float(prob)}
                    for node_id, prob in top_risk_nodes
                ],
                'total_nodes_at_risk': len(high_risk_nodes),
                'timestamp': datetime.now().isoformat()
            }
    
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
                sequence = scenario['sequence']
                
                # Convert to tensors
                node_features = torch.stack([
                    torch.tensor(timestep['node_features'], dtype=torch.float32)
                    for timestep in sequence
                ])
                edge_features = torch.stack([
                    torch.tensor(timestep['edge_features'], dtype=torch.float32)
                    for timestep in sequence
                ])
                
                # Make prediction
                prediction = self.predict(node_features, edge_features)
                
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    failed_nodes = metadata.get('failed_nodes', [])
                    time_to_cascade = metadata.get('time_to_cascade', -1)
                    
                    # Convert NumPy types to Python types
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.tolist()
                    elif isinstance(failed_nodes, list):
                        failed_nodes = [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in failed_nodes]
                    
                    if isinstance(time_to_cascade, (np.integer, np.floating)):
                        time_to_cascade = float(time_to_cascade)
                    
                    prediction['ground_truth'] = {
                        'is_cascade': bool(metadata.get('cascade', False)),
                        'failed_nodes': failed_nodes,
                        'time_to_cascade': time_to_cascade
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
                sequence = scenario['sequence']
                
                # Convert to tensors
                node_features = torch.stack([
                    torch.tensor(timestep['node_features'], dtype=torch.float32)
                    for timestep in sequence
                ])
                edge_features = torch.stack([
                    torch.tensor(timestep['edge_features'], dtype=torch.float32)
                    for timestep in sequence
                ])
                
                pred = self.predict(node_features, edge_features)
                
                # Add ground truth if available
                if 'metadata' in scenario:
                    metadata = scenario['metadata']
                    failed_nodes = metadata.get('failed_nodes', [])
                    time_to_cascade = metadata.get('time_to_cascade', -1)
                    
                    if isinstance(failed_nodes, np.ndarray):
                        failed_nodes = failed_nodes.tolist()
                    elif isinstance(failed_nodes, list):
                        failed_nodes = [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in failed_nodes]
                    
                    if isinstance(time_to_cascade, (np.integer, np.floating)):
                        time_to_cascade = float(time_to_cascade)
                    
                    pred['ground_truth'] = {
                        'is_cascade': bool(metadata.get('cascade', False)),
                        'failed_nodes': failed_nodes,
                        'time_to_cascade': time_to_cascade
                    }
                
                predictions.append(pred)
                processed += 1
                
            except Exception as e:
                print(f"  Error processing scenario {processed}: {e}")
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
    parser.add_argument("--topology_path", type=str, default="data/grid_topology.pkl", help="Path to topology file")
    parser.add_argument("--data_path", type=str, default="data/test_data.pkl", help="Path to test data file or directory")
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
