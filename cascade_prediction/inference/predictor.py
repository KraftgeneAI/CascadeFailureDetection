"""
Predictor Module
===============
Provides the CascadePredictor class for inference on cascade scenarios.
"""

import torch
from torch.utils.data import DataLoader
import pickle
import glob
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from ..models import UnifiedCascadePredictionModel
from ..data import collate_cascade_batch
from .dataset import ScenarioInferenceDataset


class CascadePredictor:
    """
    Predictor for cascade failure scenarios.
    
    Loads a trained model and performs inference on test scenarios,
    predicting failure probabilities, cascade paths, and risk assessments.
    
    Args:
        model_path: Path to trained model checkpoint
        topology_path: Path to grid topology file
        device: Device to run inference on (cuda/cpu)
        base_mva: Base MVA for power normalization
        base_freq: Base frequency for normalization
    """
    
    def __init__(
        self,
        model_path: str,
        topology_path: str,
        device: torch.device,
        base_mva: float = 100.0,
        base_freq: float = 60.0
    ):
        self.device = device
        self.base_mva = base_mva
        self.base_freq = base_freq
        
        # Load topology
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index']
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.model = UnifiedCascadePredictionModel(
            embedding_dim=128,
            hidden_dim=128,
            num_gnn_layers=3,
            heads=4,
            dropout=0.1
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load thresholds
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.1)
        self.node_threshold = checkpoint.get('node_threshold', 0.35)
        
        print(f"✓ Model loaded. Thresholds: Cascade={self.cascade_threshold:.2f}, "
              f"Node={self.node_threshold:.2f}")
    
    def predict_scenario(
        self,
        data_path: str,
        scenario_idx: int,
        window_size: int = 30,
        batch_size: int = 32
    ) -> Dict:
        """
        Predict cascade failure for a specific scenario.
        
        Args:
            data_path: Path to directory containing scenario files
            scenario_idx: Index of scenario to predict
            window_size: Sliding window size for inference
            batch_size: Batch size for inference
            
        Returns:
            Dictionary containing prediction results
        """
        # Load scenario
        scenario = self._load_scenario(data_path, scenario_idx)
        
        # Create dataset and dataloader
        dataset = ScenarioInferenceDataset(
            scenario, 
            window_size,
            base_mva=self.base_mva,
            base_frequency=self.base_freq
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_cascade_batch,
            shuffle=False
        )
        
        print(f"Running inference on {len(dataset)} steps...")
        
        # Run inference
        predictions = self._run_inference(loader)
        
        # Analyze results
        results = self._analyze_predictions(predictions, scenario)
        
        return results
    
    def _load_scenario(self, data_path: str, scenario_idx: int) -> Dict:
        """Load a scenario from disk."""
        files = sorted(glob.glob(f"{data_path}/scenario_*.pkl"))
        if not files:
            files = sorted(glob.glob(f"{data_path}/scenarios_batch_*.pkl"))
        
        if not files:
            raise FileNotFoundError(f"No scenario files found in {data_path}")
        
        target_file = files[scenario_idx]
        print(f"Loading: {target_file}")
        
        with open(target_file, 'rb') as f:
            data = pickle.load(f)
        
        scenario = data[0] if isinstance(data, list) else data
        scenario['edge_index'] = self.edge_index
        
        return scenario
    
    def _run_inference(self, loader: DataLoader) -> Dict:
        """Run inference on all timesteps."""
        max_probs = {}
        first_time = {}
        final_risk_scores = None
        final_sys_state = None
        
        current_t = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                # Move batch to device
                batch_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(batch_dev)
                
                # Extract failure probabilities
                probs = outputs['failure_probability'].squeeze(-1).cpu().numpy()
                if len(probs.shape) == 1:
                    probs = probs.reshape(1, -1)
                
                # Track maximum probabilities and first occurrence times
                for b in range(probs.shape[0]):
                    t = current_t + 1
                    step_probs = probs[b]
                    
                    for n, p in enumerate(step_probs):
                        p = float(p)
                        if n not in max_probs or p > max_probs[n]:
                            max_probs[n] = p
                            first_time[n] = t
                    
                    current_t += 1
                
                # Save final outputs
                if i == len(loader) - 1:
                    final_risk_scores = outputs['risk_scores'][-1].mean(dim=0).cpu().numpy().tolist()
                    final_sys_state = {
                        'frequency': float(outputs['frequency'].mean().item()),
                        'voltages': outputs['voltages'][-1].reshape(-1).cpu().numpy().tolist()
                    }
        
        return {
            'max_probs': max_probs,
            'first_time': first_time,
            'risk_scores': final_risk_scores,
            'system_state': final_sys_state
        }
    
    def _analyze_predictions(self, predictions: Dict, scenario: Dict) -> Dict:
        """Analyze predictions and format results."""
        max_probs = predictions['max_probs']
        first_time = predictions['first_time']
        
        # Identify risky nodes
        risky_nodes = [
            n for n, p in max_probs.items() 
            if p > self.node_threshold
        ]
        
        # Rank nodes by score
        ranked_nodes = []
        for n in risky_nodes:
            ranked_nodes.append({
                'node_id': n,
                'score': max_probs[n],
                'peak_time': first_time[n]
            })
        
        ranked_nodes.sort(key=lambda x: -x['score'])
        
        # Generate cascade path with ranking
        cascade_path = self._generate_cascade_path(ranked_nodes)
        
        # Extract ground truth
        ground_truth = self._extract_ground_truth(scenario)
        
        # Format results
        results = {
            'inference_time': 0.0,
            'cascade_detected': bool(ranked_nodes),
            'cascade_probability': ranked_nodes[0]['score'] if ranked_nodes else 0.0,
            'ground_truth': ground_truth,
            'high_risk_nodes': risky_nodes,
            'risk_assessment': predictions['risk_scores'] if predictions['risk_scores'] else [0.0] * 7,
            'top_nodes': ranked_nodes,
            'cascade_path': cascade_path,
            'system_state': predictions['system_state'] if predictions['system_state'] else {
                'frequency': 0.0,
                'voltages': []
            }
        }
        
        return results
    
    def _generate_cascade_path(self, ranked_nodes: List[Dict]) -> List[Dict]:
        """Generate cascade path with ranking order."""
        cascade_path = []
        
        if not ranked_nodes:
            return cascade_path
        
        current_rank = 1
        last_score = ranked_nodes[0]['score']
        
        for node in ranked_nodes:
            # Increment rank if score difference is significant
            if (last_score - node['score']) > 0.002:
                current_rank += 1
                last_score = node['score']
            
            cascade_path.append({
                'order': current_rank,
                'node_id': node['node_id'],
                'ranking_score': node['score']
            })
        
        return cascade_path
    
    def _extract_ground_truth(self, scenario: Dict) -> Dict:
        """Extract ground truth information from scenario."""
        meta = scenario.get('metadata', {})
        
        gt_path = []
        if 'failed_nodes' in meta and 'failure_times' in meta:
            gt_path = sorted([
                {
                    'node_id': int(n),
                    'time_minutes': float(t)
                }
                for n, t in zip(meta['failed_nodes'], meta['failure_times'])
            ], key=lambda x: x['time_minutes'])
        
        return {
            'is_cascade': meta.get('is_cascade'),
            'failed_nodes': meta.get('failed_nodes', []),
            'cascade_path': gt_path,
            'ground_truth_risk': meta.get('ground_truth_risk', [])
        }
