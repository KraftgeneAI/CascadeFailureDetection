"""
Cascade Failure Prediction Inference Script
============================================================
ALIGNED TO TRAINING: Sliding Window / Full Sequence Analysis
- Replicates Validation Methodology (Teacher Forcing).
- FIXED: Dimension handling for (Batch, Nodes, 1) output.
- FIXED: Sequence generation based on Risk Score (Ranking Loss logic).
- FIXED: 'base_freq' argument name in CascadePredictor instantiation.
- IMPROVED: Human-readable Risk Assessment output.
============================================================

Author: Kraftgene AI Inc.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import argparse
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
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
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

# ============================================================================
# INFERENCE DATASET
# ============================================================================
class ScenarioInferenceDataset(Dataset):
    def __init__(self, scenario: Dict, window_size: int, base_mva: float = 100.0, base_frequency: float = 60.0):
        self.window_size = window_size
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        self.sequence_original = scenario.get('sequence', [])
        self.edge_index = scenario['edge_index']
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.from_numpy(self.edge_index).long()
        self.total_steps = len(self.sequence_original)
        self.preprocessed_sequence = self._preprocess_full_sequence()
        
    def _normalize_power(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / self.base_mva
    def _normalize_frequency(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / self.base_frequency

    def _preprocess_full_sequence(self):
        if self.total_steps == 0: return {}
        data_dicts = {'scada_data': [], 'pmu_sequence': [], 'satellite_data': [], 'weather_sequence': [], 
                      'threat_indicators': [], 'equipment_status': [], 'visual_data': [], 'thermal_data': [], 'sensor_data': [], 'edge_mask': []}
        
        num_edges = self.edge_index.shape[1]
        
        for i, ts in enumerate(self.sequence_original):
            s = torch.tensor(ts.get('scada_data'), dtype=torch.float32)
            if s.shape[1] >= 13: s = s[:, :13]
            if s.shape[1] >= 6:
                s[:, 2] = self._normalize_power(s[:, 2])
                s[:, 3] = self._normalize_power(s[:, 3])
                s[:, 4] = self._normalize_power(s[:, 4])
                s[:, 5] = self._normalize_power(s[:, 5])
            data_dicts['scada_data'].append(s)
            
            p = torch.tensor(ts.get('pmu_sequence'), dtype=torch.float32)
            if p.shape[1] >= 6: p[:, 5] = self._normalize_frequency(p[:, 5])
            data_dicts['pmu_sequence'].append(p)
            
            data_dicts['satellite_data'].append(torch.tensor(ts.get('satellite_data'), dtype=torch.float32))
            data_dicts['weather_sequence'].append(torch.tensor(ts.get('weather_sequence'), dtype=torch.float32).reshape(s.shape[0], -1))
            data_dicts['threat_indicators'].append(torch.tensor(ts.get('threat_indicators'), dtype=torch.float32))
            data_dicts['equipment_status'].append(torch.tensor(ts.get('equipment_status'), dtype=torch.float32))
            data_dicts['visual_data'].append(torch.tensor(ts.get('visual_data'), dtype=torch.float32))
            data_dicts['thermal_data'].append(torch.tensor(ts.get('thermal_data'), dtype=torch.float32))
            data_dicts['sensor_data'].append(torch.tensor(ts.get('sensor_data'), dtype=torch.float32))
            
            # Mask Logic (Teacher Forcing)
            mask = torch.ones(num_edges, dtype=torch.float32)
            if i > 0:
                prev = self.sequence_original[i-1]
                labels = prev.get('node_labels')
                if labels is not None:
                    failed = np.where(labels > 0.5)[0]
                    if len(failed) > 0:
                        src, dst = self.edge_index.numpy()
                        edge_failed = np.isin(src, failed) | np.isin(dst, failed)
                        mask[edge_failed] = 0.0
            data_dicts['edge_mask'].append(mask)

        for k, v in data_dicts.items(): data_dicts[k] = torch.stack(v)
        return data_dicts

    def __len__(self): return self.total_steps
    def __getitem__(self, idx):
        end_idx = idx + 1
        start_idx = max(0, end_idx - self.window_size)
        item = {k: v[start_idx:end_idx] for k, v in self.preprocessed_sequence.items()}
        
        last_step = self.sequence_original[idx]
        edge_attr = torch.tensor(last_step.get('edge_attr'), dtype=torch.float32)
        if edge_attr.shape[1] >= 2: edge_attr[:, 1] = self._normalize_power(edge_attr[:, 1])
        item['edge_attr'] = edge_attr
        item['edge_index'] = self.edge_index
        item['sequence_length'] = end_idx - start_idx
        item['temporal_sequence'] = item['scada_data']
        item['graph_properties'] = {}
        return item

# ============================================================================
# PREDICTOR
# ============================================================================
class CascadePredictor:
    def __init__(self, model_path, topology_path, device, base_mva, base_freq):
        self.device = device
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index']
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = UnifiedCascadePredictionModel(
            embedding_dim=128, hidden_dim=128, num_gnn_layers=3, heads=4, dropout=0.1
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.cascade_threshold = checkpoint.get('cascade_threshold', 0.1)
        self.node_threshold = checkpoint.get('node_threshold', 0.35)
        print(f"✓ Model loaded. Thresholds: Cascade={self.cascade_threshold:.2f}, Node={self.node_threshold:.2f}")

    def predict_scenario(self, data_path, scenario_idx, window_size=30, batch_size=32):
        files = sorted(glob.glob(f"{data_path}/scenario_*.pkl"))
        if not files: files = sorted(glob.glob(f"{data_path}/scenarios_batch_*.pkl"))
        target_file = files[scenario_idx]
        print(f"Loading: {target_file}")
        
        with open(target_file, 'rb') as f:
            data = pickle.load(f)
        scenario = data[0] if isinstance(data, list) else data
        scenario['edge_index'] = self.edge_index 
        
        dataset = ScenarioInferenceDataset(scenario, window_size)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_cascade_batch, shuffle=False)
        
        print(f"Running Inference on {len(dataset)} steps...")
        
        max_probs = {}     
        first_time = {}    
        final_risk_scores = None
        final_sys_state = None

        current_t = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                outputs = self.model(batch_dev, return_sequence=False)
                probs = outputs['failure_probability'].squeeze(-1).cpu().numpy()
                if len(probs.shape) == 1: probs = probs.reshape(1, -1)
                
                for b in range(probs.shape[0]):
                    t = current_t + 1
                    step_probs = probs[b]
                    
                    for n, p in enumerate(step_probs):
                        p = float(p)
                        if n not in max_probs or p > max_probs[n]:
                            max_probs[n] = p
                            first_time[n] = t 
                            
                    current_t += 1
                
                if i == len(loader) - 1:
                    final_risk_scores = outputs['risk_scores'][-1].mean(dim=0).cpu().numpy().tolist()
                    final_sys_state = {
                        'frequency': float(outputs['frequency'].mean().item()),
                        'voltages': outputs['voltages'][-1].reshape(-1).cpu().numpy().tolist()
                    }
        
        risky_nodes = [n for n, p in max_probs.items() if p > self.node_threshold]
        
        ranked_nodes = []
        for n in risky_nodes:
            ranked_nodes.append({
                'node_id': n,
                'score': max_probs[n],
                'peak_time': first_time[n]
            })
            
        ranked_nodes.sort(key=lambda x: -x['score'])
        
        cascade_path = []
        if ranked_nodes:
            current_rank = 1
            last_score = ranked_nodes[0]['score']
            for i, node in enumerate(ranked_nodes):
                if (last_score - node['score']) > 0.002: 
                    current_rank += 1
                    last_score = node['score']
                cascade_path.append({
                    'order': current_rank,
                    'node_id': node['node_id'],
                    'ranking_score': node['score']
                })

        meta = scenario.get('metadata', {})
        gt_path = []
        if 'failed_nodes' in meta and 'failure_times' in meta:
            gt_path = sorted([
                {'node_id': int(n), 'time_minutes': float(t)} 
                for n,t in zip(meta['failed_nodes'], meta['failure_times'])
            ], key=lambda x: x['time_minutes'])

        return {
            'inference_time': 0.0,
            'cascade_detected': bool(ranked_nodes),
            'cascade_probability': ranked_nodes[0]['score'] if ranked_nodes else 0.0,
            'ground_truth': {'is_cascade': meta.get('is_cascade'), 'failed_nodes': meta.get('failed_nodes', []), 'cascade_path': gt_path, 'ground_truth_risk': meta.get('ground_truth_risk', [])},
            'high_risk_nodes': risky_nodes,
            'risk_assessment': final_risk_scores if final_risk_scores else [0.0]*7,
            'top_nodes': ranked_nodes,
            'cascade_path': cascade_path,
            'system_state': final_sys_state if final_sys_state else {'frequency': 0.0, 'voltages': []}
        }

def print_report(res: Dict, cascade_thresh: float, node_thresh: float):
    print("\n" + "="*80)
    print("PREDICTION RESULTS (Scenario Analysis)")
    print("="*80)
    print(f"Inference Time: {res['inference_time']:.4f} seconds\n")
    
    gt = res['ground_truth']
    pred = res['cascade_detected']
    actual = gt['is_cascade']
    
    print("--- 1. Overall Verdict ---")
    if pred and actual: print("✅ Correctly detected a cascade.")
    elif not pred and not actual: print("✅ Correctly identified a normal scenario.")
    elif pred and not actual: print("⚠️ FALSE POSITIVE (False Alarm)")
    elif not pred and actual: print("❌ FALSE NEGATIVE (Missed Cascade)")
    
    print(f"Prediction: {pred} (Prob: {res['cascade_probability']:.3f} / Thresh: {cascade_thresh:.3f})")
    print(f"Ground Truth: {actual}")

    if actual or pred:
        print("\n--- 2. Node-Level Analysis ---")
        pred_nodes = set(res['high_risk_nodes'])
        actual_nodes = set(gt.get('failed_nodes', []))
        tp = len(pred_nodes.intersection(actual_nodes))
        fp = len(pred_nodes - actual_nodes)
        fn = len(actual_nodes - pred_nodes)
        
        print(f"Predicted Nodes at Risk: {len(pred_nodes)} (Thresh: {node_thresh:.3f})")
        print(f"Actual Failed Nodes:     {len(actual_nodes)}")
        print(f"  - Correctly Identified (TP): {tp}")
        print(f"  - Missed Nodes (FN):         {fn}")
        print(f"  - False Alarms (FP):         {fp}")

    if actual or pred:
        print("\n--- 3. Timing Analysis ---")
        print(f"  Metric                      | Predicted       | Ground Truth")
        print(f"  ----------------------------|-----------------|-----------------")
        
        scores = [n['score'] for n in res['top_nodes']]
        min_s, max_s = (min(scores), max(scores)) if scores else (0.0, 0.0)
        score_spread = max_s - min_s
        
        act_path = gt.get('cascade_path', [])
        min_t, max_t = 0.0, 0.0
        if act_path:
            times = [x['time_minutes'] for x in act_path]
            min_t, max_t = min(times), max(times)
        
        print(f"  Prediction Mode             | Relative Rank   | Absolute Time")
        print(f"  Range (Start -> End)        | {max_s:.3f} -> {min_s:.3f} | {min_t:.2f} -> {max_t:.2f} min")
        print(f"  Sequence Spread             | {score_spread:.3f} (Score)  | {max_t - min_t:.2f} minutes")

    print("\n--- 4. Critical Information ---")
    print(f"System Frequency: {res['system_state']['frequency']:.2f} Hz")
    v_all = res['system_state']['voltages']
    if v_all:
        print(f"Voltage Range:    [{min(v_all):.3f}, {max(v_all):.3f}] p.u.")
    
    if pred and res['top_nodes']:
        print("\nTop 5 High-Risk Nodes:")
        actual_nodes = set(gt.get('failed_nodes', []))
        for node in res['top_nodes'][:5]:
            nid = node['node_id']
            status = "✓ (Actual)" if nid in actual_nodes else "✗ (Not Actual)"
            print(f"  - Node {nid:<3}: {node['score']:.4f} {status}")

    r = res['risk_assessment']
    def get_lvl(s): return "(Critical)" if s>0.8 else "(Severe)" if s>0.6 else "(Medium)" if s>0.3 else "(Low)"
    
    print("\nAggregated Risk Assessment (7-Dimensions):")
    labels = ["Threat", "Vulnerability", "Impact", "Cascade Prob", "Response", "Safety", "Urgency"]
    if len(r) >= 7:
        line1 = [f"{l}: {s:.3f} {get_lvl(s):<10}" for l,s in zip(labels[:3], r[:3])]
        line2 = [f"{l}: {s:.3f} {get_lvl(s):<10}" for l,s in zip(labels[3:6], r[3:6])]
        print("  - " + " | ".join(line1))
        print("  - " + " | ".join(line2))
        print(f"  - {labels[6]}: {r[6]:.3f} {get_lvl(r[6]):<10}")

    gt_risk = gt.get('ground_truth_risk', [])
    if gt_risk is not None and len(gt_risk) >= 7:
        print("\n  Ground Truth Risk Assessment:")
        g_line1 = [f"{l}: {s:.3f} {get_lvl(s):<10}" for l,s in zip(labels[:3], gt_risk[:3])]
        g_line2 = [f"{l}: {s:.3f} {get_lvl(s):<10}" for l,s in zip(labels[3:6], gt_risk[3:6])]
        print("  - " + " | ".join(g_line1))
        print("  - " + " | ".join(g_line2))
        print(f"  - {labels[6]}: {gt_risk[6]:.3f} {get_lvl(gt_risk[6]):<10}")
        
    print("\n--- Risk Definitions ---")
    print("  Critical (0.8+): Immediate Failure | Severe (0.6+): High Danger | Medium (0.3+): Caution")
    print("  Dimensions: Threat (Stress), Vulnerability (Weakness), Impact (Consequence),")
    print("              Cascade Prob (Propagation), Urgency (Time Sensitivity).")

    print("\n--- 5. Cascade Path Analysis (Sequence Order) ---")
    pred_path = res['cascade_path']
    actual_path = gt.get('cascade_path', [])
    
    print(f"  {'Seq #':<6} | {'Predicted Node':<15} | {'Score':<8} | {'Actual Seq #':<15} | {'Actual Node':<15} | {'Delta T (min)':<15}")
    print(f"  {'-'*6} | {'-'*15} | {'-'*8} | {'-'*15} | {'-'*15} | {'-'*15}")
    
    max_rows = max(len(pred_path), len(actual_path))
    curr_act_seq = 0
    last_act_time = -999.0
    
    for i in range(max_rows):
        p_seq, p_node, p_score = "", "", ""
        if i < len(pred_path):
            p_item = pred_path[i]
            p_seq = str(p_item['order'])
            p_node = f"Node {p_item['node_id']}"
            p_score = f"{p_item['ranking_score']:.3f}"
        
        a_seq, a_node, a_time = "", "", ""
        if i < len(actual_path):
            a_item = actual_path[i]
            t = a_item['time_minutes']
            if t > last_act_time + 0.1:
                curr_act_seq += 1
                last_act_time = t
            a_seq = str(curr_act_seq)
            a_node = f"Node {a_item['node_id']}"
            a_time = f"{t:.2f}"
        
        print(f"  {p_seq:<6} | {p_node:<15} | {p_score:<8} | {a_seq:<15} | {a_node:<15} | {a_time:<15}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="data/test")
    parser.add_argument("--scenario_idx", type=int, default=0)
    parser.add_argument("--topology_path", default="data/grid_topology.pkl")
    parser.add_argument("--output", default="prediction.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {dev}")
    
    predictor = CascadePredictor(args.model_path, args.topology_path, device=dev, base_mva=100.0, base_freq=60.0)
    
    print("\n" + "="*80)
    print("CASCADE FAILURE PREDICTION - PHYSICS-INFORMED INFERENCE")
    print("="*80 + "\n")
    
    try:
        start_time = time.time()
        res = predictor.predict_scenario(args.data_path, args.scenario_idx, args.window_size, args.batch_size)
        res['inference_time'] = time.time() - start_time
        
        print_report(res, predictor.cascade_threshold, predictor.node_threshold)
        
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f"Full prediction details saved to {args.output}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()