"""
Cascade Failure Prediction Inference Script
============================================================
UPDATED: Using refactored cascade_prediction module classes
- Uses cascade_prediction.models.unified_model.UnifiedCascadePredictionModel
- Uses cascade_prediction.data.collation.collate_cascade_batch
- Uses cascade_prediction.data.preprocessing for normalization
- Replicates Validation Methodology (Teacher Forcing)
- FIXED: Dimension handling for (Batch, Nodes, 1) output
- FIXED: Sequence generation based on Risk Score (Ranking Loss logic)
- IMPROVED: Human-readable Risk Assessment output
============================================================

Author: Kraftgene AI Inc.
"""

import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import argparse
import time
import sys

# Import refactored classes
try:
    from cascade_prediction.models.unified_model import UnifiedCascadePredictionModel
    from cascade_prediction.data.collation import collate_cascade_batch
    from cascade_prediction.data.dataset import CascadeDataset
    from cascade_prediction.data.generator.config import Settings
except ImportError as e:
    print(f"Error: Could not import from cascade_prediction module: {e}")
    print("Make sure the cascade_prediction package is in your Python path.")
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

# ============================================================================
# PREDICTOR
# ============================================================================
class CascadePredictor:
    def __init__(self, model_path, topology_path=None, device=None, base_mva=Settings.Dataset.BASE_MVA, base_freq=Settings.Dataset.BASE_FREQUENCY):
        self.device = device or torch.device('cpu')
        self.base_mva = base_mva
        self.base_freq = base_freq

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = UnifiedCascadePredictionModel(
            embedding_dim=Settings.Model.EMBEDDING_DIM,
            hidden_dim=Settings.Model.HIDDEN_DIM,
            num_gnn_layers=Settings.Model.NUM_GNN_LAYERS,
            heads=Settings.Model.HEADS,
            dropout=Settings.Model.GAT_DROPOUT
        )
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(device)
        self.model.eval()

        self.cascade_threshold = checkpoint.get('cascade_threshold', Settings.Training.CASCADE_THRESHOLD)
        self.node_threshold = checkpoint.get('node_threshold', Settings.Training.NODE_THRESHOLD)
        print(f"✓ Model loaded. Thresholds: Cascade={self.cascade_threshold:.2f}, Node={self.node_threshold:.2f}")

    def predict_scenario(self, data_path, scenario_idx,
                         window_size=Settings.Simulation.DEFAULT_SEQUENCE_LENGTH,
                         batch_size=Settings.Training.BATCH_SIZE):
        # Use CascadeDataset to load and preprocess — identical to training pipeline
        dataset = CascadeDataset(
            data_path,
            mode='full_sequence',
            base_mva=self.base_mva,
            base_frequency=self.base_freq,
        )

        print(f"Running inference on scenario {scenario_idx} / {len(dataset) - 1}...")
        item = dataset[scenario_idx]
        if not item:
            raise ValueError(f"Scenario {scenario_idx} could not be loaded from {data_path}")

        # Collate single item into a batch of 1
        batch = collate_cascade_batch([item])
        batch_dev = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = self.model(batch_dev)

        # Node failure probabilities [1, N] → [N]
        probs = outputs['failure_probability'].squeeze(-1).squeeze(0).cpu().numpy()  # [N]

        # ── IMPROVED TIMING DECODE (v2) ──────────────────────────────────────
        # The improved TimingHead now outputs Sigmoid-normalised values in (0,1)
        # where 0 = beginning of sequence and 1 = end of sequence.
        #
        # To recover absolute minutes:
        #   1. Multiply by DEFAULT_SEQUENCE_LENGTH (30 timesteps) → timestep index
        #   2. Multiply by DT_MINUTES (2.0 min/timestep) → minutes
        #
        # Previously the head output raw linear values with no activation, which
        # were interpreted directly as minutes — producing predictions of ~2-8 min
        # while actual failure times were ~44-58 min (MAE ≈ 47 min).
        # ─────────────────────────────────────────────────────────────────────
        DT_MINUTES = 2.0   # minutes per simulation timestep (ThermalConfig.DT_MINUTES)
        pred_timing_normed = outputs['cascade_timing'].squeeze(-1).squeeze(0).cpu().numpy()  # [N] in (0,1)
        pred_timing_minutes = pred_timing_normed * window_size * DT_MINUTES   # → minutes
        final_risk_scores = outputs['risk_scores'][0].mean(dim=0).cpu().numpy().tolist()
        final_sys_state = {
            'frequency': float(outputs['frequency'].mean().item()),
            'voltages': outputs['voltages'][0].reshape(-1).cpu().numpy().tolist()
        }

        # Build per-node results — include predicted failure time for risky nodes
        max_probs = {n: float(probs[n]) for n in range(len(probs))}
        risky_nodes = [n for n, p in max_probs.items() if p > self.node_threshold]

        ranked_nodes = sorted(
            [{
                'node_id': n,
                'score': max_probs[n],
                'pred_time_minutes': float(pred_timing_minutes[n]),
            } for n in risky_nodes],
            key=lambda x: x['pred_time_minutes']  # sort by predicted failure time
        )

        # Assign sequence order based on predicted failure time (nodes failing at
        # the same minute get the same order number)
        cascade_path = []
        if ranked_nodes:
            current_rank = 1
            last_time = ranked_nodes[0]['pred_time_minutes']
            for node in ranked_nodes:
                if node['pred_time_minutes'] > last_time + 0.5:  # >0.5 min gap = new rank
                    current_rank += 1
                    last_time = node['pred_time_minutes']
                cascade_path.append({
                    'order': current_rank,
                    'node_id': node['node_id'],
                    'ranking_score': node['score'],
                    'pred_time_minutes': node['pred_time_minutes'],
                })

        # Ground truth from dataset item
        meta = dataset.scenario_files[scenario_idx]
        with open(meta, 'rb') as f:
            raw = pickle.load(f)
        scenario_meta = (raw[0] if isinstance(raw, list) else raw).get('metadata', {})

        gt_path = []
        if 'failed_nodes' in scenario_meta and 'failure_times' in scenario_meta:
            # Ground truth failure_times are stored as absolute timestep indices.
            # Convert to minutes using DT_MINUTES = 2.0 (ThermalConfig.DT_MINUTES).
            GT_DT = 2.0   # minutes per simulation timestep
            gt_path = sorted([
                {'node_id': int(n), 'time_minutes': float(t) * GT_DT}
                for n, t in zip(scenario_meta['failed_nodes'], scenario_meta['failure_times'])
            ], key=lambda x: x['time_minutes'])

        return {
            'inference_time': 0.0,
            'cascade_detected': bool(risky_nodes),
            'cascade_probability': ranked_nodes[0]['score'] if ranked_nodes else 0.0,
            'ground_truth': {
                'is_cascade': scenario_meta.get('is_cascade'),
                'failed_nodes': scenario_meta.get('failed_nodes', []),
                'cascade_path': gt_path,
                'ground_truth_risk': scenario_meta.get('ground_truth_risk', [])
            },
            'high_risk_nodes': risky_nodes,
            'risk_assessment': final_risk_scores,
            'top_nodes': ranked_nodes,
            'cascade_path': cascade_path,
            'system_state': final_sys_state,
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
        pred_path = res['cascade_path']
        act_path = gt.get('cascade_path', [])

        pred_times = [n['pred_time_minutes'] for n in pred_path] if pred_path else []
        act_times  = [x['time_minutes'] for x in act_path] if act_path else []

        min_pt, max_pt = (min(pred_times), max(pred_times)) if pred_times else (0.0, 0.0)
        min_at, max_at = (min(act_times),  max(act_times))  if act_times  else (0.0, 0.0)

        print(f"  {'Metric':<28} | {'Predicted':>12} | {'Actual':>12}")
        print(f"  {'-'*28}-+-{'-'*12}-+-{'-'*12}")
        print(f"  {'First failure (min)':<28} | {min_pt:>12.2f} | {min_at:>12.2f}")
        print(f"  {'Last failure (min)':<28} | {max_pt:>12.2f} | {max_at:>12.2f}")
        print(f"  {'Spread (min)':<28} | {max_pt-min_pt:>12.2f} | {max_at-min_at:>12.2f}")
        print(f"  {'Nodes at risk':<28} | {len(pred_path):>12} | {len(act_path):>12}")

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

    print(f"  {'Seq#':<5} | {'Pred Node':<10} | {'Prob':>6} | {'Pred(min)':>9} | {'Act Seq#':<8} | {'Act Node':<10} | {'Act(min)':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")

    curr_act_seq = 0
    last_act_time = -999.0
    max_rows = max(len(pred_path), len(actual_path))

    for i in range(max_rows):
        p_seq = p_node = p_score = p_time = ""
        if i < len(pred_path):
            p = pred_path[i]
            p_seq   = str(p['order'])
            p_node  = f"Node {p['node_id']}"
            p_score = f"{p['ranking_score']:.3f}"
            p_time  = f"{p['pred_time_minutes']:.2f}"

        a_seq = a_node = a_time = ""
        if i < len(actual_path):
            a = actual_path[i]
            t = a['time_minutes']
            if t > last_act_time + 0.1:
                curr_act_seq += 1
                last_act_time = t
            a_seq  = str(curr_act_seq)
            a_node = f"Node {a['node_id']}"
            a_time = f"{t:.2f}"

        print(f"  {p_seq:<5} | {p_node:<10} | {p_score:>6} | {p_time:>9} | {a_seq:<8} | {a_node:<10} | {a_time:>8}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="data/test")
    parser.add_argument("--scenario_idx", type=int, default=0)
    parser.add_argument("--topology_path", default="data/grid_topology.pkl")
    parser.add_argument("--output", default="prediction.json")
    parser.add_argument("--batch_size", type=int, default=Settings.Training.BATCH_SIZE)
    parser.add_argument("--window_size", type=int, default=Settings.Simulation.DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {dev}")
    
    predictor = CascadePredictor(args.model_path, args.topology_path, device=dev,
                                 base_mva=Settings.Dataset.BASE_MVA,
                                 base_freq=Settings.Dataset.BASE_FREQUENCY)
    
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