"""
GridStateForecaster Inference
==============================
Autoregressive rollout from a seed window to the end of a scenario.

Usage:
    python inference_grid_state_forecaster.py \\
        --scenario data/train/scenario_0001.pkl \\
        --checkpoint checkpoints_grid_state_forecaster/best_model.pth \\
        --start_t 0

Metrics reported:
    • Node failure precision / recall / F1  (voltage-based failure detection)
    • Per-node failure timestep: predicted vs actual, and absolute error
"""

import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from cascade_prediction.models import (
    GridStateForecaster,
    assemble_full_scada,
    assemble_full_equip,
    SCADA_CONST_IDX,
    EQUIP_CONST_IDX,
)
from cascade_prediction.data import WINDOW_SIZE, load_scenario
from cascade_prediction.data.sliding_window_dataset import _build_node_features
from cascade_prediction.data.generator.config import Settings

VOLTAGE_THRESHOLD = 0.9   # per-unit; voltage below this → node failing


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout(model, tensors, start_t, device):
    """
    Run autoregressive rollout starting from seed window [start_t : start_t+W].

    Returns:
        pred_voltages [steps, N]  — predicted voltage at each predicted step
        gt_voltages   [steps, N]  — ground-truth voltage at each predicted step
        pred_range    range        — absolute timestep indices of predicted steps
    """
    W   = WINDOW_SIZE
    T   = tensors['seq_len']
    ei  = tensors['edge_index'].to(device)  # [2, E]

    # Rolling window tensors (kept on CPU, moved to device per step)
    w_scada = tensors['scada'][start_t : start_t + W].clone()    # [W, N, 18]
    w_pmu   = tensors['pmu'][start_t : start_t + W].clone()      # [W, N,  8]
    w_equip = tensors['equip'][start_t : start_t + W].clone()    # [W, N, 10]
    w_p_inj = tensors['p_inj'][start_t : start_t + W].clone()    # [W, N,  1]
    w_q_inj = tensors['q_inj'][start_t : start_t + W].clone()    # [W, N,  1]
    # Edge attributes: use seed-window slice; carry last step forward during rollout
    w_ea    = tensors['edge_attr'][start_t : start_t + W].clone()  # [W, E, 7]
    w_mask  = tensors['edge_mask'][start_t : start_t + W].clone()  # [W, E]

    pred_voltages = []
    gt_voltages   = []
    pred_range    = range(start_t + W, T)

    for abs_t in pred_range:
        # Build node features for current window
        nf = _build_node_features(
            w_scada, w_pmu, w_equip, w_p_inj, w_q_inj,
            start_t=abs_t - W,
            seq_len=T,
        )   # [W, N, 119]

        # Batch dim = 1
        batch = {
            'scada_data':       w_scada.unsqueeze(0).to(device),   # [1, W, N, 18]
            'pmu_sequence':     w_pmu.unsqueeze(0).to(device),
            'equipment_status': w_equip.unsqueeze(0).to(device),
            'node_features':    nf.unsqueeze(0).to(device),
            'edge_index':       ei,
            'edge_attr':        w_ea.unsqueeze(0).to(device),
            'edge_mask':        w_mask.unsqueeze(0).to(device),
        }

        out = model(batch)   # next_scada_vars [1,N,13], next_pmu [1,N,8], next_equip_vars [1,N,3]

        # --- Reconstruct full next-step tensors ---
        # Constants carried forward from last window step
        scada_const = w_scada[-1:, :, :][:, :, SCADA_CONST_IDX]   # [1, N, 5]
        equip_const = w_equip[-1:, :, :][:, :, EQUIP_CONST_IDX]   # [1, N, 7]

        next_scada = assemble_full_scada(
            out['next_scada_vars'].cpu(), scada_const
        )   # [1, N, 18]
        next_pmu   = out['next_pmu'].cpu()                          # [1, N,  8]
        next_equip = assemble_full_equip(
            out['next_equip_vars'].cpu(), equip_const
        )   # [1, N, 10]

        # Approximate injections from predicted generation/load
        next_p_inj = (next_scada[:, :, 2] - next_scada[:, :, 4]).unsqueeze(-1)  # [1, N, 1]
        next_q_inj = next_scada[:, :, 3].unsqueeze(-1)                           # [1, N, 1]

        # Record predicted voltage [N]
        pred_voltages.append(out['next_scada_vars'].cpu()[0, :, 0].numpy())  # SCADA_VAR_IDX[0]=col0

        # Record ground-truth voltage [N]
        gt_voltages.append(tensors['scada'][abs_t, :, 0].numpy())

        # Slide window
        w_scada = torch.cat([w_scada[1:], next_scada[0].unsqueeze(0)],  dim=0)
        w_pmu   = torch.cat([w_pmu[1:],   next_pmu[0].unsqueeze(0)],    dim=0)
        w_equip = torch.cat([w_equip[1:], next_equip[0].unsqueeze(0)],  dim=0)
        w_p_inj = torch.cat([w_p_inj[1:], next_p_inj[0].unsqueeze(0)], dim=0)
        w_q_inj = torch.cat([w_q_inj[1:], next_q_inj[0].unsqueeze(0)], dim=0)
        # Edge attrs: carry last known step forward
        w_ea   = torch.cat([w_ea[1:],   w_ea[-1:]], dim=0)
        w_mask = torch.cat([w_mask[1:], w_mask[-1:]], dim=0)

    pred_voltages = np.stack(pred_voltages, axis=0)   # [steps, N]
    gt_voltages   = np.stack(gt_voltages,   axis=0)   # [steps, N]
    return pred_voltages, gt_voltages, pred_range


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_voltages, gt_voltages, pred_range, metadata):
    """
    Args:
        pred_voltages [steps, N]
        gt_voltages   [steps, N]
        pred_range    range — absolute timestep indices of predicted steps
        metadata      dict from scenario pkl

    Returns:
        metrics dict
    """
    steps, N = pred_voltages.shape
    pred_range_list = list(pred_range)

    # Binary failure per node: did voltage ever drop below threshold?
    pred_fail  = (pred_voltages < VOLTAGE_THRESHOLD).any(axis=0).astype(int)  # [N]
    gt_fail    = (gt_voltages   < VOLTAGE_THRESHOLD).any(axis=0).astype(int)  # [N]

    precision = precision_score(gt_fail, pred_fail, zero_division=0)
    recall    = recall_score(gt_fail,    pred_fail, zero_division=0)
    f1        = f1_score(gt_fail,        pred_fail, zero_division=0)

    # --- Failure timestep per node ---
    # actual_fail_t[n] = first predicted-range step where gt < threshold, or -1
    # pred_fail_t[n]   = first predicted-range step where pred < threshold, or -1
    def first_fail_t(voltages):
        result = np.full(N, -1, dtype=int)
        for t in range(steps):
            for n in range(N):
                if result[n] == -1 and voltages[t, n] < VOLTAGE_THRESHOLD:
                    result[n] = pred_range_list[t]
        return result

    gt_fail_t   = first_fail_t(gt_voltages)
    pred_fail_t = first_fail_t(pred_voltages)

    # Cross-reference with metadata failure_times (ground truth from simulator)
    meta_fail_nodes = list(metadata.get('failed_nodes',  []))
    meta_fail_times = list(metadata.get('failure_times', []))
    meta_fail_t = {int(n): int(t) for n, t in zip(meta_fail_nodes, meta_fail_times)}

    # Timing error: only for nodes that actually failed AND are in the predicted range
    timing_errors = []
    for n in range(N):
        if meta_fail_t.get(n, -1) in pred_range_list and pred_fail_t[n] != -1:
            timing_errors.append(abs(pred_fail_t[n] - meta_fail_t[n]))

    timing_mae = float(np.mean(timing_errors)) if timing_errors else float('nan')

    return {
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'timing_mae':  timing_mae,
        'gt_fail':     gt_fail,
        'pred_fail':   pred_fail,
        'gt_fail_t':   gt_fail_t,
        'pred_fail_t': pred_fail_t,
        'meta_fail_t': meta_fail_t,
        'num_gt_fail': int(gt_fail.sum()),
        'num_pred_fail': int(pred_fail.sum()),
    }


def print_results(metrics, N, start_t, pred_range):
    print()
    print('=' * 60)
    print('ROLLOUT EVALUATION')
    print('=' * 60)
    print(f'  Seed window   : steps {start_t} – {start_t + WINDOW_SIZE - 1}')
    print(f'  Predicted     : steps {pred_range.start} – {pred_range.stop - 1}  '
          f'({len(pred_range)} steps)')
    print()
    print(f'  Ground-truth failing nodes : {metrics["num_gt_fail"]} / {N}')
    print(f'  Predicted failing nodes    : {metrics["num_pred_fail"]} / {N}')
    print()
    print(f'  Precision : {metrics["precision"]:.4f}')
    print(f'  Recall    : {metrics["recall"]:.4f}')
    print(f'  F1        : {metrics["f1"]:.4f}')
    print(f'  Timing MAE: {metrics["timing_mae"]:.2f} steps  '
          f'(failing nodes visible in predicted range)')
    print()

    # Per-node failure table (only nodes with activity)
    header = f'{"Node":>5}  {"GT fail":>7}  {"Pred fail":>9}  '  \
             f'{"GT time":>7}  {"Pred time":>9}  {"Time err":>8}'
    rows = []
    for n in range(N):
        gt_f = metrics['gt_fail'][n]
        pr_f = metrics['pred_fail'][n]
        if gt_f == 0 and pr_f == 0:
            continue
        gt_t  = metrics['gt_fail_t'][n]   if gt_f else '—'
        pr_t  = metrics['pred_fail_t'][n] if pr_f else '—'
        t_err = abs(metrics['pred_fail_t'][n] - metrics['gt_fail_t'][n]) \
                if (gt_f and pr_f) else '—'
        rows.append(f'{n:>5}  {str(gt_f):>7}  {str(pr_f):>9}  '
                    f'{str(gt_t):>7}  {str(pr_t):>9}  {str(t_err):>8}')

    if rows:
        print(header)
        print('-' * len(header))
        for row in rows:
            print(row)
    else:
        print('  No failing nodes detected in either ground truth or predictions.')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GridStateForecaster autoregressive inference'
    )
    parser.add_argument('--scenario',   type=str, required=True,
                        help='Path to scenario pkl file')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_grid_state_forecaster/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--start_t',    type=int, default=0,
                        help='First timestep of the seed window')
    parser.add_argument('--base_mva',   type=float, default=Settings.Dataset.BASE_MVA)
    parser.add_argument('--base_freq',  type=float, default=Settings.Dataset.BASE_FREQUENCY)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load scenario ---
    print(f'Loading scenario: {args.scenario}')
    tensors, metadata, _ = load_scenario(args.scenario, args.base_mva, args.base_freq)
    if tensors is None:
        print('[ERROR] Failed to load scenario.')
        sys.exit(1)
    T = tensors['seq_len']
    N = tensors['scada'].shape[1]
    print(f'  Timesteps: {T}   Nodes: {N}')
    print(f'  Cascade  : {metadata.get("is_cascade", False)}')
    print(f'  Failed nodes (ground truth): {metadata.get("failed_nodes", [])}')

    if args.start_t + WINDOW_SIZE >= T:
        print(f'[ERROR] start_t={args.start_t} leaves no steps to predict '
              f'(need start_t + {WINDOW_SIZE} < {T})')
        sys.exit(1)

    # --- Load model ---
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE)
    model = GridStateForecaster().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # --- Rollout ---
    print(f'\nRunning rollout from t={args.start_t} (seed: '
          f'{args.start_t}–{args.start_t + WINDOW_SIZE - 1}, '
          f'predicting: {args.start_t + WINDOW_SIZE}–{T - 1})...')

    pred_voltages, gt_voltages, pred_range = rollout(
        model, tensors, args.start_t, DEVICE
    )

    # --- Metrics ---
    metrics = compute_metrics(pred_voltages, gt_voltages, pred_range, metadata)
    print_results(metrics, N, args.start_t, pred_range)


if __name__ == '__main__':
    main()
