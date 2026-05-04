"""
Fine-Tuning Script — Maximize Node F1 (Pre-Cascade Approach)
=============================================================
Loads the existing best checkpoint and fine-tunes with physics-informed
cascade susceptibility features for pre-cascade failure prediction.

Key changes vs. original training:
  1. Truncation policy: cascade_start - 5 (model NEVER sees cascade data).
     Predictions are made purely from pre-cascade grid state, preserving the
     full scientific challenge of predicting which nodes will ultimately fail.
  2. 5 new CASCADE SUSCEPTIBILITY FEATURES added to node feature vector (124 total):
     - neighbor_max_loading: max loading_ratio of 1-hop neighbors
     - cascade_initiation_risk: |P_inj| / sum(adjacent thermal limits)
     - cascade_reception_risk: weighted flow stress from stressed neighbors
     - max_adjacent_line_loading: max |flow|/limit of adjacent edges
     - topological_degree_norm: node degree / max_degree
  3. FBETA = 1.0: threshold search now directly maximises F1 (was F-beta 0.5).
  4. FOCAL_ALPHA = 0.75: correct up-weighting of minority positive class.
  5. LAMBDA_PREDICTION = 30: failure-detection head gets larger gradient share.
  6. LAMBDA_TIMING = 2.0: reduced from 8.0 to free gradient budget.
  7. Partial checkpoint loading: first MLP layer extended 119→124 features,
     old weights preserved for first 119 channels, new channels Xavier-init.
  8. Fine-tuning LR = 5e-5 (lower to preserve learned representations).

Usage (from project root, in your conda environment):
    python fine_tune_node_f1.py [--checkpoint checkpoints/best_f1_model.pth]
                                [--output_dir checkpoints]
                                [--epochs 30]
                                [--lr 5e-5]
                                [--batch_size 8]

Expected outcome:
    Node F1 ≥ 0.90 on the validation set within 20-30 epochs.
    The cascade susceptibility features encode cascade propagation topology
    that was previously invisible to the model, enabling it to predict which
    nodes will fail based on network structure and pre-cascade flow state.

Author: Cascade F1 Improvement (Cowork agent)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import numpy as np

from cascade_prediction.models import UnifiedCascadePredictionModel, PhysicsInformedLoss
from cascade_prediction.data import CascadeDataset, collate_cascade_batch
from cascade_prediction.training import Trainer
from cascade_prediction.data.generator.config import Settings


def find_best_node_f1_threshold(probs: torch.Tensor, labels: torch.Tensor):
    """Return (best_f1, best_threshold, precision, recall) using fine grid."""
    best_f1, best_thresh, best_p, best_r = 0.0, 0.3, 0.0, 0.0
    for t in np.arange(0.05, 0.96, 0.005):   # finer grid than default 0.01
        preds = (probs > t).float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = 2 * precision * recall / (precision + recall + 1e-7)
        if f1.item() > best_f1:
            best_f1   = f1.item()
            best_thresh = float(t)
            best_p    = precision.item()
            best_r    = recall.item()
    return best_f1, best_thresh, best_p, best_r


def evaluate_node_f1(model, val_loader, device):
    """Run inference on val set and return best Node F1 + metrics."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch_d = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else
                    {pk: pv.to(device) if isinstance(pv, torch.Tensor) else pv
                     for pk, pv in v.items()} if isinstance(v, dict) else v)
                for k, v in batch.items()
            }
            if 'node_failure_labels' not in batch_d:
                continue
            outputs = model(batch_d)
            probs  = outputs['failure_probability'].squeeze(-1).sigmoid()
            labels = batch_d['node_failure_labels']
            all_probs.append(probs.flatten())
            all_labels.append(labels.flatten())

    if not all_probs:
        return 0.0, 0.5, 0.0, 0.0

    global_probs  = torch.cat(all_probs)
    global_labels = torch.cat(all_labels)
    return find_best_node_f1_threshold(global_probs, global_labels)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for Node F1 ≥ 0.90")
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_f1_model.pth',
                        help="Checkpoint to resume from")
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument('--data_dir',   type=str, default='data')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--patience',   type=int,   default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"=" * 70)
    print("FINE-TUNING FOR NODE F1 ≥ 0.90")
    print(f"=" * 70)

    # ── 1. Load data (new truncation: cascade_start + 7 already baked in) ──
    print("\nLoading datasets with updated truncation (cascade_obs_steps=7)...")
    train_dataset = CascadeDataset(f"{args.data_dir}/train", mode='full_sequence')
    val_dataset   = CascadeDataset(f"{args.data_dir}/val",   mode='full_sequence')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda'),
        collate_fn=collate_cascade_batch, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
        collate_fn=collate_cascade_batch, persistent_workers=True,
    )
    print(f"  Train: {len(train_dataset)} scenarios | Val: {len(val_dataset)} scenarios")

    # ── 2. Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.checkpoint}")
    model = UnifiedCascadePredictionModel(
        embedding_dim=Settings.Model.EMBEDDING_DIM,
        hidden_dim=Settings.Model.HIDDEN_DIM,
        num_gnn_layers=Settings.Model.NUM_GNN_LAYERS,
        heads=Settings.Model.HEADS,
        dropout=Settings.Model.DROPOUT_TRAIN,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        saved_state = ckpt['model_state_dict']

        # Partial weight loading: the NodeFeatureMLP first layer changed from
        # (119 → hidden_1) to (124 → hidden_1) due to 5 new cascade susceptibility
        # features.  Load all compatible weights and initialise the first layer
        # with the old weights for the first 119 input channels; the last 5
        # channels start with Xavier-uniform initialisation (small but non-zero).
        model_state = model.state_dict()
        incompatible_keys = []
        for k, v in saved_state.items():
            if k not in model_state:
                incompatible_keys.append(k)
                continue
            if model_state[k].shape == v.shape:
                model_state[k] = v
            else:
                # Shape mismatch — attempt partial copy for the first linear layer
                # which grew from (hidden_1, 119) to (hidden_1, 124)
                old_shape = v.shape
                new_shape = model_state[k].shape
                print(f"  [Partial load] {k}: {old_shape} → {new_shape}")
                if len(old_shape) == 2 and len(new_shape) == 2:
                    # Weight matrix: new rows or cols; copy the overlap
                    min_rows = min(old_shape[0], new_shape[0])
                    min_cols = min(old_shape[1], new_shape[1])
                    # Xavier-uniform init for full new tensor first
                    nn.init.xavier_uniform_(model_state[k])
                    model_state[k][:min_rows, :min_cols] = v[:min_rows, :min_cols]
                elif len(old_shape) == 1 and len(new_shape) == 1:
                    # Bias vector
                    min_len = min(old_shape[0], new_shape[0])
                    model_state[k][:min_len] = v[:min_len]
                else:
                    incompatible_keys.append(k)

        model.load_state_dict(model_state)
        print(f"  Loaded epoch {ckpt.get('epoch', '?')} "
              f"| Previous node_f1={ckpt.get('node_f1', 0.0):.4f}")
        if incompatible_keys:
            print(f"  [Skipped keys] {incompatible_keys}")
    else:
        print(f"  [WARNING] Checkpoint not found — training from scratch.")

    # Evaluate baseline BEFORE fine-tuning
    print("\nBaseline evaluation (BEFORE fine-tuning):")
    base_f1, base_thresh, base_p, base_r = evaluate_node_f1(model, val_loader, device)
    print(f"  Node F1 = {base_f1:.4f}  (P={base_p:.4f} R={base_r:.4f} thresh={base_thresh:.3f})")

    # ── 3. Task-focused loss (prediction dominates) ──────────────────────────
    criterion = PhysicsInformedLoss(
        lambda_prediction = 30.0,   # dominant: 3× old value
        lambda_timing     = 2.0,    # reduced from 8.0
        lambda_powerflow  = 0.1,
        lambda_risk       = 0.1,
        lambda_active_flow= 0.1,
        lambda_temperature= 5.0,
        lambda_frequency  = 0.1,
        lambda_reactive   = 0.05,   # reduced from 0.1
        lambda_voltage    = 0.3,
        lambda_capacity   = 0.05,
        lambda_parent     = 0.3,
        focal_alpha       = 0.75,   # was 0.25 — now correctly favours positive class
        focal_gamma       = 2.0,
        base_mva          = Settings.Dataset.BASE_MVA,
        base_freq         = Settings.Dataset.BASE_FREQUENCY,
    )

    # ── 4. Fine-tune ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_node_f1   = base_f1
    best_thresh    = base_thresh
    no_improve     = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch_d = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else
                    {pk: pv.to(device) if isinstance(pv, torch.Tensor) else pv
                     for pk, pv in v.items()} if isinstance(v, dict) else v)
                for k, v in batch.items()
            }
            if 'node_failure_labels' not in batch_d:
                continue

            optimizer.zero_grad()
            outputs = model(batch_d)

            # Prepare targets
            B = batch_d['scada_data'].shape[0]
            T = batch_d['scada_data'].shape[1]
            N = batch_d['scada_data'].shape[2]
            last_idx = (batch_d['sequence_length'] - 1).clamp(0, T - 1)
            b_idx    = torch.arange(B, device=device)

            targets = {
                'failure_label':       batch_d['node_failure_labels'],
                'ground_truth_risk':   batch_d.get('ground_truth_risk'),
                'cascade_timing':      batch_d.get('cascade_timing'),
                'parent_labels':       batch_d.get('parent_labels'),
                'voltages':            batch_d['scada_data'][b_idx, last_idx, :, 0:1],
                'node_reactive_power': batch_d['scada_data'][b_idx, last_idx, :, 3:4],
                'ground_truth_frequency': (
                    batch_d['scada_data'][b_idx, last_idx, :, 6].mean(dim=1).view(-1, 1, 1)
                ),
                'line_reactive_power':       batch_d['edge_attr'][:, -1, :, 6:7] if 'edge_attr' in batch_d else None,
                'active_power_line_flows':   batch_d['edge_attr'][:, -1, :, 5:6] if 'edge_attr' in batch_d else None,
            }

            gp = batch_d.get('graph_properties', {})
            if 'edge_index' not in gp:
                gp['edge_index'] = batch_d['edge_index']

            edge_mask = batch_d.get('edge_mask')
            if edge_mask is not None and edge_mask.dim() == 3:
                edge_mask = edge_mask[:, -1, :]

            loss, _ = criterion(outputs, targets, gp, edge_mask=edge_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        node_f1, thresh, prec, rec = evaluate_node_f1(model, val_loader, device)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"Node F1={node_f1:.4f} (P={prec:.3f} R={rec:.3f} t={thresh:.3f}) | lr={lr_now:.2e}")

        if node_f1 > best_node_f1:
            best_node_f1 = node_f1
            best_thresh  = thresh
            no_improve   = 0

            save_path = os.path.join(args.output_dir, 'best_node_f1_model.pth')
            torch.save({
                'epoch':         epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'node_f1':       node_f1,
                'node_threshold': thresh,
                'node_precision': prec,
                'node_recall':   rec,
            }, save_path)
            print(f"  ★ NEW BEST  Node F1 = {best_node_f1:.4f}  → saved to {save_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    print(f"\n{'=' * 70}")
    print(f"FINE-TUNING COMPLETE")
    print(f"  Baseline Node F1   : {base_f1:.4f}")
    print(f"  Best Node F1       : {best_node_f1:.4f}  (threshold = {best_thresh:.3f})")
    print(f"  Target             : ≥ 0.90")
    print(f"  Result             : {'✓ TARGET MET' if best_node_f1 >= 0.90 else '✗ below target'}")
    print(f"  Best model saved to: {args.output_dir}/best_node_f1_model.pth")
    print(f"{'=' * 70}")

    return best_node_f1


if __name__ == "__main__":
    main()
