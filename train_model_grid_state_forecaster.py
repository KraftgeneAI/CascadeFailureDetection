"""
Training Script for GridStateForecaster
========================================
Trains the autoregressive next-step prediction model and saves:
  - checkpoints_grid_state_forecaster/best_model.pth   (best val MSE loss)
  - checkpoints_grid_state_forecaster/best_f1_model.pth (best node-level voltage F1)

Author: Kraftgene AI Inc. (R&D)
Date: April 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from cascade_prediction.models import (
    GridStateForecaster,
    extract_next_step_targets,
    SCADA_VAR_IDX,
)
from cascade_prediction.data import SlidingWindowDataset, collate_cascade_batch, TEMPORAL_KEYS
from cascade_prediction.data.generator.config import Settings
from cascade_prediction.utils import find_best_fbeta



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_history(history: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # F1 curve
    axes[1].plot(history['val_f1'], label='Val node F1', color='green')
    axes[1].set_title('Node Voltage F1')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=100)
    plt.close(fig)


def compute_node_f1(
    pred_labels:   torch.Tensor,  # [B, N]  — sigmoid probabilities
    target_labels: torch.Tensor,  # [B, N]  — binary ground truth
    threshold: float = 0.5,
) -> float:
    """Binary per-node F1 from predicted node_labels probabilities vs ground truth."""
    pred_fail   = (pred_labels > threshold).int().reshape(-1).numpy()
    actual_fail = target_labels.int().reshape(-1).numpy()

    if actual_fail.sum() == 0 and pred_fail.sum() == 0:
        return 1.0  # Both agree: no failures
    return float(f1_score(actual_fail, pred_fail, zero_division=0))


def train_one_epoch(
    model: GridStateForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
) -> dict:
    model.train()
    total_loss = total_scada = total_pmu = total_equip = total_labels = 0.0

    for batch in tqdm(loader, desc='  train', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            predictions = model(_make_input_batch(batch))
            targets = extract_next_step_targets(
                batch['scada_data'],
                batch['pmu_sequence'],
                batch['equipment_status'],
                batch.get('node_labels'),
            )
            targets['node_label_weights'] = _node_label_weights(batch)
            loss, components = model.compute_loss(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss   += components['total']
        total_scada  += components['loss_scada']
        total_pmu    += components['loss_pmu']
        total_equip  += components['loss_equip']
        total_labels += components['loss_labels']

    n = len(loader)
    return {
        'total':       total_loss   / n,
        'loss_scada':  total_scada  / n,
        'loss_pmu':    total_pmu    / n,
        'loss_equip':  total_equip  / n,
        'loss_labels': total_labels / n,
    }


@torch.no_grad()
def validate(
    model: GridStateForecaster,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple:
    """Returns (loss_dict, node_f1)."""
    model.eval()
    total_loss = total_scada = total_pmu = total_equip = total_labels = 0.0
    all_pred_labels = []
    all_tgt_labels  = []

    for batch in tqdm(loader, desc='  val  ', leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast('cuda', enabled=use_amp):
            predictions = model(_make_input_batch(batch))
            targets = extract_next_step_targets(
                batch['scada_data'],
                batch['pmu_sequence'],
                batch['equipment_status'],
                batch.get('node_labels'),
            )
            targets['node_label_weights'] = _node_label_weights(batch)
            _, components = model.compute_loss(predictions, targets)

        total_loss   += components['total']
        total_scada  += components['loss_scada']
        total_pmu    += components['loss_pmu']
        total_equip  += components['loss_equip']
        total_labels += components['loss_labels']

        all_pred_labels.append(predictions['node_labels'].cpu())
        all_tgt_labels.append(targets['node_labels'].cpu())

    n = len(loader)
    loss_dict = {
        'total':       total_loss   / n,
        'loss_scada':  total_scada  / n,
        'loss_pmu':    total_pmu    / n,
        'loss_equip':  total_equip  / n,
        'loss_labels': total_labels / n,
    }

    pred_cat = torch.cat(all_pred_labels, dim=0)
    tgt_cat  = torch.cat(all_tgt_labels,  dim=0)

    _, best_thresh = find_best_fbeta(
        pred_cat.flatten(), tgt_cat.flatten(),
        beta=Settings.Training.FBETA,
    )
    node_f1 = compute_node_f1(pred_cat, tgt_cat, threshold=best_thresh)

    return loss_dict, node_f1, best_thresh


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _node_label_weights(batch: dict) -> torch.Tensor:
    """
    Per-node BCE weights based on label transition in the window.
      0 → 1  (first failure)   : FIRST_FAILURE_WEIGHT
      1 → 1  (already failed)  : ALREADY_FAILED_WEIGHT
      otherwise                : 1.0
    Returns [B, N] weight tensor on the same device as the batch.
    """
    labels = batch['node_labels']            # [B, W+1, N]
    last_input = labels[:, -2, :]            # [B, N]  last input step
    target     = labels[:, -1, :]            # [B, N]  target step

    weights = torch.ones_like(target)
    first_failure  = (last_input < 0.5) & (target > 0.5)
    already_failed = (last_input > 0.5) & (target > 0.5)
    weights[first_failure]  = Settings.Training.FIRST_FAILURE_WEIGHT
    weights[already_failed] = Settings.Training.ALREADY_FAILED_WEIGHT
    return weights


def _make_input_batch(batch: dict) -> dict:
    """Slice temporal keys to first W steps, leaving the target step for extract_next_step_targets."""
    return {
        k: v[:, :-1] if k in TEMPORAL_KEYS and isinstance(v, torch.Tensor) and v.dim() >= 2 else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GridStateForecaster (next-step autoregressive model)"
    )

    # Data / output
    parser.add_argument('--data_dir',   type=str, default='data',
                        help='Root directory containing train/val data folders')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints_grid_state_forecaster',
                        help='Directory to save checkpoints and logs')

    # Training
    parser.add_argument('--epochs',     type=int,   default=Settings.Training.EPOCHS)
    parser.add_argument('--batch_size', type=int,   default=Settings.Training.BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=Settings.Training.LEARNING_RATE)
    parser.add_argument('--grad_clip',  type=float, default=Settings.Training.GRAD_CLIP)
    parser.add_argument('--patience',   type=int,   default=Settings.Training.PATIENCE)

    # Model
    parser.add_argument('--embedding_dim',  type=int,   default=Settings.Model.EMBEDDING_DIM)
    parser.add_argument('--num_gnn_layers', type=int,   default=Settings.Model.NUM_GNN_LAYERS)
    parser.add_argument('--heads',          type=int,   default=Settings.Model.HEADS)
    parser.add_argument('--dropout',        type=float, default=Settings.Model.DROPOUT_TRAIN)

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Dataset physics
    parser.add_argument('--base_mva',  type=float, default=Settings.Dataset.BASE_MVA)
    parser.add_argument('--base_freq', type=float, default=Settings.Dataset.BASE_FREQUENCY)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    print('=' * 70)
    print('GRID STATE FORECASTER — TRAINING')
    print('=' * 70)

    DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = DEVICE.type == 'cuda'

    print(f'\n  Data dir   : {args.data_dir}')
    print(f'  Output dir : {args.output_dir}')
    print(f'  Device     : {DEVICE}')
    print(f'  AMP        : {USE_AMP}')
    print(f'  Epochs     : {args.epochs}')
    print(f'  Batch size : {args.batch_size}')
    print(f'  LR         : {args.lr}')
    print(f'  Patience   : {args.patience}')

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print('\nLoading datasets...')
    try:
        train_dataset = SlidingWindowDataset(
            f'{args.data_dir}/train',
            base_mva=args.base_mva,
            base_frequency=args.base_freq,
        )
        val_dataset = SlidingWindowDataset(
            f'{args.data_dir}/val',
            base_mva=args.base_mva,
            base_frequency=args.base_freq,
        )
    except Exception as e:
        print(f'\n[ERROR] Failed to load dataset: {e}')
        sys.exit(1)

    print(f'  Train samples : {len(train_dataset)}')
    print(f'  Val samples   : {len(val_dataset)}')

    if len(train_dataset) == 0:
        print('\n[ERROR] No training data found.')
        sys.exit(1)

    # Balanced sampler (same as train_model.py)
    positive_count = sum(
        train_dataset.get_cascade_label(i) for i in range(len(train_dataset))
    )
    negative_count = len(train_dataset) - positive_count

    if positive_count == 0 or negative_count == 0:
        print('  [WARNING] Single-class dataset — using uniform weights.')
        sample_weights = [1.0] * len(train_dataset)
    else:
        n_total = len(train_dataset)
        pos_w = n_total / positive_count
        neg_w = n_total / negative_count
        sample_weights = [
            pos_w if train_dataset.get_cascade_label(i) else neg_w
            for i in range(n_total)
        ]
        print(f'  Pos/Neg: {positive_count}/{negative_count}  '
              f'weights: {pos_w:.2f}/{neg_w:.2f}')

    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_cascade_batch,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False,               **loader_kwargs)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print('\nInitializing model...')
    model = GridStateForecaster(
        embedding_dim=args.embedding_dim,
        num_gnn_layers=args.num_gnn_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=Settings.Training.SCHEDULER_PATIENCE
    )
    scaler = GradScaler('cuda', enabled=USE_AMP)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch     = 0
    best_val_loss   = float('inf')
    best_val_f1     = 0.0
    no_improve      = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_f1': [],
        'val_loss_scada': [], 'val_loss_pmu': [], 'val_loss_equip': [], 'val_loss_labels': [],
    }

    if args.resume and os.path.exists(args.resume):
        print(f'\nResuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_f1   = ckpt.get('best_val_f1',   0.0)
        history       = ckpt.get('history',        history)
        print(f'  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}')
    elif args.resume:
        print(f'[WARNING] Checkpoint not found: {args.resume}')

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss_path = os.path.join(args.output_dir, 'best_model.pth')
    best_f1_path   = os.path.join(args.output_dir, 'best_f1_model.pth')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f'\n{"="*70}')
    print('STARTING TRAINING')
    print(f'{"="*70}\n')

    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}  (lr={optimizer.param_groups[0]["lr"]:.2e})')

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE, USE_AMP, args.grad_clip
        )
        val_metrics, val_f1, best_thresh = validate(model, val_loader, DEVICE, USE_AMP)

        scheduler.step(val_metrics['total'])

        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['val_f1'].append(val_f1)
        history['val_loss_scada'].append(val_metrics['loss_scada'])
        history['val_loss_pmu'].append(val_metrics['loss_pmu'])
        history['val_loss_equip'].append(val_metrics['loss_equip'])
        history['val_loss_labels'].append(val_metrics['loss_labels'])

        print(f'  train : total={train_metrics["total"]:.6f}  '
              f'scada={train_metrics["loss_scada"]:.4f}  '
              f'pmu={train_metrics["loss_pmu"]:.4f}  '
              f'equip={train_metrics["loss_equip"]:.4f}  '
              f'labels={train_metrics["loss_labels"]:.4f}')
        print(f'  val   : total={val_metrics["total"]:.6f}  '
              f'scada={val_metrics["loss_scada"]:.4f}  '
              f'pmu={val_metrics["loss_pmu"]:.4f}  '
              f'equip={val_metrics["loss_equip"]:.4f}  '
              f'labels={val_metrics["loss_labels"]:.4f}  '
              f'f1={val_f1:.4f}  thresh={best_thresh:.2f}')

        checkpoint_state = {
            'epoch':                  epoch,
            'model_state_dict':       model.state_dict(),
            'optimizer_state_dict':   optimizer.state_dict(),
            'best_val_loss':          best_val_loss,
            'best_val_f1':            best_val_f1,
            'node_label_threshold':   best_thresh,
            'history':                history,
            'args':                   vars(args),
        }

        # Save best val-loss checkpoint
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            checkpoint_state['best_val_loss'] = best_val_loss
            save_checkpoint(checkpoint_state, best_loss_path)
            print(f'  [✓] Saved best_model.pth  (val_loss={best_val_loss:.6f})')
            no_improve = 0
        else:
            no_improve += 1

        # Save best node-F1 checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_state['best_val_f1'] = best_val_f1
            save_checkpoint(checkpoint_state, best_f1_path)
            print(f'  [✓] Saved best_f1_model.pth  (val_f1={best_val_f1:.4f})')

        # Early stopping
        if no_improve >= args.patience:
            print(f'\nEarly stopping: no val_loss improvement for {args.patience} epochs.')
            break

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    history_path = os.path.join(args.output_dir, 'training_history.json')
    save_history(history, history_path)
    plot_training_curves(history, args.output_dir)

    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'  Best val MSE : {best_val_loss:.6f}  →  {best_loss_path}')
    print(f'  Best node F1 : {best_val_f1:.4f}   →  {best_f1_path}')
    print(f'  History      : {history_path}')


if __name__ == '__main__':
    main()