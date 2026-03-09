"""
Training Script for Cascade Prediction Model (Refactored)
==========================================================

This is a refactored version of train_model.py that uses the modular
components from the cascade_prediction package.

Key differences from original:
- Uses cascade_prediction.models.UnifiedCascadePredictionModel
- Uses cascade_prediction.models.PhysicsInformedLoss
- Uses cascade_prediction.data.CascadeDataset
- Uses cascade_prediction.training.Trainer
- Cleaner, more maintainable code

Author: Kraftgene AI Inc. (R&D)
Date: March 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import os
import sys
from pathlib import Path

# Import from refactored modules
from cascade_prediction.models import UnifiedCascadePredictionModel, PhysicsInformedLoss
from cascade_prediction.data import CascadeDataset, collate_cascade_batch
from cascade_prediction.training import Trainer


def main():
    """Main training function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Cascade Prediction Model (Refactored Version)"
    )
    
    # Data and output
    parser.add_argument(
        '--data_dir', type=str, default="data",
        help="Root directory containing train/val/test data folders"
    )
    parser.add_argument(
        '--output_dir', type=str, default="checkpoints",
        help="Directory to save checkpoints and logs"
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=100,
        help="Number of epochs to train"
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help="Training and validation batch size"
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help="Initial learning rate"
    )
    parser.add_argument(
        '--grad_clip', type=float, default=20.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        '--patience', type=int, default=25,
        help="Epochs for early stopping patience"
    )
    
    # Model parameters
    parser.add_argument(
        '--embedding_dim', type=int, default=128,
        help="Embedding dimension"
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        '--num_gnn_layers', type=int, default=3,
        help="Number of GNN layers"
    )
    parser.add_argument(
        '--heads', type=int, default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        '--dropout', type=float, default=0.5,
        help="Dropout rate"
    )
    
    # Resume training
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Path to checkpoint file to resume training"
    )
    
    # Physics parameters
    parser.add_argument(
        '--base_mva', type=float, default=100.0,
        help="Base MVA for physics normalization"
    )
    parser.add_argument(
        '--base_freq', type=float, default=60.0,
        help="Base frequency (Hz) for physics normalization"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("CASCADE FAILURE PREDICTION - TRAINING SCRIPT (REFACTORED)")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MAX_GRAD_NORM = args.grad_clip
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = (DEVICE.type == 'cuda')  # Enable AMP only on CUDA devices
    
    MODEL_OUTPUTS_LOGITS = False 

    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    if not torch.cuda.is_available():
        print("  (WARNING: CUDA not available, training will be slow on CPU)")
    print(f"  Gradient clipping: {MAX_GRAD_NORM}")
    print(f"  Mixed precision (AMP): {USE_AMP}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Resume training: {args.resume is not None}")
    
    # Load datasets
    print(f"\nLoading datasets...")
    try:
        train_dataset = CascadeDataset(
            f"{DATA_DIR}/train",
            mode='full_sequence',
            base_mva=args.base_mva,
            base_frequency=args.base_freq
        )
        val_dataset = CascadeDataset(
            f"{DATA_DIR}/val",
            mode='full_sequence',
            base_mva=args.base_mva,
            base_frequency=args.base_freq
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        print(f"Please ensure data exists in '{DATA_DIR}/train' and '{DATA_DIR}/val'")
        print("Run multimodal_data_generator_new.py to generate data.")
        sys.exit(1)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("\n[ERROR] No training data found. Please generate data first.")
        sys.exit(1)
    
    # Compute sample weights for balanced sampling
    print(f"\nComputing sample weights for balanced sampling...")
    
    positive_count = sum(
        train_dataset.get_cascade_label(idx)
        for idx in range(len(train_dataset))
    )
    negative_count = len(train_dataset) - positive_count
    
    if positive_count == 0 or negative_count == 0:
        print("  [WARNING] Training data contains only one class. Using uniform weights.")
        sample_weights = [1.0] * len(train_dataset)
    else:
        total_samples = len(train_dataset)
        pos_weight_val = total_samples / positive_count
        neg_weight_val = total_samples / negative_count
        
        sample_weights = []
        for idx in range(len(train_dataset)):
            if train_dataset.get_cascade_label(idx):
                sample_weights.append(pos_weight_val)
            else:
                sample_weights.append(neg_weight_val)
        
        print(f"  Positive samples: {positive_count} ({positive_count/total_samples*100:.1f}%)")
        print(f"  Negative samples: {negative_count} ({negative_count/total_samples*100:.1f}%)")
        print(f"  Calculated weights -> Pos: {pos_weight_val:.2f}, Neg: {neg_weight_val:.2f}")
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,  # Don't shuffle when using sampler
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_cascade_batch,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_cascade_batch,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = UnifiedCascadePredictionModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize loss function with dynamic calibration
    print(f"\nInitializing loss function...")
    
    from cascade_prediction.training import calibrate_loss_weights, apply_calibrated_weights
    
    # Perform dynamic loss weight calibration (always enabled)
    calibrated_lambdas = calibrate_loss_weights(
        model=model,
        train_loader=train_loader,
        criterion=None,  # Will create dummy criterion internally
        device=DEVICE,
        num_batches=20,  # Use 20 batches for calibration
        model_outputs_logits=False,
        base_mva=args.base_mva,
        base_freq=args.base_freq
    )
    
    # Create loss function with calibrated weights
    if calibrated_lambdas:
        criterion = PhysicsInformedLoss(
            lambda_prediction=calibrated_lambdas.get('lambda_prediction', 1.0),
            lambda_powerflow=calibrated_lambdas.get('lambda_powerflow', 0.1),
            lambda_temperature=calibrated_lambdas.get('lambda_temperature', 0.05),
            lambda_frequency=calibrated_lambdas.get('lambda_frequency', 0.08) ,
            lambda_reactive=calibrated_lambdas.get('lambda_reactive', 0.1) ,
            lambda_risk=calibrated_lambdas.get('lambda_risk', 0.1) ,
            lambda_timing=calibrated_lambdas.get('lambda_timing', 0.1) ,
            lambda_active_flow=calibrated_lambdas.get('lambda_active_flow', 0.1) ,
            lambda_voltage=calibrated_lambdas.get('lambda_voltage', 1.0),
            lambda_capacity=calibrated_lambdas.get('lambda_capacity', 0.05) ,
            pos_weight=1.0,
            focal_alpha=0.15,
            focal_gamma=2.0,
            label_smoothing=0.0,
            use_logits=False,
            base_mva=args.base_mva,
            base_freq=args.base_freq
        )
        print(f"  Calibrated loss weights :")
        print(f"    Prediction:      {calibrated_lambdas.get('lambda_prediction', 1.0) :.6f}")
        print(f"    Powerflow:       {calibrated_lambdas.get('lambda_powerflow', 0.1) :.6f}")
        print(f"    Temperature:     {calibrated_lambdas.get('lambda_temperature', 0.05) :.6f}")
        print(f"    Frequency:       {calibrated_lambdas.get('lambda_frequency', 0.08) :.6f}")
        print(f"    Reactive:        {calibrated_lambdas.get('lambda_reactive', 0.1) :.6f}")
        print(f"    Risk:            {calibrated_lambdas.get('lambda_risk', 0.1) :.6f}")
        print(f"    Timing:          {calibrated_lambdas.get('lambda_timing', 0.1) :.6f}")
        print(f"    Active flow:     {calibrated_lambdas.get('lambda_active_flow', 0.1) :.6f}")
        print(f"    Voltage:         {calibrated_lambdas.get('lambda_voltage', 1.0) :.6f}")
        print(f"    Capacity:        {calibrated_lambdas.get('lambda_capacity', 0.05) :.6f}")
    else:
        print("[WARNING] Calibration failed, using default lambda values")
        criterion = PhysicsInformedLoss(
            lambda_prediction=1.0,
            lambda_powerflow=0.1,
            lambda_temperature=0.05,
            lambda_frequency=0.08,
            lambda_reactive=0.1,
            lambda_risk=0.1,
            lambda_timing=0.1 ,
            lambda_active_flow=0.1 ,
            lambda_voltage=1.0 ,
            lambda_capacity=0.05 ,
            pos_weight=1.0,
            focal_alpha=0.15,
            focal_gamma=2.0,
            label_smoothing=0.0,
            use_logits=False,
            base_mva=args.base_mva,
            base_freq=args.base_freq
        )
        print(f"  Default loss weights:")
        print(f"    Prediction:      1.0")
        print(f"    Powerflow:       0.1")
        print(f"    Temperature:     0.05")
        print(f"    Stability:       0.05")
        print(f"    Frequency:       0.08")
        print(f"    Reactive:        0.1")
        print(f"    Risk:            0.1")
        print(f"    Timing:          0.1")
        print(f"    Flow:            0.05")
        print(f"    Active flow:     0.1")
        print(f"    Voltage:         1.0")
        print(f"    Capacity:        0.05")
    
    # Initialize trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        max_grad_norm=MAX_GRAD_NORM,
        patience=args.patience,
        use_amp=USE_AMP,
        model_outputs_logits=False,  # Model outputs probabilities, not logits
        base_mva=args.base_mva,
        base_freq=args.base_freq
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = args.resume 
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            
            # --- NEW: FORCE LR RESET FOR PHASE 2 ---
            print(f"\n[PHASE 2 RESET] Manually resetting Learning Rate to {LEARNING_RATE}...")
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE  # Resets to 0.0001 (or whatever arg you passed)
            
            # OPTIONAL: Reset Scheduler to forget "patience" history
            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer, 'min', patience=5
            )
            print("[PHASE 2 RESET] Scheduler reset.")
            # ---------------------------------------

        else:
            print(f"Warning: Checkpoint file not found...")
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")
    print(f"\nYou can now run inference using:")
    print(f"  python inference.py --model {OUTPUT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
