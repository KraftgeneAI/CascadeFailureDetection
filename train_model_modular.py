"""
Training Script for Cascade Prediction Model (Modular Version)
==============================================================
Uses modular components from cascade_prediction package.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path

# Import modular components
try:
    from cascade_prediction.models import UnifiedCascadePredictionModel
    from cascade_prediction.data import CascadeDataset, collate_cascade_batch
    from cascade_prediction.training import Trainer
except ImportError as e:
    print(f"Error: Could not import cascade_prediction modules. {e}")
    print("Make sure the cascade_prediction package is properly installed.")
    sys.exit(1)


def create_dataloaders(
    train_path: str,
    val_path: str,
    topology_path: str,
    batch_size: int = 16,
    max_sequence_length: int = 30,
    num_workers: int = 0
):
    """
    Create training and validation dataloaders.
    
    Args:
        train_path: Path to training data directory
        val_path: Path to validation data directory
        topology_path: Path to grid topology file
        batch_size: Batch size for training
        max_sequence_length: Maximum sequence length
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CascadeDataset(
        data_path=train_path,
        topology_path=topology_path,
        max_sequence_length=max_sequence_length,
        is_training=True
    )
    
    val_dataset = CascadeDataset(
        data_path=val_path,
        topology_path=topology_path,
        max_sequence_length=max_sequence_length,
        is_training=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cascade_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cascade_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train cascade prediction model"
    )
    parser.add_argument("--train_path", default="data/train", help="Path to training data")
    parser.add_argument("--val_path", default="data/test", help="Path to validation data")
    parser.add_argument("--topology_path", default="data/grid_topology.pkl", help="Path to topology file")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_sequence_length", type=int, default=30, help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        args.train_path,
        args.val_path,
        args.topology_path,
        args.batch_size,
        args.max_sequence_length
    )
    
    # Create model
    print("\nCreating model...")
    model = UnifiedCascadePredictionModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        heads=args.heads,
        dropout=args.dropout
    )
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        max_grad_norm=1.0,
        accumulation_steps=1
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_every=5
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
