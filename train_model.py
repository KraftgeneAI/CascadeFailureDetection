"""
Cascade Failure Prediction Model Training Script
Complete training pipeline with data loading, training, validation, and model saving.

Author: Kraftgene AI Inc.
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from cascade_prediction_model import (
    CompleteCascadePredictionModel,
    PhysicsInformedLoss
)


class CascadeDataset(Dataset):
    """PyTorch Dataset for cascade failure scenarios."""
    
    def __init__(self, data_file: str, topology_file: str):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to pickle file containing scenarios
            topology_file: Path to pickle file containing grid topology
        """
        print(f"Loading data from {data_file}...")
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loading topology from {topology_file}...")
        with open(topology_file, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index']
            self.num_nodes = topology['num_nodes']
        
        print(f"Loaded {len(self.data)} scenarios")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single scenario."""
        scenario = self.data[idx]
        sequence = scenario['sequence']
        metadata = scenario['metadata']
        
        # Convert sequence to tensors
        node_features_seq = []
        edge_features_seq = []
        
        for timestep in sequence:
            node_features_seq.append(torch.tensor(timestep['node_features'], dtype=torch.float32))
            edge_features_seq.append(torch.tensor(timestep['edge_features'], dtype=torch.float32))
        
        # Stack into temporal sequences
        node_features = torch.stack(node_features_seq)  # [T, N, F_node]
        edge_features = torch.stack(edge_features_seq)  # [T, E, F_edge]
        
        # Labels
        cascade_label = torch.tensor([1.0 if metadata['cascade'] else 0.0], dtype=torch.float32)
        
        # Node failure labels
        node_failure_labels = torch.zeros(self.num_nodes, dtype=torch.float32)
        if metadata['cascade']:
            failed_nodes = metadata['failed_nodes']
            node_failure_labels[failed_nodes] = 1.0
        
        # Time to cascade
        time_to_cascade = torch.tensor(
            [metadata.get('time_to_cascade', -1.0)], 
            dtype=torch.float32
        )
        
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_index': self.edge_index,
            'cascade_label': cascade_label,
            'node_failure_labels': node_failure_labels,
            'time_to_cascade': time_to_cascade,
            'metadata': metadata
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching graph data."""
    # Since all graphs have the same topology, we can batch them
    node_features = torch.stack([item['node_features'] for item in batch])
    edge_features = torch.stack([item['edge_features'] for item in batch])
    edge_index = batch[0]['edge_index']  # Same for all
    
    cascade_labels = torch.stack([item['cascade_label'] for item in batch])
    node_failure_labels = torch.stack([item['node_failure_labels'] for item in batch])
    time_to_cascade = torch.stack([item['time_to_cascade'] for item in batch])
    
    return {
        'node_features': node_features,
        'edge_features': edge_features,
        'edge_index': edge_index,
        'cascade_label': cascade_labels,
        'node_failure_labels': node_failure_labels,
        'time_to_cascade': time_to_cascade
    }


class Trainer:
    """Training manager for cascade prediction model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        output_dir: str = "checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The cascade prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            output_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function
        self.criterion = PhysicsInformedLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cascade_acc': [],
            'val_cascade_acc': [],
            'train_node_acc': [],
            'val_node_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        cascade_correct = 0
        cascade_total = 0
        node_correct = 0
        node_total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            node_features = batch['node_features'].to(self.device)
            edge_features = batch['edge_features'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            cascade_labels = batch['cascade_label'].to(self.device)
            node_failure_labels = batch['node_failure_labels'].to(self.device)
            time_labels = batch['time_to_cascade'].to(self.device)
            
            batch_size, seq_len, num_nodes, node_feat_dim = node_features.shape
            x_sequence = [node_features[:, t, :, :] for t in range(seq_len)]
            
            # Create graph_properties for physics extractor
            # Note: These are averaged over time, which is fine for the extractor
            graph_properties_extractor = {
                'thermal_limits': edge_features[:, :, 1:2].mean(dim=1),
                'susceptance': edge_features[:, :, 1:2].mean(dim=1),
                'line_flows': edge_features[:, :, 2:3].mean(dim=1)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                x_sequence=x_sequence,
                edge_index=edge_index,
                edge_attr=edge_features[:, -1, :, :],  # Use last timestep edge features
                graph_properties=graph_properties_extractor
            )
            
            # Process outputs
            # failure_probability is [B, N, 1], cascade_prob should be max prob
            node_failure_prob = outputs['failure_probability']
            cascade_prob, _ = torch.max(node_failure_prob, dim=1) # [B, 1]
            time_to_cascade = outputs['failure_timing']
            final_node_features = outputs['node_embeddings']
            
            # Organize predictions and targets into dictionaries
            predictions = {
                'failure_probability': node_failure_prob,
                'failure_timing': time_to_cascade,
                'voltages': outputs['voltages'],
                'angles': outputs['angles'],
                'line_flows': outputs['line_flows']
            }
            targets = {
                'failure_label': node_failure_labels.view_as(node_failure_prob), # Ensure shape matches [B, N, 1]
                'failure_time': time_labels
            }
            
            # Re-create graph_properties to match batch size
            graph_properties_batch = {
                'edge_index': edge_index,
                'conductance': edge_features[:, -1, :, 0:1], # Use last timestep
                'susceptance': edge_features[:, -1, :, 1:2],
                'thermal_limits': edge_features[:, -1, :, 1:2] * 100 + 50, # Approximate
                'power_injection': node_features[:, -1, :, 2:3] - node_features[:, -1, :, 4:5], # P_gen - P_load
            }

            # Compute loss
            loss, loss_components = self.criterion(
                predictions=predictions,
                targets=targets,
                graph_properties=graph_properties_batch,
                prev_predictions=None # Add logic for this if needed
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Cascade accuracy
            cascade_pred = (cascade_prob > 0.5).float()
            cascade_correct += (cascade_pred == cascade_labels).sum().item()
            cascade_total += cascade_labels.size(0)
            
            # Node failure accuracy (all scenarios)
            node_pred = (node_failure_prob > 0.5).float()
            node_true = node_failure_labels.view_as(node_pred)
            node_correct += (node_pred == node_true).sum().item()
            node_total += node_true.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cascade_acc': f"{cascade_correct/cascade_total:.4f}" if cascade_total > 0 else "0.0000"
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'cascade_acc': cascade_correct / cascade_total if cascade_total > 0 else 0.0,
            'node_acc': node_correct / node_total if node_total > 0 else 0.0
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        cascade_correct = 0
        cascade_total = 0
        node_correct = 0
        node_total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                node_features = batch['node_features'].to(self.device)
                edge_features = batch['edge_features'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                cascade_labels = batch['cascade_label'].to(self.device)
                node_failure_labels = batch['node_failure_labels'].to(self.device)
                time_labels = batch['time_to_cascade'].to(self.device)
                
                batch_size, seq_len, num_nodes, node_feat_dim = node_features.shape
                x_sequence = [node_features[:, t, :, :] for t in range(seq_len)]
                
                # Create graph_properties for physics extractor
                graph_properties_extractor = {
                    'thermal_limits': edge_features[:, :, 1:2].mean(dim=1),
                    'susceptance': edge_features[:, :, 1:2].mean(dim=1),
                    'line_flows': edge_features[:, :, 2:3].mean(dim=1)
                }
                
                # Forward pass
                outputs = self.model(
                    x_sequence=x_sequence,
                    edge_index=edge_index,
                    edge_attr=edge_features[:, -1, :, :],
                    graph_properties=graph_properties_extractor
                )
                
                # Process outputs
                node_failure_prob = outputs['failure_probability']
                cascade_prob, _ = torch.max(node_failure_prob, dim=1)
                time_to_cascade = outputs['failure_timing']
                final_node_features = outputs['node_embeddings']
                
                # Organize predictions and targets into dictionaries
                predictions = {
                    'failure_probability': node_failure_prob,
                    'failure_timing': time_to_cascade,
                    'voltages': outputs['voltages'],
                    'angles': outputs['angles'],
                    'line_flows': outputs['line_flows']
                }
                targets = {
                    'failure_label': node_failure_labels.view_as(node_failure_prob), # Ensure shape matches [B, N, 1]
                    'failure_time': time_labels
                }

                # Re-create graph_properties to match batch size
                graph_properties_batch = {
                    'edge_index': edge_index,
                    'conductance': edge_features[:, -1, :, 0:1], # Use last timestep
                    'susceptance': edge_features[:, -1, :, 1:2],
                    'thermal_limits': edge_features[:, -1, :, 1:2] * 100 + 50, # Approximate
                    'power_injection': node_features[:, -1, :, 2:3] - node_features[:, -1, :, 4:5], # P_gen - P_load
                }

                # Compute loss
                loss, loss_components = self.criterion(
                    predictions=predictions,
                    targets=targets,
                    graph_properties=graph_properties_batch,
                    prev_predictions=None # Add logic for this if needed
                )
                
                total_loss += loss.item()
                
                # Cascade accuracy
                cascade_pred = (cascade_prob > 0.5).float()
                cascade_correct += (cascade_pred == cascade_labels).sum().item()
                cascade_total += cascade_labels.size(0)
                
                # Node failure accuracy
                node_pred = (node_failure_prob > 0.5).float()
                node_true = node_failure_labels.view_as(node_pred)
                node_correct += (node_pred == node_true).sum().item()
                node_total += node_true.numel()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cascade_acc': f"{cascade_correct/cascade_total:.4f}" if cascade_total > 0 else "0.0000"
                })
        
        return {
            'loss': total_loss / len(self.val_loader),
            'cascade_acc': cascade_correct / cascade_total if cascade_total > 0 else 0.0,
            'node_acc': node_correct / node_total if node_total > 0 else 0.0
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
        """
        print("\n" + "=" * 80)
        print("TRAINING CASCADE PREDICTION MODEL")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("=" * 80 + "\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_cascade_acc'].append(train_metrics['cascade_acc'])
            self.history['val_cascade_acc'].append(val_metrics['cascade_acc'])
            self.history['train_node_acc'].append(train_metrics['node_acc'])
            self.history['val_node_acc'].append(val_metrics['node_acc'])
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Cascade Acc: {train_metrics['cascade_acc']:.4f} | Val Cascade Acc: {val_metrics['cascade_acc']:.4f}")
            print(f"  Train Node Acc: {train_metrics['node_acc']:.4f} | Val Node Acc: {val_metrics['node_acc']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model and history
        self.save_checkpoint(epoch, is_best=False, filename="final_model.pt")
        self.save_training_history()
        self.plot_training_curves()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved in: {self.output_dir}")
        print()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_file}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Cascade accuracy
        axes[0, 1].plot(self.history['train_cascade_acc'], label='Train')
        axes[0, 1].plot(self.history['val_cascade_acc'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Cascade Detection Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Node failure accuracy
        axes[1, 0].plot(self.history['train_node_acc'], label='Train')
        axes[1, 0].plot(self.history['val_node_acc'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Node Failure Prediction Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot([self.optimizer.param_groups[0]['lr']] * len(self.history['train_loss']))
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_file = self.output_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {plot_file}")
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Cascade Prediction Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset = CascadeDataset(
        data_file=str(data_dir / "train_data.pkl"),
        topology_file=str(data_dir / "grid_topology.pkl")
    )
    val_dataset = CascadeDataset(
        data_file=str(data_dir / "val_data.pkl"),
        topology_file=str(data_dir / "grid_topology.pkl")
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize model
    model = CompleteCascadePredictionModel(
        node_features=45,
        edge_features=28,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_heads=args.num_attention_heads,
        dropout=0.1
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, early_stopping_patience=args.early_stopping)


if __name__ == "__main__":
    main()