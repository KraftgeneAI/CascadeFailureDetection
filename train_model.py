"""
Cascade Failure Prediction Model Training Script
Complete training pipeline with data loading, training, validation, and model saving.
Updated to work with unified multi-modal model and batch-based data generation.

Author: Kraftgene AI Inc.
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import glob

from multimodal_cascade_model import (
    UnifiedCascadePredictionModel,
    PhysicsInformedLoss
)


class BatchStreamingDataset(Dataset):
    """
    Memory-efficient dataset that loads batches on-demand.
    Only keeps batch file paths in memory, loads data when needed.
    """
    
    def __init__(self, batch_dir: str, topology_file: str):
        """
        Initialize streaming dataset.
        
        Args:
            batch_dir: Directory containing batch files
            topology_file: Path to pickle file containing grid topology
        """
        self.batch_dir = Path(batch_dir)
        
        self.batch_files = sorted(self.batch_dir.glob("batch_*.pkl"))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        print(f"Found {len(self.batch_files)} batch files in {batch_dir}")
        
        # Load batch info
        info_file = self.batch_dir / "batch_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.batch_info = json.load(f)
                self.total_scenarios = self.batch_info['total_scenarios']
        else:
            # Count scenarios by loading batch files
            print("Counting scenarios...")
            self.total_scenarios = 0
            for batch_file in self.batch_files:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    self.total_scenarios += len(batch_data)
        
        # Load topology
        print(f"Loading topology from {topology_file}...")
        with open(topology_file, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index']
            self.num_nodes = topology['num_nodes']
        
        self.scenario_index = []
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                for i in range(len(batch_data)):
                    self.scenario_index.append((batch_file, i))
        
        print(f"Indexed {len(self.scenario_index)} scenarios")
    
    def __len__(self) -> int:
        return len(self.scenario_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single scenario on-demand.
        This avoids loading all data into memory.
        """
        batch_file, idx_in_batch = self.scenario_index[idx]
        
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            scenario = batch_data[idx_in_batch]
        
        sequence = scenario['sequence']
        metadata = scenario['metadata']
        
        sequence_length = len(sequence)
        
        # Stack all timesteps for temporal processing
        satellite_sequence = torch.stack([
            torch.tensor(timestep['satellite_data'], dtype=torch.float32) 
            for timestep in sequence
        ])  # [T, N, C, H, W]
        
        weather_sequences = torch.stack([
            torch.tensor(timestep['weather_sequence'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, seq_len, features]
        
        threat_sequence = torch.stack([
            torch.tensor(timestep['threat_indicators'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, features]
        
        scada_sequence = torch.stack([
            torch.tensor(timestep['scada_data'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, features]
        
        pmu_sequences = torch.stack([
            torch.tensor(timestep['pmu_sequence'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, pmu_seq_len, features]
        
        equipment_sequence = torch.stack([
            torch.tensor(timestep['equipment_status'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, features]
        
        visual_sequence = torch.stack([
            torch.tensor(timestep['visual_data'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, C, H, W]
        
        thermal_sequence = torch.stack([
            torch.tensor(timestep['thermal_data'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, C, H, W]
        
        sensor_sequence = torch.stack([
            torch.tensor(timestep['sensor_data'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N, features]
        
        edge_attr_sequence = torch.stack([
            torch.tensor(timestep['edge_attr'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, E, features]
        
        # Assuming 'node_labels' are present in each timestep of 'scenario'
        # If not, this part will need adjustment based on actual data structure
        if 'node_labels' not in sequence[0]:
            raise ValueError("Missing 'node_labels' in scenario data. Please ensure your data generator includes it.")
            
        node_labels_sequence = torch.stack([
            torch.tensor(timestep['node_labels'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N]
        
        # Use the last timestep labels as target
        node_failure_labels = node_labels_sequence[-1].unsqueeze(-1)  # [N, 1]
        
        # For nodes that fail, compute when they fail
        failure_timing_labels = torch.zeros(self.num_nodes, 1)
        cascade_start = metadata.get('cascade_start_time', -1)
        
        if cascade_start >= 0:
            # Find when each node first fails
            for node_idx in range(self.num_nodes):
                node_failure_times = []
                for t, timestep in enumerate(sequence):
                    # Ensure 'node_labels' exists and is a list/array of correct size
                    if node_idx < len(timestep['node_labels']) and timestep['node_labels'][node_idx] > 0.5:
                        # Node failed at timestep t
                        time_minutes = (t - cascade_start) * 2.0 / 60.0  # Convert to minutes
                        node_failure_times.append(max(0, time_minutes))
                        break
                
                if node_failure_times:
                    failure_timing_labels[node_idx, 0] = node_failure_times[0]
                else:
                    failure_timing_labels[node_idx, 0] = -1.0  # Did not fail
        
        # Use last timestep for edge attributes
        last_timestep = sequence[-1]
        
        edge_attr_np = last_timestep['edge_attr']
        reactance = edge_attr_np[:, 0]
        thermal_limits_np = edge_attr_np[:, 1]
        
        # Compute conductance and susceptance (1/R and 1/X)
        conductance = torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32)
        susceptance = torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32)
        thermal_limits = torch.tensor(thermal_limits_np, dtype=torch.float32)
        
        return {
            'satellite_sequence': satellite_sequence,  # [T, N, C, H, W]
            'weather_sequence': weather_sequences,  # [T, N, seq_len, features]
            'threat_sequence': threat_sequence,  # [T, N, features]
            'scada_sequence': scada_sequence,  # [T, N, features]
            'pmu_sequence': pmu_sequences,  # [T, N, pmu_seq_len, features]
            'equipment_sequence': equipment_sequence,  # [T, N, features]
            'visual_sequence': visual_sequence,  # [T, N, C, H, W]
            'thermal_sequence': thermal_sequence,  # [T, N, C, H, W]
            'sensor_sequence': sensor_sequence,  # [T, N, features]
            'edge_attr_sequence': edge_attr_sequence,  # [T, E, features]
            
            # Graph structure
            'edge_index': self.edge_index,
            'edge_attr': torch.tensor(last_timestep['edge_attr'], dtype=torch.float32),
            
            'node_failure_labels': node_failure_labels,  # [N, 1]
            'failure_timing_labels': failure_timing_labels,  # [N, 1]
            'node_labels_sequence': node_labels_sequence,  # [T, N] for sequence prediction
            
            # Physics properties from last timestep
            'conductance': conductance,
            'susceptance': susceptance,
            'thermal_limits': thermal_limits,
            'power_injection': torch.tensor(last_timestep['scada_data'][:, 2:3], dtype=torch.float32),
            
            'metadata': metadata,
            'sequence_length': sequence_length
        }


class CascadeDataset(Dataset):
    """PyTorch Dataset for cascade failure scenarios with batch file support."""
    
    def __init__(self, data_path: str, topology_file: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file or directory containing batch files
            topology_file: Path to pickle file containing grid topology
        """
        data_path = Path(data_path)
        
        if data_path.is_dir() and (data_path / "batch_info.json").exists():
            print(f"Using memory-efficient batch streaming from: {data_path}")
            # Use the streaming dataset instead
            self._use_streaming = True
            self._streaming_dataset = BatchStreamingDataset(str(data_path), topology_file)
            return
        
        self._use_streaming = False
        
        if data_path.is_file():
            # Single combined file
            print(f"Loading data from single file: {data_path}...")
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif data_path.is_dir():
            # Load from batch files (old method - loads all into memory)
            print(f"Loading data from batch directory: {data_path}...")
            batch_files = sorted(glob.glob(str(data_path / "batch_*.pkl")))
            if not batch_files:
                raise ValueError(f"No batch files found in {data_path}")
            
            print(f"Found {len(batch_files)} batch files")
            self.data = []
            for batch_file in tqdm(batch_files, desc="Loading batches"):
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    self.data.extend(batch_data)
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
        
        print(f"Loading topology from {topology_file}...")
        with open(topology_file, 'rb') as f:
            topology = pickle.load(f)
            self.edge_index = topology['edge_index']
            self.num_nodes = topology['num_nodes']
        
        print(f"Loaded {len(self.data)} scenarios")
    
    def __len__(self) -> int:
        if self._use_streaming:
            return len(self._streaming_dataset)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single scenario."""
        if self._use_streaming:
            return self._streaming_dataset[idx]
        
        scenario = self.data[idx]
        sequence = scenario['sequence']
        metadata = scenario['metadata']
        
        sequence_length = len(sequence)
        
        satellite_sequence = torch.stack([
            torch.tensor(timestep['satellite_data'], dtype=torch.float32) 
            for timestep in sequence
        ])
        
        weather_sequences = torch.stack([
            torch.tensor(timestep['weather_sequence'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        threat_sequence = torch.stack([
            torch.tensor(timestep['threat_indicators'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        scada_sequence = torch.stack([
            torch.tensor(timestep['scada_data'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        pmu_sequences = torch.stack([
            torch.tensor(timestep['pmu_sequence'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        equipment_sequence = torch.stack([
            torch.tensor(timestep['equipment_status'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        visual_sequence = torch.stack([
            torch.tensor(timestep['visual_data'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        thermal_sequence = torch.stack([
            torch.tensor(timestep['thermal_data'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        sensor_sequence = torch.stack([
            torch.tensor(timestep['sensor_data'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        edge_attr_sequence = torch.stack([
            torch.tensor(timestep['edge_attr'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        if 'node_labels' not in sequence[0]:
            raise ValueError("Missing 'node_labels' in scenario data. Please ensure your data generator includes it.")
            
        node_labels_sequence = torch.stack([
            torch.tensor(timestep['node_labels'], dtype=torch.float32)
            for timestep in sequence
        ])
        
        node_failure_labels = node_labels_sequence[-1].unsqueeze(-1)
        
        failure_timing_labels = torch.zeros(self.num_nodes, 1)
        cascade_start = metadata.get('cascade_start_time', -1)
        
        if cascade_start >= 0:
            for node_idx in range(self.num_nodes):
                for t, timestep in enumerate(sequence):
                    if node_idx < len(timestep['node_labels']) and timestep['node_labels'][node_idx] > 0.5:
                        time_minutes = (t - cascade_start) * 2.0 / 60.0
                        failure_timing_labels[node_idx, 0] = max(0, time_minutes)
                        break
                else:
                    failure_timing_labels[node_idx, 0] = -1.0
        
        last_timestep = sequence[-1]
        
        edge_attr_np = last_timestep['edge_attr']
        reactance = edge_attr_np[:, 0]
        thermal_limits_np = edge_attr_np[:, 1]
        
        # Compute conductance and susceptance (1/R and 1/X)
        conductance = torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32)
        susceptance = torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32)
        thermal_limits = torch.tensor(thermal_limits_np, dtype=torch.float32)
        
        return {
            'satellite_sequence': satellite_sequence,
            'weather_sequence': weather_sequences,
            'threat_sequence': threat_sequence,
            'scada_sequence': scada_sequence,
            'pmu_sequence': pmu_sequences,
            'equipment_sequence': equipment_sequence,
            'visual_sequence': visual_sequence,
            'thermal_sequence': thermal_sequence,
            'sensor_sequence': sensor_sequence,
            'edge_attr_sequence': edge_attr_sequence,
            
            'edge_index': self.edge_index,
            'edge_attr': torch.tensor(last_timestep['edge_attr'], dtype=torch.float32),
            
            'node_failure_labels': node_failure_labels,
            'failure_timing_labels': failure_timing_labels,
            'node_labels_sequence': node_labels_sequence,
            
            'conductance': conductance,
            'susceptance': susceptance,
            'thermal_limits': thermal_limits,
            'power_injection': torch.tensor(last_timestep['scada_data'][:, 2:3], dtype=torch.float32),
            
            'metadata': metadata,
            'sequence_length': sequence_length
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching graph data."""
    return {
        'satellite_sequence': torch.stack([item['satellite_sequence'] for item in batch]),
        'weather_sequence': torch.stack([item['weather_sequence'] for item in batch]),
        'threat_sequence': torch.stack([item['threat_sequence'] for item in batch]),
        'scada_sequence': torch.stack([item['scada_sequence'] for item in batch]),
        'pmu_sequence': torch.stack([item['pmu_sequence'] for item in batch]),
        'equipment_sequence': torch.stack([item['equipment_sequence'] for item in batch]),
        'visual_sequence': torch.stack([item['visual_sequence'] for item in batch]),
        'thermal_sequence': torch.stack([item['thermal_sequence'] for item in batch]),
        'sensor_sequence': torch.stack([item['sensor_sequence'] for item in batch]),
        'edge_attr_sequence': torch.stack([item['edge_attr_sequence'] for item in batch]),
        
        # Graph
        'edge_index': batch[0]['edge_index'],
        'edge_attr': torch.stack([item['edge_attr'] for item in batch]),
        
        'node_failure_labels': torch.stack([item['node_failure_labels'] for item in batch]),
        'failure_timing_labels': torch.stack([item['failure_timing_labels'] for item in batch]),
        'node_labels_sequence': torch.stack([item['node_labels_sequence'] for item in batch]),
        
        # Physics
        'conductance': torch.stack([item['conductance'] for item in batch]),
        'susceptance': torch.stack([item['susceptance'] for item in batch]),
        'thermal_limits': torch.stack([item['thermal_limits'] for item in batch]),
        'power_injection': torch.stack([item['power_injection'] for item in batch]),
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
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            # Extract last timestep from sequences and rename keys
            B, T, N = batch_device['satellite_sequence'].shape[:3]
            
            model_batch = {
                # Extract last timestep from sequences
                'satellite_data': batch_device['satellite_sequence'][:, -1],  # [B, N, C, H, W]
                'weather_sequence': batch_device['weather_sequence'][:, -1],  # [B, N, seq_len, features]
                'threat_indicators': batch_device['threat_sequence'][:, -1],  # [B, N, features]
                'scada_data': batch_device['scada_sequence'][:, -1],  # [B, N, features]
                'pmu_sequence': batch_device['pmu_sequence'][:, -1],  # [B, N, pmu_seq_len, features]
                'equipment_status': batch_device['equipment_sequence'][:, -1],  # [B, N, features]
                'visual_data': batch_device['visual_sequence'][:, -1],  # [B, N, C, H, W]
                'thermal_data': batch_device['thermal_sequence'][:, -1],  # [B, N, C, H, W]
                'sensor_data': batch_device['sensor_sequence'][:, -1],  # [B, N, features]
                
                # Add dummy positions and timestamps (model requires them)
                'positions': torch.randn(B, N, 2, device=self.device),  # Dummy lat/lon
                'timestamps': torch.zeros(B, N, device=self.device),  # All at same time
                
                # Graph structure
                'edge_index': batch_device['edge_index'],
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(model_batch)
            
            targets = {
                'failure_label': batch_device['node_failure_labels'],  # [B, N, 1]
                'failure_time': batch_device['failure_timing_labels']  # [B, N, 1]
            }
            
            graph_properties = {
                'edge_index': batch_device['edge_index'],
                'conductance': batch_device['conductance'],
                'susceptance': batch_device['susceptance'],
                'thermal_limits': batch_device['thermal_limits'],
                'power_injection': batch_device['power_injection']
            }
            
            # Compute loss
            loss, loss_components = self.model.compute_loss(outputs, targets, graph_properties)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            cascade_prob = outputs['failure_probability'].max(dim=1)[0]  # [B, 1]
            cascade_pred = (cascade_prob > 0.5).float()
            # Assuming node_failure_labels has shape [B, N, 1], taking max over N gives [B]
            cascade_labels = (batch_device['node_failure_labels'].max(dim=1)[0] > 0.5).float()
            cascade_correct += (cascade_pred == cascade_labels).sum().item()
            cascade_total += cascade_labels.size(0)
            
            node_pred = (outputs['failure_probability'] > 0.5).float()  # [B, N, 1]
            node_labels = batch_device['node_failure_labels']  # [B, N, 1]
            node_correct += (node_pred == node_labels).sum().item()
            node_total += node_labels.numel()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cascade_acc': f"{cascade_correct/cascade_total:.4f}" if cascade_total > 0 else "0.0000",
                'node_acc': f"{node_correct/node_total:.4f}" if node_total > 0 else "0.0000"
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
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                B, T, N = batch_device['satellite_sequence'].shape[:3]
                
                model_batch = {
                    'satellite_data': batch_device['satellite_sequence'][:, -1],
                    'weather_sequence': batch_device['weather_sequence'][:, -1],
                    'threat_indicators': batch_device['threat_sequence'][:, -1],
                    'scada_data': batch_device['scada_sequence'][:, -1],
                    'pmu_sequence': batch_device['pmu_sequence'][:, -1],
                    'equipment_status': batch_device['equipment_sequence'][:, -1],
                    'visual_data': batch_device['visual_sequence'][:, -1],
                    'thermal_data': batch_device['thermal_sequence'][:, -1],
                    'sensor_data': batch_device['sensor_sequence'][:, -1],
                    'positions': torch.randn(B, N, 2, device=self.device),
                    'timestamps': torch.zeros(B, N, device=self.device),
                    'edge_index': batch_device['edge_index'],
                }
                
                # Forward pass
                outputs = self.model(model_batch)
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'failure_time': batch_device['failure_timing_labels']
                }
                
                graph_properties = {
                    'edge_index': batch_device['edge_index'],
                    'conductance': batch_device['conductance'],
                    'susceptance': batch_device['susceptance'],
                    'thermal_limits': batch_device['thermal_limits'],
                    'power_injection': batch_device['power_injection']
                }
                
                # Compute loss
                loss, _ = self.model.compute_loss(outputs, targets, graph_properties)
                
                total_loss += loss.item()
                
                # Cascade accuracy
                cascade_prob = outputs['failure_probability'].max(dim=1)[0]
                cascade_pred = (cascade_prob > 0.5).float()
                cascade_labels = (batch_device['node_failure_labels'].max(dim=1)[0] > 0.5).float()
                cascade_correct += (cascade_pred == cascade_labels).sum().item()
                cascade_total += cascade_labels.size(0)
                
                node_pred = (outputs['failure_probability'] > 0.5).float()
                node_labels = batch_device['node_failure_labels']
                node_correct += (node_pred == node_labels).sum().item()
                node_total += node_labels.numel()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cascade_acc': f"{cascade_correct/cascade_total:.4f}" if cascade_total > 0 else "0.0000",
                    'node_acc': f"{node_correct/node_total:.4f}" if node_total > 0 else "0.0000"
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
    parser.add_argument("--data_dir", type=str, default="data_unified", help="Data directory")
    parser.add_argument("--train_data", type=str, default=None, help="Training data directory")
    parser.add_argument("--val_data", type=str, default=None, help="Validation data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Debug output to check data paths
    print(f"\n[DEBUG] Checking data directory: {data_dir}")
    print(f"[DEBUG] Directory exists: {data_dir.exists()}")
    if data_dir.exists():
        print(f"[DEBUG] Contents: {list(data_dir.iterdir())}")
    
    if args.train_data:
        train_data_path = args.train_data
    elif (data_dir / "train_batches").exists():
        train_data_path = str(data_dir / "train_batches")
    else:
        print(f"[DEBUG] train_batches not found, checking alternatives...")
        if (data_dir / "train").exists():
            train_data_path = str(data_dir / "train")
            print(f"[DEBUG] Using train directory: {train_data_path}")
        elif (data_dir / "batches").exists():
            train_data_path = str(data_dir / "batches")
            print(f"[DEBUG] Using batches directory: {train_data_path}")
        else:
            raise ValueError(f"No training data found in {data_dir}. Please generate data first using unified_data_generator.py")
    
    if args.val_data:
        val_data_path = args.val_data
    elif (data_dir / "val_batches").exists():
        val_data_path = str(data_dir / "val_batches")
    else:
        print(f"[DEBUG] val_batches not found, using train data for validation...")
        val_data_path = train_data_path
    
    topology_path = str(data_dir / "grid_topology.pkl")
    
    if not Path(topology_path).exists():
        raise ValueError(f"Topology file not found: {topology_path}. Please generate data first using unified_data_generator.py")
    
    print(f"\nData Configuration:")
    print(f"  Training data: {train_data_path}")
    print(f"  Validation data: {val_data_path}")
    print(f"  Topology: {topology_path}\n")
    
    try:
        print("[DEBUG] Loading training dataset...")
        train_dataset = BatchStreamingDataset(
            batch_dir=train_data_path,
            topology_file=topology_path
        )
        print(f"[DEBUG] Training dataset loaded: {len(train_dataset)} scenarios")
        
        print("[DEBUG] Loading validation dataset...")
        val_dataset = BatchStreamingDataset(
            batch_dir=val_data_path,
            topology_file=topology_path
        )
        print(f"[DEBUG] Validation dataset loaded: {len(val_dataset)} scenarios")
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Please generate data first using unified_data_generator.py")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty! Please generate data first using unified_data_generator.py")
    
    # Create data loaders
    print("[DEBUG] Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for streaming dataset
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"[DEBUG] Train loader batches: {len(train_loader)}")
    print(f"[DEBUG] Val loader batches: {len(val_loader)}")
    
    print("[DEBUG] Testing batch loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"[DEBUG] Successfully loaded test batch with keys: {test_batch.keys()}")
        print(f"[DEBUG] Batch shapes:")
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load test batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("[DEBUG] Initializing model...")
    model = UnifiedCascadePredictionModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        heads=args.num_heads,
        dropout=0.1
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    print("[DEBUG] Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Train
    print("[DEBUG] Starting training...")
    trainer.train(num_epochs=args.num_epochs, early_stopping_patience=args.early_stopping)


if __name__ == "__main__":
    main()
