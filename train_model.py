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
        
        last_timestep = sequence[-1]
        
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
        
        if 'node_labels' not in sequence[0]:
            raise ValueError("Missing 'node_labels' in scenario data. Please ensure your data generator includes it.")
            
        node_labels_sequence = torch.stack([
            torch.tensor(timestep['node_labels'], dtype=torch.float32)
            for timestep in sequence
        ])  # [T, N]
        
        # Use the last timestep labels as target
        node_failure_labels = node_labels_sequence[-1].unsqueeze(-1)  # [N, 1]
        
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
        
        frequency_sequence = torch.tensor([
            timestep['scada_data'][0, 0] if 'scada_data' in timestep and len(timestep['scada_data']) > 0 and len(timestep['scada_data'][0]) > 0 else 60.0
            for timestep in sequence
        ], dtype=torch.float32)  # [T]
        
        voltage_sequence = torch.stack([
            torch.tensor(timestep['scada_data'][:, 1:2], dtype=torch.float32)  # Voltage magnitude
            for timestep in sequence
        ])  # [T, N, 1]
        
        angle_sequence = torch.stack([
            torch.tensor(timestep['scada_data'][:, 3:4], dtype=torch.float32)  # Voltage angle
            for timestep in sequence
        ])  # [T, N, 1]
        
        line_flow_sequence = torch.stack([
            torch.tensor(timestep['edge_attr'][:, 2:3], dtype=torch.float32)  # Line loading
            for timestep in sequence
        ])  # [T, E, 1]
        
        risk_labels = torch.zeros(self.num_nodes, 3)  # [N, 3]
        
        # Overload risk: based on line loading connected to node
        line_loading = last_timestep['edge_attr'][:, 2]  # Line loading percentage
        thermal_limits = last_timestep['edge_attr'][:, 1]
        # Check for potential division by zero if thermal_limits is 0
        overload_condition = (line_loading > 0.8 * thermal_limits) if torch.any(thermal_limits > 0) else (line_loading > 0.8)
        overload_risk_edges = overload_condition.float()  # [E] - edge-level risk

        # Aggregate edge-level overload risk to node-level
        edge_index = self.edge_index
        node_overload_risk = torch.zeros(self.num_nodes)
        for i in range(len(overload_risk_edges)):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_overload_risk[src] = max(node_overload_risk[src], overload_risk_edges[i])
            node_overload_risk[dst] = max(node_overload_risk[dst], overload_risk_edges[i])

        # Voltage risk: based on voltage deviation
        voltages = last_timestep['scada_data'][:, 1]
        voltage_risk = ((voltages < 0.95) | (voltages > 1.05)).float()
        
        # Frequency risk: based on frequency deviation
        frequency = last_timestep['scada_data'][0, 0] if 'scada_data' in last_timestep and len(last_timestep['scada_data']) > 0 and len(last_timestep['scada_data'][0]) > 0 else 60.0
        frequency_risk = torch.full((self.num_nodes,), 1.0 if abs(frequency - 60.0) > 0.5 else 0.0)
        
        risk_labels[:, 0] = node_overload_risk  # Overload risk (was voltage_risk before)
        risk_labels[:, 1] = voltage_risk  # Voltage risk
        risk_labels[:, 2] = frequency_risk  # Frequency risk
        
        # Assuming relay settings are in edge_attr columns 3-6
        # Add checks for edge_attr shape to prevent index errors
        relay_labels = {
            'time_dial': torch.tensor(last_timestep['edge_attr'][:, 3] if last_timestep['edge_attr'].shape[1] > 3 else np.ones(len(last_timestep['edge_attr'])) * 0.5, dtype=torch.float32),
            'pickup_current': torch.tensor(last_timestep['edge_attr'][:, 4] if last_timestep['edge_attr'].shape[1] > 4 else np.ones(len(last_timestep['edge_attr'])) * 1.5, dtype=torch.float32),
            'operating_time': torch.tensor(last_timestep['edge_attr'][:, 5] if last_timestep['edge_attr'].shape[1] > 5 else np.ones(len(last_timestep['edge_attr'])) * 0.1, dtype=torch.float32),
            'will_operate': torch.tensor((line_loading > 1.0).astype(float), dtype=torch.float32)
        }
        
        # Resistance is in column 0, Reactance is in column 6 (or estimate from resistance)
        resistance = last_timestep['edge_attr'][:, 0]
        reactance = last_timestep['edge_attr'][:, 6] if last_timestep['edge_attr'].shape[1] > 6 else resistance * 3.0  # X ≈ 3*R for transmission lines
        
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
            
            # Graph structure
            'edge_index': self.edge_index,
            'edge_attr': torch.tensor(last_timestep['edge_attr'], dtype=torch.float32),
            
            'node_failure_labels': node_failure_labels,
            'failure_timing_labels': failure_timing_labels,
            'node_labels_sequence': node_labels_sequence,
            'frequency_labels': frequency_sequence,  # [T]
            'voltage_labels': voltage_sequence,  # [T, N, 1]
            'angle_labels': angle_sequence,  # [T, N, 1]
            'line_flow_labels': line_flow_sequence,  # [T, E, 1]
            'risk_labels': risk_labels,  # [N, 3]
            'relay_labels': relay_labels,  # Dict with relay parameters
            
            'conductance': torch.tensor(1.0 / (resistance + 1e-6), dtype=torch.float32),
            'susceptance': torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32),
            'thermal_limits': torch.tensor(last_timestep['edge_attr'][:, 1], dtype=torch.float32),
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
        
        last_timestep = sequence[-1]
        
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
        
        frequency_sequence = torch.tensor([
            timestep['scada_data'][0, 0] if 'scada_data' in timestep and len(timestep['scada_data']) > 0 and len(timestep['scada_data'][0]) > 0 else 60.0
            for timestep in sequence
        ], dtype=torch.float32)  # [T]
        
        voltage_sequence = torch.stack([
            torch.tensor(timestep['scada_data'][:, 1:2], dtype=torch.float32)  # Voltage magnitude
            for timestep in sequence
        ])  # [T, N, 1]
        
        angle_sequence = torch.stack([
            torch.tensor(timestep['scada_data'][:, 3:4], dtype=torch.float32)  # Voltage angle
            for timestep in sequence
        ])  # [T, N, 1]
        
        line_flow_sequence = torch.stack([
            torch.tensor(timestep['edge_attr'][:, 2:3], dtype=torch.float32)  # Line loading
            for timestep in sequence
        ])  # [T, E, 1]
        
        risk_labels = torch.zeros(self.num_nodes, 3)  # [N, 3]
        
        # Overload risk: based on line loading connected to node
        line_loading = last_timestep['edge_attr'][:, 2]  # Line loading percentage
        thermal_limits = last_timestep['edge_attr'][:, 1]
        overload_condition = (line_loading > 0.8 * thermal_limits) if torch.any(thermal_limits > 0) else (line_loading > 0.8)
        overload_risk_edges = overload_condition.float()  # [E] - edge-level risk

        # Aggregate edge-level overload risk to node-level
        edge_index = self.edge_index
        node_overload_risk = torch.zeros(self.num_nodes)
        for i in range(len(overload_risk_edges)):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_overload_risk[src] = max(node_overload_risk[src], overload_risk_edges[i])
            node_overload_risk[dst] = max(node_overload_risk[dst], overload_risk_edges[i])

        # Voltage risk: based on voltage deviation
        voltages = last_timestep['scada_data'][:, 1]
        voltage_risk = ((voltages < 0.95) | (voltages > 1.05)).float()
        
        # Frequency risk: based on frequency deviation
        frequency = last_timestep['scada_data'][0, 0] if 'scada_data' in last_timestep and len(last_timestep['scada_data']) > 0 and len(last_timestep['scada_data'][0]) > 0 else 60.0
        frequency_risk = torch.full((self.num_nodes,), 1.0 if abs(frequency - 60.0) > 0.5 else 0.0)
        
        risk_labels[:, 0] = node_overload_risk  # Overload risk (was voltage_risk before)
        risk_labels[:, 1] = voltage_risk  # Voltage risk
        risk_labels[:, 2] = frequency_risk  # Frequency risk
        
        relay_labels = {
            'time_dial': torch.tensor(last_timestep['edge_attr'][:, 3] if last_timestep['edge_attr'].shape[1] > 3 else np.ones(len(last_timestep['edge_attr'])) * 0.5, dtype=torch.float32),
            'pickup_current': torch.tensor(last_timestep['edge_attr'][:, 4] if last_timestep['edge_attr'].shape[1] > 4 else np.ones(len(last_timestep['edge_attr'])) * 1.5, dtype=torch.float32),
            'operating_time': torch.tensor(last_timestep['edge_attr'][:, 5] if last_timestep['edge_attr'].shape[1] > 5 else np.ones(len(last_timestep['edge_attr'])) * 0.1, dtype=torch.float32),
            'will_operate': torch.tensor((line_loading > 1.0).astype(float), dtype=torch.float32)
        }
        
        resistance = last_timestep['edge_attr'][:, 0]
        reactance = last_timestep['edge_attr'][:, 6] if last_timestep['edge_attr'].shape[1] > 6 else resistance * 3.0
        
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
            'frequency_labels': frequency_sequence,  # [T]
            'voltage_labels': voltage_sequence,  # [T, N, 1]
            'angle_labels': angle_sequence,  # [T, N, 1]
            'line_flow_labels': line_flow_sequence,  # [T, E, 1]
            'risk_labels': risk_labels,  # [N, 3]
            'relay_labels': relay_labels,  # Dict with relay parameters
            
            'conductance': torch.tensor(1.0 / (resistance + 1e-6), dtype=torch.float32),
            'susceptance': torch.tensor(1.0 / (reactance + 1e-6), dtype=torch.float32),
            'thermal_limits': torch.tensor(last_timestep['edge_attr'][:, 1], dtype=torch.float32),
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
        
        # Labels
        'node_failure_labels': torch.stack([item['node_failure_labels'] for item in batch]),
        'failure_timing_labels': torch.stack([item['failure_timing_labels'] for item in batch]),
        'node_labels_sequence': torch.stack([item['node_labels_sequence'] for item in batch]),
        'frequency_labels': torch.stack([item['frequency_labels'] for item in batch]),
        'voltage_labels': torch.stack([item['voltage_labels'] for item in batch]),
        'angle_labels': torch.stack([item['angle_labels'] for item in batch]),
        'line_flow_labels': torch.stack([item['line_flow_labels'] for item in batch]),
        'risk_labels': torch.stack([item['risk_labels'] for item in batch]),
        'relay_labels': {
            'time_dial': torch.stack([item['relay_labels']['time_dial'] for item in batch]),
            'pickup_current': torch.stack([item['relay_labels']['pickup_current'] for item in batch]),
            'operating_time': torch.stack([item['relay_labels']['operating_time'] for item in batch]),
            'will_operate': torch.stack([item['relay_labels']['will_operate'] for item in batch])
        },
        
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
        learning_rate: float = 0.0003,
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
            'val_node_acc': [],
            'learning_rates': []  # Track learning rates for plotting
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        self.scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
        
        self.warmup_epochs = 3
        from torch.optim.lr_scheduler import LinearLR
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # Start at 10% of base LR
            total_iters=self.warmup_epochs
        )
        
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
            
            if 'relay_labels' in batch_device and isinstance(batch_device['relay_labels'], dict):
                batch_device['relay_labels'] = {
                    k: v.to(self.device) for k, v in batch_device['relay_labels'].items()
                }
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_device, return_sequence=True)
                    
                    targets = {
                        'failure_label': batch_device['node_failure_labels'],
                        'failure_time': batch_device['failure_timing_labels'],
                        'frequency': batch_device['frequency_labels'],
                        'voltage': batch_device['voltage_labels'],
                        'angle': batch_device['angle_labels'],
                        'line_flow': batch_device['line_flow_labels'],
                        'risk': batch_device['risk_labels'],
                        'relay': batch_device['relay_labels']
                    }
                    
                    graph_properties = {
                        'edge_index': batch_device['edge_index'],
                        'conductance': batch_device['conductance'],
                        'susceptance': batch_device['susceptance'],
                        'thermal_limits': batch_device['thermal_limits'],
                        'power_injection': batch_device['power_injection']
                    }
                    
                    loss, loss_components = self.model.compute_loss(outputs, targets, graph_properties)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_device, return_sequence=True)
                
                targets = {
                    'failure_label': batch_device['node_failure_labels'],
                    'failure_time': batch_device['failure_timing_labels'],
                    'frequency': batch_device['frequency_labels'],
                    'voltage': batch_device['voltage_labels'],
                    'angle': batch_device['angle_labels'],
                    'line_flow': batch_device['line_flow_labels'],
                    'risk': batch_device['risk_labels'],
                    'relay': batch_device['relay_labels']
                }
                
                graph_properties = {
                    'edge_index': batch_device['edge_index'],
                    'conductance': batch_device['conductance'],
                    'susceptance': batch_device['susceptance'],
                    'thermal_limits': batch_device['thermal_limits'],
                    'power_injection': batch_device['power_injection']
                }
                
                loss, loss_components = self.model.compute_loss(outputs, targets, graph_properties)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
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
                
                if 'relay_labels' in batch_device and isinstance(batch_device['relay_labels'], dict):
                    batch_device['relay_labels'] = {
                        k: v.to(self.device) for k, v in batch_device['relay_labels'].items()
                    }
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_device, return_sequence=True)
                        
                        targets = {
                            'failure_label': batch_device['node_failure_labels'],
                            'failure_time': batch_device['failure_timing_labels'],
                            'frequency': batch_device['frequency_labels'],
                            'voltage': batch_device['voltage_labels'],
                            'angle': batch_device['angle_labels'],
                            'line_flow': batch_device['line_flow_labels'],
                            'risk': batch_device['risk_labels'],
                            'relay': batch_device['relay_labels']
                        }
                        
                        graph_properties = {
                            'edge_index': batch_device['edge_index'],
                            'conductance': batch_device['conductance'],
                            'susceptance': batch_device['susceptance'],
                            'thermal_limits': batch_device['thermal_limits'],
                            'power_injection': batch_device['power_injection']
                        }
                        
                        loss, _ = self.model.compute_loss(outputs, targets, graph_properties)
                else:
                    outputs = self.model(batch_device, return_sequence=True)
                    
                    targets = {
                        'failure_label': batch_device['node_failure_labels'],
                        'failure_time': batch_device['failure_timing_labels'],
                        'frequency': batch_device['frequency_labels'],
                        'voltage': batch_device['voltage_labels'],
                        'angle': batch_device['angle_labels'],
                        'line_flow': batch_device['line_flow_labels'],
                        'risk': batch_device['risk_labels'],
                        'relay': batch_device['relay_labels']
                    }
                    
                    graph_properties = {
                        'edge_index': batch_device['edge_index'],
                        'conductance': batch_device['conductance'],
                        'susceptance': batch_device['susceptance'],
                        'thermal_limits': batch_device['thermal_limits'],
                        'power_injection': batch_device['power_injection']
                    }
                    
                    loss, _ = self.model.compute_loss(outputs, targets, graph_properties)
                
                total_loss += loss.item()
                
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
            
            # Apply learning rate warmup if enabled
            if epoch < self.warmup_epochs and self.warmup_scheduler is not None:
                self.warmup_scheduler.step()

            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate (only after warmup if applicable)
            if epoch >= self.warmup_epochs:
                self.scheduler.step(val_metrics['loss'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
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
            print(f"  Learning Rate: {current_lr:.6f}")  # Print current LR
            
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
        if 'learning_rates' in self.history and len(self.history['learning_rates']) > 0:
            epochs = range(1, len(self.history['learning_rates']) + 1)
            axes[1, 1].plot(epochs, self.history['learning_rates'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')  # Log scale for better visualization
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
    parser.add_argument("--train_data", type=str, default=None, help="Training data directory")
    parser.add_argument("--val_data", type=str, default=None, help="Validation data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
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
    print("Using UnifiedCascadePredictionModel")
    
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
