"""
Loss Weight Calibration Utility
================================

Provides dynamic loss weight calibration to balance physics-informed loss components.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm

# Import the loss function
from cascade_prediction.models import PhysicsInformedLoss
from cascade_prediction.data.generator.config import Settings

def calibrate_loss_weights(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = Settings.Loss.CALIB_NUM_BATCHES,
    model_outputs_logits: bool = False,
    base_mva: float = Settings.Dataset.BASE_MVA,
    base_freq: float = Settings.Dataset.BASE_FREQUENCY
) -> Dict[str, float]:
    """
    Run a few batches to find the average raw loss for each component.
    
    This function:
    1. Creates a dummy criterion with all weights = 1.0
    2. Runs forward passes on a few batches
    3. Measures the raw (unweighted) loss magnitude for each component
    4. Calculates balanced lambda weights based on those magnitudes
    
    Parameters:
    -----------
    model : nn.Module
        The model to calibrate with
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function class (will create dummy instance)
    device : torch.device
        Device to run on
    num_batches : int
        Number of batches to use for calibration
    model_outputs_logits : bool
        Whether model outputs logits or probabilities
    base_mva : float
        Base MVA for physics normalization
    base_freq : float
        Base frequency for physics normalization
    
    Returns:
    --------
    calibrated_lambdas : Dict[str, float]
        Dictionary of calibrated lambda weights
    """
    print("\n" + "="*80)
    print("STARTING DYNAMIC LOSS WEIGHT CALIBRATION")
    print("="*80)
    print(f"Running loss calibration for {num_batches} batches...")
    
    # Set model to eval mode for calibration
    model.eval()
    
    # Create dummy criterion with calibration weights (matching original train_model.py)
    dummy_criterion = PhysicsInformedLoss(
        lambda_prediction=Settings.Loss.CALIB_LAMBDA_PREDICTION,
        lambda_powerflow=1.0,
        lambda_temperature=1.0,
        lambda_frequency=1.0,
        lambda_reactive=1.0,
        lambda_risk=1.0,
        lambda_timing=1.0,
        lambda_active_flow=1.0,
        lambda_voltage=1.0,
        lambda_capacity=1.0,
        pos_weight=1.0,
        use_logits=model_outputs_logits,
        base_mva=base_mva,
        base_freq=base_freq
    )
    
    loss_sums = {}
    total_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            
            # Move batch to device
            batch_device = {}
            for k, v in batch.items():
                if k == 'graph_properties':
                    batch_device[k] = {
                        prop_k: prop_v.to(device) if isinstance(prop_v, torch.Tensor) else prop_v
                        for prop_k, prop_v in v.items()
                    }
                elif isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v
            
            if 'node_failure_labels' not in batch_device:
                continue
            
            # Forward pass
            outputs = model(batch_device, return_sequence=True)
            
            # Prepare targets and graph_properties (matching original train_model.py)
            graph_properties = batch_device.get('graph_properties', {})
            if 'edge_index' not in graph_properties:
                graph_properties['edge_index'] = batch_device['edge_index']
            
            targets = {
                'failure_label': batch_device['node_failure_labels'],
                'ground_truth_risk': batch_device.get('ground_truth_risk'),
                'cascade_timing': batch_device.get('cascade_timing'),
                # Extract Voltage (Feature 0) from the LAST timestep (-1) of the SCADA sequence
                'voltages': batch_device['scada_data'][:, -1, :, 0:1] if 'scada_data' in batch_device else None,
                'node_reactive_power': batch_device['scada_data'][:, -1, :, 3:4] if 'scada_data' in batch_device else None,
                'line_reactive_power': batch_device['edge_attr'][:, :, 6:7] if 'edge_attr' in batch_device else None,
                'active_power_line_flows': batch_device['edge_attr'][:, :, 5:6] if 'edge_attr' in batch_device else None,
            }
            
            # Extract edge mask
            edge_mask = batch_device.get('edge_mask')
            if edge_mask is not None and edge_mask.dim() == 3:
                # Take the last timestep to match the model's prediction
                edge_mask = edge_mask[:, -1, :]
            
            # Calculate loss components
            try:
                _, loss_components = dummy_criterion(
                    outputs,
                    targets,
                    graph_properties,
                    edge_mask=edge_mask
                )
                
                # Accumulate loss components
                for key, val in loss_components.items():
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    if not np.isnan(val) and not np.isinf(val):
                        loss_sums[key] = loss_sums.get(key, 0.0) + val
                
                total_batches += 1
            except Exception as e:
                print(f"[WARNING] Error during calibration batch {i}: {e}")
                continue
    
    if total_batches == 0:
        print("[ERROR] Calibration failed: No data loaded.")
        print("="*80)
        return {}
    
    # Calculate average losses
    avg_losses = {key: val / total_batches for key, val in loss_sums.items()}
    
    print("  Average raw loss components (unweighted):")
    for key, val in sorted(avg_losses.items()):
        print(f"    {key: <20}: {val:10.6f}")
    
    print("="*80)
    print("CALIBRATION COMPLETE")
    print("="*80 + "\n")
    
    # Calculate balanced lambda weights
    print("Balancing loss weights dynamically...")
    
    # Use prediction loss as target magnitude
    target_magnitude = avg_losses.get('prediction', 0.1)
    if target_magnitude < Settings.Loss.CALIB_MIN_MAGNITUDE:
        target_magnitude = Settings.Loss.CALIB_MIN_MAGNITUDE

    calibrated_lambdas = {}

    # Add prediction lambda (boosted weight for calibration)
    calibrated_lambdas['lambda_prediction'] = Settings.Loss.CALIB_LAMBDA_PREDICTION
    
    # Define which loss components to calibrate (matching original train_model.py)
    physics_loss_keys = [
        'powerflow', 'temperature', 'voltage', 'frequency', 
        'reactive', 'risk', 'timing',  
        'active_flow', 'capacity'
    ]
    
    for key in physics_loss_keys:
        raw_loss = avg_losses.get(key, 0.0)
        
        # Calculate lambda to balance this component with prediction loss
        if raw_loss >= 1e-6:
            lambda_val = target_magnitude / raw_loss
        else:
            lambda_val = 1.0  # Default if component is too small
        
        # Store with lambda_ prefix
        lambda_key = f"lambda_{key}"
        calibrated_lambdas[lambda_key] = lambda_val
    
    # Print calibration report
    print(f"  Target Magnitude (from prediction loss): {target_magnitude:.4f}")
    print("\n  Final Loss Weights (Fully Dynamic):")
    print(f"  {'Component':<20} | {'Raw Loss':<12} | {'Final Lambda':<12} | {'Weighted Loss'}")
    print(f"  {'-'*20} | {'-'*12} | {'-'*12} | {'-'*20}")
    
    # Print prediction first
    pred_raw = avg_losses.get('prediction', 0.0)
    pred_lambda = calibrated_lambdas.get('lambda_prediction', 1.0)
    print(f"  {'prediction':<20} | {pred_raw:<12.4f} | {pred_lambda:<12.4f} | {pred_raw * pred_lambda:12.4f}")
    
    # Print physics losses
    for key in physics_loss_keys:
        raw = avg_losses.get(key, 0.0)
        lambda_key = f"lambda_{key}"
        final = calibrated_lambdas.get(lambda_key, 0.0)
        weighted = raw * final
        print(f"  {key:<20} | {raw:<12.4f} | {final:<12.4f} | {weighted:12.4f}")
    
    print(f"\n✓ Loss weights calibrated successfully.")
    
    # Set model back to train mode
    model.train()
    
    return calibrated_lambdas


def apply_calibrated_weights(
    base_lambdas: Dict[str, float],
    calibrated_lambdas: Dict[str, float],
    scaling_factor: float = 1.0
) -> Dict[str, float]:
    """
    Apply calibrated weights to base lambda values.
    
    Parameters:
    -----------
    base_lambdas : Dict[str, float]
        Base lambda values (from command-line args or defaults)
    calibrated_lambdas : Dict[str, float]
        Calibrated lambda values from calibration
    scaling_factor : float
        Optional scaling factor to adjust all physics losses
    
    Returns:
    --------
    final_lambdas : Dict[str, float]
        Final lambda values to use
    """
    final_lambdas = {}
    
    for key, base_val in base_lambdas.items():
        if key in calibrated_lambdas:
            # Use calibrated value
            final_lambdas[key] = calibrated_lambdas[key] * scaling_factor
        else:
            # Keep base value
            final_lambdas[key] = base_val * scaling_factor
    
    return final_lambdas
