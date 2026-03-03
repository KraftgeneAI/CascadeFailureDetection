# Cascade Prediction System - Modular Architecture

## Overview

This package provides a complete, modular implementation of a physics-informed deep learning system for power grid cascade failure prediction.

## Package Structure

```
cascade_prediction/
├── __init__.py                 # Main package exports
├── README.md                   # This file
│
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── unified_model.py        # Main model class
│   │
│   ├── embeddings/             # Multi-modal embeddings
│   │   ├── __init__.py
│   │   ├── environmental.py    # Satellite, weather, threat data
│   │   ├── infrastructure.py   # SCADA, PMU, equipment data
│   │   └── robotic.py          # Visual, thermal, sensor data
│   │
│   ├── layers/                 # Graph neural network layers
│   │   ├── __init__.py
│   │   ├── graph_attention.py  # GAT with physics-aware messaging
│   │   └── temporal_gnn.py     # Temporal GNN with LSTM
│   │
│   ├── heads/                  # Prediction heads
│   │   ├── __init__.py
│   │   └── prediction_heads.py # All task-specific heads
│   │
│   └── loss/                   # Loss functions
│       ├── __init__.py
│       └── physics_informed.py # Physics-informed loss
│
└── data/                       # Data loading and preprocessing
    ├── __init__.py
    ├── dataset.py              # Main dataset class
    ├── collation.py            # Batch collation
    │
    └── preprocessing/          # Preprocessing utilities
        ├── __init__.py
        ├── normalization.py    # Power/frequency normalization
        ├── truncation.py       # Sliding window truncation
        └── edge_masking.py     # Dynamic topology masking
```

## Quick Start

### Basic Usage

```python
# Import main components
from cascade_prediction import (
    UnifiedCascadePredictionModel,
    PhysicsInformedLoss,
    CascadeDataset,
    collate_cascade_batch
)

# Create model
model = UnifiedCascadePredictionModel(
    embedding_dim=128,
    hidden_dim=128,
    num_gnn_layers=3,
    heads=4,
    dropout=0.3
)

# Create dataset
dataset = CascadeDataset(
    data_dir='data/train',
    mode='full_sequence',
    base_mva=100.0,
    base_frequency=60.0
)

# Create dataloader
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_cascade_batch
)

# Training loop
for batch in loader:
    outputs = model(batch, return_sequence=True)
    # ... training logic
```

### Using Individual Components

```python
# Import specific embeddings
from cascade_prediction.models.embeddings import (
    EnvironmentalEmbedding,
    InfrastructureEmbedding,
    RoboticEmbedding
)

# Import layers
from cascade_prediction.models.layers import (
    GraphAttentionLayer,
    TemporalGNNCell
)

# Import prediction heads
from cascade_prediction.models.heads import (
    FailureProbabilityHead,
    VoltageHead,
    TimingHead,
    RiskHead
)

# Import preprocessing utilities
from cascade_prediction.data.preprocessing import (
    normalize_power,
    normalize_frequency,
    calculate_truncation_window,
    create_edge_mask_from_failures
)
```

## Component Details

### Models Package

#### Embeddings
- **EnvironmentalEmbedding**: Processes satellite imagery, weather sequences, and threat indicators
- **InfrastructureEmbedding**: Processes SCADA data, PMU measurements, and equipment status
- **RoboticEmbedding**: Processes visual data, thermal imagery, and sensor readings

#### Layers
- **GraphAttentionLayer**: Graph attention with physics-aware message passing and edge masking
- **TemporalGNNCell**: Combines graph attention with LSTM for temporal dynamics

#### Prediction Heads
- **FailureProbabilityHead**: Node failure probability (0-1)
- **VoltageHead**: Node voltages (per-unit)
- **AngleHead**: Voltage angles (radians)
- **FrequencyHead**: System frequency (Hz)
- **TemperatureHead**: Node/line temperatures
- **LineFlowHead**: Reactive power flows on lines
- **ReactiveFlowHead**: Reactive power at nodes
- **ActivePowerLineFlowHead**: Active power flows on lines
- **RiskHead**: 7-dimensional risk assessment
- **TimingHead**: Cascade failure timing

#### Loss Functions
- **PhysicsInformedLoss**: Combines prediction accuracy with physics constraints
  - Focal loss for imbalanced classification
  - Power flow conservation
  - Thermal capacity constraints
  - Voltage stability
  - Frequency dynamics
  - Reactive power balance
  - Timing accuracy
  - Risk assessment

### Data Package

#### Dataset
- **CascadeDataset**: Memory-efficient dataset with:
  - One-file-per-scenario loading
  - Physics-based normalization
  - Sliding window truncation
  - Dynamic topology masking
  - Metadata caching

#### Preprocessing
- **Normalization**: Power (MW to p.u.) and frequency (Hz to p.u.) normalization
- **Truncation**: Random sliding window to prevent data leakage
- **Edge Masking**: Dynamic topology masking using t-1 failures

## Migration Guide

### From Old Structure

If you're using the old monolithic files:

**Old:**
```python
from multimodal_cascade_model import UnifiedCascadePredictionModel
from cascade_dataset import CascadeDataset, collate_cascade_batch
```

**New:**
```python
from cascade_prediction.models import UnifiedCascadePredictionModel
from cascade_prediction.data import CascadeDataset, collate_cascade_batch
```

### Backward Compatibility

The old files (`multimodal_cascade_model.py`, `cascade_dataset.py`) are maintained as compatibility wrappers that import from the new structure. They will issue deprecation warnings.

## Training and Inference

The training and inference scripts (`train_model.py`, `inference.py`) have been updated to use the new modular structure. See those files for complete examples.

## Benefits of Modular Structure

1. **Better Organization**: Related code is grouped together
2. **Easier Testing**: Individual components can be tested in isolation
3. **Improved Maintainability**: Changes to one component don't affect others
4. **Clearer Dependencies**: Import statements show exactly what's being used
5. **Reusability**: Components can be easily reused in other projects
6. **Documentation**: Each module has focused documentation
7. **Extensibility**: New components can be added without modifying existing code

## Development

### Adding New Components

#### New Prediction Head
```python
# cascade_prediction/models/heads/prediction_heads.py

class MyNewHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
```

Then add to `__init__.py` and use in `unified_model.py`.

#### New Preprocessing Function
```python
# cascade_prediction/data/preprocessing/my_module.py

def my_preprocessing_function(data, param):
    """
    My preprocessing function.
    
    Args:
        data: Input data
        param: Parameter
    
    Returns:
        Processed data
    """
    # Implementation
    return processed_data
```

Then add to `preprocessing/__init__.py`.

## Testing

Each module can be tested independently:

```python
# Test embedding
from cascade_prediction.models.embeddings import EnvironmentalEmbedding
import torch

embedding = EnvironmentalEmbedding(embedding_dim=128)
satellite = torch.randn(2, 10, 118, 12, 16, 16)
weather = torch.randn(2, 10, 118, 80)
threat = torch.randn(2, 10, 118, 6)

output = embedding(satellite, weather, threat)
assert output.shape == (2, 10, 118, 128)
```

## Performance Considerations

- **Memory Efficiency**: Dataset loads one file at a time
- **Caching**: Metadata is cached for fast initialization
- **Batch Processing**: Efficient collation with padding
- **GPU Support**: All components support CUDA
- **Mixed Precision**: Compatible with automatic mixed precision training

## Citation

If you use this code, please cite:

```bibtex
@software{cascade_prediction_2025,
  title={Physics-Informed Deep Learning for Power Grid Cascade Failure Prediction},
  author={Kraftgene AI Inc.},
  year={2025},
  version={1.0.0}
}
```

## License

Copyright © 2025 Kraftgene AI Inc. All rights reserved.

## Support

For questions or issues, please contact the development team.
