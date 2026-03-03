# Cascade Prediction System - Modular Structure

This document describes the modular architecture of the cascade prediction system.

## Package Structure

```
cascade_prediction/
├── __init__.py                 # Main package initialization
├── data/                       # Data handling and preprocessing
│   ├── __init__.py
│   ├── dataset.py             # CascadeDataset class
│   ├── collation.py           # Batch collation functions
│   └── preprocessing/         # Data preprocessing modules
│       ├── __init__.py
│       ├── normalization.py   # Feature normalization
│       ├── truncation.py      # Sequence truncation
│       └── edge_masking.py    # Edge mask generation
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── unified_model.py       # Main UnifiedCascadePredictionModel
│   ├── embeddings/            # Embedding modules
│   │   ├── __init__.py
│   │   ├── infrastructure.py  # SCADA, PMU embeddings
│   │   ├── environmental.py   # Weather, satellite embeddings
│   │   └── robotic.py         # Sensor, visual embeddings
│   ├── layers/                # Neural network layers
│   │   ├── __init__.py
│   │   ├── temporal_gnn.py    # Temporal GNN layers
│   │   └── graph_attention.py # Graph attention mechanisms
│   ├── heads/                 # Prediction heads
│   │   ├── __init__.py
│   │   └── prediction_heads.py # Failure, timing, risk heads
│   └── loss/                  # Loss functions
│       ├── __init__.py
│       └── physics_informed.py # Physics-informed loss
├── training/                   # Training utilities
│   ├── __init__.py
│   ├── trainer.py             # Trainer class
│   ├── metrics.py             # Metric computation
│   └── checkpointing.py       # Checkpoint management
├── inference/                  # Inference utilities
│   ├── __init__.py
│   ├── predictor.py           # CascadePredictor class
│   ├── dataset.py             # Inference dataset
│   └── reporting.py           # Result reporting
└── utils/                      # Utility functions
    ├── __init__.py
    └── json_encoder.py        # JSON encoding utilities
```

## Module Descriptions

### Data Package (`cascade_prediction/data/`)

Handles all data loading, preprocessing, and batch preparation.

- `dataset.py`: Main `CascadeDataset` class for loading and preprocessing scenarios
- `collation.py`: Functions for collating variable-length sequences into batches
- `preprocessing/`: Subpackage containing preprocessing modules
  - `normalization.py`: Feature normalization (power, frequency, temperature)
  - `truncation.py`: Sliding window truncation
  - `edge_masking.py`: Edge mask generation for teacher forcing

### Models Package (`cascade_prediction/models/`)

Contains all model architectures and components.

- `unified_model.py`: Main `UnifiedCascadePredictionModel` class
- `embeddings/`: Embedding modules for different data modalities
  - `infrastructure.py`: SCADA and PMU embeddings
  - `environmental.py`: Weather and satellite embeddings
  - `robotic.py`: Sensor, visual, and thermal embeddings
- `layers/`: Neural network layer implementations
  - `temporal_gnn.py`: Temporal graph neural network layers
  - `graph_attention.py`: Graph attention mechanisms
- `heads/`: Prediction head modules
  - `prediction_heads.py`: Failure, timing, and risk prediction heads
- `loss/`: Loss function implementations
  - `physics_informed.py`: Physics-informed loss functions

### Training Package (`cascade_prediction/training/`)

Provides training loop, metrics, and checkpoint management.

- `trainer.py`: Main `Trainer` class with training loop
- `metrics.py`: Metric computation functions (cascade, node, timing, risk)
- `checkpointing.py`: Checkpoint saving and loading utilities

### Inference Package (`cascade_prediction/inference/`)

Handles model inference and result reporting.

- `predictor.py`: `CascadePredictor` class for running inference
- `dataset.py`: `ScenarioInferenceDataset` for inference data loading
- `reporting.py`: Functions for formatting and printing prediction reports

### Utils Package (`cascade_prediction/utils/`)

General utility functions.

- `json_encoder.py`: Custom JSON encoder for NumPy types

## Usage

### Training

```python
from cascade_prediction.models import UnifiedCascadePredictionModel
from cascade_prediction.data import CascadeDataset, collate_cascade_batch
from cascade_prediction.training import Trainer
from torch.utils.data import DataLoader
import torch

# Create datasets
train_dataset = CascadeDataset(
    data_path="data/train",
    topology_path="data/grid_topology.pkl",
    max_sequence_length=30,
    is_training=True
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_cascade_batch
)

# Create model
model = UnifiedCascadePredictionModel(
    embedding_dim=128,
    hidden_dim=128,
    num_gnn_layers=3,
    heads=4,
    dropout=0.1
)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=torch.device('cuda'),
    checkpoint_dir='checkpoints'
)

# Train
history = trainer.train(num_epochs=50)
```

### Inference

```python
from cascade_prediction.inference import CascadePredictor, print_prediction_report
import torch

# Create predictor
predictor = CascadePredictor(
    model_path="checkpoints/best_model.pt",
    topology_path="data/grid_topology.pkl",
    device=torch.device('cuda'),
    base_mva=100.0,
    base_freq=60.0
)

# Run prediction
results = predictor.predict_scenario(
    data_path="data/test",
    scenario_idx=0,
    window_size=30,
    batch_size=32
)

# Print report
print_prediction_report(
    results,
    predictor.cascade_threshold,
    predictor.node_threshold
)
```

## Command-Line Scripts

Two modular command-line scripts are provided:

### Training Script (`train_model_modular.py`)

```bash
python train_model_modular.py \
    --train_path data/train \
    --val_path data/test \
    --topology_path data/grid_topology.pkl \
    --batch_size 16 \
    --num_epochs 50 \
    --lr 0.001
```

### Inference Script (`inference_modular.py`)

```bash
python inference_modular.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/test \
    --scenario_idx 0 \
    --topology_path data/grid_topology.pkl \
    --output prediction.json
```

## Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a clear, focused responsibility
2. **Reusability**: Components can be easily reused in different contexts
3. **Testability**: Individual modules can be tested in isolation
4. **Maintainability**: Changes to one module don't affect others
5. **Extensibility**: New features can be added without modifying existing code
6. **Documentation**: Each module is self-contained and well-documented

## Migration from Legacy Code

The legacy monolithic scripts (`train_model.py`, `inference.py`) have been refactored into:

- Modular packages under `cascade_prediction/`
- New command-line scripts (`train_model_modular.py`, `inference_modular.py`)

The legacy scripts are preserved for backward compatibility but should be considered deprecated.
