# AI-Driven Predictive Cascade Failure Analysis

We present a Graph Neural Network (GNN) architecture for predictive cascade failure analysis in electrical power grids, capable of forecasting events 15–35 minutes prior to occurrence. This system achieves a holistic view by fusing multi-modal environmental and infrastructure data. This specific implementation focuses on validating the core GNN and its physics-informed learning (PIL) constraints, serving as a robust proof-of-concept for the full production-scale data integration pipeline developed in our research.

## Overview

This project presents a novel approach to power grid failure prediction by combining:

- **Infrastructure Data**: Real-time telemetry from SCADA systems and Phasor Measurement Units (PMUs)
- **Environmental Data**: Satellite imagery, meteorological data, and threat indicators (wildfire, flooding)
- **Robotic Sensor Data**: High-resolution visual, thermal, and sensor data from autonomous robotic platforms

The model's predictions are enhanced by a physics-informed loss function that ensures all predictions are consistent with the physical laws of power flow.

## System Architecture

The framework is built on a four-layer architecture that processes data from ingestion to actionable decisions:

1. **Data Ingestion Layer**: Collects and preprocesses multi-modal data streams
2. **Feature Extraction Layer**: Processes heterogeneous data into unified representations
3. **Prediction Layer**: Spatio-temporal GNN models grid dynamics
4. **Decision Layer**: Generates actionable risk assessments

## Core Features

- **Multi-Modal Fusion**: Attention-based architecture fuses heterogeneous data (time-series, imagery, graph data) into a unified latent representation
- **Spatio-Temporal GNN**: Graph Attention Network (GAT) combined with multi-layer LSTM models complex spatio-temporal dynamics
- **Physics-Informed Learning**: Novel loss function incorporates AC power flow equations as soft constraints
- **Dynamic Loss Balancing**: Automatic calibration of loss component weights for stable training
- **7-Dimensional Risk Assessment**: Detailed risk vectors for every grid node enabling fine-grained decision support

## Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/KraftgeneAI/CascadeFailureDetection.git
cd cascade-failure-prediction
```

Install dependencies (virtual environment recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch torch_geometric numpy matplotlib scipy
```

### Step 1: Generate Dataset

Generate simulated multi-modal data for training:

```bash
python multimodal_data_generator.py --normal 6000 --cascade 4000 --batch-size 100 --output-dir data
```

This creates 10,000 scenarios (6,000 normal, 4,000 cascade) and saves them in the `data/` directory along with `grid_topology.pkl`.

**Note**: Thousands of samples are required for effective training. Small datasets (50-100 samples) will not produce meaningful results.

### Step 2: Train the Model

Start training with automatic loss calibration:

```bash
# Default training
python train_model.py # default config: learning_rate = 0.0005, batch_size = 4, epoch = 50
# Basic training
python train_model.py --data_dir ./data --output_dir ./checkpoints --epochs 100 --batch_size 8 --lr 0.0005

# Adjust learning rate and other parameters
python train_model.py --data_dir ./data --epochs 50 --batch_size 16 --lr 0.0001 --grad_clip 5.0

# Resume training from a checkpoint
python train_model.py --resume
```

The script will:
- Automatically locate training and validation data
- Run dynamic loss calibration
- Use WeightedRandomSampler and Focal Loss for class imbalance
- Save the best model to `checkpoints/best_model.pth`
- Generate training charts in the `checkpoints/` directory

### Step 3: Run Inference

#### Batch Predictions

Run predictions on the entire test set:

```bash
python inference.py --model_path checkpoints/best_model.pth --data_path data/test --batch --output test_predictions.json
```

This outputs full predictions with risk assessments and evaluation metrics (Accuracy, Precision, Recall, F1 Score).

#### Single Scenario Prediction

Predict a specific scenario:

```bash
python inference.py --model_path checkpoints/best_model.pth --data_path data/test --scenario_idx 0 --output single_prediction.json
```

## Project Structure

```
.
├── checkpoints/
│   ├── best_model.pth           # Best model weights
│   ├── training_curves.png      # Training/validation metrics chart
│   └── training_history.json    # Raw metrics per epoch
│
├── data/
│   ├── grid_topology.pkl        # Grid topology map
│   ├── train/                   # Training data
│   ├── val/                     # Validation data
│   └── test/                    # Test data
│
├── multimodal_data_generator.py # Data generation script
├── cascade_dataset.py           # PyTorch Dataset class
├── multimodal_cascade_model.py  # Model architecture
├── train_model.py               # Training script
├── inference.py                 # Inference script
└── README.md                    # This file
```

## Model Architecture

The system uses a unified architecture combining:

- **Graph Attention Networks (GAT)**: Captures spatial relationships in grid topology
- **LSTM Networks**: Models temporal dynamics and failure propagation
- **Attention Mechanisms**: Fuses multi-modal data streams
- **Physics-Informed Constraints**: Ensures predictions obey power flow laws

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- NumPy
- Matplotlib
- SciPy

## License

This project is licensed under dual License.

## Contributing

Contributions are welcome! 
