# AI-Driven Predictive Cascade Failure Analysis

We present a Graph Neural Network (GNN) architecture for predictive cascade failure analysis in electrical power grids, capable of forecasting the **causal chain of failures** (e.g., `Node B -> Node A -> Node C`) minutes before they occur.

This system achieves a holistic view by fusing multi-modal data. For example, it can correlate **robotic data** (a drone spotting a landslide) with **environmental data** (heavy rain) and **infrastructure data** (line flow rerouting) to predict not just *that* a failure will happen, but the *exact sequence* of subsequent overloads and failures.

This specific implementation validates the core GNN and its physics-informed learning (PIL) constraints, serving as a robust proof-of-concept for a full production-scale data integration pipeline.

## Core Features

-   **Multi-Modal Fusion**: Attention-based architecture fuses heterogeneous data (SCADA time-series, satellite imagery, robotic sensor data) into a unified latent representation.
-   **Causal Path Prediction**: The model predicts the *specific time* of failure for every node, not just the probability. Sorting these predictions reveals the predicted causal chain of the cascade.
-   **Spatio-Temporal GNN**: A Graph Attention Network (GAT) combined with a multi-layer LSTM models how failures propagate through the grid's topology over time.
-   **Physics-Informed Learning**: A multi-component loss function includes AC power flow equations as soft constraints, ensuring predictions (voltage, frequency) are physically plausible.
-   **7-Dimensional Risk Assessment**: Predicts a detailed risk vector (Threat, Vulnerability, Impact, etc.) for every node, enabling fine-grained decision support.
-   **Multi-Modal Ground Truth**: The data generator creates a rich, labeled dataset that includes not just failure states but also the ground truth *causal path*, *failure timings*, and *risk profiles* for every scenario.
-   **Dynamic Loss Balancing**: An automatic calibration step in the training script balances the weights of all loss components (physics, timing, risk, etc.) for stable and effective training.

## System Architecture

The model is a multi-task spatio-temporal GNN. An input scenario is processed in parallel by three encoders, fused, and then passed through a GNN-LSTM core to make simultaneous predictions.

1.  **Feature Extraction Layer (Multi-Modal Encoders)**:
    * `EnvironmentalEmbedding`: CNNs process satellite imagery and threat maps.
    * `InfrastructureEmbedding`: MLPs process SCADA/PMU time-series data.
    * `RoboticEmbedding`: CNNs process visual/thermal drone imagery and sensor data.

2.  **Spatio-Temporal Prediction Layer (The "Brain")**:
    * An attention mechanism fuses the three embeddings into a single vector for each node.
    * `TemporalGNNCell`: A **Graph Attention (GAT)** layer shares information between neighboring nodes, while an **LSTM** processes the *sequence* of these messages over time. This is what allows the model to learn `B -> A -> C` causal relationships.

3.  **Multi-Task Decision Layer (The "Output")**:
    The final node embeddings are fed into parallel heads to predict:
    * **`failure_prob_head`**: *Will* it fail? (Probability)
    * **`failure_time_head`**: *When* will it fail? (Causal Path)
    * **`risk_head`**: *Why* will it fail? (7-D Risk Vector)
    * **Physics Heads**: `voltage_head`, `angle_head`, `line_flow_head`, etc.

## The Multi-Component Loss Function

The "enormous gaps" in timing are fixed by training the model on a comprehensive loss function that grades *all* of its predictions, not just the final yes/no answer.

* **`focal_loss`**: Trains the `failure_prob_head`. (Is it a cascade: Yes/No?)
* **`timing_loss`**: Trains the `failure_time_head`. (Is the predicted causal path `B -> A -> C` correct?)
* **`risk_loss`**: Trains the `risk_head`. (Is the predicted *reason* for the failure correct?)
* **Physics Losses** (`powerflow_loss`, `capacity_loss`, etc.): Trains the physics heads. (Are the predicted voltages and line flows physically possible?)

## Detailed Example: The Landslide Scenario

This system achieves a holistic view by fusing multi-modal data. The following example demonstrates how the system correlates **robotic data** (a drone spotting a landslide) with **environmental data** (heavy rain) and **infrastructure data** (line flow rerouting) to predict not just *that* a failure will happen, but the *exact sequence* of subsequent overloads and failures.

### Step 1: Data Ingestion (The "What")

A drone, a weather satellite, and the grid's own sensors all report data simultaneously for the same timestep:

- **Robotic Data**: The drone's visual feed shows mud and debris at the base of a transmission tower connecting Node 33 and Node 40. Its acoustic/vibration sensors detect the "groan" of stressed metal.

- **Environmental Data**: The threat indicators module flags a high "Geohazard" (landslide) risk for the area. Satellite weather sequence data confirms heavy precipitation causing the landslide.

- **Infrastructure Data**: The tower foundation shifts, causing the line to sag. PMU data for Node 33 and Node 40 shows their voltage angles drifting apart.

### Step 2: The Physical Event (The "So What")

The tower fails. The transmission line (Edge 33-40) breaks and its circuit breaker trips, taking the line offline.

- **Immediate Consequence**: The 800 MW of power flowing from Node 33 to Node 40 vanishes from that path.

- **The Reroute**: That 800 MW instantly (at the speed of light) tries to find another path through the grid. It surges onto adjacent, parallel lines, such as Node 33 → Node 35 and Node 40 → Node 47.

- **The Overload**: These adjacent lines were already at 70% capacity and are now suddenly at 150% capacity, violating their thermal limits.

### Step 3: The Model's Prediction (The "What's Next")

The system ingests all new data and the model recognizes the pattern:

1. **EnvironmentalEmbedding** and **RoboticEmbedding** modules process the drone/weather data and output a high "threat" vector.

2. **InfrastructureEmbedding** processes the SCADA/PMU data and outputs a high "vulnerability" vector, detecting the line outage at 33-40 and critical overloads at 33-35 and 40-47.

3. **Graph Attention Layer (GNN)** propagates a "shock message" from the now-offline Nodes 33 and 40 to their neighbors, Nodes 35 and 47.

4. **LSTM (TemporalGNNCell)** at Nodes 35 and 47 receives this GNN shock message and sees its own internal state is now critical (150% overload).

5. This combined state (external shock + internal stress) is fed to the multi-task prediction heads.

### Step 4: The Final Inference Report

The inference script produces a detailed report:

**Overall Verdict**: Cascade Predicted (Probability: 0.99)

**Time-to-Cascade**: Predicted Lead Time: 0.15 minutes (9 seconds until first subsequent failure)

**Top 5 High-Risk Nodes**:
- Node 35: 0.998 (The first overload)
- Node 47: 0.997 (The second overload)
- Node 33: 0.980 (Part of the initial event)
- Node 40: 0.979 (Part of the initial event)
- Node 51: 0.850 (The next node in the cascade path)

**Aggregated Risk Assessment**:
- Threat: 0.85 (Critical) - from the landslide
- Vulnerability: 0.92 (Critical) - from the line outage
- Impact: 0.90 (Critical) - from the high power reroute
- CascadeProb: 0.95 (Critical) - the model sees the dominoes

**Cascade Path Analysis (The Causal Chain)**:

| Predicted Time | Predicted Node | Predicted Risk | Actual Time | Actual Node | Actual Cause |
|:--------------|:---------------|:---------------|:------------|:------------|:-------------|
| 0.15m | Node 35 | Vulnerability: 0.92 | 0.00m | Node 33 | Environmental |
| 0.18m | Node 47 | Vulnerability: 0.90 | 0.00m | Node 40 | Environmental |
| 0.45m | Node 51 | CascadeProb: 0.85 | 0.16m | Node 35 | Loading |
| 0.62m | Node 29 | CascadeProb: 0.78 | 0.21m | Node 47 | Loading |
| — | — | — | 0.48m | Node 51 | Loading |

This report doesn't just say "there is a problem." It tells you **what** will fail (Nodes 35, 47, 51...), **when** it will fail (0.15m, 0.45m...), and **why** (high Vulnerability from the outage, then high CascadeProb from subsequent overloads).


## Quick Start

### Installation

Clone the repository:

```bash
git clone [https://github.com/KraftgeneAI/CascadeFailureDetection.git](https://github.com/KraftgeneAI/CascadeFailureDetection.git)
cd CascadeFailureDetection
```

Install dependencies (virtual environment recommended):

```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install torch torch_geometric numpy matplotlib scipy tqdm psutil
```

Step 1: Generate Master Topology
First, generate the single, master grid_topology.pkl file. All data must be generated using this one file.

```bash
python multimodal_data_generator.py --normal 1 --cascade 0 --output-dir ./data
```
This creates data/grid_topology.pkl.

Step 2: Generate Dataset
Now, generate the full dataset using the master topology file. Thousands of samples are required for effective training, they can be generated on multiple machines at the same time due to the memory efficiency of the generator.

```bash
# Generate 10,000 scenarios for training/validation/testing
python multimodal_data_generator.py --normal 6000 --cascade 4000 --batch-size 100 --output-dir data --topology-file data/grid_topology.pkl
```

To generate data on multiple machines (Parallel Generation):

Copy the grid_topology.pkl file to all machines.

Run the generator on each machine with a different start_batch number.

```bash
# Machine 1
python multimodal_data_generator.py --normal 3000 --cascade 2000 --output-dir data_p1 --start_batch 0 --topology-file grid_topology.pkl

# Machine 2
python multimodal_data_generator.py --normal 3000 --cascade 2000 --output-dir data_p2 --start_batch 5000 --topology-file grid_topology.pkl
```
Then, merge the data_p1 and data_p2 train, val, and test folders.

Step 3: Train the Model
Start training with automatic loss calibration. This script will use the new timing_loss and risk_loss to train the model on the causal path.

```bash

# Recommended training command
python train_model.py --data_dir ./data --output_dir ./checkpoints --epochs 100 --batch_size 8 --lr 0.0001

# Resume training from a checkpoint
python train_model.py --resume
```
The script will:

- Automatically locate data/train and data/val.

- Run dynamic loss calibration to balance all loss components.

- Use WeightedRandomSampler and FocalLoss to handle class imbalance.

- Save the best model to checkpoints/best_model.pth.

- Generate checkpoints/training_curves.png.

Step 4: Run Inference

Batch Predictions
Run predictions on the entire test set to get evaluation metrics.

```bash
python inference.py --model_path checkpoints/best_model.pth --data_path data/test --batch --output test_predictions.json
```

Single Scenario Prediction
Predict a specific scenario to see the full causal path analysis.

```bash

# Use a known cascade scenario index from your test set
python inference.py --model_path checkpoints/best_model.pth --data_path data/test --scenario_idx 10 --output single_prediction.json

```
This outputs the detailed report, including the Predicted Causal Path and the 7-D Risk Assessment for comparison against the ground truth.

Project Structure

```bash
.
├── checkpoints/
│   ├── best_model.pth           # Best model weights
│   ├── training_curves.png      # Training/validation metrics chart
│   └── training_history.json    # Raw metrics per epoch
│
├── data/
│   ├── grid_topology.pkl        # The MASTER grid topology file
│   ├── train/                   # Training data batches
│   ├── val/                     # Validation data batches
│   └── test/                    # Test data batches
│
├── multimodal_data_generator.py # Data generation script (Run this first)
├── cascade_dataset.py           # PyTorch Dataset class (Loads data)
├── multimodal_cascade_model.py  # Model architecture (The "Brain")
├── train_model.py               # Training script (The "Teacher")
├── inference.py                 # Inference script (The "Report")
└── README.md                    # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- NumPy
- Matplotlib
- SciPy
- tqdm
- psutil

## License

This project is licensed under dual license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.