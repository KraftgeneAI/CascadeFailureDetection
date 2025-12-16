# AI-Driven Predictive Cascade Failure Analysis （End-To-End）

We present an end-to-end Graph Neural Network (GNN) architecture for predictive cascade failure analysis in electrical power grids, capable of forecasting the **causal chain of failures** (e.g., `Node B -> Node A -> Node C`) minutes before they occur.

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

The inference script produces a detailed report. Below is an example output from an early-stage trained model analyzing a complex cascade scenario.

**Overall Verdict**: 
  - ✅ Correctly detected a cascade.
  - Prediction: True (Prob: 0.401 / Thresh: 0.100)
  - Ground Truth: True

**Node-Level Analysis**: 
  - Predicted Nodes at Risk: 95 (Thresh: 0.350)
  - Actual Failed Nodes:     69
  - Correctly Identified (TP): 69
  - Missed Nodes (FN):         0
  - False Alarms (FP):         26

**Critical Information**:
  - System Frequency: 64.84 Hz
  - Voltage Range:    [0.000, 0.217] p.u.


**Top High-Risk Nodes**:
  - Node 40 : 0.4007 ✓ (Actual)
  - Node 115: 0.3960 ✓ (Actual)
  - Node 76 : 0.3951 ✓ (Actual)
  - Node 46 : 0.3951 ✓ (Actual)
  - Node 28 : 0.3950 ✓ (Actual)

**Aggregated Risk Assessment**:
Aggregated Risk Assessment (7-Dimensions):
  - Threat: 0.867 (Critical) | Vulnerability: 0.820 (Critical) | Impact: 0.796 (Severe)  
  - Cascade Prob: 0.767 (Severe)   | Response: 0.038 (Low)      | Safety: 0.038 (Low)     
  - Urgency: 0.802 (Critical)

Ground Truth Risk Assessment:
  - Threat: 0.974 (Critical) | Vulnerability: 0.700 (Severe)   | Impact: 0.900 (Critical)
  - Cascade Prob: 0.792 (Severe)   | Response: 0.000 (Low)      | Safety: 0.000 (Low)     
  - Urgency: 0.900 (Critical)

Risk Definitions: 
  - Critical (0.8+): Immediate Failure | Severe (0.6+): High Danger | Medium (0.3+): Caution
  - Dimensions: Threat (Stress), Vulnerability (Weakness), Impact (Consequence),
              Cascade Prob (Propagation), Urgency (Time Sensitivity).

**Cascade Path Analysis (Sequence Order)**:
The model predicts the *causal sequence* of failures by ranking nodes based on their failure priority score.

```bash
  Seq /#  | Predicted Node  | Score    | Actual Seq /#  | Actual Node     | Delta T (min)  
  ------ | --------------- | -------- | --------------- | --------------- | ---------------
  1      | Node 82         | 0.401    | 1               | Node 82         | 0.00           
  2      | Node 115        | 0.396    | 2               | Node 115        | 0.22           
  2      | Node 76         | 0.395    | 2               | Node 64         | 0.28           
  2      | Node 46         | 0.395    | 2               | Node 78         | 0.29           
  2      | Node 28         | 0.395    | 3               | Node 112        | 0.37           
  2      | Node 103        | 0.395    | 3               | Node 65         | 0.43           
  2      | Node 114        | 0.394    | 3               | Node 71         | 0.43           
  2      | Node 80         | 0.394    | 3               | Node 76         | 0.46           
  3      | Node 67         | 0.394    | 4               | Node 108        | 0.48           
  3      | Node 36         | 0.393    | 4               | Node 99         | 0.50           
  3      | Node 21         | 0.393    | 4               | Node 61         | 0.50           
  3      | Node 70         | 0.393    | 4               | Node 70         | 0.58           
  3      | Node 85         | 0.393    | 4               | Node 91         | 0.58           
  3      | Node 104        | 0.393    | 5               | Node 80         | 0.60           
  3      | Node 40         | 0.393    | 5               | Node 117        | 0.60           
  3      | Node 56         | 0.393    | 5               | Node 92         | 0.63           
  3      | Node 38         | 0.392    | 5               | Node 60         | 0.64           
  3      | Node 91         | 0.392    | 5               | Node 59         | 0.67           
  3      | Node 10         | 0.392    | 6               | Node 89         | 0.72           
  4      | Node 100        | 0.392    | 6               | Node 85         | 0.73           
  4      | Node 58         | 0.391    | 6               | Node 84         | 0.74           
  4      | Node 48         | 0.391    | 6               | Node 98         | 0.75           
  4      | Node 4          | 0.391    | 6               | Node 110        | 0.76           
  4      | Node 59         | 0.391    | 6               | Node 103        | 0.79           
  4      | Node 74         | 0.391    | 6               | Node 107        | 0.80           
  4      | Node 106        | 0.391    | 6               | Node 111        | 0.80           
  4      | Node 72         | 0.391    | 6               | Node 101        | 0.80           
  4      | Node 68         | 0.390    | 7               | Node 81         | 0.82           
  5      | Node 9          | 0.389    | 7               | Node 94         | 0.82           
  5      | Node 109        | 0.389    | 7               | Node 106        | 0.88           
  5      | Node 99         | 0.388    | 7               | Node 36         | 0.90           
  5      | Node 94         | 0.388    | 7               | Node 93         | 0.91           
  5      | Node 64         | 0.388    | 8               | Node 97         | 0.92           
  5      | Node 8          | 0.388    | 8               | Node 79         | 0.94           
  5      | Node 110        | 0.388    | 8               | Node 68         | 0.96           
  5      | Node 87         | 0.388    | 8               | Node 100        | 1.00           
  5      | Node 101        | 0.387    | 9               | Node 62         | 1.03           
  6      | Node 93         | 0.387    | 9               | Node 52         | 1.05           
  6      | Node 117        | 0.386    | 9               | Node 83         | 1.05           
  6      | Node 97         | 0.386    | 9               | Node 38         | 1.12           
  6      | Node 3          | 0.386    | 10              | Node 87         | 1.15           
  6      | Node 51         | 0.386    | 10              | Node 35         | 1.23           
  6      | Node 25         | 0.385    | 11              | Node 48         | 1.28           
  6      | Node 42         | 0.385    | 11              | Node 29         | 1.31           
  6      | Node 14         | 0.385    | 11              | Node 58         | 1.31           
  6      | Node 39         | 0.385    | 11              | Node 46         | 1.35           
  6      | Node 22         | 0.385    | 12              | Node 57         | 1.52           
  6      | Node 81         | 0.385    | 12              | Node 40         | 1.52           
  6      | Node 78         | 0.385    | 12              | Node 53         | 1.54           
  6      | Node 55         | 0.385    | 12              | Node 47         | 1.56           
  7      | Node 57         | 0.385    | 12              | Node 56         | 1.59           
  7      | Node 15         | 0.384    | 13              | Node 44         | 1.66           
  7      | Node 33         | 0.384    | 13              | Node 28         | 1.71           
  7      | Node 35         | 0.384    | 13              | Node 33         | 1.72           
  7      | Node 47         | 0.384    | 14              | Node 34         | 1.78           
  7      | Node 60         | 0.384    | 15              | Node 4          | 1.88           
  7      | Node 23         | 0.384    | 16              | Node 17         | 2.02           
  7      | Node 65         | 0.383    | 16              | Node 3          | 2.06           
  7      | Node 71         | 0.383    | 16              | Node 55         | 2.08           
  8      | Node 53         | 0.383    | 16              | Node 15         | 2.09           
  8      | Node 107        | 0.383    | 17              | Node 24         | 2.15           
  8      | Node 79         | 0.382    | 17              | Node 2          | 2.17           
  8      | Node 62         | 0.382    | 17              | Node 23         | 2.19           
  8      | Node 17         | 0.382    | 17              | Node 20         | 2.22           
  8      | Node 108        | 0.382    | 18              | Node 8          | 2.31           
  8      | Node 29         | 0.382    | 18              | Node 14         | 2.34           
  8      | Node 16         | 0.382    | 18              | Node 26         | 2.34           
  8      | Node 52         | 0.382    | 19              | Node 16         | 2.45           
  8      | Node 18         | 0.381    | 20              | Node 25         | 2.61           
  8      | Node 92         | 0.381    |                 |                 |                
  8      | Node 20         | 0.381    |                 |                 |                
  8      | Node 112        | 0.381    |                 |                 |                
  8      | Node 83         | 0.381    |                 |                 |                
  9      | Node 31         | 0.380    |                 |                 |                
  9      | Node 24         | 0.380    |                 |                 |                
  9      | Node 6          | 0.380    |                 |                 |                
  9      | Node 34         | 0.380    |                 |                 |                
  9      | Node 30         | 0.380    |                 |                 |                
  9      | Node 88         | 0.379    |                 |                 |                
  9      | Node 89         | 0.379    |                 |                 |                
  10     | Node 111        | 0.378    |                 |                 |                
  10     | Node 98         | 0.377    |                 |                 |                
  10     | Node 2          | 0.377    |                 |                 |                
  10     | Node 32         | 0.377    |                 |                 |                
  10     | Node 61         | 0.377    |                 |                 |                
  10     | Node 26         | 0.376    |                 |                 |                
  11     | Node 49         | 0.376    |                 |                 |                
  11     | Node 50         | 0.376    |                 |                 |                
  11     | Node 44         | 0.375    |                 |                 |                
  11     | Node 84         | 0.374    |                 |                 |                
  11     | Node 66         | 0.374    |                 |                 |                
  12     | Node 7          | 0.373    |                 |                 |                
  13     | Node 75         | 0.371    |                 |                 |                
  14     | Node 1          | 0.356    |                 |                 |                
  15     | Node 86         | 0.350    |                 |                 |                
================================================================================
```
This report doesn't just say "there is a problem." It tells you **what** will fail, and provides a **priority ranking** of the failure sequence, allowing operators to focus on the root causes (Seq #) before the cascading effects.


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
python multimodal_data_generator.py --normal 5000 --cascade 4000 --stressed 1000 --output-dir data --topology-file data/grid_topology.pkl
```

To generate data on multiple machines (Parallel Generation):

Copy the data/grid_topology.pkl file to all machines.

Run the generator on each machine with a different start_batch number.

```bash
# Machine 1
python multimodal_data_generator.py --normal 2000 --cascade 2000 --stressed 1000  --output-dir data_p1 --start_batch 0 --topology-file data/grid_topology.pkl

# Machine 2
python multimodal_data_generator.py --normal 3000 --cascade 2000 --stressed 1000 --output-dir data_p2 --start_batch 5000 --topology-file data/grid_topology.pkl
```
Then, merge the data_p1 and data_p2 train, val, and test folders.

Step 3: Train the Model
Start training with automatic loss calibration. This script will use the new timing_loss and risk_loss to train the model on the causal path.

```bash

# Recommended training command (batch size of 4 for cpu, 16 or 32 or 64 for gpus)
python train_model.py --data_dir ./data --output_dir ./checkpoints --epochs 100 --batch_size 4 --lr 0.0001

# Resume training from the latest checkpoint
python train_model.py --resume checkpoints/latest_checkpoint.pth
```
The script will:

- Automatically locate data/train and data/val.

- Run dynamic loss calibration to balance all loss components.

- Use WeightedRandomSampler and FocalLoss to handle class imbalance.

- Save the best model to checkpoints/best_f1_model.pth.

- Generate checkpoints/training_curves.png.


Step 4: Run Inference

Batch Predictions
Run predictions on the entire test set to get evaluation metrics.

```bash
python inference.py --model_path checkpoints/best_f1_model.pth --data_path data/test --batch --output test_predictions.json
```

Single Scenario Prediction
Predict a specific scenario to see the full causal path analysis.

```bash

# Use a known cascade scenario index from your test set
python inference.py --model_path checkpoints/best_f1_model.pth --data_path data/test --scenario_idx 10 --output single_prediction.json

```
This outputs the detailed report, including the Predicted Causal Path and the 7-D Risk Assessment for comparison against the ground truth.

Project Structure

```bash
.
├── checkpoints/
│   ├── best_model.pth           # Best va loss weights
│   ├── best_f1_model.pth      # Best f1 model weights
│   └── latest_checkpoint.pth    # Raw metrics per epoch
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

Contributions are welcome! 