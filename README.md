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

The training process is like the following, with continously decreasing tMAE showing the model is getting better and better at predicting correct cascade path (since cascade_f1 & node_f1 are easy to train as shown below, so tMAE is the metric to guide the training process): 

```bash
PS D:\CascadeFailureDetection> python .\train_model.py
================================================================================
CASCADE FAILURE PREDICTION - TRAINING SCRIPT (IMPROVED)
================================================================================

Configuration:
  Data directory: data
  Output directory: checkpoints
  Batch size: 4
  Epochs: 100
  Learning rate: 0.0001
  Device: cpu
  (WARNING: CUDA (GPU) not available, training will be slow on CPU.)
  Gradient clipping: 10.0
  Mixed precision: False
  Resume training: False

Loading datasets...
Indexing scenarios from: data/train
  [Warning] Found 'scenarios_batch_*.pkl' files. Assuming batch_size=1 and attempting to load.
  [Info] For performance, run rebatch_data.py to create 'scenario_*.pkl' files.
Physics normalization: base_mva=100.0, base_frequency=60.0
Scanning 1315 files for cascade labels...
Warning: Skipping corrupted or unreadable file: data\train\scenarios_batch_8738.pkl. Error: pickle data was truncated
Indexed 1315 scenarios.
  Cascade scenarios: 622 (47.3%)
  Normal scenarios: 693 (52.7%)
Ultra-memory-efficient mode: Loading 1 file per sample.
Indexing scenarios from: data/val
  [Warning] Found 'scenarios_batch_*.pkl' files. Assuming batch_size=1 and attempting to load.
  [Info] For performance, run rebatch_data.py to create 'scenario_*.pkl' files.
Physics normalization: base_mva=100.0, base_frequency=60.0
Scanning 239 files for cascade labels...
Indexed 239 scenarios.
  Cascade scenarios: 118 (49.4%)
  Normal scenarios: 121 (50.6%)
Ultra-memory-efficient mode: Loading 1 file per sample.
  Training samples: 1315
  Validation samples: 239
  Mode: full_sequence (utilizing 3-layer LSTM for temporal modeling)

Computing sample weights for balanced sampling...
  Positive samples: 622 (47.3%)
  Negative samples: 693 (52.7%)
  Oversampling ratio: 20:1 (positive:negative)

Initializing model...
  Total parameters: 765,245
  Trainable parameters: 765,245

================================================================================
STARTING DYNAMIC LOSS WEIGHT CALIBRATION
================================================================================
Running loss calibration for 20 batches...
  Average raw loss components (unweighted):
    prediction: 0.356208
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.049858
    risk: 0.125549
    timing: 1.242879

  Target magnitude (from prediction loss): 0.356208

  Calibrating lambda weights (Target-Value Strategy):
    lambda_powerflow:     0.0356  (Raw Loss: 10.000000, Denom: 10.000000, Initial Weighted Loss: 0.3562)
    lambda_capacity:     1.0000  (Raw Loss: 0.000000, Denom: 0.356208, Initial Weighted Loss: 0.0000)
    voltage:     1.0000  (Raw Loss: 0.000000, Denom: 0.356208, Initial Weighted Loss: 0.0000)
    lambda_frequency:     7.1444  (Raw Loss: 0.049858, Denom: 0.049858, Initial Weighted Loss: 0.3562)
    lambda_risk:     2.8372  (Raw Loss: 0.125549, Denom: 0.125549, Initial Weighted Loss: 0.3562)
    lambda_timing:     0.2866  (Raw Loss: 1.242879, Denom: 1.242879, Initial Weighted Loss: 0.3562)
================================================================================
CALIBRATION COMPLETE
================================================================================


✓ PhysicsInformedLoss initialized with MANUALLY SET lambda_timing=10.0

================================================================================
STARTING TRAINING
================================================================================


Epoch 1/100
--------------------------------------------------------------------------------
Training:   0%|                                                                                                                                                                                | 0/329 [00:00<?, ?it/s] 
================================================================================
MODEL OUTPUT VALIDATION (First Batch)
================================================================================
Checking required outputs for loss calculation...
  ✓ failure_probability: shape (4, 118, 1) (Matches expected)
  ✓ voltages: shape (4, 118, 1) (Matches expected)
  ✓ angles: shape (4, 118, 1) (Matches expected)
  ✓ line_flows: shape (4, 686, 1) (Matches expected)
  ✓ frequency: shape (4, 1, 1) (Matches expected)
  ✓ risk_scores: shape (4, 118, 7) (Matches expected)
  ✓ cascade_timing: shape (4, 118, 1) (Matches expected)

Checking other model outputs...
  ✓ reactive_flows: shape (4, 686, 1) (Matches expected)

Temporal sequence detected: B=4, T=60, N=118
  ✓ 3-layer LSTM IS BEING UTILIZED.
================================================================================

Training (Loss: 12.2372, Grad: 59.01): 100%|█████████████████████████████████████████████████████████████████████| 329/329 [1:15:32<00:00, 13.78s/it, cF1=0.971, nF1=0.711, tMAE=0.84m, rMSE=0.070, pL=0.277, tL=1.130]

  Average gradient norm: 25.7455
  Average loss components:
    prediction: 0.314406
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.035605
    risk: 0.069799
    timing: 0.906871
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [06:01<00:00,  6.02s/it, loss=8.8180, casc_f1=1.0000, time_mae=0.77m]
  ✓ Improved F1 score: 0.9384 (thresholds: cascade=0.500, node=0.500)

Epoch 1 Results:
  Train Loss: 10.1917 | Val Loss: 9.2214
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 0.9710 | Val 1.0000
    Precision: Train 0.9436 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.7109 | Val 0.8767
    Precision: Train 0.5580 | Val 0.8858
    Recall:    Train 0.9794 | Val 0.8678

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.836 | Val 0.770
    Risk MSE:          Train 0.0698 | Val 0.1059
  ✓ Saved best model (val_loss: 9.2214) - (F1 scores not high enough yet)

Epoch 2/100
--------------------------------------------------------------------------------
Training (Loss: 6.7178, Grad: 24.02): 100%|██████████████████████████████████████████████████████████████████████| 329/329 [1:13:39<00:00, 13.43s/it, cF1=0.974, nF1=0.826, tMAE=0.68m, rMSE=0.063, pL=0.212, tL=0.584] 

  Average gradient norm: 73.3611
  Average loss components:
    prediction: 0.223230
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.031482
    risk: 0.062625
    timing: 0.734327
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [06:04<00:00,  6.07s/it, loss=4.7906, casc_f1=1.0000, time_mae=0.69m]

Epoch 2 Results:
  Train Loss: 8.3253 | Val Loss: 8.9878
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 0.9744 | Val 1.0000
    Precision: Train 0.9502 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.8256 | Val 0.7757
    Precision: Train 0.7216 | Val 0.9467
    Recall:    Train 0.9648 | Val 0.6570

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.677 | Val 0.688
    Risk MSE:          Train 0.0626 | Val 0.0938
  ✓ Saved best model (val_loss: 8.9878) - (F1 scores not high enough yet)

Epoch 3/100
--------------------------------------------------------------------------------
Training (Loss: 5.4477, Grad: 266.23): 100%|█████████████████████████████████████████████████████████████████████| 329/329 [1:13:28<00:00, 13.40s/it, cF1=0.988, nF1=0.839, tMAE=0.57m, rMSE=0.062, pL=0.175, tL=0.424] 

  Average gradient norm: 153.6640
  Average loss components:
    prediction: 0.198799
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.030553
    risk: 0.061600
    timing: 0.580553
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [06:00<00:00,  6.01s/it, loss=3.5566, casc_f1=1.0000, time_mae=0.64m]
  ✓ Improved F1 score: 0.9404 (thresholds: cascade=0.500, node=0.500)

Epoch 3 Results:
  Train Loss: 6.7536 | Val Loss: 8.4497
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 0.9879 | Val 1.0000
    Precision: Train 0.9762 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.8392 | Val 0.8807
    Precision: Train 0.7395 | Val 0.9342
    Recall:    Train 0.9699 | Val 0.8330

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.566 | Val 0.636
    Risk MSE:          Train 0.0616 | Val 0.0885
  ✓ Saved best model (val_loss: 8.4497) - (F1 scores not high enough yet)

Epoch 4/100
--------------------------------------------------------------------------------
Training (Loss: 3.6709, Grad: 107.37): 100%|█████████████████████████████████████████████████████████████████████| 329/329 [1:13:41<00:00, 13.44s/it, cF1=0.999, nF1=0.854, tMAE=0.47m, rMSE=0.057, pL=0.138, tL=0.274] 

  Average gradient norm: 164.0272
  Average loss components:
    prediction: 0.164700
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.030736
    risk: 0.056984
    timing: 0.411607
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [06:04<00:00,  6.07s/it, loss=3.2047, casc_f1=1.0000, time_mae=0.32m]
  ✓ Improved F1 score: 0.9682 (thresholds: cascade=0.500, node=0.500)

Epoch 4 Results:
  Train Loss: 5.0182 | Val Loss: 2.8543
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 0.9992 | Val 1.0000
    Precision: Train 0.9984 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.8540 | Val 0.9364
    Precision: Train 0.7597 | Val 0.9078
    Recall:    Train 0.9749 | Val 0.9668

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.472 | Val 0.325
    Risk MSE:          Train 0.0570 | Val 0.0857
  ✓ Saved best model (New best MAE: 0.3248m, F1s: 1.000/0.936)

Epoch 5/100
--------------------------------------------------------------------------------
Training (Loss: 2.5984, Grad: 69.24): 100%|██████████████████████████████████████████████████████████████████████| 329/329 [1:14:30<00:00, 13.59s/it, cF1=1.000, nF1=0.922, tMAE=0.38m, rMSE=0.056, pL=0.054, tL=0.192] 

  Average gradient norm: 109.5636
  Average loss components:
    prediction: 0.088019
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.030320
    risk: 0.055775
    timing: 0.271086
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [06:03<00:00,  6.06s/it, loss=1.4928, casc_f1=1.0000, time_mae=0.20m]
  ✓ Improved F1 score: 0.9982 (thresholds: cascade=0.500, node=0.500)

Epoch 5 Results:
  Train Loss: 3.5299 | Val Loss: 1.6012
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 1.0000 | Val 1.0000
    Precision: Train 1.0000 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.9224 | Val 0.9963
    Precision: Train 0.8662 | Val 0.9988
    Recall:    Train 0.9863 | Val 0.9939

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.382 | Val 0.202
    Risk MSE:          Train 0.0558 | Val 0.0623
  ✓ Saved best model (New best MAE: 0.2023m, F1s: 1.000/0.996)

Epoch 6/100
--------------------------------------------------------------------------------
Training (Loss: 2.2654, Grad: 88.89): 100%|██████████████████████████████████████████████████████████████████████| 329/329 [1:14:46<00:00, 13.64s/it, cF1=1.000, nF1=0.990, tMAE=0.30m, rMSE=0.054, pL=0.013, tL=0.149] 

  Average gradient norm: 80.1866
  Average loss components:
    prediction: 0.024521
    powerflow: 10.000000
    capacity: 0.000000
    voltage: 0.000000
    frequency: 0.030650
    risk: 0.054115
    timing: 0.192599
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [07:11<00:00,  7.19s/it, loss=1.0928, casc_f1=1.0000, time_mae=0.14m]
  ✓ Improved F1 score: 0.9997 (thresholds: cascade=0.500, node=0.500)

Epoch 6 Results:
  Train Loss: 2.6792 | Val Loss: 1.2281
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 1.0000 | Val 1.0000
    Precision: Train 1.0000 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.9896 | Val 0.9994
    Precision: Train 0.9816 | Val 0.9989
    Recall:    Train 0.9978 | Val 0.9999

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.305 | Val 0.140
    Risk MSE:          Train 0.0541 | Val 0.0636
  ✓ Saved best model (New best MAE: 0.1403m, F1s: 1.000/0.999)

Epoch 7/100
--------------------------------------------------------------------------------
Training (Loss: 1.8649, Grad: 11.33): 100%|██████████████████████████████████████████████████████████████████████| 329/329 [1:22:44<00:00, 15.09s/it, cF1=1.000, nF1=0.998, tMAE=0.26m, rMSE=0.051, pL=0.008, tL=0.123] 

  Average gradient norm: 67.7706
  Average loss components:
    prediction: 0.009582
    powerflow: 10.000000
    frequency: 0.030398
    risk: 0.050803
    timing: 0.155299
Validation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [05:59<00:00,  5.99s/it, loss=1.2527, casc_f1=1.0000, time_mae=0.13m] 
  ✓ Improved F1 score: 1.0000 (thresholds: cascade=0.500, node=0.500)

Epoch 7 Results:
  Train Loss: 2.2801 | Val Loss: 1.1611
  Learning Rate: 0.000100

  CASCADE DETECTION (Thresh: 0.500):
    F1 Score:  Train 1.0000 | Val 1.0000
    Precision: Train 1.0000 | Val 1.0000
    Recall:    Train 1.0000 | Val 1.0000

  NODE FAILURE (Thresh: 0.500):
    F1 Score:  Train 0.9980 | Val 0.9999
    Precision: Train 0.9969 | Val 1.0000
    Recall:    Train 0.9990 | Val 0.9999

  CAUSAL PATH & RISK METRICS:
    Timing MAE (mins): Train 0.261 | Val 0.127
    Risk MSE:          Train 0.0508 | Val 0.0628
  ✓ Saved best model (New best MAE: 0.1271m, F1s: 1.000/1.000)
```


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