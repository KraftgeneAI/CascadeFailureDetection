# How to Run the Agentic Cascade Failure Detection System

## Quick Start (3 Steps)

### Step 1: Create Test Environment
```bash
cd scripts/agentic_cascade_system
python create_test_environment.py --data_dir ./data --num_nodes 50 --verbose
```

This creates:
- `data/grid_topology.pkl` - Synthetic power grid topology
- `checkpoints/test_model.pth` - Mock model checkpoint for testing

### Step 2: Run the System
```bash
python main.py --model_path checkpoints/test_model.pth --data_dir ./data --duration 30  --verbose
```

### Step 3: Watch the Output
You'll see logs showing:
- Agent initialization
- Data acquisition cycles
- Predictions being made
- Risk assessments
- Alerts (if cascade detected)

---

## What Each Agent Does

### 1. Data Acquisition Agent
**Purpose**: Collects sensor data from the power grid

**What it does every 0.5 seconds**:
- Generates synthetic SCADA data (voltage, power, load)
- Generates synthetic PMU data (frequency, phase angles)
- Generates environmental data (satellite, weather)
- Generates robotic sensor data (thermal, visual)
- Stores data in a rolling buffer (last 60 timesteps)
- Broadcasts "data_ready" message

**How to verify it's working**:
```
Look for logs like:
"Starting data acquisition loop..."
"DATA message received from data_acquisition"
```

### 2. Prediction Agent
**Purpose**: Runs the cascade failure prediction model

**What it does when triggered**:
- Receives batch data from coordinator
- Converts data to PyTorch tensors
- Runs model inference (GNN forward pass)
- Extracts failure probabilities per node
- Identifies high-risk nodes (probability > threshold)
- Builds cascade propagation path
- Broadcasts prediction results

**How to verify it's working**:
```
Look for logs like:
"Prediction Agent ready for inference requests..."
"cascade_detected: True/False"
"cascade_probability: 0.XXX"
```

### 3. Risk Assessment Agent
**Purpose**: Evaluates risk and generates alerts

**What it does for each prediction**:
- Receives prediction results
- Calculates 7-dimensional risk score:
  1. Cascade probability
  2. Affected node count
  3. Propagation speed
  4. Critical infrastructure involvement
  5. Geographic spread
  6. Load impact
  7. Confidence score
- Determines alert level (NORMAL/LOW/MEDIUM/HIGH/CRITICAL)
- Generates mitigation recommendations
- Logs comprehensive risk report

**How to verify it's working**:
```
Look for logs like:
"Risk Assessment Agent ready..."
"ALERT LEVEL: [level]"
"RISK REPORT..."
```

### 4. Coordination Agent (Orchestrator)
**Purpose**: Manages all other agents

**What it does**:
- Starts/stops all agents
- Schedules prediction pipeline (every 1 second)
- Routes messages between agents
- Monitors system health
- Tracks total predictions and alerts

**How to verify it's working**:
```
Look for logs like:
"Starting N agents..."
"All agents started"
"Total Predictions: X"
```

---

## Understanding the Output

### Normal Operation
```
2025-01-01 12:00:00 | Agent.DataAcquisitionAgent | INFO | Starting data acquisition loop...
2025-01-01 12:00:01 | Agent.PredictionAgent      | INFO | Prediction Agent ready...
2025-01-01 12:00:02 | Agent.CoordinationAgent    | INFO | All agents started
2025-01-01 12:00:03 | Agent.PredictionAgent      | DEBUG| Prediction completed: cascade_detected=False
```

### Cascade Detected
```
2025-01-01 12:00:05 | Agent.PredictionAgent      | WARNING | Cascade detected! Probability: 0.652
2025-01-01 12:00:05 | Agent.RiskAssessmentAgent  | WARNING | ALERT LEVEL: HIGH
2025-01-01 12:00:05 | Agent.RiskAssessmentAgent  | INFO    | High risk nodes: [12, 34, 7]
```

---

## Running with Real Model

To use your trained model instead of the mock:

```bash
python main.py --model_path ../checkpoints/best_f1_model.pth --data_dir ../data --verbose
```

The trained model checkpoint should contain:
- `model_state_dict`: PyTorch state dict
- `model_config`: Model configuration (optional)
- `cascade_threshold`: Optimized threshold (optional)
- `node_threshold`: Optimized threshold (optional)

---

## Troubleshooting

### "Could not load topology"
Run `create_test_environment.py` first to create the topology file.

### "No handler for message type"
This warning is normal - it means an agent received a broadcast it doesn't handle.

### "Using MOCK model"
This is expected when using `test_model.pth`. Replace with your trained model for real predictions.

### System seems stuck
Check if data acquisition is generating data:
- Look for "data_ready" messages in logs
- Ensure at least 30 timesteps collected before predictions start

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | Required | Path to model checkpoint |
| `--data_dir` | `./data` | Directory with topology file |
| `--topology_path` | `{data_dir}/grid_topology.pkl` | Explicit topology path |
| `--device` | Auto | `cpu` or `cuda` |
| `--prediction_interval` | `1.0` | Seconds between predictions |
| `--duration` | `60` | Run duration (0 = indefinite) |
| `--log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
