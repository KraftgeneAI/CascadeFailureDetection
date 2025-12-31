# Agent Monitoring Guide

## How to Know If Agents Are Working Correctly

This guide explains how to verify that the multi-agent cascade detection system is functioning properly and what each agent is doing.

## Quick Start: Launch the Monitor

```bash
python monitor_agents.py --model_path checkpoints/best_f1_model.pth --data_dir ./data --verbose
```

This launches a real-time dashboard showing all agent activity.

---

## What Each Agent Does

### 1. **DataAcquisitionAgent** (Runs every ~500ms)

**Job:** Continuously collects and preprocesses multi-modal sensor data

**What to look for:**
- ✅ **Messages per second:** Should send ~2 DATA messages/second
- ✅ **Message type:** `MessageType.DATA` with event `"data_ready"`
- ✅ **Payload shows:** `num_timesteps` increasing from 1 to 60

**Signs it's working:**
```
[12:34:56] data_acquisition → broadcast | data
           └─ Data ready: 32 timesteps
```

**What it's doing internally:**
1. Simulates SCADA data (13 dimensions per node): voltage, power, frequency, etc.
2. Simulates PMU data (7 dimensions): high-speed measurements
3. Simulates environmental data (64+8 dimensions): satellite imagery, weather
4. Simulates robotic data (64+64+8 dimensions): visual, thermal, sensors
5. Normalizes all data using physics-based scaling (power/MVA, freq/60Hz)
6. Stores last 60 timesteps in rolling buffer
7. Broadcasts "data_ready" message every cycle

**Debugging:**
- If no DATA messages appear → Check if agent started (look in AGENT STATUS table)
- If error_count increasing → Check logs for data generation errors

---

### 2. **CoordinationAgent** (Runs prediction pipeline every 1-5 seconds)

**Job:** Orchestrates the prediction pipeline and manages all other agents

**What to look for:**
- ✅ **Scheduled tasks:** `prediction_pipeline` and `health_check` should be enabled
- ✅ **System state:** `agents_running` should show 4/4
- ✅ **Pipeline execution:** Every 1-5 seconds, should trigger prediction

**Signs it's working:**
```
[12:35:00] coordinator → data_acquisition | data
           └─ Request: batch

[12:35:00] coordinator → prediction | data
           └─ Prediction request
```

**What it's doing internally:**
1. **Pipeline coordination:**
   - Waits for DataAcquisition buffer to have ≥30 timesteps
   - Requests batch data from DataAcquisitionAgent
   - Sends batch to PredictionAgent
   - Forwards predictions to RiskAssessmentAgent

2. **Health monitoring:**
   - Checks all agents every 30 seconds
   - Monitors error counts
   - Could auto-restart failed agents (not yet implemented)

3. **Task scheduling:**
   - Manages periodic tasks with priorities
   - Ensures tasks run at correct intervals

**Debugging:**
- If predictions never run → Check if data buffer has reached 30 timesteps
- If "pipeline_running" stuck true → Check PredictionAgent for errors

---

### 3. **PredictionAgent** (Runs on-demand when coordinator requests)

**Job:** Runs the trained GNN model to predict cascade failures

**What to look for:**
- ✅ **Processing messages:** Should process DATA messages with event="predict"
- ✅ **Inference time:** Should be 50-500ms depending on hardware
- ✅ **Output:** PREDICTION messages with cascade_detected, probabilities, paths

**Signs it's working:**
```
[12:35:01] prediction → coordinator | prediction
           └─ CASCADE DETECTED! Probability: 0.742

OR

[12:35:01] prediction → coordinator | prediction
           └─ No cascade detected
```

**What it's doing internally:**
1. **Input preparation:**
   - Converts JSON batch_data to PyTorch tensors
   - Moves data to GPU if available
   - Ensures all required tensors present (scada, pmu, satellite, weather, visual, thermal, sensors, edge_attr, edge_index)

2. **Model inference:**
   - Forward pass through UnifiedCascadePredictionModel
   - Multi-modal embeddings → Graph Attention → Temporal GNN (LSTM) → Prediction heads
   - Outputs: failure_probability (per node), risk_scores (7-dim), voltages, frequency

3. **Post-processing:**
   - Applies thresholds (cascade_threshold=0.1, node_threshold=0.35)
   - Identifies high-risk nodes
   - Reconstructs cascade propagation path
   - Calculates severity

4. **Alert broadcasting:**
   - If cascade detected → sends high-priority ALERT message

**Debugging:**
- If "Model not initialized" error → Check model_path is correct
- If CUDA errors → Add `--device cpu` to run on CPU
- If all predictions show cascade → Check thresholds, may need adjustment

---

### 4. **RiskAssessmentAgent** (Processes predictions as they arrive)

**Job:** Translates predictions into actionable risk assessments and recommendations

**What to look for:**
- ✅ **Handles PREDICTION messages** from PredictionAgent
- ✅ **Generates ALERT messages** for HIGH/CRITICAL severity
- ✅ **7-dimensional risk scoring:** threat, vulnerability, impact, cascade prob, complexity, safety, urgency

**Signs it's working:**
```
[12:35:02] risk_assessment → broadcast | alert
           └─ Alert: CRITICAL

LOG OUTPUT:
2025-12-31 12:35:02 - WARNING - CASCADE ALERT: CRITICAL - 15 nodes at risk
```

**What it's doing internally:**
1. **Risk aggregation:**
   - Takes 7-dimensional risk vector from model
   - Applies weights: operational_impact (20%), cascade_prob (20%), threat (15%), vulnerability (15%), others (10% each)
   - Calculates aggregate risk score

2. **Severity determination:**
   - CRITICAL: risk > 0.75 OR >20 nodes OR cascade_prob > 0.8
   - HIGH: risk > 0.50 OR >10 nodes OR cascade_prob > 0.6
   - MODERATE: risk > 0.25 OR >5 nodes
   - LOW: everything else

3. **Time estimation:**
   - Estimates time until critical (15-35 min lead time from research paper)
   - Adjusts based on cascade probability

4. **Recommendation generation:**
   - CRITICAL: "IMMEDIATE: Initiate emergency response protocol"
   - HIGH: "Activate enhanced monitoring for affected region"
   - Component-specific: "Consider preemptive isolation of high-vulnerability nodes"
   - Node-specific: "Priority monitoring for nodes: [1, 5, 12, 23, 45]"

**Debugging:**
- If no alerts ever generated → Predictions might be below threshold
- If too many alerts → Adjust severity thresholds in risk_assessment_agent.py

---

## Message Flow Diagram

```
┌─────────────────┐
│ Every 500ms:    │
│ DataAcquisition │───┐
│ generates data  │   │
└─────────────────┘   │
                      ▼
                 MessageBus
                 (DATA msg)
                      │
                      │ (buffering... 30 timesteps)
                      │
┌─────────────────┐   │
│ Every 1-5s:     │◄──┘
│ Coordinator     │
│ checks buffer   │
└─────────────────┘
         │
         ▼
    Request batch
         │
         ▼
┌─────────────────┐
│ DataAcquisition │
│ prepares batch  │
└─────────────────┘
         │
         ▼
    Send batch
         │
         ▼
┌─────────────────┐
│ Prediction      │
│ runs GNN model  │───── 50-500ms inference
└─────────────────┘
         │
         ▼
  PREDICTION msg
         │
         ├──────────────┐
         │              │
         ▼              ▼
┌──────────────┐  ┌──────────────┐
│ Coordinator  │  │ RiskAssess   │
│ (logs)       │  │ (evaluate)   │
└──────────────┘  └──────────────┘
                         │
                         ▼
                    ALERT msg
                    (if HIGH/CRITICAL)
```

---

## Key Performance Indicators

### Normal Operation

| Metric | Expected Value | What It Means |
|--------|----------------|---------------|
| **DATA messages/sec** | 1-2 | DataAcquisition sending data |
| **Predictions/min** | 12-60 | Coordinator running pipeline |
| **Avg inference time** | 50-500ms | Prediction speed (GPU faster) |
| **Agent errors** | 0 | All agents healthy |
| **Agents running** | 4/4 | Full system operational |

### Warning Signs

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| No DATA messages | DataAcquisition not started | Check agent status, restart |
| No predictions | Buffer not full (< 30 timesteps) | Wait 15-30 seconds |
| High error count | Model loading failed | Check model_path, CUDA availability |
| Predictions very slow (>2s) | CPU inference, large network | Use GPU or smaller network |
| All predictions = cascade | Thresholds too low | Adjust cascade_threshold in checkpoint |

---

## Example: Successful System Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CASCADE DETECTION AGENT MONITOR                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Uptime: 45s | Iteration: 45 | Time: 12:35:23                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 SYSTEM STATUS
────────────────────────────────────────────────────────────────────────────────
  Status: RUNNING
  Agents Running: 4/4
  Total Predictions: 8
  Active Alerts: 0
  Last Prediction: 2025-12-31T12:35:20

🤖 AGENT STATUS
────────────────────────────────────────────────────────────────────────────────
Agent                     State        Messages   Errors   Last Activity
────────────────────────────────────────────────────────────────────────────────
DataAcquisitionAgent      🟢 RUNNING   180        0        0.3s ago
PredictionAgent           🟢 RUNNING   8          0        3.2s ago
RiskAssessmentAgent       🟢 RUNNING   8          0        3.1s ago
CoordinationAgent         🟢 RUNNING   45         0        0.1s ago

📨 MESSAGE BUS ACTIVITY
────────────────────────────────────────────────────────────────────────────────
  Total Messages: 241

  Message Types:
    data                  172 ( 71.4%) ████████████████████████████████████
    prediction             16 (  6.6%) ███
    response               24 (  10.0%) █████
    query                  12 (  5.0%) ██
    alert                   8 (  3.3%) █
    heartbeat               9 (  3.7%) █

📬 RECENT MESSAGES
────────────────────────────────────────────────────────────────────────────────
  [12:35:23] data_acquisition → broadcast | data
           └─ Data ready: 46 timesteps
  [12:35:22] data_acquisition → broadcast | data
           └─ Data ready: 45 timesteps
  [12:35:20] prediction → coordinator | prediction
           └─ No cascade detected
  [12:35:20] coordinator → prediction | data
           └─ Prediction request
  [12:35:20] coordinator → data_acquisition | data
           └─ Request: batch
```

---

## Advanced Monitoring

### Enable Debug Logging

Add to your main.py or monitor_agents.py:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Inspect Message Payloads

Check `message_bus.message_log` for full message history:

```python
# In Python debugger or script
for msg in coordinator.message_bus.message_log[-10:]:
    print(f"{msg.sender} → {msg.receiver}")
    print(f"  Type: {msg.message_type.value}")
    print(f"  Payload: {msg.payload}")
```

### Query Individual Agents

```python
# Get prediction statistics
from agents.base_agent import AgentMessage, MessageType

msg = AgentMessage(
    receiver='prediction',
    message_type=MessageType.QUERY,
    payload={'query': 'prediction_stats'},
    requires_response=True
)

response = await coordinator.send_and_wait(msg)
print(response.payload)
```

---

## Troubleshooting

### No Agents Start

**Symptom:** Monitor shows 0/4 agents running

**Causes:**
1. Model file not found
2. Topology file not found
3. Import errors

**Fix:**
```bash
# Check files exist
ls -la checkpoints/best_f1_model.pth
ls -la data/grid_topology.pkl

# Check imports
python -c "from scripts.multimodal_cascade_model import UnifiedCascadePredictionModel"
```

### Agents Hang

**Symptom:** State shows "PROCESSING" for >30 seconds

**Causes:**
1. Deadlock in message handling
2. Infinite loop in agent code
3. Blocking I/O operation

**Fix:**
- Add timeout to message handling
- Check for unhandled exceptions
- Use async/await properly

### Memory Issues

**Symptom:** System slows down over time

**Causes:**
1. Message log growing unbounded
2. Prediction history not cleared
3. Sequence buffer too large

**Fix:**
```python
# Limit message log size
if len(self.message_bus.message_log) > 10000:
    self.message_bus.message_log = self.message_bus.message_log[-5000:]
```

---

## Summary

**The agents are working correctly when:**

1. ✅ DataAcquisitionAgent sends DATA messages every ~500ms
2. ✅ CoordinationAgent triggers predictions every 1-5 seconds
3. ✅ PredictionAgent processes requests in 50-500ms
4. ✅ RiskAssessmentAgent generates alerts for high-risk scenarios
5. ✅ All agents show state=RUNNING with error_count=0
6. ✅ Message bus shows healthy mix of message types
7. ✅ System makes 12-60 predictions per minute

**Use the monitor to:**
- Verify real-time agent activity
- Debug message flow
- Track performance metrics
- Identify bottlenecks
- Validate predictions

The monitoring dashboard provides complete visibility into your multi-agent system!
