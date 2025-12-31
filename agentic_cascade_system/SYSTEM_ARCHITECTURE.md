# Agentic Cascade Failure Detection System - Complete Architecture Guide

## 1. How the Entire System Works (Step-by-Step)

### System Overview

This is a **Multi-Agent Agentic AI System** for real-time cascade failure prediction in power grids. It consists of 4 specialized agents that work together autonomously to detect, predict, and assess cascade failure risks.

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│            Coordination Agent (Orchestrator)                 │
│  - Schedules pipeline execution                             │
│  - Manages agent communication via MessageBus               │
│  - Handles agent lifecycle and health monitoring            │
└─────────────────────────────────────────────────────────────┘
                            ▼
        ┌───────────────────┴───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Data          │   │ Prediction    │   │ Risk          │
│ Acquisition   │──▶│ Agent         │──▶│ Assessment    │
│ Agent         │   │               │   │ Agent         │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

### Step-by-Step Execution Flow

#### **STEP 1: System Initialization** (`main.py`)

```python
# 1.1 Parse command-line arguments
args = parse_arguments()
# - model_path: Path to trained PyTorch model
# - data_dir: Directory containing topology and sensor data
# - duration: How long to run (60 seconds default)

# 1.2 Create MessageBus (communication backbone)
message_bus = MessageBus()
# - Shared queue for inter-agent messages
# - Supports pub/sub pattern with topic filtering
# - Thread-safe for concurrent agent operations

# 1.3 Instantiate all agents
data_agent = DataAcquisitionAgent(...)
prediction_agent = PredictionAgent(...)
risk_agent = RiskAssessmentAgent(...)
coordinator = CoordinationAgent(...)

# 1.4 Register agents with coordinator
coordinator.register_agent(data_agent)
coordinator.register_agent(prediction_agent)
coordinator.register_agent(risk_agent)

# 1.5 Start coordinator (which starts all other agents)
await coordinator.start()
```

---

#### **STEP 2: Data Acquisition Agent Startup** (`data_acquisition_agent.py`)

```python
async def start(self):
    # 2.1 Load grid topology (edge connections)
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
    self.edge_index = topology['edge_index']  # [2, num_edges] tensor
    self.num_nodes = self.edge_index.max().item() + 1
    
    # 2.2 Load sensor data files
    env_data = np.load('environmental_data.npy')      # [T, N, 10] features
    infra_data = np.load('infrastructure_data.npy')   # [T, N, 15] features
    robot_data = np.load('robotic_platform_data.npy') # [T, N, 12] features
    
    # 2.3 Start background collection loop
    asyncio.create_task(self._collection_loop())
```

**What happens in `_collection_loop()`:**

```python
while self.running:
    # Every 5 seconds:
    # 1. Read current timestep data
    timestamp = datetime.now()
    env_features = self.env_data[self.current_idx]    # [N, 10]
    infra_features = self.infra_data[self.current_idx] # [N, 15]
    robot_features = self.robot_data[self.current_idx] # [N, 12]
    
    # 2. Package into message
    data_message = Message(
        type=MessageType.DATA_READY,
        sender='data_acquisition',
        data={
            'timestamp': timestamp,
            'environmental': env_features,
            'infrastructure': infra_features,
            'robotic': robot_features,
            'edge_index': self.edge_index
        }
    )
    
    # 3. Publish to MessageBus
    self.message_bus.publish(data_message)
    
    # 4. Log statistics
    self.logger.info(f"Collected data at t={self.current_idx}")
    
    # 5. Wait 5 seconds
    await asyncio.sleep(5.0)
```

---

#### **STEP 3: Prediction Agent Processing** (`prediction_agent.py`)

The Prediction Agent **listens** for `DATA_READY` messages:

```python
async def _process_messages(self):
    while self.running:
        # 3.1 Check for new messages
        message = await self.message_bus.get_message('prediction')
        
        if message.type == MessageType.DATA_READY:
            # 3.2 Extract data from message
            env_data = message.data['environmental']   # [N, 10]
            infra_data = message.data['infrastructure'] # [N, 15]
            robot_data = message.data['robotic']       # [N, 12]
            edge_index = message.data['edge_index']    # [2, E]
            
            # 3.3 Run inference through the neural network
            prediction = await self._run_inference(
                env_data, infra_data, robot_data, edge_index
            )
            # Returns:
            # - cascade_prob: Probability of cascade [0-1]
            # - node_risk: Risk score per node [N]
            # - cascade_paths: Predicted propagation sequence
            # - timing: When each node fails
            
            # 3.4 Send prediction to Risk Assessment Agent
            pred_message = Message(
                type=MessageType.PREDICTION_COMPLETE,
                sender='prediction',
                recipient='risk_assessment',
                data={
                    'timestamp': message.data['timestamp'],
                    'cascade_prob': prediction['cascade_prob'],
                    'node_risk': prediction['node_risk'],
                    'cascade_paths': prediction['cascade_paths'],
                    'timing': prediction['timing']
                }
            )
            self.message_bus.publish(pred_message)
```

**Inside `_run_inference()` - The Neural Network Forward Pass:**

```python
async def _run_inference(self, env, infra, robot, edge_index):
    # Convert numpy arrays to PyTorch tensors
    env_tensor = torch.FloatTensor(env).to(self.device)       # [N, 10]
    infra_tensor = torch.FloatTensor(infra).to(self.device)   # [N, 15]
    robot_tensor = torch.FloatTensor(robot).to(self.device)   # [N, 12]
    edge_index_tensor = edge_index.to(self.device)            # [2, E]
    
    # Add batch dimension
    env_tensor = env_tensor.unsqueeze(0)      # [1, N, 10]
    infra_tensor = infra_tensor.unsqueeze(0)  # [1, N, 15]
    robot_tensor = robot_tensor.unsqueeze(0)  # [1, N, 12]
    
    # Forward pass through UnifiedCascadePredictionModel
    with torch.no_grad():
        outputs = self.model(
            environmental_features=env_tensor,
            infrastructure_features=infra_tensor,
            robotic_features=robot_tensor,
            edge_index=edge_index_tensor
        )
    
    # Extract predictions
    cascade_logits = outputs['cascade_prediction']  # [1, 2]
    cascade_prob = F.softmax(cascade_logits, dim=1)[0, 1].item()
    
    node_risk_scores = outputs['node_risk'].squeeze(0).cpu().numpy()  # [N]
    
    # Detect cascade if probability > threshold
    is_cascade = cascade_prob > self.cascade_threshold
    
    if is_cascade:
        # Reconstruct cascade propagation path
        cascade_paths = self._reconstruct_cascade_path(
            node_risk_scores, edge_index
        )
    else:
        cascade_paths = []
    
    return {
        'cascade_prob': cascade_prob,
        'node_risk': node_risk_scores,
        'cascade_paths': cascade_paths,
        'timing': outputs.get('timing_pred', None)
    }
```

---

#### **STEP 4: Risk Assessment Agent** (`risk_assessment_agent.py`)

The Risk Agent **listens** for `PREDICTION_COMPLETE` messages:

```python
async def _process_messages(self):
    while self.running:
        message = await self.message_bus.get_message('risk_assessment')
        
        if message.type == MessageType.PREDICTION_COMPLETE:
            # 4.1 Extract prediction results
            cascade_prob = message.data['cascade_prob']
            node_risk = message.data['node_risk']
            cascade_paths = message.data['cascade_paths']
            
            # 4.2 Calculate 7-dimensional risk scores
            risk_metrics = self._calculate_risk_metrics(
                cascade_prob, node_risk, cascade_paths
            )
            # Returns:
            # - severity: How bad is the cascade? [0-1]
            # - propagation_speed: How fast does it spread? [0-1]
            # - spatial_extent: How many nodes affected? [0-1]
            # - critical_nodes: Are key substations at risk? [0-1]
            # - recovery_difficulty: Hard to restore? [0-1]
            # - economic_impact: Financial loss estimation [0-1]
            # - safety_risk: Human safety concerns [0-1]
            
            # 4.3 Calculate overall risk level
            overall_risk = np.mean(list(risk_metrics.values()))
            
            # 4.4 Determine alert level
            if overall_risk >= 0.7:
                alert_level = "CRITICAL"
            elif overall_risk >= 0.5:
                alert_level = "HIGH"
            elif overall_risk >= 0.3:
                alert_level = "MEDIUM"
            else:
                alert_level = "LOW"
            
            # 4.5 Generate mitigation recommendations
            recommendations = self._generate_recommendations(
                risk_metrics, cascade_paths, node_risk
            )
            # Examples:
            # - "Immediately isolate node 47 (critical substation)"
            # - "Reroute power flow away from high-risk corridor"
            # - "Deploy mobile generation units to nodes [12, 34, 56]"
            
            # 4.6 Create alert message
            alert_message = Message(
                type=MessageType.ALERT_GENERATED,
                sender='risk_assessment',
                recipient='coordinator',
                data={
                    'timestamp': message.data['timestamp'],
                    'alert_level': alert_level,
                    'overall_risk': overall_risk,
                    'risk_breakdown': risk_metrics,
                    'cascade_probability': cascade_prob,
                    'affected_nodes': len(cascade_paths),
                    'recommendations': recommendations
                }
            )
            
            # 4.7 Publish alert
            self.message_bus.publish(alert_message)
            
            # 4.8 Log to file and console
            self.logger.warning(
                f"[{alert_level}] Risk={overall_risk:.3f} | "
                f"Cascade_Prob={cascade_prob:.3f} | "
                f"Nodes_Affected={len(cascade_paths)}"
            )
```

**Risk Calculation Details:**

```python
def _calculate_risk_metrics(self, cascade_prob, node_risk, cascade_paths):
    # Severity: Based on cascade probability and affected nodes
    severity = cascade_prob * (len(cascade_paths) / self.num_nodes)
    
    # Propagation Speed: Based on timing predictions
    if cascade_paths:
        avg_time_to_failure = np.mean([p['time'] for p in cascade_paths])
        propagation_speed = 1.0 / (avg_time_to_failure + 1e-6)
    else:
        propagation_speed = 0.0
    
    # Spatial Extent: Percentage of network affected
    spatial_extent = len(cascade_paths) / self.num_nodes
    
    # Critical Nodes: Check if high-importance nodes are at risk
    critical_threshold = 0.7
    critical_nodes_affected = np.sum(node_risk > critical_threshold)
    critical_nodes = critical_nodes_affected / self.num_nodes
    
    # Recovery Difficulty: Complex cascades are harder to recover
    recovery_difficulty = severity * (1 + 0.5 * propagation_speed)
    
    # Economic Impact: Estimate based on affected load
    economic_impact = severity * 0.8 + spatial_extent * 0.2
    
    # Safety Risk: High if critical infrastructure affected
    safety_risk = critical_nodes * 0.6 + severity * 0.4
    
    return {
        'severity': float(np.clip(severity, 0, 1)),
        'propagation_speed': float(np.clip(propagation_speed, 0, 1)),
        'spatial_extent': float(np.clip(spatial_extent, 0, 1)),
        'critical_nodes': float(np.clip(critical_nodes, 0, 1)),
        'recovery_difficulty': float(np.clip(recovery_difficulty, 0, 1)),
        'economic_impact': float(np.clip(economic_impact, 0, 1)),
        'safety_risk': float(np.clip(safety_risk, 0, 1))
    }
```

---

#### **STEP 5: Coordination Agent** (`coordination_agent.py`)

The Coordinator **orchestrates** the entire pipeline:

```python
async def _orchestrate_pipeline(self):
    """Main orchestration loop"""
    while self.running:
        # 5.1 Wait for next cycle (aligned with data collection)
        await asyncio.sleep(self.prediction_interval)  # 5 seconds
        
        # 5.2 Check agent health
        for agent_id, agent in self.agents.items():
            if agent.state != AgentState.RUNNING:
                self.logger.error(f"Agent {agent_id} is not running!")
        
        # 5.3 Monitor message flow
        # The coordinator doesn't actively trigger agents
        # It just monitors that messages are flowing correctly
        
        # 5.4 Log system statistics
        self.logger.info(
            f"Cycle {self.cycle_count}: "
            f"Messages_Processed={self.message_count}"
        )
        
        self.cycle_count += 1


async def _monitor_alerts(self):
    """Listen for critical alerts"""
    while self.running:
        message = await self.message_bus.get_message('coordinator')
        
        if message.type == MessageType.ALERT_GENERATED:
            alert_level = message.data['alert_level']
            
            if alert_level in ['CRITICAL', 'HIGH']:
                # Log to file
                self._log_critical_alert(message.data)
                
                # Could trigger external actions here:
                # - Send SMS/email to operators
                # - Trigger automated grid controls
                # - Escalate to emergency response team
                
                self.logger.critical(
                    f"⚠️  {alert_level} ALERT: "
                    f"Risk={message.data['overall_risk']:.3f}"
                )
```

---

### Complete Data Flow Example

```
Time t=0:
1. DataAgent reads sensor data at timestep 0
2. DataAgent publishes DATA_READY message

Time t=0.1s:
3. PredictionAgent receives DATA_READY
4. PredictionAgent runs neural network inference
5. Model predicts: cascade_prob=0.82, node_risk=[0.1, 0.3, ..., 0.9]
6. PredictionAgent publishes PREDICTION_COMPLETE

Time t=0.2s:
7. RiskAgent receives PREDICTION_COMPLETE
8. RiskAgent calculates 7D risk metrics
9. RiskAgent determines: CRITICAL alert (overall_risk=0.85)
10. RiskAgent generates recommendations: ["Isolate node 47", ...]
11. RiskAgent publishes ALERT_GENERATED

Time t=0.3s:
12. Coordinator receives ALERT_GENERATED
13. Coordinator logs critical alert to file
14. System waits for next cycle (5 seconds)

Time t=5s:
15. Repeat from step 1 with timestep 1
```

---

## 2. Why Don't We Need LLMs? What Are the Agents Using?

### Current Implementation: **Rule-Based Agents**

The current system uses **deterministic, rule-based agents** - NOT large language models. Here's what each agent actually does:

#### Data Acquisition Agent
- **No AI/ML**: Just reads numpy arrays from disk
- **Logic**: Simple file I/O and data packaging
- **Code**: Pure Python with asyncio

#### Prediction Agent
- **Uses ML**: Your trained PyTorch model (`UnifiedCascadePredictionModel`)
- **Type**: Specialized graph neural network (GNN)
- **Purpose**: Predict cascade failures based on physics and learned patterns
- **NOT an LLM**: It's a domain-specific neural network trained on power grid data

#### Risk Assessment Agent
- **No AI/ML**: Uses mathematical formulas and thresholds
- **Logic**: 
  ```python
  severity = cascade_prob * (affected_nodes / total_nodes)
  alert_level = "CRITICAL" if overall_risk >= 0.7 else "HIGH" if ...
  ```
- **Code**: Deterministic calculations

#### Coordination Agent
- **No AI/ML**: Simple orchestration logic
- **Logic**: Start agents, monitor health, log statistics
- **Code**: Pure Python asyncio coordination

### What Makes It "Agentic"?

The system is "agentic" because:

1. **Autonomy**: Each agent runs independently in its own async loop
2. **Communication**: Agents exchange messages via the MessageBus
3. **Specialization**: Each agent has a specific role (data, prediction, risk, coordination)
4. **Decentralization**: No central controller dictating every step
5. **Reactive**: Agents respond to events (messages) rather than following a script

**This is NOT the same as "AI agents" with LLMs!**

---

## 3. Rule-Based vs LLM-Based Agents: Which Is Better?

### Current System: Rule-Based Agents

**Pros:**
- ✅ **Fast**: Sub-second response time
- ✅ **Deterministic**: Same input → same output (reproducible)
- ✅ **Transparent**: You can see exactly why a decision was made
- ✅ **No API costs**: Runs entirely on your hardware
- ✅ **Reliable**: No hallucinations or unexpected behavior
- ✅ **Low latency**: Critical for real-time grid control (< 1 second)
- ✅ **Explainable**: Required for safety-critical infrastructure

**Cons:**
- ❌ Limited to predefined logic
- ❌ Can't handle novel situations outside rules
- ❌ Requires manual updates for new scenarios

### LLM-Based Agents (e.g., Claude, GPT-4o)

**Pros:**
- ✅ Can handle novel, unexpected situations
- ✅ Natural language reasoning
- ✅ Can explain decisions in human terms
- ✅ Adaptable without retraining

**Cons:**
- ❌ **Slow**: 2-10 seconds per decision (unacceptable for real-time control)
- ❌ **Expensive**: $0.01-0.10 per API call × 1000s of calls/day
- ❌ **Non-deterministic**: Same input → different outputs
- ❌ **Unpredictable**: Can hallucinate or make bizarre decisions
- ❌ **Not explainable**: Can't prove why it made a decision (regulatory issue)
- ❌ **Requires internet**: Not suitable for air-gapped critical infrastructure
- ❌ **Security risks**: Prompt injection, data leakage

### Hybrid Approach: Rule-Based + Domain-Specific ML

**Your current system is optimal for power grid applications:**

```
┌─────────────────────────────────────────────────────────┐
│  Rule-Based Agents (Fast, Deterministic)                │
│  - Data collection                                      │
│  - Risk assessment                                      │
│  - Coordination                                         │
└─────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Domain-Specific Neural Network (Accurate, Fast)        │
│  - Trained on power grid physics                        │
│  - Graph neural network for topology                    │
│  - Temporal prediction for cascade propagation          │
│  - Inference time: ~50ms                                │
└─────────────────────────────────────────────────────────┘
```

### When to Use LLMs in This System

LLMs could be useful for **non-critical tasks**:

1. **Generating operator reports**: Convert risk metrics to natural language
2. **Answering operator questions**: "Why did node 47 fail?"
3. **Strategy planning**: Long-term grid upgrades (not real-time)
4. **Training simulations**: Generate synthetic failure scenarios

But **NEVER** use LLMs for:
- Real-time cascade detection (too slow)
- Safety-critical decisions (not reliable)
- High-frequency control (too expensive)

### Recommendation for Your Use Case

**Stick with the current rule-based + domain ML approach.**

**Why:**
- Power grid control requires < 100ms latency
- Safety-critical systems need deterministic behavior
- Regulatory compliance requires explainability
- Your GNN model is already achieving excellent results (see training logs)

**If you want to add LLMs**, use them as an **advisory layer**:
```python
# After risk assessment (non-blocking)
async def _generate_operator_report(risk_data):
    prompt = f"Explain this cascade failure risk: {risk_data}"
    report = await llm_api.generate(prompt)  # Runs in parallel
    # Operator can read this while system handles the emergency
```

---

## Summary

1. **The system works by**: 4 specialized agents communicating via MessageBus, with a domain-specific GNN doing the heavy ML lifting
2. **No LLMs needed because**: Rule-based agents are faster, more reliable, and deterministic - perfect for real-time critical infrastructure
3. **Better approach**: Your current hybrid (rules + domain ML) is optimal. Only add LLMs for non-critical advisory tasks.

The fix ensures PyTorch tensors use correct syntax, and the system is now fully operational for real-time cascade failure detection.
