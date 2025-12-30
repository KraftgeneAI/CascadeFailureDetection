# Agentic Cascade Failure Detection System

Multi-agent AI system for real-time cascade failure prediction in power grid infrastructure.

## Overview

This system implements the proof-of-concept agentic AI architecture described in the research paper "AI-Driven Predictive Cascade Failure Analysis Using Multi-Modal Environmental-Infrastructure Data Fusion".

## Architecture

The system consists of four specialized agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Coordination Agent                           │
│                      (Orchestrator)                             │
│  - Agent lifecycle management                                   │
│  - Task scheduling                                              │
│  - Pipeline coordination                                        │
└───────────────┬─────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────────────┐
    ▼           ▼           ▼                   ▼
┌─────────┐ ┌─────────┐ ┌─────────┐      ┌─────────┐
│  Data   │ │Predict- │ │  Risk   │      │ Future  │
│Acquisit-│ │  ion    │ │ Assess- │ ...  │ Agents  │
│  ion    │ │ Agent   │ │  ment   │      │         │
│ Agent   │ │         │ │ Agent   │      │         │
└─────────┘ └─────────┘ └─────────┘      └─────────┘
```

### Agent Responsibilities

1. **Data Acquisition Agent**
   - Collects multi-modal data (environmental, infrastructure, robotic)
   - Preprocesses and normalizes data
   - Maintains temporal sequence buffer

2. **Prediction Agent**
   - Loads trained physics-informed GNN model
   - Runs cascade failure predictions
   - Identifies high-risk nodes and cascade paths

3. **Risk Assessment Agent**
   - Computes 7-dimensional risk vectors
   - Generates tiered alerts
   - Produces actionable recommendations

4. **Coordination Agent**
   - Orchestrates agent communication
   - Schedules prediction pipeline
   - Monitors system health

## Installation

```bash
# Clone the repository
git clone https://github.com/KraftgeneAI/CascadeFailureDetection.git
cd CascadeFailureDetection/agentic_cascade_system

# Install dependencies 
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run the multi-agent system
python main.py \
    --model_path ../../checkpoints/best_f1_model.pth \
    --data_dir ../../data \
    --duration 60
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to trained model checkpoint | Required |
| `--data_dir` | Directory containing data and topology | `./data` |
| `--topology_path` | Path to grid topology file | `data_dir/grid_topology.pkl` |
| `--device` | Device to use (cpu/cuda) | Auto-detect |
| `--prediction_interval` | Interval between predictions (seconds) | 1.0 |
| `--duration` | Duration to run (seconds, 0=indefinite) | 60.0 |
| `--log_level` | Logging level | INFO |

### Programmatic Usage

```python
import asyncio
from main import AgenticCascadeSystem

async def run_system():
    system = AgenticCascadeSystem(
        model_path="checkpoints/best_f1_model.pth",
        data_dir="./data",
        prediction_interval=0.5
    )
    
    await system.start()
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Get status
    status = system.get_status()
    print(f"Total predictions: {status['system_state']['total_predictions']}")
    
    await system.stop()

asyncio.run(run_system())
```

## Agent Communication

Agents communicate via an asynchronous message bus:

```python
from agents.base_agent import AgentMessage, MessageType

# Send a message
message = AgentMessage(
    receiver="prediction",
    message_type=MessageType.DATA,
    payload={"event": "predict", "batch_data": data}
)
await agent.send_message(message)

# Send and wait for response
response = await agent.send_and_wait(message, timeout=5.0)
```

## Message Types

| Type | Description |
|------|-------------|
| `DATA` | Data transfer between agents |
| `COMMAND` | Control commands |
| `QUERY` | Information requests |
| `RESPONSE` | Query/command responses |
| `ALERT` | Risk alerts |
| `PREDICTION` | Prediction results |
| `RISK_ASSESSMENT` | Risk assessment results |
| `COORDINATION` | Orchestration messages |

## Risk Assessment Framework

The system uses a 7-dimensional risk vector as described in the research paper:

| Dimension | Description |
|-----------|-------------|
| R₁: Threat Severity | External threat level from environmental factors |
| R₂: Vulnerability | Infrastructure weakness and susceptibility |
| R₃: Operational Impact | Potential operational consequences |
| R₄: Cascade Probability | Likelihood of failure propagation |
| R₅: Response Complexity | Difficulty of implementing response actions |
| R₆: Public Safety | Risk to public safety and welfare |
| R₇: Urgency | Time sensitivity of required response |

## Testing

```bash
# Run all tests
pytest test_agents.py -v

# Run specific test class
pytest test_agents.py::TestRiskAssessmentAgent -v
```

## Project Structure

```
agentic_cascade_system/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Base agent class and message bus
│   ├── data_acquisition_agent.py
│   ├── prediction_agent.py
│   ├── risk_assessment_agent.py
│   └── coordination_agent.py
├── main.py                     # Entry point
├── test_agents.py              # Test suite
├── requirements.txt
└── README.md
```

## License

Dual license - see LICENSE file.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
