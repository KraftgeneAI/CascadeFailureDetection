"""
Agent Validation Script
=======================
Validates that each agent in the multi-agent system is working correctly.
Tests each component independently and then tests the full pipeline.

Run this to verify your agentic framework is functioning properly.

Author: Kraftgene AI Inc.
"""

import asyncio
import sys
import os
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import MessageBus, AgentMessage, MessageType, AgentState
from agents.data_acquisition_agent import DataAcquisitionAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.coordination_agent import CoordinationAgent


class AgentValidator:
    """Validates each agent component independently."""
    
    def __init__(self, model_path: str, data_dir: str):
        self.model_path = model_path
        self.data_dir = data_dir
        self.results = {}
        
    def print_header(self, title: str):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        
    def print_result(self, test_name: str, passed: bool, details: str = ""):
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if details:
            print(f"         {details}")
        self.results[test_name] = passed
        
    async def validate_message_bus(self):
        """Test 1: Validate MessageBus communication."""
        self.print_header("TEST 1: MessageBus Communication")
        
        message_bus = MessageBus()
        received_messages = []
        
        # Create a mock handler
        async def mock_handler(msg):
            received_messages.append(msg)
            
        # Test subscription
        message_bus.subscribe("test_agent", MessageType.DATA, mock_handler)
        self.print_result(
            "Subscription registration",
            MessageType.DATA in message_bus.subscriptions,
            f"Subscriptions: {list(message_bus.subscriptions.keys())}"
        )
        
        # Test message publishing
        test_message = AgentMessage(
            sender="sender_agent",
            receiver="test_agent",
            message_type=MessageType.DATA,
            payload={"test": "data"},
            priority=1
        )
        
        await message_bus.publish(test_message)
        await asyncio.sleep(0.1)  # Allow message to be processed
        
        self.print_result(
            "Message delivery",
            len(received_messages) == 1,
            f"Received {len(received_messages)} message(s)"
        )
        
        # Test broadcast
        received_messages.clear()
        broadcast_message = AgentMessage(
            sender="broadcaster",
            receiver=None,  # Broadcast
            message_type=MessageType.DATA,
            payload={"broadcast": True},
            priority=1
        )
        
        await message_bus.publish(broadcast_message)
        await asyncio.sleep(0.1)
        
        self.print_result(
            "Broadcast delivery",
            len(received_messages) == 1,
            f"Broadcast reached {len(received_messages)} subscriber(s)"
        )
        
    async def validate_data_acquisition_agent(self):
        """Test 2: Validate DataAcquisitionAgent."""
        self.print_header("TEST 2: DataAcquisitionAgent")
        
        message_bus = MessageBus()
        agent = DataAcquisitionAgent(
            agent_id="test_data_agent",
            message_bus=message_bus,
            data_dir=self.data_dir
        )
        
        # Test initialization
        await agent.initialize()
        self.print_result(
            "Agent initialization",
            agent.state == AgentState.RUNNING,
            f"State: {agent.state.value}, Nodes: {agent.num_nodes}"
        )
        
        # Test data generation
        env_data = agent._collect_environmental_data()
        self.print_result(
            "Environmental data generation",
            env_data is not None and 'satellite' in env_data,
            f"Keys: {list(env_data.keys()) if env_data else 'None'}"
        )
        
        infra_data = agent._collect_infrastructure_data()
        self.print_result(
            "Infrastructure data generation",
            infra_data is not None and 'scada' in infra_data,
            f"Keys: {list(infra_data.keys()) if infra_data else 'None'}"
        )
        
        # Test data buffer
        for _ in range(5):
            agent._process_and_store_data(env_data, infra_data)
            
        self.print_result(
            "Data buffer storage",
            len(agent.data_buffer) == 5,
            f"Buffer size: {len(agent.data_buffer)}"
        )
        
        # Test batch preparation
        batch = agent._prepare_batch_data()
        self.print_result(
            "Batch data preparation",
            batch is not None and 'node_features' in batch,
            f"Batch keys: {list(batch.keys()) if batch else 'None'}"
        )
        
        # Verify tensor shapes
        if batch:
            nf_shape = batch['node_features'].shape
            self.print_result(
                "Node features shape",
                len(nf_shape) == 3,  # [seq_len, num_nodes, features]
                f"Shape: {nf_shape}"
            )
            
        await agent.stop()
        
    async def validate_prediction_agent(self):
        """Test 3: Validate PredictionAgent."""
        self.print_header("TEST 3: PredictionAgent")
        
        message_bus = MessageBus()
        agent = PredictionAgent(
            agent_id="test_pred_agent",
            message_bus=message_bus,
            model_path=self.model_path,
            device="cpu"
        )
        
        # Test initialization
        await agent.initialize()
        self.print_result(
            "Agent initialization",
            agent.state == AgentState.RUNNING,
            f"State: {agent.state.value}, Model type: {'TRAINED' if not agent.is_mock_model else 'MOCK'}"
        )
        
        # Create synthetic batch for testing
        num_nodes = 118
        seq_len = 10
        
        test_batch = {
            'node_features': torch.randn(seq_len, num_nodes, 24),
            'edge_index': torch.randint(0, num_nodes, (2, 200)),
            'edge_attr': torch.randn(200, 8),
            'env_features': torch.randn(seq_len, 32),
            'infra_features': torch.randn(seq_len, 64),
            'robotic_features': torch.randn(seq_len, 16),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test prediction
        try:
            result = await agent._run_prediction(test_batch)
            self.print_result(
                "Model inference",
                result is not None and 'cascade_detected' in result,
                f"Cascade detected: {result.get('cascade_detected', 'N/A')}, Probability: {result.get('cascade_probability', 0):.4f}"
            )
            
            self.print_result(
                "Node risk scores",
                'node_risks' in result and len(result['node_risks']) > 0,
                f"High risk nodes: {len(result.get('high_risk_nodes', []))}"
            )
            
            self.print_result(
                "Timing predictions",
                'timing_predictions' in result,
                f"Has timing data: {'timing_predictions' in result}"
            )
            
        except Exception as e:
            self.print_result("Model inference", False, f"Error: {str(e)}")
            
        await agent.stop()
        
    async def validate_risk_assessment_agent(self):
        """Test 4: Validate RiskAssessmentAgent."""
        self.print_header("TEST 4: RiskAssessmentAgent")
        
        message_bus = MessageBus()
        agent = RiskAssessmentAgent(
            agent_id="test_risk_agent",
            message_bus=message_bus
        )
        
        # Test initialization
        await agent.initialize()
        self.print_result(
            "Agent initialization",
            agent.state == AgentState.RUNNING,
            f"State: {agent.state.value}"
        )
        
        # Create mock prediction result
        mock_prediction = {
            'cascade_detected': True,
            'cascade_probability': 0.85,
            'node_risks': np.random.rand(118).tolist(),
            'high_risk_nodes': [5, 12, 23, 45, 67],
            'timing_predictions': np.random.rand(118).tolist(),
            'path_predictions': np.random.rand(118, 118).tolist(),
            'confidence': 0.92,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test risk calculation
        risk_score = agent._calculate_risk_score(mock_prediction)
        self.print_result(
            "Risk score calculation",
            0 <= risk_score <= 1,
            f"Risk score: {risk_score:.4f}"
        )
        
        # Test alert level determination
        alert_level = agent._determine_alert_level(risk_score, mock_prediction)
        self.print_result(
            "Alert level determination",
            alert_level in ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            f"Alert level: {alert_level}"
        )
        
        # Test mitigation generation
        mitigations = agent._generate_mitigations(mock_prediction, alert_level)
        self.print_result(
            "Mitigation recommendations",
            len(mitigations) > 0,
            f"Generated {len(mitigations)} recommendation(s)"
        )
        
        # Test full assessment
        assessment = await agent._perform_assessment(mock_prediction)
        self.print_result(
            "Full risk assessment",
            assessment is not None and 'alert_level' in assessment,
            f"Assessment keys: {list(assessment.keys()) if assessment else 'None'}"
        )
        
        await agent.stop()
        
    async def validate_full_pipeline(self):
        """Test 5: Validate full agent pipeline integration."""
        self.print_header("TEST 5: Full Pipeline Integration")
        
        message_bus = MessageBus()
        
        # Track messages through the pipeline
        pipeline_events = []
        
        # Create all agents
        data_agent = DataAcquisitionAgent(
            agent_id="data_acquisition",
            message_bus=message_bus,
            data_dir=self.data_dir
        )
        
        pred_agent = PredictionAgent(
            agent_id="prediction",
            message_bus=message_bus,
            model_path=self.model_path,
            device="cpu"
        )
        
        risk_agent = RiskAssessmentAgent(
            agent_id="risk_assessment",
            message_bus=message_bus
        )
        
        coord_agent = CoordinationAgent(
            agent_id="coordinator",
            message_bus=message_bus,
            prediction_interval=2.0  # Fast for testing
        )
        
        # Register agents
        coord_agent.register_agent(data_agent)
        coord_agent.register_agent(pred_agent)
        coord_agent.register_agent(risk_agent)
        
        self.print_result(
            "Agent registration",
            len(coord_agent.agents) == 3,
            f"Registered {len(coord_agent.agents)} agents"
        )
        
        # Initialize all agents
        await data_agent.initialize()
        await pred_agent.initialize()
        await risk_agent.initialize()
        await coord_agent.initialize()
        
        all_running = all(a.state == AgentState.RUNNING for a in [data_agent, pred_agent, risk_agent, coord_agent])
        self.print_result(
            "All agents initialized",
            all_running,
            f"States: DA={data_agent.state.value}, PA={pred_agent.state.value}, RA={risk_agent.state.value}, CA={coord_agent.state.value}"
        )
        
        # Collect data manually
        print("\n  [INFO] Collecting data samples...")
        for i in range(15):
            env_data = data_agent._collect_environmental_data()
            infra_data = data_agent._collect_infrastructure_data()
            data_agent._process_and_store_data(env_data, infra_data)
            
        self.print_result(
            "Data collection",
            len(data_agent.data_buffer) >= 10,
            f"Collected {len(data_agent.data_buffer)} samples"
        )
        
        # Prepare batch and run prediction
        print("  [INFO] Running prediction pipeline...")
        batch = data_agent._prepare_batch_data()
        
        if batch:
            prediction_result = await pred_agent._run_prediction(batch)
            self.print_result(
                "Prediction execution",
                prediction_result is not None,
                f"Cascade detected: {prediction_result.get('cascade_detected', 'N/A')}"
            )
            
            # Run risk assessment
            if prediction_result:
                risk_assessment = await risk_agent._perform_assessment(prediction_result)
                self.print_result(
                    "Risk assessment execution",
                    risk_assessment is not None,
                    f"Alert level: {risk_assessment.get('alert_level', 'N/A')}"
                )
                
                # Print detailed results
                print("\n  " + "-" * 50)
                print("  PIPELINE OUTPUT SUMMARY:")
                print("  " + "-" * 50)
                print(f"  Cascade Probability: {prediction_result.get('cascade_probability', 0):.4f}")
                print(f"  High Risk Nodes: {len(prediction_result.get('high_risk_nodes', []))}")
                print(f"  Risk Score: {risk_assessment.get('risk_score', 0):.4f}")
                print(f"  Alert Level: {risk_assessment.get('alert_level', 'N/A')}")
                print(f"  Mitigations: {len(risk_assessment.get('mitigations', []))}")
                
        # Cleanup
        await data_agent.stop()
        await pred_agent.stop()
        await risk_agent.stop()
        await coord_agent.stop()
        
    def print_summary(self):
        """Print final validation summary."""
        self.print_header("VALIDATION SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v)
        failed = total - passed
        
        print(f"\n  Total Tests: {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Success Rate: {100 * passed / total:.1f}%")
        
        if failed > 0:
            print("\n  Failed Tests:")
            for test, result in self.results.items():
                if not result:
                    print(f"    - {test}")
                    
        print("\n" + "=" * 60)
        
        return failed == 0


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Multi-Agent System")
    parser.add_argument("--model_path", type=str, default="../checkpoints/best_f1_model.pth")
    parser.add_argument("--data_dir", type=str, default="../data")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MULTI-AGENT SYSTEM VALIDATION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print(f"\n  Model Path: {args.model_path}")
    print(f"  Data Dir: {args.data_dir}")
    
    validator = AgentValidator(args.model_path, args.data_dir)
    
    # Run all validations
    await validator.validate_message_bus()
    await validator.validate_data_acquisition_agent()
    await validator.validate_prediction_agent()
    await validator.validate_risk_assessment_agent()
    await validator.validate_full_pipeline()
    
    # Print summary
    success = validator.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
