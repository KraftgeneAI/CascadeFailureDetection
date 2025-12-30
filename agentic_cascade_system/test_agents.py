"""
Test Suite for Multi-Agent Cascade Failure Detection System
============================================================
Unit tests and integration tests for all agents.

Author: Kraftgene AI Inc.
"""

import asyncio
import pytest
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import tempfile
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agents.base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, 
    AgentState, MessageBus
)
from agents.data_acquisition_agent import DataAcquisitionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent, AlertSeverity


class TestBaseAgent:
    """Tests for BaseAgent class"""
    
    def test_message_creation(self):
        """Test message creation and serialization"""
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=MessageType.DATA,
            payload={"key": "value"},
            priority=7
        )
        
        # Test serialization
        msg_dict = msg.to_dict()
        assert msg_dict["sender"] == "agent1"
        assert msg_dict["receiver"] == "agent2"
        assert msg_dict["priority"] == 7
        
        # Test deserialization
        msg2 = AgentMessage.from_dict(msg_dict)
        assert msg2.sender == msg.sender
        assert msg2.message_type == msg.message_type
    
    def test_message_bus_registration(self):
        """Test agent registration with message bus"""
        bus = MessageBus()
        
        class DummyAgent(BaseAgent):
            async def initialize(self):
                pass
            async def execute(self):
                pass
        
        agent = DummyAgent("test_id", "TestAgent")
        bus.register_agent(agent)
        
        assert "test_id" in bus.agents
        assert agent._message_bus == bus


class TestDataAcquisitionAgent:
    """Tests for DataAcquisitionAgent"""
    
    @pytest.fixture
    def temp_topology(self):
        """Create temporary topology file"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            topology = {
                'edge_index': np.array([[0, 1, 2], [1, 2, 0]]),
            }
            pickle.dump(topology, f)
            return f.name
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_topology):
        """Test agent initialization"""
        agent = DataAcquisitionAgent(
            agent_id="test_data",
            data_dir="./data",
            topology_path=temp_topology
        )
        
        await agent.initialize()
        
        assert agent.num_nodes == 3
        assert agent.num_edges == 3
        assert agent.state != AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_data_preprocessing(self, temp_topology):
        """Test data preprocessing"""
        agent = DataAcquisitionAgent(
            agent_id="test_data",
            data_dir="./data",
            topology_path=temp_topology
        )
        
        await agent.initialize()
        
        # Acquire data
        await agent._acquire_environmental_data()
        await agent._acquire_infrastructure_data()
        await agent._acquire_robotic_data()
        
        # Preprocess
        preprocessed = agent._preprocess_data()
        
        # Check normalization
        assert 'scada_data' in preprocessed
        assert 'pmu_sequence' in preprocessed
        
        # Power values should be normalized by base_mva
        scada = preprocessed['scada_data']
        assert scada.shape == (3, 13)


class TestRiskAssessmentAgent:
    """Tests for RiskAssessmentAgent"""
    
    @pytest.fixture
    def risk_agent(self):
        """Create risk assessment agent"""
        agent = RiskAssessmentAgent(agent_id="test_risk")
        asyncio.get_event_loop().run_until_complete(agent.initialize())
        return agent
    
    def test_severity_determination(self, risk_agent):
        """Test severity determination logic"""
        # Low risk prediction
        low_risk_pred = {
            'cascade_detected': False,
            'cascade_probability': 0.1,
            'high_risk_nodes': [],
            'risk_assessment': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }
        
        result = risk_agent.assess_risk(low_risk_pred)
        assert result['severity'] == AlertSeverity.LOW
        
        # Critical risk prediction
        critical_pred = {
            'cascade_detected': True,
            'cascade_probability': 0.9,
            'high_risk_nodes': list(range(20)),
            'risk_assessment': [0.9, 0.8, 0.9, 0.95, 0.7, 0.8, 0.9]
        }
        
        result = risk_agent.assess_risk(critical_pred)
        assert result['severity'] == AlertSeverity.CRITICAL
    
    def test_recommendation_generation(self, risk_agent):
        """Test recommendation generation"""
        prediction = {
            'cascade_detected': True,
            'cascade_probability': 0.7,
            'high_risk_nodes': [1, 2, 3],
            'risk_assessment': [0.6, 0.7, 0.5, 0.6, 0.4, 0.5, 0.6],
            'cascade_path': [
                {'order': 1, 'node_id': 1, 'probability': 0.8},
                {'order': 2, 'node_id': 2, 'probability': 0.6}
            ]
        }
        
        result = risk_agent.assess_risk(prediction)
        
        assert len(result['recommendations']) > 0
        assert any('Node 1' in r for r in result['recommendations'])


class TestIntegration:
    """Integration tests for the multi-agent system"""
    
    @pytest.fixture
    def temp_topology(self):
        """Create temporary topology file"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Create IEEE 14-bus like topology
            edges = []
            for i in range(13):
                edges.append([i, i+1])
                edges.append([i+1, i])
            
            topology = {
                'edge_index': np.array(edges).T,
            }
            pickle.dump(topology, f)
            return f.name
    
    @pytest.mark.asyncio
    async def test_data_to_risk_pipeline(self, temp_topology):
        """Test data acquisition to risk assessment pipeline"""
        # Create agents
        data_agent = DataAcquisitionAgent(
            agent_id="data",
            topology_path=temp_topology
        )
        
        risk_agent = RiskAssessmentAgent(agent_id="risk")
        
        # Initialize
        await data_agent.initialize()
        await risk_agent.initialize()
        
        # Acquire and preprocess data
        await data_agent._acquire_environmental_data()
        await data_agent._acquire_infrastructure_data()
        await data_agent._acquire_robotic_data()
        
        preprocessed = data_agent._preprocess_data()
        data_agent._update_sequence_buffer(preprocessed)
        
        # Create mock prediction
        mock_prediction = {
            'cascade_detected': True,
            'cascade_probability': 0.6,
            'high_risk_nodes': [0, 3, 7],
            'risk_assessment': [0.5, 0.6, 0.4, 0.6, 0.3, 0.4, 0.5],
            'cascade_path': [
                {'order': 1, 'node_id': 0, 'probability': 0.7},
                {'order': 2, 'node_id': 3, 'probability': 0.5}
            ]
        }
        
        # Assess risk
        assessment = risk_agent.assess_risk(mock_prediction)
        
        assert 'aggregate_risk' in assessment
        assert 'severity' in assessment
        assert 'recommendations' in assessment
        assert len(assessment['component_risks']) == 7


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
