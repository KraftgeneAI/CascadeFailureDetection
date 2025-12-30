"""
Agentic Cascade Failure Detection System - Agents Module
=========================================================

This module contains all agent implementations for the multi-agent
cascade failure detection system.

Agents:
- BaseAgent: Abstract base class for all agents
- DataAcquisitionAgent: Multi-modal data collection and preprocessing
- PredictionAgent: Physics-informed GNN cascade prediction
- RiskAssessmentAgent: Risk evaluation and alert generation
- CoordinationAgent: System orchestration

Author: Kraftgene AI Inc.
"""

from .base_agent import (
    BaseAgent,
    AgentMessage,
    MessageType,
    AgentState,
    AgentCapability,
    MessageBus
)

from .data_acquisition_agent import DataAcquisitionAgent
from .prediction_agent import PredictionAgent
from .risk_assessment_agent import RiskAssessmentAgent, AlertSeverity, RiskCategory
from .coordination_agent import CoordinationAgent, TaskPriority

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentMessage',
    'MessageType',
    'AgentState',
    'AgentCapability',
    'MessageBus',
    
    # Agents
    'DataAcquisitionAgent',
    'PredictionAgent',
    'RiskAssessmentAgent',
    'CoordinationAgent',
    
    # Enums
    'AlertSeverity',
    'RiskCategory',
    'TaskPriority',
]
