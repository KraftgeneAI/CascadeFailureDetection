"""
Prediction Agent
================
Core agent responsible for cascade failure prediction using the 
physics-informed Graph Neural Network model.

Implements the Prediction Layer from the research paper:
- Spatio-Temporal GNN
- Physics-informed constraints
- Multi-task prediction heads

Author: Kraftgene AI Inc.
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pickle
import sys

from .base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, AgentState
)

# Import the model - adjust path as needed
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from multimodal_cascade_model import UnifiedCascadePredictionModel
except ImportError:
    UnifiedCascadePredictionModel = None


class PredictionAgent(BaseAgent):
    """
    Agent responsible for running cascade failure predictions using the
    trained physics-informed GNN model.
    
    Corresponds to the Prediction Layer in the research paper:
    - Graph construction from infrastructure topology
    - Spatio-temporal GNN for cascade prediction
    - Multi-task prediction heads (failure probability, risk assessment, timing)
    """
    
    def __init__(
        self,
        agent_id: str,
        model_path: str,
        device: str = None,
        cascade_threshold: float = 0.1,
        node_threshold: float = 0.35
    ):
        super().__init__(
            agent_id=agent_id,
            name="PredictionAgent",
            description="Physics-informed GNN cascade failure prediction"
        )
        
        self.model_path = model_path
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Thresholds (can be loaded from checkpoint)
        self.cascade_threshold = cascade_threshold
        self.node_threshold = node_threshold
        
        # Model
        self.model: Optional[nn.Module] = None
        
        # Prediction state
        self.last_prediction: Optional[Dict] = None
        self.prediction_history: List[Dict] = []
        self.prediction_count: int = 0
        
        # Performance tracking
        self.avg_inference_time_ms: float = 0.0
        self.total_inference_time_ms: float = 0.0
        
        # Register capabilities
        self._register_capabilities()
        
        # Register handlers
        self.message_handlers[MessageType.DATA] = self._handle_data
        self.message_handlers[MessageType.QUERY] = self._handle_query
        self.message_handlers[MessageType.COMMAND] = self._handle_command
    
    def _register_capabilities(self):
        """Register agent capabilities"""
        self.register_capability(AgentCapability(
            name="cascade_prediction",
            description="Predict cascade failures using physics-informed GNN",
            input_types=["batch_data", "sequence_data"],
            output_types=["failure_probability", "risk_assessment", "cascade_path"],
            latency_ms=100.0,
            reliability=0.95
        ))
        
        self.register_capability(AgentCapability(
            name="node_risk_assessment",
            description="Assess failure risk for individual nodes",
            input_types=["batch_data"],
            output_types=["node_probabilities", "risk_scores"],
            latency_ms=50.0,
            reliability=0.95
        ))
    
    async def initialize(self):
        """Initialize the prediction model"""
        self.logger.info(f"Initializing Prediction Agent on {self.device}...")
        
        if UnifiedCascadePredictionModel is None:
            raise ImportError("Could not import UnifiedCascadePredictionModel")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            self.model = UnifiedCascadePredictionModel(
                embedding_dim=128,
                hidden_dim=128,
                num_gnn_layers=3,
                heads=4,
                dropout=0.1
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load thresholds if available
            if 'cascade_threshold' in checkpoint:
                self.cascade_threshold = checkpoint['cascade_threshold']
            if 'node_threshold' in checkpoint:
                self.node_threshold = checkpoint['node_threshold']
            
            self.logger.info(f"Model loaded successfully. Thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    async def execute(self):
        """Main execution loop - wait for prediction requests"""
        self.logger.info("Prediction Agent ready for inference requests...")
        
        while self._running:
            await asyncio.sleep(0.1)  # Wait for messages
    
    async def _handle_data(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming data for prediction"""
        event = message.payload.get("event")
        
        if event == "data_ready":
            # Data is ready - request it and run prediction
            return None  # Will be triggered by orchestrator
        
        elif event == "predict":
            # Run prediction on provided data
            batch_data = message.payload.get("batch_data")
            if batch_data:
                result = await self.predict(batch_data)
                return AgentMessage(
                    message_type=MessageType.PREDICTION,
                    payload=result,
                    priority=9
                )
        
        return None
    
    async def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle query messages"""
        query_type = message.payload.get("query")
        
        if query_type == "last_prediction":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={"prediction": self.last_prediction}
            )
        
        elif query_type == "prediction_stats":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "prediction_count": self.prediction_count,
                    "avg_inference_time_ms": self.avg_inference_time_ms,
                    "thresholds": {
                        "cascade": self.cascade_threshold,
                        "node": self.node_threshold
                    }
                }
            )
        
        return await super()._handle_query(message)
    
    async def _handle_command(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle command messages"""
        command = message.payload.get("command")
        
        if command == "set_thresholds":
            if "cascade_threshold" in message.payload:
                self.cascade_threshold = message.payload["cascade_threshold"]
            if "node_threshold" in message.payload:
                self.node_threshold = message.payload["node_threshold"]
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "status": "thresholds_updated",
                    "cascade_threshold": self.cascade_threshold,
                    "node_threshold": self.node_threshold
                }
            )
        
        return await super()._handle_command(message)
    
    async def predict(self, batch_data: Dict) -> Dict:
        """
        Run cascade failure prediction on batch data.
        
        Args:
            batch_data: Dictionary containing model inputs
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        start_time = datetime.now()
        
        try:
            # Convert to tensors
            model_input = self._prepare_model_input(batch_data)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(model_input, return_sequence=False)
            
            # Process outputs
            result = self._process_outputs(outputs, batch_data)
            
            # Track timing
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_timing_stats(inference_time)
            
            result['inference_time_ms'] = inference_time
            result['timestamp'] = datetime.now().isoformat()
            
            # Store prediction
            self.last_prediction = result
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            self.prediction_count += 1
            
            # Broadcast if cascade detected
            if result['cascade_detected']:
                await self._broadcast_cascade_alert(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            self.error_count += 1
            return {"error": str(e)}
    
    def _prepare_model_input(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Convert batch data to model input tensors"""
        model_input = {}
        
        # Convert lists to tensors
        for key, value in batch_data.items():
            if isinstance(value, list):
                tensor = torch.tensor(value, dtype=torch.float32)
                if key == 'edge_index':
                    tensor = tensor.long()
                model_input[key] = tensor.to(self.device)
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                if key == 'edge_index':
                    tensor = tensor.long()
                else:
                    tensor = tensor.float()
                model_input[key] = tensor.to(self.device)
            elif isinstance(value, torch.Tensor):
                model_input[key] = value.to(self.device)
        
        # Ensure temporal_sequence exists
        if 'temporal_sequence' not in model_input and 'scada_data' in model_input:
            model_input['temporal_sequence'] = model_input['scada_data']
        
        # Ensure graph_properties exists
        if 'graph_properties' not in model_input:
            model_input['graph_properties'] = {}
        
        return model_input
    
    def _process_outputs(self, outputs: Dict[str, torch.Tensor], batch_data: Dict) -> Dict:
        """Process model outputs into prediction results"""
        # Get failure probabilities
        failure_probs = outputs['failure_probability'].squeeze(-1).cpu().numpy()
        if len(failure_probs.shape) == 1:
            failure_probs = failure_probs.reshape(1, -1)
        
        # Take last timestep if batched
        if len(failure_probs.shape) > 1:
            node_probs = failure_probs[-1] if failure_probs.shape[0] > 1 else failure_probs[0]
        else:
            node_probs = failure_probs
        
        # Identify high-risk nodes
        high_risk_nodes = np.where(node_probs > self.node_threshold)[0].tolist()
        
        # Build cascade path based on probability ranking
        ranked_indices = np.argsort(-node_probs)
        cascade_path = []
        current_rank = 1
        last_score = float(node_probs[ranked_indices[0]]) if len(ranked_indices) > 0 else 0
        
        for i, idx in enumerate(ranked_indices):
            prob = float(node_probs[idx])
            if prob < self.node_threshold:
                break
            
            if (last_score - prob) > 0.002:
                current_rank += 1
                last_score = prob
            
            cascade_path.append({
                'order': current_rank,
                'node_id': int(idx),
                'probability': prob
            })
        
        # Get risk assessment (7-dimensional)
        risk_scores = outputs['risk_scores']
        if len(risk_scores.shape) > 1:
            risk_scores = risk_scores[-1].mean(dim=0)
        risk_assessment = risk_scores.cpu().numpy().tolist()
        
        # Get system state
        frequency = float(outputs['frequency'].mean().item())
        voltages = outputs['voltages'][-1].reshape(-1).cpu().numpy().tolist() if 'voltages' in outputs else []
        
        # Determine cascade detection
        max_prob = float(np.max(node_probs)) if len(node_probs) > 0 else 0.0
        cascade_detected = max_prob > self.cascade_threshold and len(high_risk_nodes) > 0
        
        return {
            'cascade_detected': cascade_detected,
            'cascade_probability': max_prob,
            'high_risk_nodes': high_risk_nodes,
            'node_probabilities': node_probs.tolist(),
            'cascade_path': cascade_path,
            'risk_assessment': risk_assessment,
            'risk_dimensions': [
                'threat_severity', 'vulnerability', 'operational_impact',
                'cascade_probability', 'response_complexity', 'public_safety', 'urgency'
            ],
            'system_state': {
                'frequency': frequency,
                'voltages': voltages
            },
            'thresholds_used': {
                'cascade': self.cascade_threshold,
                'node': self.node_threshold
            }
        }
    
    async def _broadcast_cascade_alert(self, prediction: Dict):
        """Broadcast cascade detection alert"""
        await self.send_message(AgentMessage(
            message_type=MessageType.ALERT,
            payload={
                "alert_type": "cascade_detected",
                "severity": self._calculate_severity(prediction),
                "prediction": prediction
            },
            priority=10  # Highest priority
        ))
    
    def _calculate_severity(self, prediction: Dict) -> str:
        """Calculate alert severity based on prediction"""
        prob = prediction['cascade_probability']
        num_nodes = len(prediction['high_risk_nodes'])
        
        if prob > 0.8 or num_nodes > 10:
            return "critical"
        elif prob > 0.6 or num_nodes > 5:
            return "high"
        elif prob > 0.4 or num_nodes > 2:
            return "medium"
        else:
            return "low"
    
    def _update_timing_stats(self, inference_time_ms: float):
        """Update inference timing statistics"""
        self.total_inference_time_ms += inference_time_ms
        self.avg_inference_time_ms = self.total_inference_time_ms / max(1, self.prediction_count + 1)
