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
import sys

from .base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, AgentState
)

# Import the model - adjust path as needed
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from multimodal_cascade_model import UnifiedCascadePredictionModel
    MODEL_AVAILABLE = True
except ImportError:
    UnifiedCascadePredictionModel = None
    MODEL_AVAILABLE = False


class MockPredictionModel(nn.Module):
    """
    Mock model for testing the agent system without a trained model.
    Generates synthetic predictions based on input statistics.
    """
    def __init__(self, num_nodes: int = 50):
        super().__init__()
        self.num_nodes = num_nodes
    
    def forward(self, batch: Dict, return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """Generate mock predictions based on input data patterns"""
        # Determine batch size and num_nodes from input
        if 'scada_data' in batch:
            data = batch['scada_data']
            if len(data.shape) == 4:  # (batch, seq, nodes, features)
                batch_size, seq_len, num_nodes, _ = data.shape
            elif len(data.shape) == 3:  # (seq, nodes, features)
                seq_len, num_nodes, _ = data.shape
                batch_size = 1
            else:
                num_nodes = self.num_nodes
                batch_size = 1
                seq_len = 1
        else:
            num_nodes = self.num_nodes
            batch_size = 1
            seq_len = 1
        
        # Generate failure probabilities with some structure
        # Base probability with random variation
        base_prob = 0.15 + np.random.randn() * 0.05
        node_probs = np.clip(
            base_prob + np.random.randn(num_nodes) * 0.1,
            0.01, 0.95
        ).astype(np.float32)
        
        # Add some high-risk nodes (simulating potential cascade)
        if np.random.random() < 0.3:  # 30% chance of cascade scenario
            num_high_risk = np.random.randint(1, min(5, num_nodes))
            high_risk_idx = np.random.choice(num_nodes, num_high_risk, replace=False)
            node_probs[high_risk_idx] = np.clip(
                0.5 + np.random.randn(num_high_risk) * 0.2,
                0.4, 0.95
            )
        
        failure_prob = torch.from_numpy(node_probs).unsqueeze(0).unsqueeze(-1)  # (1, nodes, 1)
        
        # Generate risk scores (7 dimensions)
        risk_scores = torch.rand(batch_size, 7) * 0.5 + 0.2
        
        # Generate system state outputs
        frequency = torch.tensor([[60.0 + np.random.randn() * 0.1]])
        voltages = torch.ones(batch_size, num_nodes) * (1.0 + np.random.randn(num_nodes) * 0.02)
        
        return {
            'failure_probability': failure_prob,
            'risk_scores': risk_scores,
            'frequency': frequency,
            'voltages': voltages,
            'cascade_logits': torch.randn(batch_size, 1),
            'timing_prediction': torch.rand(batch_size, num_nodes) * 10
        }


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
        cascade_threshold: float = 0.4,
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
        self.is_mock_model: bool = False
        
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
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Check if this is a mock checkpoint
            if checkpoint.get('is_mock', False):
                self.logger.warning("Using MOCK model - predictions are synthetic!")
                self.logger.warning("Replace with trained model for real predictions.")
                num_nodes = checkpoint.get('model_config', {}).get('num_nodes', 50)
                self.model = MockPredictionModel(num_nodes=num_nodes)
                self.is_mock_model = True
            elif MODEL_AVAILABLE and UnifiedCascadePredictionModel is not None:
                # Initialize real model
                config = checkpoint.get('model_config', {})
                self.model = UnifiedCascadePredictionModel(
                    embedding_dim=config.get('embed_dim', 128),
                    hidden_dim=config.get('hidden_dim', 128),
                    num_gnn_layers=config.get('num_gat_layers', 3),
                    heads=config.get('num_heads', 4),
                    dropout=config.get('dropout', 0.1)
                )
                
                # --- CRITICAL FIX START ---
                print(">>> APPLYING ROBUST MODEL LOADING FIX (FUSION + EDGE) <<<")
                
                # 1. Get the state dictionary
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                # 2. Identify keys to remove (fusion layer AND edge embedding layer)
                keys_to_remove = []
                for k in list(state_dict.keys()):
                    if 'robot_embedding.fusion' in k:
                        keys_to_remove.append(k)
                    elif 'edge_embedding' in k:
                        keys_to_remove.append(k)
                
                # 3. Delete them
                if keys_to_remove:
                    self.logger.info(f"Removing {len(keys_to_remove)} mismatched keys to allow loading.")
                    for k in keys_to_remove:
                        del state_dict[k]

                # 4. Load remaining valid weights
                self.model.load_state_dict(state_dict, strict=False)
                self.logger.info("Loaded trained model weights (Partial load due to architecture update)")
                # --- CRITICAL FIX END ---
                
                self.is_mock_model = False
            else:
                # Fall back to mock if real model not available
                self.logger.warning("UnifiedCascadePredictionModel not available - using mock model")
                self.model = MockPredictionModel(num_nodes=50)
                self.is_mock_model = True
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load thresholds if available
            if 'cascade_threshold' in checkpoint:
                self.cascade_threshold = checkpoint['cascade_threshold']
            if 'node_threshold' in checkpoint:
                self.node_threshold = checkpoint['node_threshold']
            
            model_type = "MOCK" if self.is_mock_model else "TRAINED"
            self.logger.info(f"Model loaded [{model_type}]. Thresholds: cascade={self.cascade_threshold:.3f}, node={self.node_threshold:.3f}")
            
        except FileNotFoundError:
            self.logger.warning(f"Model file not found: {self.model_path}")
            self.logger.warning("Using mock model for testing")
            self.model = MockPredictionModel(num_nodes=50)
            self.is_mock_model = True
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Falling back to mock model")
            self.model = MockPredictionModel(num_nodes=50)
            self.is_mock_model = True
            self.model.to(self.device)
            self.model.eval()
    
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
                    "is_mock_model": self.is_mock_model,
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
            result['is_mock_prediction'] = self.is_mock_model
            
            # Store prediction
            self.last_prediction = result
            self.prediction_history.append(result)
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            self.prediction_count += 1
            
            # Broadcast if cascade detected
            if result['cascade_detected']:
                await self._broadcast_cascade_alert(result)
            
            # --- DEBUG LOGGING ---
            self.logger.info(f"Prediction cycle complete. Max Cascade Probability: {result.get('cascade_probability', 0):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.error_count += 1
            return {"error": str(e)}
    
    def _prepare_model_input(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """Convert batch data to model input tensors with feature safety"""
        model_input = {}
        
        # 1. Basic Tensor Conversion (Keep existing logic)
        for key, value in batch_data.items():
            if isinstance(value, list):
                tensor = torch.tensor(value, dtype=torch.float32)
                if key == 'edge_index': tensor = tensor.long()
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

        # 2. SAFETY FIX: Ensure 'temporal_sequence' matches Model Physics Input
        # Many trained models expect only 4 features (V, A, P, Q) but Agent sends 13.
        if 'scada_data' in model_input:
            scada = model_input['scada_data'] # (Batch, Time, Nodes, 13)
            
            # Slice to standard 4 physics variables (Voltage, Angle, P, Q)
            # This is the most robust default for physics-informed GNNs.
            physics_input = scada[..., :4] 
            
            model_input['temporal_sequence'] = physics_input
            model_input['x'] = physics_input # Map 'x' for PyG compatibility
            
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
            risk_scores = risk_scores[-1] if risk_scores.shape[0] > 1 else risk_scores[0]
        risk_assessment = risk_scores.cpu().numpy().tolist()
        
        # Get system state
        frequency = float(outputs['frequency'].mean().item())
        voltages = outputs['voltages'].reshape(-1).cpu().numpy().tolist() if 'voltages' in outputs else []
        
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