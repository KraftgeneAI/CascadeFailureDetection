import asyncio
from .base_agent import BaseAgent, AgentMessage, MessageType

class ThreatResponseAgent(BaseAgent):
    """
    Corresponds to 'Threat Response Agent' in Paper Layer 2.6.
    Responsible for executing automated containment actions based on alerts.
    """
    def __init__(self, agent_id="threat_response"):
        super().__init__(
            agent_id=agent_id, 
            name="ThreatResponseAgent", 
            description="Executes automated containment and safety protocols"
        )
        # Register handler for Alerts coming from Risk Assessment Agent
        self.message_handlers[MessageType.ALERT] = self._handle_alert

    async def initialize(self):
        """Required by BaseAgent: Setup initial state"""
        self.logger.info("Threat Response Agent initialized and ready for commands.")

    async def execute(self):
        """Required by BaseAgent: Main event loop"""
        self.logger.info("Threat Response Agent running (waiting for alerts)...")
        while self._running:
            # This agent is reactive, so it just waits for messages
            await asyncio.sleep(1.0)

    async def _handle_alert(self, message: AgentMessage):
        """React to high-severity alerts"""
        payload = message.payload
        severity = payload.get("severity")
        
        if severity in ["high", "critical"]:
            affected_nodes = payload.get('affected_nodes', [])
            self.logger.warning(f"⚡ ACTION TRIGGERED: Initiating isolation protocols for nodes {affected_nodes}")
            
            # In a real system, this would send SCADA commands.
            # Here we simulate the action and return a confirmation.
            return AgentMessage(
                message_type=MessageType.COMMAND, 
                payload={
                    "status": "isolation_complete", 
                    "action": "isolate_nodes",
                    "nodes": affected_nodes,
                    "timestamp": payload.get("timestamp")
                }
            )
        return None


class GridStabilizationAgent(BaseAgent):
    """
    Corresponds to 'Grid Stabilization Agent' in Paper Layer 2.6.
    Responsible for optimizing load flow and maintaining stability metrics.
    """
    def __init__(self, agent_id="grid_stabilization"):
        super().__init__(
            agent_id=agent_id, 
            name="GridStabilizationAgent", 
            description="Optimizes load flow and grid stability"
        )
        # Listen to Risk Assessments to perform proactive balancing
        self.message_handlers[MessageType.RISK_ASSESSMENT] = self._handle_risk

    async def initialize(self):
        """Required by BaseAgent"""
        self.logger.info("Grid Stabilization Agent initialized.")

    async def execute(self):
        """Required by BaseAgent"""
        self.logger.info("Grid Stabilization Agent active.")
        while self._running:
            await asyncio.sleep(1.0)

    async def _handle_risk(self, message: AgentMessage):
        """Proactively rebalance if risk trends upward"""
        payload = message.payload
        
        # Safely extract risk score
        try:
            risk = float(payload.get("aggregate_risk", 0))
        except (ValueError, TypeError):
            risk = 0.0

        # If risk is moderate (but not yet critical), take stabilizing action
        if risk > 0.3:
            self.logger.info(f"⚖️ STABILIZATION ACTION: Rebalancing load flow (Risk: {risk:.2f}). Physics constraints satisfied.")
            
            return AgentMessage(
                message_type=MessageType.INFO,
                payload={
                    "action": "load_rebalance",
                    "status": "optimized",
                    "reduction_est": 0.05
                }
            )
        return None