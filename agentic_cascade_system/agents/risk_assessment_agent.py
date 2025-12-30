"""
Risk Assessment Agent
=====================
Agent responsible for translating predictions into actionable intelligence.

Implements the Decision Support Layer from the research paper:
- 7-dimensional risk assessment framework
- Alert generation and prioritization
- Mitigation recommendation generation

Author: Kraftgene AI Inc.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, AgentState
)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """7-dimensional risk categories from research paper"""
    THREAT_SEVERITY = 0
    VULNERABILITY = 1
    OPERATIONAL_IMPACT = 2
    CASCADE_PROBABILITY = 3
    RESPONSE_COMPLEXITY = 4
    PUBLIC_SAFETY = 5
    URGENCY = 6


@dataclass
class RiskAlert:
    """Structured risk alert"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    affected_nodes: List[int]
    cascade_probability: float
    risk_vector: List[float]
    recommended_actions: List[str]
    estimated_impact: Dict[str, Any]
    time_to_action_minutes: float


class RiskAssessmentAgent(BaseAgent):
    """
    Agent responsible for comprehensive risk assessment and alert generation.
    
    Corresponds to the Decision Support Layer in the research paper:
    - Aggregates predictions into actionable risk assessments
    - Generates tiered alerts based on severity
    - Recommends mitigation strategies
    """
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="RiskAssessmentAgent",
            description="Risk assessment and decision support"
        )
        
        # Risk assessment configuration
        self.risk_weights = {
            RiskCategory.THREAT_SEVERITY: 0.15,
            RiskCategory.VULNERABILITY: 0.15,
            RiskCategory.OPERATIONAL_IMPACT: 0.20,
            RiskCategory.CASCADE_PROBABILITY: 0.20,
            RiskCategory.RESPONSE_COMPLEXITY: 0.10,
            RiskCategory.PUBLIC_SAFETY: 0.10,
            RiskCategory.URGENCY: 0.10
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            AlertSeverity.CRITICAL: 0.75,
            AlertSeverity.HIGH: 0.50,
            AlertSeverity.MODERATE: 0.25,
            AlertSeverity.LOW: 0.0
        }
        
        # Alert history
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        
        # Current risk state
        self.current_risk_state: Dict[str, Any] = {}
        
        # Register capabilities
        self._register_capabilities()
        
        # Register handlers
        self.message_handlers[MessageType.PREDICTION] = self._handle_prediction
        self.message_handlers[MessageType.ALERT] = self._handle_alert
        self.message_handlers[MessageType.QUERY] = self._handle_query
    
    def _register_capabilities(self):
        """Register agent capabilities"""
        self.register_capability(AgentCapability(
            name="risk_aggregation",
            description="Aggregate multi-dimensional risk scores",
            input_types=["risk_vector", "prediction"],
            output_types=["aggregated_risk", "severity"],
            latency_ms=10.0,
            reliability=0.99
        ))
        
        self.register_capability(AgentCapability(
            name="alert_generation",
            description="Generate prioritized alerts",
            input_types=["risk_assessment"],
            output_types=["alert", "recommendations"],
            latency_ms=5.0,
            reliability=0.99
        ))
        
        self.register_capability(AgentCapability(
            name="mitigation_planning",
            description="Generate mitigation recommendations",
            input_types=["cascade_path", "risk_vector"],
            output_types=["action_plan", "priority_list"],
            latency_ms=20.0,
            reliability=0.95
        ))
    
    async def initialize(self):
        """Initialize risk assessment systems"""
        self.logger.info("Initializing Risk Assessment Agent...")
        
        # Initialize risk state
        self.current_risk_state = {
            'overall_risk': 0.0,
            'risk_vector': [0.0] * 7,
            'trend': 'stable',
            'alerts_active': 0
        }
        
        self.logger.info("Risk Assessment Agent initialized")
    
    async def execute(self):
        """Main execution loop"""
        self.logger.info("Risk Assessment Agent running...")
        
        while self._running:
            # Periodic risk state update
            await self._update_risk_state()
            await asyncio.sleep(1.0)
    
    async def _handle_prediction(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming predictions from Prediction Agent"""
        prediction = message.payload
        
        # Assess risk
        risk_assessment = self.assess_risk(prediction)
        
        # Generate alert if necessary
        if risk_assessment['severity'] in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            alert = self._create_alert(prediction, risk_assessment)
            self.active_alerts[alert.id] = alert
            
            # Broadcast alert
            await self.send_message(AgentMessage(
                message_type=MessageType.ALERT,
                payload={
                    "alert_id": alert.id,
                    "severity": alert.severity.value,
                    "affected_nodes": alert.affected_nodes,
                    "recommended_actions": alert.recommended_actions,
                    "time_to_action": alert.time_to_action_minutes
                },
                priority=10 if alert.severity == AlertSeverity.CRITICAL else 8
            ))
        
        # Return risk assessment
        return AgentMessage(
            message_type=MessageType.RISK_ASSESSMENT,
            payload=risk_assessment
        )
    
    async def _handle_alert(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle alerts from other agents"""
        alert_type = message.payload.get("alert_type")
        
        if alert_type == "cascade_detected":
            # Process cascade detection alert
            prediction = message.payload.get("prediction", {})
            risk_assessment = self.assess_risk(prediction)
            
            # Create structured alert
            alert = self._create_alert(prediction, risk_assessment)
            self.active_alerts[alert.id] = alert
            
            self.logger.warning(
                f"CASCADE ALERT: {alert.severity.value.upper()} - "
                f"{len(alert.affected_nodes)} nodes at risk"
            )
        
        return None
    
    async def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle query messages"""
        query_type = message.payload.get("query")
        
        if query_type == "current_risk":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload=self.current_risk_state
            )
        
        elif query_type == "active_alerts":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "count": len(self.active_alerts),
                    "alerts": [
                        {
                            "id": a.id,
                            "severity": a.severity.value,
                            "affected_nodes": a.affected_nodes,
                            "timestamp": a.timestamp.isoformat()
                        }
                        for a in self.active_alerts.values()
                    ]
                }
            )
        
        elif query_type == "risk_trend":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload=self._calculate_risk_trend()
            )
        
        return await super()._handle_query(message)
    
    def assess_risk(self, prediction: Dict) -> Dict:
        """
        Perform comprehensive risk assessment based on prediction.
        
        Args:
            prediction: Dictionary containing model prediction outputs
            
        Returns:
            Dictionary with risk assessment results
        """
        # Extract risk vector (7 dimensions)
        risk_vector = prediction.get('risk_assessment', [0.0] * 7)
        if len(risk_vector) < 7:
            risk_vector = risk_vector + [0.0] * (7 - len(risk_vector))
        
        # Calculate weighted aggregate risk
        aggregate_risk = sum(
            risk_vector[cat.value] * weight
            for cat, weight in self.risk_weights.items()
        )
        
        # Determine severity
        severity = self._determine_severity(aggregate_risk, prediction)
        
        # Calculate component risks
        component_risks = {
            'threat_severity': {
                'value': risk_vector[0],
                'level': self._get_level(risk_vector[0]),
                'description': 'External threat level from environmental factors'
            },
            'vulnerability': {
                'value': risk_vector[1],
                'level': self._get_level(risk_vector[1]),
                'description': 'Infrastructure weakness and susceptibility'
            },
            'operational_impact': {
                'value': risk_vector[2],
                'level': self._get_level(risk_vector[2]),
                'description': 'Potential operational consequences'
            },
            'cascade_probability': {
                'value': risk_vector[3],
                'level': self._get_level(risk_vector[3]),
                'description': 'Likelihood of failure propagation'
            },
            'response_complexity': {
                'value': risk_vector[4],
                'level': self._get_level(risk_vector[4]),
                'description': 'Difficulty of implementing response actions'
            },
            'public_safety': {
                'value': risk_vector[5],
                'level': self._get_level(risk_vector[5]),
                'description': 'Risk to public safety and welfare'
            },
            'urgency': {
                'value': risk_vector[6],
                'level': self._get_level(risk_vector[6]),
                'description': 'Time sensitivity of required response'
            }
        }
        
        # Estimate time to critical
        time_to_critical = self._estimate_time_to_critical(prediction)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, component_risks, prediction
        )
        
        return {
            'aggregate_risk': aggregate_risk,
            'severity': severity,
            'risk_vector': risk_vector,
            'component_risks': component_risks,
            'cascade_detected': prediction.get('cascade_detected', False),
            'affected_nodes': prediction.get('high_risk_nodes', []),
            'cascade_probability': prediction.get('cascade_probability', 0.0),
            'time_to_critical_minutes': time_to_critical,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_severity(self, aggregate_risk: float, prediction: Dict) -> AlertSeverity:
        """Determine alert severity based on risk and prediction"""
        # Base severity from aggregate risk
        for severity, threshold in sorted(
            self.severity_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if aggregate_risk >= threshold:
                base_severity = severity
                break
        else:
            base_severity = AlertSeverity.LOW
        
        # Escalate based on number of affected nodes
        num_affected = len(prediction.get('high_risk_nodes', []))
        if num_affected > 10 and base_severity.value in ['low', 'moderate']:
            base_severity = AlertSeverity.HIGH
        elif num_affected > 20:
            base_severity = AlertSeverity.CRITICAL
        
        # Escalate based on cascade probability
        cascade_prob = prediction.get('cascade_probability', 0.0)
        if cascade_prob > 0.8:
            base_severity = AlertSeverity.CRITICAL
        
        return base_severity
    
    def _get_level(self, score: float) -> str:
        """Convert score to human-readable level"""
        if score > 0.8:
            return "Critical"
        elif score > 0.6:
            return "Severe"
        elif score > 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _estimate_time_to_critical(self, prediction: Dict) -> float:
        """Estimate time until situation becomes critical"""
        # Based on cascade path and probabilities
        cascade_path = prediction.get('cascade_path', [])
        
        if not cascade_path:
            return float('inf')
        
        # Estimate based on prediction lead time (15-35 minutes from paper)
        base_time = 25.0  # Average prediction lead time
        
        # Adjust based on cascade probability
        cascade_prob = prediction.get('cascade_probability', 0.0)
        if cascade_prob > 0.8:
            return base_time * 0.5  # ~12 minutes
        elif cascade_prob > 0.6:
            return base_time * 0.7  # ~17 minutes
        elif cascade_prob > 0.4:
            return base_time  # ~25 minutes
        else:
            return base_time * 1.5  # ~37 minutes
    
    def _generate_recommendations(
        self,
        severity: AlertSeverity,
        component_risks: Dict,
        prediction: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        # Severity-based general recommendations
        if severity == AlertSeverity.CRITICAL:
            recommendations.extend([
                "IMMEDIATE: Initiate emergency response protocol",
                "IMMEDIATE: Alert all control room operators",
                "IMMEDIATE: Prepare for controlled load shedding"
            ])
        elif severity == AlertSeverity.HIGH:
            recommendations.extend([
                "Activate enhanced monitoring for affected region",
                "Prepare emergency response teams",
                "Review contingency switching plans"
            ])
        elif severity == AlertSeverity.MODERATE:
            recommendations.extend([
                "Increase monitoring frequency",
                "Verify protection system readiness",
                "Review available generation reserves"
            ])
        
        # Component-specific recommendations
        if component_risks['vulnerability']['value'] > 0.6:
            recommendations.append(
                "Consider preemptive isolation of high-vulnerability nodes"
            )
        
        if component_risks['cascade_probability']['value'] > 0.5:
            recommendations.append(
                "Implement topology splitting to limit cascade propagation"
            )
        
        if component_risks['threat_severity']['value'] > 0.7:
            recommendations.append(
                "Dispatch inspection teams to high-threat areas"
            )
        
        # Node-specific recommendations
        high_risk_nodes = prediction.get('high_risk_nodes', [])
        if high_risk_nodes:
            recommendations.append(
                f"Priority monitoring for nodes: {high_risk_nodes[:5]}"
            )
        
        cascade_path = prediction.get('cascade_path', [])
        if cascade_path and len(cascade_path) > 0:
            root_node = cascade_path[0].get('node_id')
            recommendations.append(
                f"Focus mitigation on potential cascade root: Node {root_node}"
            )
        
        return recommendations
    
    def _create_alert(self, prediction: Dict, risk_assessment: Dict) -> RiskAlert:
        """Create a structured risk alert"""
        import uuid
        
        return RiskAlert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=risk_assessment['severity'],
            affected_nodes=risk_assessment['affected_nodes'],
            cascade_probability=risk_assessment['cascade_probability'],
            risk_vector=risk_assessment['risk_vector'],
            recommended_actions=risk_assessment['recommendations'],
            estimated_impact={
                'nodes_at_risk': len(risk_assessment['affected_nodes']),
                'aggregate_risk': risk_assessment['aggregate_risk']
            },
            time_to_action_minutes=risk_assessment['time_to_critical_minutes']
        )
    
    async def _update_risk_state(self):
        """Update current risk state"""
        if self.alert_history:
            recent_alerts = [
                a for a in self.alert_history
                if (datetime.now() - a.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if recent_alerts:
                avg_risk = np.mean([
                    sum(a.risk_vector) / len(a.risk_vector)
                    for a in recent_alerts
                ])
                self.current_risk_state['overall_risk'] = avg_risk
        
        self.current_risk_state['alerts_active'] = len(self.active_alerts)
    
    def _calculate_risk_trend(self) -> Dict:
        """Calculate risk trend over time"""
        if len(self.alert_history) < 2:
            return {'trend': 'stable', 'change': 0.0}
        
        recent = self.alert_history[-10:]
        risks = [sum(a.risk_vector) / len(a.risk_vector) for a in recent]
        
        if len(risks) >= 2:
            change = risks[-1] - risks[0]
            if change > 0.1:
                trend = 'increasing'
            elif change < -0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {'trend': trend, 'change': change, 'values': risks}
        
        return {'trend': 'stable', 'change': 0.0}
