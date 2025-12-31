"""
Coordination Agent (Orchestrator)
=================================
Master agent responsible for coordinating all other agents in the system.

Implements:
- Agent lifecycle management
- Task scheduling and distribution
- System-wide monitoring
- Inter-agent communication orchestration

Author: Kraftgene AI Inc.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, 
    AgentState, MessageBus
)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: str
    name: str
    agent_id: str
    priority: TaskPriority
    interval_seconds: float
    last_run: Optional[datetime]
    next_run: datetime
    callback: Callable
    enabled: bool = True


class CoordinationAgent(BaseAgent):
    """
    Master coordination agent that orchestrates the multi-agent system.
    
    Responsibilities:
    - Manage agent lifecycle (start, stop, restart)
    - Schedule periodic tasks
    - Route messages between agents
    - Monitor system health
    - Coordinate prediction pipeline
    """
    
    def __init__(self, agent_id: str = "coordinator"):
        super().__init__(
            agent_id=agent_id,
            name="CoordinationAgent",
            description="System orchestrator and coordinator"
        )
        
        # Agent registry
        self.managed_agents: Dict[str, BaseAgent] = {}
        
        # Message bus
        self.message_bus = MessageBus()
        
        # Task scheduler
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        
        # System state
        self.system_state = {
            'status': 'initializing',
            'agents_running': 0,
            'total_predictions': 0,
            'active_alerts': 0,
            'last_prediction_time': None
        }
        
        # Prediction pipeline state
        self.pipeline_running = False
        self.prediction_interval = 1.0  # seconds
        
        # Register capabilities
        self._register_capabilities()
        
        # Register handlers
        self.message_handlers[MessageType.ALERT] = self._handle_alert
        self.message_handlers[MessageType.PREDICTION] = self._handle_prediction
        self.message_handlers[MessageType.COORDINATION] = self._handle_coordination
        self.message_handlers[MessageType.DATA] = self._handle_data  # New handler registration
    
    def _register_capabilities(self):
        """Register orchestrator capabilities"""
        self.register_capability(AgentCapability(
            name="agent_management",
            description="Manage agent lifecycle",
            input_types=["command"],
            output_types=["status"],
            latency_ms=5.0,
            reliability=0.99
        ))
        
        self.register_capability(AgentCapability(
            name="task_scheduling",
            description="Schedule and manage periodic tasks",
            input_types=["task_definition"],
            output_types=["schedule_confirmation"],
            latency_ms=5.0,
            reliability=0.99
        ))
        
        self.register_capability(AgentCapability(
            name="pipeline_coordination",
            description="Coordinate prediction pipeline execution",
            input_types=["trigger"],
            output_types=["pipeline_result"],
            latency_ms=10.0,
            reliability=0.98
        ))
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.managed_agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.managed_agents:
            del self.managed_agents[agent_id]
            self.message_bus.unregister_agent(agent_id)
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def initialize(self):
        """Initialize the coordination system"""
        self.logger.info("Initializing Coordination Agent...")
        
        # Register self with message bus
        self.message_bus.register_agent(self)
        
        # Subscribe to important message types
        self.message_bus.subscribe(self.agent_id, MessageType.ALERT)
        self.message_bus.subscribe(self.agent_id, MessageType.PREDICTION)
        self.message_bus.subscribe(self.agent_id, MessageType.DATA)  # Subscribe to DATA messages
        
        self.system_state['status'] = 'initialized'
        self.logger.info("Coordination Agent initialized")
    
    async def execute(self):
        """Main execution loop"""
        self.logger.info("Coordination Agent starting execution...")
        
        # Start all managed agents
        await self._start_all_agents()
        
        # Schedule default tasks
        self._schedule_default_tasks()
        
        # Main loop
        while self._running:
            try:
                # Run scheduled tasks
                await self._run_scheduled_tasks()
                
                # Update system state
                self._update_system_state()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Coordination error: {e}")
                self.error_count += 1
                await asyncio.sleep(1.0)
    
    async def _start_all_agents(self):
        """Start all registered agents"""
        self.logger.info(f"Starting {len(self.managed_agents)} agents...")
        
        # Start each agent in the background without waiting
        for agent_id, agent in self.managed_agents.items():
            if agent_id != self.agent_id:
                asyncio.create_task(agent.start())
        
        # Give agents a moment to initialize
        await asyncio.sleep(0.5)
        
        self.system_state['agents_running'] = len(self.managed_agents)
        self.system_state['status'] = 'running'
        self.logger.info("All agents started")
    
    async def stop_all_agents(self):
        """Stop all managed agents"""
        self.logger.info("Stopping all agents...")
        
        for agent_id, agent in self.managed_agents.items():
            if agent_id != self.agent_id:
                await agent.stop()
        
        self.system_state['status'] = 'stopped'
    
    def _schedule_default_tasks(self):
        """Schedule default periodic tasks"""
        # Health check task
        self.schedule_task(
            name="health_check",
            agent_id=self.agent_id,
            callback=self._health_check,
            interval_seconds=30.0,
            priority=TaskPriority.LOW
        )
        
        # Prediction pipeline task
        self.schedule_task(
            name="prediction_pipeline",
            agent_id=self.agent_id,
            callback=self._run_prediction_pipeline,
            interval_seconds=self.prediction_interval,
            priority=TaskPriority.HIGH
        )
    
    def schedule_task(
        self,
        name: str,
        agent_id: str,
        callback: Callable,
        interval_seconds: float,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Schedule a periodic task"""
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            agent_id=agent_id,
            priority=priority,
            interval_seconds=interval_seconds,
            last_run=None,
            next_run=datetime.now(),
            callback=callback
        )
        
        self.scheduled_tasks[task_id] = task
        self.logger.debug(f"Scheduled task: {name} (every {interval_seconds}s)")
        
        return task_id
    
    async def _run_scheduled_tasks(self):
        """Run due scheduled tasks"""
        now = datetime.now()
        
        for task_id, task in self.scheduled_tasks.items():
            if not task.enabled:
                continue
            
            if now >= task.next_run:
                try:
                    await task.callback()
                    task.last_run = now
                    task.next_run = now + timedelta(seconds=task.interval_seconds)
                except Exception as e:
                    self.logger.error(f"Task {task.name} failed: {e}")
    
    async def _health_check(self):
        """Perform health check on all agents"""
        for agent_id, agent in self.managed_agents.items():
            status = agent.get_status()
            
            if status['state'] == 'error':
                self.logger.warning(f"Agent {agent.name} in error state")
                # Could implement auto-restart here
            
            elif status['error_count'] > 10:
                self.logger.warning(f"Agent {agent.name} has high error count: {status['error_count']}")
    
    async def _run_prediction_pipeline(self):
        """Run the prediction pipeline"""
        if self.pipeline_running:
            return  # Already running
        
        self.pipeline_running = True
        
        try:
            # Step 1: Get data from Data Acquisition Agent
            data_agent = self.managed_agents.get('data_acquisition')
            if not data_agent:
                return
            
            # Request batch data
            data_response = await self._request_data(data_agent, window_size=12)
            if not data_response or 'batch_data' not in data_response:
                return
            
            # Step 2: Send to Prediction Agent
            prediction_agent = self.managed_agents.get('prediction')
            if not prediction_agent:
                return
            
            prediction = await self._request_prediction(
                prediction_agent,
                data_response['batch_data']
            )
            
            if prediction and not prediction.get('error'):
                self.system_state['total_predictions'] += 1
                self.system_state['last_prediction_time'] = datetime.now().isoformat()
                
                # Step 3: Send to Risk Assessment Agent
                risk_agent = self.managed_agents.get('risk_assessment')
                if risk_agent:
                    await self._request_risk_assessment(risk_agent, prediction)
        
        finally:
            self.pipeline_running = False
    
    async def _request_data(self, agent: BaseAgent, window_size: int) -> Optional[Dict]:
        """Request data from Data Acquisition Agent"""
        message = AgentMessage(
            receiver=agent.agent_id,
            message_type=MessageType.DATA,
            payload={
                "request_type": "batch",
                "window_size": window_size
            },
            requires_response=True
        )
        
        response = await self.send_and_wait(message, timeout=5.0)
        return response.payload if response else None
    
    async def _request_prediction(self, agent: BaseAgent, batch_data: Dict) -> Optional[Dict]:
        """Request prediction from Prediction Agent"""
        message = AgentMessage(
            receiver=agent.agent_id,
            message_type=MessageType.DATA,
            payload={
                "event": "predict",
                "batch_data": batch_data
            },
            requires_response=True
        )
        
        response = await self.send_and_wait(message, timeout=10.0)
        return response.payload if response else None
    
    async def _request_risk_assessment(self, agent: BaseAgent, prediction: Dict):
        """Request risk assessment"""
        message = AgentMessage(
            receiver=agent.agent_id,
            message_type=MessageType.PREDICTION,
            payload=prediction
        )
        
        await self.send_message(message)
    
    async def _handle_alert(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle alerts from agents"""
        severity = message.payload.get('severity', 'unknown')
        
        self.system_state['active_alerts'] += 1
        
        self.logger.warning(f"ALERT received: {severity}")
        
        # Could implement escalation logic here
        if severity == 'critical':
            self.logger.critical("CRITICAL ALERT - Immediate action required!")
        
        return None
    
    async def _handle_prediction(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle predictions"""
        prediction = message.payload
        
        if prediction.get('cascade_detected'):
            self.logger.warning(
                f"Cascade detected! Probability: {prediction.get('cascade_probability', 0):.3f}"
            )
        
        return None
    
    async def _handle_coordination(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle coordination messages"""
        command = message.payload.get('command')
        
        if command == 'status':
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload=self.get_system_status()
            )
        
        elif command == 'start_pipeline':
            self.prediction_interval = message.payload.get('interval', 1.0)
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={'status': 'pipeline_started'}
            )
        
        elif command == 'stop_pipeline':
            # Disable prediction task
            for task in self.scheduled_tasks.values():
                if task.name == 'prediction_pipeline':
                    task.enabled = False
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={'status': 'pipeline_stopped'}
            )
        
        return None
    
    async def _handle_data(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle data messages"""
        self.logger.debug(f"DATA message received from {message.sender}")
        # Implement data handling logic here if needed
        
        return None
    
    def _update_system_state(self):
        """Update system state"""
        running_count = sum(
            1 for a in self.managed_agents.values()
            if a.state == AgentState.RUNNING
        )
        self.system_state['agents_running'] = running_count
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        agent_statuses = {
            agent_id: agent.get_status()
            for agent_id, agent in self.managed_agents.items()
        }
        
        return {
            'system_state': self.system_state,
            'agents': agent_statuses,
            'scheduled_tasks': {
                task.name: {
                    'enabled': task.enabled,
                    'interval': task.interval_seconds,
                    'last_run': task.last_run.isoformat() if task.last_run else None
                }
                for task in self.scheduled_tasks.values()
            },
            'message_bus_stats': {
                'registered_agents': len(self.message_bus.agents),
                'total_messages': len(self.message_bus.message_log)
            }
        }
