"""
Base Agent Class for Multi-Agent Cascade Failure Detection System
==================================================================
Implements the foundational agent architecture with communication,
state management, and lifecycle methods.

Author: Kraftgene AI Inc.
"""

import asyncio
import uuid
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import queue
import threading


class AgentState(Enum):
    """Agent lifecycle states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class MessageType(Enum):
    """Inter-agent message types"""
    DATA = "data"
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    PREDICTION = "prediction"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""  # Empty string means broadcast
    message_type: MessageType = MessageType.DATA
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, 10 is highest
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response tracking
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        return cls(
            id=data["id"],
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data["priority"],
            requires_response=data["requires_response"],
            correlation_id=data.get("correlation_id")
        )


@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    latency_ms: float = 0.0
    reliability: float = 1.0


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the cascade failure detection system.
    
    Implements:
    - Asynchronous message handling
    - State management
    - Capability advertisement
    - Logging and monitoring
    """
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.capabilities: List[AgentCapability] = []
        
        # Communication
        self.message_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # State tracking
        self.internal_state: Dict[str, Any] = {}
        self.last_activity: datetime = datetime.now()
        self.message_count: int = 0
        self.error_count: int = 0
        
        # Message bus reference (set by orchestrator)
        self._message_bus: Optional['MessageBus'] = None
        
        # Logging
        self.logger = logging.getLogger(f"Agent.{self.name}")
        
        # Threading
        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.COMMAND] = self._handle_command
        self.message_handlers[MessageType.QUERY] = self._handle_query
    
    def set_message_bus(self, bus: 'MessageBus'):
        """Set the message bus for communication"""
        self._message_bus = bus
    
    async def send_message(self, message: AgentMessage):
        """Send a message through the message bus"""
        message.sender = self.agent_id
        if self._message_bus:
            await self._message_bus.publish(message)
        else:
            self.logger.warning(f"No message bus configured for {self.name}")
    
    async def send_and_wait(self, message: AgentMessage, timeout: float = 30.0) -> Optional[AgentMessage]:
        """Send a message and wait for response"""
        message.requires_response = True
        future = asyncio.get_event_loop().create_future()
        self.pending_responses[message.id] = future
        
        await self.send_message(message)
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to {message.id}")
            return None
        finally:
            self.pending_responses.pop(message.id, None)
    
    def receive_message(self, message: AgentMessage):
        """Receive a message from the message bus"""
        # Priority queue uses (priority, timestamp, message) for ordering
        # Negate priority so higher priority = lower number = processed first
        self.message_queue.put(
            (-message.priority, message.timestamp.timestamp(), message)
        )
    
    async def process_messages(self):
        """Process incoming messages from the queue"""
        while self._running:
            try:
                if not self.message_queue.empty():
                    _, _, message = self.message_queue.get_nowait()
                    await self._process_single_message(message)
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.error_count += 1
    
    async def _process_single_message(self, message: AgentMessage):
        """Process a single message"""
        self.message_count += 1
        self.last_activity = datetime.now()
        
        # Check if this is a response to a pending request
        if message.correlation_id and message.correlation_id in self.pending_responses:
            self.pending_responses[message.correlation_id].set_result(message)
            return
        
        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                self.state = AgentState.PROCESSING
                response = await handler(message)
                
                # Send response if required
                if message.requires_response and response:
                    response.correlation_id = message.id
                    response.receiver = message.sender
                    await self.send_message(response)
                    
            except Exception as e:
                self.logger.error(f"Handler error for {message.message_type}: {e}")
                self.error_count += 1
            finally:
                self.state = AgentState.RUNNING
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
    
    async def _handle_heartbeat(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle heartbeat messages"""
        return AgentMessage(
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            payload={
                "status": self.state.value,
                "uptime": (datetime.now() - self.last_activity).total_seconds(),
                "message_count": self.message_count,
                "error_count": self.error_count
            }
        )
    
    async def _handle_command(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle command messages - override in subclasses"""
        command = message.payload.get("command")
        if command == "status":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload=self.get_status()
            )
        elif command == "stop":
            await self.stop()
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={"status": "stopped"}
            )
        return None
    
    async def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle query messages - override in subclasses"""
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": [c.name for c in self.capabilities],
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_activity": self.last_activity.isoformat()
        }
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def execute(self):
        """Main execution loop - must be implemented by subclasses"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.state = AgentState.INITIALIZING
        self.logger.info(f"Starting agent: {self.name}")
        
        try:
            await self.initialize()
            self._running = True
            self.state = AgentState.RUNNING
            
            # Start message processing in background
            asyncio.create_task(self.process_messages())
            
            # Start main execution
            await self.execute()
            
        except Exception as e:
            self.logger.error(f"Agent startup failed: {e}")
            self.state = AgentState.ERROR
            raise
    
    async def stop(self):
        """Stop the agent"""
        self.logger.info(f"Stopping agent: {self.name}")
        self._running = False
        self.state = AgentState.TERMINATED
    
    def register_capability(self, capability: AgentCapability):
        """Register a capability"""
        self.capabilities.append(capability)
        self.logger.debug(f"Registered capability: {capability.name}")


class MessageBus:
    """
    Central message bus for inter-agent communication.
    Implements pub/sub pattern with topic filtering.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.subscriptions: Dict[MessageType, List[str]] = {}
        self.message_log: List[AgentMessage] = []
        self.logger = logging.getLogger("MessageBus")
        self._lock = asyncio.Lock()
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        agent.set_message_bus(self)
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # Remove from subscriptions
            for msg_type in self.subscriptions:
                if agent_id in self.subscriptions[msg_type]:
                    self.subscriptions[msg_type].remove(agent_id)
    
    def subscribe(self, agent_id: str, message_type: MessageType):
        """Subscribe an agent to a message type"""
        if message_type not in self.subscriptions:
            self.subscriptions[message_type] = []
        if agent_id not in self.subscriptions[message_type]:
            self.subscriptions[message_type].append(agent_id)
    
    async def publish(self, message: AgentMessage):
        """Publish a message to the bus"""
        async with self._lock:
            self.message_log.append(message)
            
            # If specific receiver, send directly
            if message.receiver and message.receiver in self.agents:
                self.agents[message.receiver].receive_message(message)
                return
            
            subscribers = self.subscriptions.get(message.message_type, [])
            delivered = False
            for agent_id in subscribers:
                if agent_id != message.sender and agent_id in self.agents:
                    self.agents[agent_id].receive_message(message)
                    delivered = True
            
            if not delivered and not message.receiver:
                self.logger.debug(f"No subscribers for message type: {message.message_type.value}")
