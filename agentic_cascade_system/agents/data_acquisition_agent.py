"""
Data Acquisition Agent
======================
Responsible for collecting and preprocessing multi-modal data from various sources:
- Environmental data (satellite, weather)
- Infrastructure data (SCADA, PMU)
- Robotic platform data (visual, thermal, sensors)

Author: Kraftgene AI Inc.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle
from pathlib import Path

from .base_agent import (
    BaseAgent, AgentMessage, MessageType, AgentCapability, AgentState
)


@dataclass
class DataStream:
    """Represents a continuous data stream"""
    name: str
    source_type: str
    sampling_rate_hz: float
    buffer_size: int
    data_buffer: List[Any]
    last_update: datetime
    is_active: bool = True


class DataAcquisitionAgent(BaseAgent):
    """
    Agent responsible for multi-modal data acquisition and preprocessing.
    
    Corresponds to the Data Acquisition Layer in the research paper:
    - Environmental data sources (satellite imagery, weather)
    - Infrastructure telemetry (SCADA, PMU)
    - Autonomous robotic platforms (visual, thermal, sensors)
    """
    
    def __init__(self, agent_id: str = "data_acquisition", data_dir: str = "./data", topology_path: str = None):
        super().__init__(
            agent_id=agent_id,
            name="DataAcquisitionAgent",
            description="Multi-modal data acquisition and preprocessing"
        )
        
        self.data_dir = Path(data_dir)
        self.topology_path = topology_path or str(self.data_dir / "grid_topology.pkl")
        
        # Data streams
        self.data_streams: Dict[str, DataStream] = {}
        
        # Grid topology
        self.edge_index: Optional[np.ndarray] = None
        self.num_nodes: int = 0
        self.num_edges: int = 0
        
        # Normalization parameters (from research paper)
        self.base_mva = 100.0
        self.base_frequency = 60.0
        
        # Current data buffers
        self.current_data: Dict[str, torch.Tensor] = {}
        self.sequence_buffer: List[Dict[str, torch.Tensor]] = []
        self.buffer_max_size = 60  # Store last 60 timesteps (30 minutes at 30s intervals)
        
        # Data quality metrics
        self.missing_data_count: Dict[str, int] = {}
        self.data_quality_scores: Dict[str, float] = {}
        
        # Register capabilities
        self._register_capabilities()
        
        # Register message handlers
        self.message_handlers[MessageType.DATA] = self._handle_data_request
        self.message_handlers[MessageType.QUERY] = self._handle_query  # Register handler for QUERY messages for data requests
    
    def _register_capabilities(self):
        """Register agent capabilities"""
        self.register_capability(AgentCapability(
            name="environmental_data_acquisition",
            description="Collect and process satellite imagery and weather data",
            input_types=["satellite_feed", "weather_api"],
            output_types=["satellite_tensor", "weather_sequence"],
            latency_ms=50.0,
            reliability=0.98
        ))
        
        self.register_capability(AgentCapability(
            name="infrastructure_data_acquisition",
            description="Collect SCADA telemetry and PMU measurements",
            input_types=["scada_feed", "pmu_feed"],
            output_types=["scada_tensor", "pmu_tensor"],
            latency_ms=10.0,
            reliability=0.99
        ))
        
        self.register_capability(AgentCapability(
            name="robotic_data_acquisition",
            description="Collect visual, thermal, and sensor data from robots",
            input_types=["robot_visual", "robot_thermal", "robot_sensors"],
            output_types=["visual_tensor", "thermal_tensor", "sensor_tensor"],
            latency_ms=100.0,
            reliability=0.95
        ))
    
    async def initialize(self):
        """Initialize data acquisition systems"""
        self.logger.info("Initializing Data Acquisition Agent...")
        
        # Load grid topology
        await self._load_topology()
        
        # Initialize data streams
        self._initialize_streams()
        
        # Initialize data buffers
        self._initialize_buffers()
        
        self.logger.info(f"Data Acquisition initialized: {self.num_nodes} nodes, {self.num_edges} edges")
    
    async def _load_topology(self):
        """Load grid topology from file"""
        try:
            with open(self.topology_path, 'rb') as f:
                topology = pickle.load(f)
            
            self.edge_index = topology['edge_index']
            self.num_edges = self.edge_index.shape[1]
            if isinstance(self.edge_index, torch.Tensor):
                self.num_nodes = int(self.edge_index.max().item()) + 1
            else:
                self.num_nodes = int(np.max(self.edge_index)) + 1
            
            # Store additional topology info if available
            self.internal_state['topology'] = {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'edge_index': self.edge_index
            }
            
            self.logger.info(f"Loaded topology: {self.num_nodes} nodes, {self.num_edges} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to load topology: {e}")
            raise
    
    def _initialize_streams(self):
        """Initialize data streams for each modality"""
        stream_configs = [
            ("scada", "infrastructure", 2.0, 120),
            ("pmu", "infrastructure", 30.0, 1800),
            ("satellite", "environmental", 0.1, 12),
            ("weather", "environmental", 0.017, 6),  # ~1 per minute
            ("visual", "robotic", 1.0, 60),
            ("thermal", "robotic", 1.0, 60),
            ("sensors", "robotic", 10.0, 600)
        ]
        
        for name, source_type, rate, buffer_size in stream_configs:
            self.data_streams[name] = DataStream(
                name=name,
                source_type=source_type,
                sampling_rate_hz=rate,
                buffer_size=buffer_size,
                data_buffer=[],
                last_update=datetime.now()
            )
            self.missing_data_count[name] = 0
            self.data_quality_scores[name] = 1.0
    
    def _initialize_buffers(self):
        """Initialize data buffers with proper dimensions"""
        # Based on the model architecture from the research paper
        self.current_data = {
            'scada_data': torch.zeros(self.num_nodes, 13),
            'pmu_sequence': torch.zeros(self.num_nodes, 7),
            'satellite_data': torch.zeros(self.num_nodes, 64),
            'weather_sequence': torch.zeros(self.num_nodes, 8),
            'threat_indicators': torch.zeros(self.num_nodes, 6),
            'equipment_status': torch.zeros(self.num_nodes, 10),
            'visual_data': torch.zeros(self.num_nodes, 64),
            'thermal_data': torch.zeros(self.num_nodes, 64),
            'sensor_data': torch.zeros(self.num_nodes, 8),
            'edge_attr': torch.zeros(self.num_edges, 4)
        }
    
    async def execute(self):
        """Main execution loop - continuously acquire and process data"""
        self.logger.info("Starting data acquisition loop...")
        
        acquisition_interval = 0.5  # 500ms acquisition cycle
        
        while self._running:
            try:
                self.state = AgentState.PROCESSING
                
                # Simulate data acquisition from various sources
                await self._acquire_environmental_data()
                await self._acquire_infrastructure_data()
                await self._acquire_robotic_data()
                
                # Preprocess and normalize data
                preprocessed = self._preprocess_data()
                
                # Update sequence buffer
                self._update_sequence_buffer(preprocessed)
                
                # Broadcast data availability
                await self._broadcast_data_ready()
                
                self.state = AgentState.RUNNING
                await asyncio.sleep(acquisition_interval)
                
            except Exception as e:
                self.logger.error(f"Data acquisition error: {e}")
                self.error_count += 1
                await asyncio.sleep(1.0)
    
    async def _acquire_environmental_data(self):
        """Acquire environmental data (satellite + weather)"""
        # In production: connect to actual data sources
        # Here: simulate with realistic synthetic data
        
        # Satellite data - 64-dim feature vector per node
        # Simulates: thermal anomalies, vegetation health, infrastructure visibility
        satellite = np.random.randn(self.num_nodes, 64).astype(np.float32)
        # Add spatial correlation
        satellite = 0.7 * satellite + 0.3 * np.mean(satellite, axis=0)
        
        # Weather data - 8-dim: temp, humidity, wind_speed, wind_dir, precip, pressure, visibility, uv
        weather_base = np.array([25.0, 60.0, 15.0, 180.0, 0.0, 1013.0, 10.0, 5.0])
        weather = weather_base + np.random.randn(self.num_nodes, 8) * np.array([5, 10, 5, 30, 1, 5, 2, 1])
        
        # Threat indicators - 6-dim: fire, flood, wind, ice, lightning, seismic
        threat = np.clip(np.random.exponential(0.1, (self.num_nodes, 6)), 0, 1).astype(np.float32)
        
        self.current_data['satellite_data'] = torch.from_numpy(satellite)
        self.current_data['weather_sequence'] = torch.from_numpy(weather.astype(np.float32))
        self.current_data['threat_indicators'] = torch.from_numpy(threat)
        
        self.data_streams['satellite'].last_update = datetime.now()
        self.data_streams['weather'].last_update = datetime.now()
    
    async def _acquire_infrastructure_data(self):
        """Acquire infrastructure data (SCADA + PMU)"""
        # SCADA data - 13 dimensions per node
        # [voltage_pu, angle_deg, p_mw, q_mvar, p_gen, q_gen, load_p, load_q, 
        #  tap_ratio, status, temperature, age_years, maintenance_score]
        
        scada = np.zeros((self.num_nodes, 13), dtype=np.float32)
        scada[:, 0] = 1.0 + np.random.randn(self.num_nodes) * 0.02  # Voltage ~1.0 pu
        scada[:, 1] = np.random.randn(self.num_nodes) * 10  # Angle
        scada[:, 2] = np.random.uniform(0, 100, self.num_nodes) * self.base_mva  # P (will be normalized)
        scada[:, 3] = np.random.uniform(-20, 50, self.num_nodes) * self.base_mva  # Q
        scada[:, 4] = np.random.uniform(0, 200, self.num_nodes) * self.base_mva  # P_gen
        scada[:, 5] = np.random.uniform(-50, 100, self.num_nodes) * self.base_mva  # Q_gen
        scada[:, 6] = np.random.uniform(0, 150, self.num_nodes)  # Load P
        scada[:, 7] = np.random.uniform(0, 50, self.num_nodes)  # Load Q
        scada[:, 8] = 1.0 + np.random.randn(self.num_nodes) * 0.05  # Tap ratio
        scada[:, 9] = np.ones(self.num_nodes)  # Status (1 = operational)
        scada[:, 10] = 40 + np.random.randn(self.num_nodes) * 10  # Temperature
        scada[:, 11] = np.random.uniform(5, 40, self.num_nodes)  # Age
        scada[:, 12] = np.random.uniform(0.5, 1.0, self.num_nodes)  # Maintenance score
        
        # PMU data - 7 dimensions per node
        # [voltage_mag, voltage_angle, current_mag, current_angle, rocof, freq, power_factor]
        pmu = np.zeros((self.num_nodes, 7), dtype=np.float32)
        pmu[:, 0] = scada[:, 0]  # Voltage magnitude
        pmu[:, 1] = scada[:, 1] * np.pi / 180  # Voltage angle (radians)
        pmu[:, 2] = np.abs(scada[:, 2] + 1j * scada[:, 3]) / scada[:, 0] / 1000  # Current
        pmu[:, 3] = np.arctan2(scada[:, 3], scada[:, 2])  # Current angle
        pmu[:, 4] = np.random.randn(self.num_nodes) * 0.01  # ROCOF
        pmu[:, 5] = self.base_frequency + np.random.randn(self.num_nodes) * 0.1  # Frequency
        pmu[:, 6] = np.clip(np.random.uniform(0.85, 1.0, self.num_nodes), 0, 1)  # Power factor
        
        # Equipment status - 10 dimensions
        # [switch_status, breaker_status, transformer_loading, line_loading, 
        #  cooling_status, alarm_count, trip_count, overload_duration, fault_current, protection_status]
        equipment = np.zeros((self.num_nodes, 10), dtype=np.float32)
        equipment[:, 0] = np.random.choice([0, 1], self.num_nodes, p=[0.02, 0.98])  # Switch
        equipment[:, 1] = np.random.choice([0, 1], self.num_nodes, p=[0.01, 0.99])  # Breaker
        equipment[:, 2] = np.random.uniform(0.3, 0.9, self.num_nodes)  # Transformer loading
        equipment[:, 3] = np.random.uniform(0.2, 0.8, self.num_nodes)  # Line loading
        equipment[:, 4] = np.ones(self.num_nodes)  # Cooling status
        equipment[:, 5] = np.random.poisson(0.5, self.num_nodes)  # Alarm count
        equipment[:, 6] = np.random.poisson(0.1, self.num_nodes)  # Trip count
        equipment[:, 7] = np.zeros(self.num_nodes)  # Overload duration
        equipment[:, 8] = np.random.exponential(100, self.num_nodes)  # Fault current capacity
        equipment[:, 9] = np.ones(self.num_nodes)  # Protection status
        
        # Edge attributes - 4 dimensions per edge
        # [impedance, capacity_mva, current_loading, thermal_limit_pct]
        edge_attr = np.zeros((self.num_edges, 4), dtype=np.float32)
        edge_attr[:, 0] = np.random.uniform(0.001, 0.1, self.num_edges)  # Impedance
        edge_attr[:, 1] = np.random.uniform(100, 500, self.num_edges) * self.base_mva  # Capacity
        edge_attr[:, 2] = np.random.uniform(0.2, 0.7, self.num_edges)  # Loading
        edge_attr[:, 3] = np.random.uniform(0.1, 0.6, self.num_edges)  # Thermal limit pct
        
        self.current_data['scada_data'] = torch.from_numpy(scada)
        self.current_data['pmu_sequence'] = torch.from_numpy(pmu)
        self.current_data['equipment_status'] = torch.from_numpy(equipment)
        self.current_data['edge_attr'] = torch.from_numpy(edge_attr)
        
        self.data_streams['scada'].last_update = datetime.now()
        self.data_streams['pmu'].last_update = datetime.now()
    
    async def _acquire_robotic_data(self):
        """Acquire robotic platform data (visual + thermal + sensors)"""
        # Visual features - 64-dim CNN feature vector per node
        visual = np.random.randn(self.num_nodes, 64).astype(np.float32) * 0.5
        
        # Thermal features - 64-dim thermal anomaly features per node
        thermal = np.random.randn(self.num_nodes, 64).astype(np.float32) * 0.3
        # Add hotspot detection signals for some nodes
        hotspot_nodes = np.random.choice(self.num_nodes, size=int(self.num_nodes * 0.1), replace=False)
        thermal[hotspot_nodes] += np.random.uniform(0.5, 2.0, (len(hotspot_nodes), 64))
        
        # Sensor data - 8 dimensions
        # [vibration, acoustic, magnetic, humidity, gas, radiation, proximity, motion]
        sensors = np.zeros((self.num_nodes, 8), dtype=np.float32)
        sensors[:, 0] = np.random.exponential(0.1, self.num_nodes)  # Vibration
        sensors[:, 1] = np.random.uniform(30, 70, self.num_nodes)  # Acoustic (dB)
        sensors[:, 2] = np.random.randn(self.num_nodes) * 10  # Magnetic
        sensors[:, 3] = np.random.uniform(30, 80, self.num_nodes)  # Humidity
        sensors[:, 4] = np.random.exponential(1, self.num_nodes)  # Gas detection
        sensors[:, 5] = np.random.exponential(0.01, self.num_nodes)  # Radiation
        sensors[:, 6] = np.random.uniform(0, 100, self.num_nodes)  # Proximity
        sensors[:, 7] = np.random.choice([0, 1], self.num_nodes, p=[0.95, 0.05])  # Motion
        
        self.current_data['visual_data'] = torch.from_numpy(visual)
        self.current_data['thermal_data'] = torch.from_numpy(thermal)
        self.current_data['sensor_data'] = torch.from_numpy(sensors)
        
        self.data_streams['visual'].last_update = datetime.now()
        self.data_streams['thermal'].last_update = datetime.now()
        self.data_streams['sensors'].last_update = datetime.now()
    
    def _preprocess_data(self) -> Dict[str, torch.Tensor]:
        """Preprocess and normalize acquired data"""
        preprocessed = {}
        
        # Copy current data
        for key, tensor in self.current_data.items():
            preprocessed[key] = tensor.clone()
        
        # Normalize power values in SCADA data
        scada = preprocessed['scada_data'].clone()
        if scada.shape[1] >= 6:
            scada[:, 2] = scada[:, 2] / self.base_mva
            scada[:, 3] = scada[:, 3] / self.base_mva
            scada[:, 4] = scada[:, 4] / self.base_mva
            scada[:, 5] = scada[:, 5] / self.base_mva
        preprocessed['scada_data'] = scada
        
        # Normalize frequency in PMU data
        pmu = preprocessed['pmu_sequence'].clone()
        if pmu.shape[1] >= 6:
            pmu[:, 5] = pmu[:, 5] / self.base_frequency
        preprocessed['pmu_sequence'] = pmu
        
        # Normalize edge capacity
        edge_attr = preprocessed['edge_attr'].clone()
        if edge_attr.shape[1] >= 2:
            edge_attr[:, 1] = edge_attr[:, 1] / self.base_mva
        preprocessed['edge_attr'] = edge_attr
        
        return preprocessed
    
    def _update_sequence_buffer(self, data: Dict[str, torch.Tensor]):
        """Update the temporal sequence buffer"""
        timestep_data = {k: v.clone() for k, v in data.items()}
        timestep_data['timestamp'] = datetime.now()
        
        self.sequence_buffer.append(timestep_data)
        
        # Maintain buffer size
        if len(self.sequence_buffer) > self.buffer_max_size:
            self.sequence_buffer.pop(0)
    
    async def _broadcast_data_ready(self):
        """Notify coordinator that new data is available"""
        await self.send_message(AgentMessage(
            receiver="coordinator",  # Specify receiver to avoid broadcast warnings
            message_type=MessageType.DATA,
            payload={
                "event": "data_ready",
                "timestamp": datetime.now().isoformat(),
                "num_timesteps": len(self.sequence_buffer),
                "data_quality": self.data_quality_scores
            },
            priority=7
        ))
    
    async def _handle_data_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle data requests from other agents"""
        request_type = message.payload.get("request_type", "current")
        
        if request_type == "current":
            # Return current preprocessed data
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "data": {k: v.numpy().tolist() for k, v in self.current_data.items()},
                    "timestamp": datetime.now().isoformat(),
                    "edge_index": self.edge_index.tolist() if self.edge_index is not None else None
                }
            )
        
        elif request_type == "sequence":
            # Return sequence data for temporal analysis
            window_size = message.payload.get("window_size", 30)
            sequence_data = self._get_sequence_data(window_size)
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "sequence": sequence_data,
                    "window_size": len(sequence_data),
                    "edge_index": self.edge_index.tolist() if self.edge_index is not None else None
                }
            )
        
        elif request_type == "batch":
            # Return batch-ready data for model inference
            batch_data = self._prepare_batch_data(
                message.payload.get("window_size", 30)
            )
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload=batch_data
            )
        
        return None
    
    def _get_sequence_data(self, window_size: int) -> List[Dict]:
        """Get sequence data for the specified window"""
        if len(self.sequence_buffer) < window_size:
            return [
                {k: v.numpy().tolist() for k, v in step.items() if isinstance(v, torch.Tensor)}
                for step in self.sequence_buffer
            ]
        return [
            {k: v.numpy().tolist() for k, v in step.items() if isinstance(v, torch.Tensor)}
            for step in self.sequence_buffer[-window_size:]
        ]
    
    def _prepare_batch_data(self, window_size: int) -> Dict:
        """Prepare data in batch format for model inference"""
        if not self.sequence_buffer:
            return {"error": "No data available"}
        
        # Stack sequence data
        sequence_keys = ['scada_data', 'pmu_sequence', 'satellite_data', 'weather_sequence',
                        'threat_indicators', 'equipment_status', 'visual_data', 
                        'thermal_data', 'sensor_data']
        
        batch_data = {}
        current_len = len(self.sequence_buffer)
        
        # --- NEW LOGIC: Handle padding/slicing ---
        for key in sequence_keys:
            if current_len < window_size:
                # Pad with the first frame repeated to fill the window
                padding_count = window_size - current_len
                first_frame = self.sequence_buffer[0][key]
                # Create a list of tensors for padding
                padding = [first_frame.clone() for _ in range(padding_count)]
                # Combine padding + actual data
                tensors = padding + [step[key] for step in self.sequence_buffer]
            else:
                # Take the last 'window_size' frames
                tensors = [step[key] for step in self.sequence_buffer[-window_size:]]
            
            stacked = torch.stack(tensors, dim=0)  # (seq_len, num_nodes, features)
            batch_data[key] = stacked.unsqueeze(0)  # Add batch dimension
        # -----------------------------------------
        
        # Edge data from latest timestep
        batch_data['edge_attr'] = self.sequence_buffer[-1]['edge_attr'].unsqueeze(0)
        
        # Handle edge_index (Tensor vs Numpy fix from before)
        if isinstance(self.edge_index, torch.Tensor):
            batch_data['edge_index'] = self.edge_index.long()
        else:
            batch_data['edge_index'] = torch.from_numpy(self.edge_index).long()
            
        batch_data['edge_mask'] = torch.ones(1, self.num_edges)
        batch_data['sequence_length'] = torch.tensor([window_size]) # Fixed length
        batch_data['temporal_sequence'] = batch_data['scada_data']
        
        return {
            "batch_data": {k: v.numpy().tolist() for k, v in batch_data.items()},
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "sequence_length": window_size
        }
    
    async def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle query messages"""
        query_type = message.payload.get("query")
        
        if query_type == "topology":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "num_nodes": self.num_nodes,
                    "num_edges": self.num_edges,
                    "edge_index": self.edge_index.tolist() if self.edge_index is not None else None
                }
            )
        
        elif query_type == "data_quality":
            return AgentMessage(
                message_type=MessageType.RESPONSE,
                payload={
                    "quality_scores": self.data_quality_scores,
                    "missing_counts": self.missing_data_count,
                    "stream_status": {
                        name: {
                            "active": stream.is_active,
                            "last_update": stream.last_update.isoformat()
                        }
                        for name, stream in self.data_streams.items()
                    }
                }
            )
        
        return await super()._handle_query(message)
