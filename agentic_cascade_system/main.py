"""
Agentic Cascade Failure Detection System - Main Entry Point
============================================================
Multi-agent system for real-time cascade failure prediction.

This implements the proof-of-concept agentic AI architecture described
in the research paper, coordinating:
- Data Acquisition Agent
- Prediction Agent  
- Risk Assessment Agent
- Coordination Agent (Orchestrator)

Author: Kraftgene AI Inc.

Usage:
    python main.py --model_path checkpoints/best_f1_model.pth --data_dir ./data
"""

import asyncio
import argparse
import logging
import sys
import signal
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import MessageBus
from agents.data_acquisition_agent import DataAcquisitionAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.coordination_agent import CoordinationAgent


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the multi-agent system"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'agentic_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


class AgenticCascadeSystem:
    """
    Main system class that initializes and runs the multi-agent cascade
    failure detection system.
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        topology_path: str = None,
        device: str = None,
        prediction_interval: float = 1.0
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.topology_path = topology_path or f"{data_dir}/grid_topology.pkl"
        self.device = device
        self.prediction_interval = prediction_interval
        
        self.logger = logging.getLogger("AgenticSystem")
        
        # Agents
        self.coordinator: CoordinationAgent = None
        self.data_agent: DataAcquisitionAgent = None
        self.prediction_agent: PredictionAgent = None
        self.risk_agent: RiskAssessmentAgent = None
        
        # Running state
        self._running = False
    
    def _create_agents(self):
        """Create all system agents"""
        self.logger.info("Creating agents...")
        
        # Create Coordination Agent (Orchestrator)
        self.coordinator = CoordinationAgent(agent_id="coordinator")
        
        # Create Data Acquisition Agent
        self.data_agent = DataAcquisitionAgent(
            agent_id="data_acquisition",
            data_dir=self.data_dir,
            topology_path=self.topology_path
        )
        
        # Create Prediction Agent
        self.prediction_agent = PredictionAgent(
            agent_id="prediction",
            model_path=self.model_path,
            device=self.device
        )
        
        # Create Risk Assessment Agent
        self.risk_agent = RiskAssessmentAgent(
            agent_id="risk_assessment"
        )
        
        self.logger.info("All agents created")
    
    def _register_agents(self):
        """Register agents with the coordinator"""
        self.logger.info("Registering agents with coordinator...")
        
        self.coordinator.register_agent(self.data_agent)
        self.coordinator.register_agent(self.prediction_agent)
        self.coordinator.register_agent(self.risk_agent)
        
        self.logger.info("All agents registered")
    
    async def start(self):
        """Start the multi-agent system"""
        self.logger.info("=" * 60)
        self.logger.info("AGENTIC CASCADE FAILURE DETECTION SYSTEM")
        self.logger.info("=" * 60)
        
        try:
            # Create agents
            self._create_agents()
            
            # Register with coordinator
            self._register_agents()
            
            # Set prediction interval
            self.coordinator.prediction_interval = self.prediction_interval
            
            # Start coordinator (will start all other agents)
            self._running = True
            
            self.logger.info("Starting multi-agent system...")
            await self.coordinator.start()
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise
    
    async def stop(self):
        """Stop the multi-agent system"""
        self.logger.info("Stopping multi-agent system...")
        
        self._running = False
        
        if self.coordinator:
            await self.coordinator.stop_all_agents()
            await self.coordinator.stop()
        
        self.logger.info("System stopped")
    
    def get_status(self) -> dict:
        """Get system status"""
        if self.coordinator:
            return self.coordinator.get_system_status()
        return {"status": "not_initialized"}


async def run_demo(args):
    """Run a demonstration of the multi-agent system"""
    logger = logging.getLogger("Demo")
    
    # Create system
    system = AgenticCascadeSystem(
        model_path=args.model_path,
        data_dir=args.data_dir,
        topology_path=args.topology_path,
        device=args.device,
        prediction_interval=args.prediction_interval
    )
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        loop.create_task(system.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    try:
        # Start system
        await system.start()
        
        # Run for specified duration
        if args.duration > 0:
            logger.info(f"Running for {args.duration} seconds...")
            await asyncio.sleep(args.duration)
            
            # Print final status
            status = system.get_status()
            logger.info("\n" + "=" * 60)
            logger.info("FINAL SYSTEM STATUS")
            logger.info("=" * 60)
            logger.info(f"Total Predictions: {status['system_state']['total_predictions']}")
            logger.info(f"Active Alerts: {status['system_state']['active_alerts']}")
            logger.info(f"Agents Running: {status['system_state']['agents_running']}")
        else:
            # Run indefinitely
            logger.info("Running indefinitely (Ctrl+C to stop)...")
            while system._running:
                await asyncio.sleep(1.0)
    
    finally:
        await system.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic Cascade Failure Detection System"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing data and topology"
    )
    
    parser.add_argument(
        "--topology_path",
        type=str,
        default=None,
        help="Path to grid topology file (default: data_dir/grid_topology.pkl)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda)"
    )
    
    parser.add_argument(
        "--prediction_interval",
        type=float,
        default=1.0,
        help="Interval between predictions in seconds"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration to run in seconds (0 for indefinite)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run system
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
