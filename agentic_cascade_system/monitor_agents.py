"""
Real-Time Agent Monitoring Dashboard
====================================
Monitors and displays agent activity, messages, and system health.

Author: Kraftgene AI Inc.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.base_agent import MessageType


class AgentMonitor:
    """Real-time monitoring dashboard for the multi-agent system"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.message_bus = coordinator.message_bus
        self.start_time = datetime.now()
        
        # Monitoring state
        self.message_counts = {}
        self.agent_activity = {}
        self.last_messages = []
        self.alerts = []
        
        # Initialize counters for all message types
        for msg_type in MessageType:
            self.message_counts[msg_type.value] = 0
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        print("\n" + "="*80)
        print("MULTI-AGENT CASCADE DETECTION SYSTEM - REAL-TIME MONITOR")
        print("="*80)
        print("\nPress Ctrl+C to stop monitoring\n")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                
                # Update statistics
                self._update_statistics()
                
                # Display dashboard
                self._display_dashboard(iteration)
                
                # Display recent activity
                if iteration % 5 == 0:  # Every 5 seconds
                    self._display_recent_activity()
                
                await asyncio.sleep(1.0)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"\n[ERROR] Monitor error: {e}")
                await asyncio.sleep(1.0)
    
    def _update_statistics(self):
        """Update monitoring statistics"""
        # Count messages by type
        for msg in self.message_bus.message_log:
            msg_type_str = msg.message_type.value
            self.message_counts[msg_type_str] = self.message_counts.get(msg_type_str, 0)
        
        # Track agent activity
        for agent_id, agent in self.coordinator.managed_agents.items():
            self.agent_activity[agent_id] = {
                'state': agent.state.value,
                'message_count': agent.message_count,
                'error_count': agent.error_count,
                'last_activity': agent.last_activity
            }
    
    def _display_dashboard(self, iteration: int):
        """Display the monitoring dashboard"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        # Header
        uptime = (datetime.now() - self.start_time).total_seconds()
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "CASCADE DETECTION AGENT MONITOR" + " "*27 + "║")
        print("╠" + "="*78 + "╣")
        print(f"║ Uptime: {uptime:.0f}s | Iteration: {iteration} | Time: {datetime.now().strftime('%H:%M:%S')}" + " "*(78-len(f"Uptime: {uptime:.0f}s | Iteration: {iteration} | Time: {datetime.now().strftime('%H:%M:%S')}")) + "║")
        print("╚" + "="*78 + "╝")
        
        # System status
        print("\n📊 SYSTEM STATUS")
        print("-" * 80)
        system_state = self.coordinator.system_state
        print(f"  Status: {system_state['status'].upper()}")
        print(f"  Agents Running: {system_state['agents_running']}/{len(self.coordinator.managed_agents)}")
        print(f"  Total Predictions: {system_state['total_predictions']}")
        print(f"  Active Alerts: {system_state['active_alerts']}")
        if system_state['last_prediction_time']:
            print(f"  Last Prediction: {system_state['last_prediction_time']}")
        
        # Agent status
        print("\n🤖 AGENT STATUS")
        print("-" * 80)
        print(f"{'Agent':<25} {'State':<12} {'Messages':<10} {'Errors':<8} {'Last Activity'}")
        print("-" * 80)
        
        for agent_id, agent in self.coordinator.managed_agents.items():
            status = agent.get_status()
            last_activity = status['last_activity']
            time_since = (datetime.now() - datetime.fromisoformat(last_activity)).total_seconds()
            
            # Color code by state
            state_str = status['state'].upper()
            if status['state'] == 'error':
                state_indicator = "🔴"
            elif status['state'] == 'processing':
                state_indicator = "🟡"
            elif status['state'] == 'running':
                state_indicator = "🟢"
            else:
                state_indicator = "⚪"
            
            print(f"{agent.name:<25} {state_indicator} {state_str:<10} {status['message_count']:<10} {status['error_count']:<8} {time_since:.1f}s ago")
        
        # Message bus statistics
        print("\n📨 MESSAGE BUS ACTIVITY")
        print("-" * 80)
        total_messages = len(self.message_bus.message_log)
        print(f"  Total Messages: {total_messages}")
        
        # Message type breakdown
        if total_messages > 0:
            print("\n  Message Types:")
            for msg_type, count in sorted(self.message_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_messages) * 100
                    bar_length = int(percentage / 2)
                    bar = "█" * bar_length
                    print(f"    {msg_type:<20} {count:>5} ({percentage:>5.1f}%) {bar}")
        
        # Recent messages (last 5)
        print("\n📬 RECENT MESSAGES")
        print("-" * 80)
        recent_messages = self.message_bus.message_log[-5:] if self.message_bus.message_log else []
        if recent_messages:
            for msg in reversed(recent_messages):
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                sender = msg.sender[:15] if msg.sender else "system"
                receiver = msg.receiver[:15] if msg.receiver else "broadcast"
                msg_type = msg.message_type.value
                
                # Get summary of payload
                payload_summary = self._summarize_payload(msg.payload)
                
                print(f"  [{timestamp}] {sender} → {receiver} | {msg_type}")
                if payload_summary:
                    print(f"           └─ {payload_summary}")
        else:
            print("  No messages yet")
        
        # Active alerts
        if system_state['active_alerts'] > 0:
            print("\n⚠️  ACTIVE ALERTS")
            print("-" * 80)
            print(f"  {system_state['active_alerts']} alert(s) currently active")
        
        # Footer
        print("\n" + "─" * 80)
        print("💡 TIP: Watch 'Messages' column to see agents processing data in real-time")
        print("🔍 Look for 'DATA' messages from DataAcquisitionAgent (should appear every 1s)")
        print("🎯 Look for 'PREDICTION' messages when coordinator triggers prediction pipeline")
        print("─" * 80)
    
    def _summarize_payload(self, payload: Dict) -> str:
        """Summarize message payload for display"""
        if not payload:
            return ""
        
        # Extract key information
        if 'event' in payload:
            event = payload['event']
            if event == 'data_ready':
                num_timesteps = payload.get('num_timesteps', 0)
                return f"Data ready: {num_timesteps} timesteps"
            elif event == 'predict':
                return "Prediction request"
        
        if 'cascade_detected' in payload:
            if payload['cascade_detected']:
                prob = payload.get('cascade_probability', 0)
                return f"CASCADE DETECTED! Probability: {prob:.3f}"
            else:
                return "No cascade detected"
        
        if 'severity' in payload:
            severity = payload['severity']
            return f"Alert: {severity.upper()}"
        
        if 'request_type' in payload:
            return f"Request: {payload['request_type']}"
        
        # Generic summary
        keys = list(payload.keys())[:3]
        return f"Keys: {', '.join(keys)}"
    
    def _display_recent_activity(self):
        """Display detailed recent activity"""
        print("\n" + "="*80)
        print("📋 DETAILED ACTIVITY LOG (Last 10 messages)")
        print("="*80)
        
        recent = self.message_bus.message_log[-10:] if self.message_bus.message_log else []
        for i, msg in enumerate(reversed(recent), 1):
            timestamp = msg.timestamp.strftime("%H:%M:%S.%f")[:-3]
            print(f"\n[{i}] {timestamp} - Priority: {msg.priority}")
            print(f"    From: {msg.sender}")
            print(f"    To: {msg.receiver if msg.receiver else 'ALL (broadcast)'}")
            print(f"    Type: {msg.message_type.value}")
            print(f"    Payload: {self._format_payload(msg.payload)}")
    
    def _format_payload(self, payload: Dict, indent: int = 4) -> str:
        """Format payload for readable display"""
        if not payload:
            return "{}"
        
        # Limit size for display
        if len(str(payload)) > 200:
            keys = list(payload.keys())
            return f"{{{', '.join(f'{k}:...' for k in keys[:5])}}}"
        
        return str(payload)


async def main():
    """Main entry point for monitoring"""
    import argparse
    from main import create_system
    
    parser = argparse.ArgumentParser(description="Monitor multi-agent cascade detection system")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    args = parser.parse_args()
    
    print("Starting multi-agent system...")
    
    # Create the system
    coordinator = create_system(
        model_path=args.model_path,
        data_dir=args.data_dir
    )
    
    # Create monitor
    monitor = AgentMonitor(coordinator)
    
    # Start coordinator
    coordinator_task = asyncio.create_task(coordinator.start())
    
    # Wait a moment for system to initialize
    await asyncio.sleep(2.0)
    
    # Start monitoring
    try:
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop coordinator
        await coordinator.stop_all_agents()
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(main())
