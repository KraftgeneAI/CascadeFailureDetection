"""
Robotic Data Generator Module
==============================

PURPOSE:
--------
Generates synthetic robotic sensor data (visual feeds, thermal cameras, and
multi-sensor arrays) that are correlated with equipment condition and failures.

This module simulates what drones and robotic sensors would observe when inspecting
power grid equipment, providing early warning signs that SCADA sensors miss.

WHY ROBOTIC DATA?
-----------------
In real grids, utilities are deploying:
- Drones with visual/thermal cameras for line inspection
- Acoustic sensors to detect corona discharge
- Vibration sensors to detect mechanical stress
- Oil quality sensors in transformers

This data provides early warning signs that SCADA sensors miss.

CORRELATION STRATEGY:
---------------------
1. Equipment age/condition affects visual appearance (rust, corrosion)
2. Temperature affects thermal camera readings
3. Precursor signals (vibration, acoustic) appear before failures
4. Partial discharge increases as equipment degrades

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import numpy as np
from typing import List, Tuple


class RoboticDataGenerator:
    """
    Generates synthetic robotic sensor data correlated with equipment condition.
    
    This class creates three types of robotic data:
    1. Visual feeds (RGB images, 32x32 pixels per node)
    2. Thermal camera (temperature maps, 32x32 pixels per node)
    3. Sensor arrays (12 features per node)
    
    All data is correlated with equipment age, condition, and impending failures.
    """
    
    def __init__(
        self,
        num_nodes: int,
        equipment_age: np.ndarray,
        equipment_condition: np.ndarray
    ):
        """
        Initialize the robotic data generator.
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the power grid
        equipment_age : np.ndarray, shape (num_nodes,)
            Age of equipment at each node (0.0 to 1.0, where 1.0 is oldest)
        equipment_condition : np.ndarray, shape (num_nodes,)
            Condition of equipment at each node (0.0 to 1.0, where 1.0 is best)
        """
        self.num_nodes = num_nodes
        self.equipment_age = equipment_age
        self.equipment_condition = equipment_condition
    
    def generate_visual_data(
        self,
        failed_nodes: List[int],
        timestep: int,
        cascade_start: int,
        precursor_duration: int = 15
    ) -> np.ndarray:
        """
        Generate synthetic visual feed from drone cameras.
        
        VISUAL FEED (3 channels RGB × 32×32 pixels per node):
        - Base: Gray image (0.5 ± 0.1) representing equipment
        - Degradation: Reddish tint (rust), dark spots (corrosion)
        - Correlation: Worse condition → more visual defects
        
        Example:
        - New equipment (condition=1.0): Clean gray
        - Aged equipment (condition=0.6): Rust spots, discoloration
        - Failing equipment: Visible damage, smoke
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed (used to add failure signatures)
        timestep : int
            Current timestep (0 to sequence_length-1)
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        
        Returns:
        --------
        visual_data : np.ndarray, shape (num_nodes, 3, 32, 32)
            RGB visual feed from drone cameras (dtype=float16)
        """
        visual_data = np.zeros((self.num_nodes, 3, 32, 32), dtype=np.float16)
        
        # Generate base gray images with noise
        for node_idx in range(self.num_nodes):
            visual_data[node_idx, 0, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            visual_data[node_idx, 1, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            visual_data[node_idx, 2, :, :] = 0.5 + np.random.randn(32, 32) * 0.1
            
            # Add degradation effects (rust, corrosion)
            degradation = 1.0 - self.equipment_condition[node_idx]
            
            # Reddish tint for rust
            if degradation > 0.3:
                visual_data[node_idx, 0, :, :] += degradation * 0.2  # More red
                visual_data[node_idx, 2, :, :] -= degradation * 0.1  # Less blue
            
            # Dark spots for corrosion
            if degradation > 0.4:
                num_spots = int(degradation * 5)
                for _ in range(num_spots):
                    x, y = np.random.randint(0, 32, 2)
                    visual_data[node_idx, :, max(0,x-2):min(32,x+3), max(0,y-2):min(32,y+3)] *= 0.5
        
        # Add precursor signals before cascade
        if timestep >= cascade_start - 10 and cascade_start > 0:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
            precursor_strength = max(0, precursor_strength)
            
            for node in failed_nodes:
                # Add smoke/damage effects
                visual_data[node, 0, :, :] += 0.3 * precursor_strength  # More red
                visual_data[node, 1:3, :, :] -= 0.2 * precursor_strength  # Less green/blue
        
        return visual_data
    
    def generate_thermal_data(
        self,
        equipment_temps: np.ndarray,
        failed_nodes: List[int],
        timestep: int,
        cascade_start: int,
        precursor_duration: int = 15
    ) -> np.ndarray:
        """
        Generate synthetic thermal camera feed.
        
        THERMAL CAMERA (1 channel × 32×32 pixels per node):
        - Base: Equipment temperature (from thermal dynamics)
        - Hotspots: Random locations with +5 to +15°C
        - Precursor: Temperature rises 10 timesteps before failure
        
        Example:
        - Normal: 60-80°C, uniform
        - Stressed: 90-100°C, some hotspots
        - Failing: 110-120°C, multiple hotspots
        
        Parameters:
        -----------
        equipment_temps : np.ndarray, shape (num_nodes,)
            Current equipment temperature at each node (°C)
        failed_nodes : List[int]
            Nodes that have failed
        timestep : int
            Current timestep
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        
        Returns:
        --------
        thermal_data : np.ndarray, shape (num_nodes, 1, 32, 32)
            Thermal camera feed (temperature in °C, dtype=float16)
        """
        # Initialize with base equipment temperature
        thermal_data = equipment_temps.reshape(-1, 1, 1, 1) * np.ones(
            (self.num_nodes, 1, 32, 32), dtype=np.float16
        )
        
        # Pre-build pixel coordinate grids once (shared across all nodes/hotspots)
        xs, ys = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')  # (32,32)

        # Add hotspots — fully vectorised: no Python loops over pixels
        for node_idx in range(self.num_nodes):
            num_hotspots = np.random.randint(2, 5)
            for _ in range(num_hotspots):
                hx, hy = np.random.randint(4, 28, 2)
                hotspot_temp = equipment_temps[node_idx] + np.random.uniform(5, 15)
                dist = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)       # (32,32)
                thermal_data[node_idx, 0] += (hotspot_temp * np.exp(-dist / 3)).astype(np.float16)
        
        # Add measurement noise
        thermal_data += np.random.uniform(-2, 2, (self.num_nodes, 1, 32, 32)).astype(np.float16)
        
        # Add precursor signals before cascade
        if timestep >= cascade_start - 10 and cascade_start > 0:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
            precursor_strength = max(0, precursor_strength)
            
            for node in failed_nodes:
                # Temperature rises before failure
                thermal_data[node] += 15.0 * precursor_strength
        
        # Ensure float16 dtype
        return thermal_data.astype(np.float16)
    
    def generate_sensor_data(
        self,
        failed_nodes: List[int],
        timestep: int,
        cascade_start: int,
        precursor_duration: int = 15
    ) -> np.ndarray:
        """
        Generate synthetic multi-sensor array readings.
        
        SENSOR ARRAY (12 features per node):
        [0-2] Vibration (3-axis accelerometer, m/s²):
              - Base: 0.5 + equipment_age × 0.02
              - Increases before failure (mechanical stress)
        
        [3-4] Acoustic (2 microphones, dB):
              - Base: 0.3 (ambient noise)
              - Spikes before failure (corona discharge, arcing)
        
        [5-7] Magnetic field (3-axis magnetometer, Tesla):
              - Base: 1.0 (normal field)
              - Random noise (not strongly correlated)
        
        [8] Oil quality (transformers only, 0-1):
            - Decreases with equipment condition
            - 0.95 = new oil, 0.70 = degraded oil
        
        [9] Oil moisture content (ppm):
            - Increases with equipment condition
            - High moisture = insulation breakdown risk
        
        [10] Oil acidity (mg KOH/g):
             - Increases with equipment condition
             - High acidity = oil degradation
        
        [11] Partial discharge (pC):
             - Increases with equipment condition
             - Spikes before failure (insulation breakdown)
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed
        timestep : int
            Current timestep
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        
        Returns:
        --------
        sensor_data : np.ndarray, shape (num_nodes, 12)
            Multi-sensor array readings (dtype=float16)
        """
        sensor_data = np.zeros((self.num_nodes, 12), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Vibration: Increases with equipment age
            base_vibration = 0.5 + self.equipment_age[node_idx] * 0.02
            sensor_data[node_idx, 0:3] = base_vibration + np.random.randn(3) * 0.2
            
            # Acoustic: Ambient noise
            sensor_data[node_idx, 3:5] = 0.3 + np.random.randn(2) * 0.1
            
            # Magnetic field: Random noise
            sensor_data[node_idx, 5:8] = 1.0 + np.random.randn(3) * 0.3
            
            # Oil quality: Decreases with equipment condition
            sensor_data[node_idx, 8] = 0.95 - (1.0 - self.equipment_condition[node_idx]) * 0.2
            
            # Oil moisture: Increases with equipment condition
            sensor_data[node_idx, 9] = 0.02 + (1.0 - self.equipment_condition[node_idx]) * 0.05
            
            # Oil acidity: Increases with equipment condition
            sensor_data[node_idx, 10] = 0.01 + (1.0 - self.equipment_condition[node_idx]) * 0.08
            
            # Partial discharge: Increases with equipment condition
            sensor_data[node_idx, 11] = (
                (1.0 - self.equipment_condition[node_idx]) * 0.5 + np.random.randn() * 0.1
            )
        
        # Add precursor signals before cascade
        if timestep >= cascade_start - 10 and cascade_start > 0:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
            precursor_strength = max(0, precursor_strength)
            
            for node in failed_nodes:
                # Vibration increases (mechanical stress)
                sensor_data[node, 0:3] += 2.0 * precursor_strength ** 2
                
                # Acoustic spikes (corona discharge, arcing)
                sensor_data[node, 3:5] += 1.5 * precursor_strength ** 2
                
                # Partial discharge spikes (insulation breakdown)
                sensor_data[node, 11] += 3.0 * precursor_strength ** 2
                
                # Oil quality degrades
                sensor_data[node, 8] -= 0.1 * precursor_strength
                sensor_data[node, 9] += 0.05 * precursor_strength
                sensor_data[node, 10] += 0.08 * precursor_strength
        
        return sensor_data
    
    def generate_correlated_robotic_data(
        self,
        failed_nodes: List[int],
        failed_lines: List[int],
        timestep: int,
        cascade_start: int,
        equipment_temps: np.ndarray,
        precursor_duration: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate all robotic data in one call.

        This is a convenience method that calls all three generation methods
        and returns the complete robotic data package.

        IMPORTANT: ``precursor_duration`` must be drawn once per scenario (not
        per timestep) and passed in here so that the precursor signal window is
        the same across every timestep of the same scenario.  If the caller
        omits it a random value is drawn as a fallback, but this breaks temporal
        coherence and should only be done in isolation testing.

        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed
        failed_lines : List[int]
            Lines that have failed (not used currently)
        timestep : int
            Current timestep (0 to sequence_length-1)
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        equipment_temps : np.ndarray, shape (num_nodes,)
            Current equipment temperature at each node (°C)
        precursor_duration : int, optional
            Number of timesteps before cascade_start over which precursor
            signals ramp up.  Should be fixed for the whole scenario.
            Defaults to a random draw in [8, 20] if not provided.

        Returns:
        --------
        visual_data : np.ndarray, shape (num_nodes, 3, 32, 32)
            RGB visual feed from drone cameras
        thermal_data : np.ndarray, shape (num_nodes, 1, 32, 32)
            Thermal camera feed (temperature in °C)
        sensor_data : np.ndarray, shape (num_nodes, 12)
            Multi-sensor array readings
        """
        if precursor_duration is None:
            precursor_duration = int(np.random.randint(8, 20))

        visual_data = self.generate_visual_data(
            failed_nodes, timestep, cascade_start, precursor_duration
        )

        thermal_data = self.generate_thermal_data(
            equipment_temps, failed_nodes, timestep, cascade_start, precursor_duration
        )

        sensor_data = self.generate_sensor_data(
            failed_nodes, timestep, cascade_start, precursor_duration
        )

        return visual_data, thermal_data, sensor_data
