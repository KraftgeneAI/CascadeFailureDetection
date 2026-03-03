"""
Environmental Data Generator Module
====================================

PURPOSE:
--------
Generates synthetic environmental data (satellite imagery, weather sequences, and
threat indicators) that are correlated with power grid state and failures.

This module creates realistic-looking environmental data that teaches the model to
fuse multi-modal information for cascade prediction.

WHY SYNTHETIC?
--------------
Real satellite/weather data for power grids is:
- Expensive to obtain
- Rarely available during actual cascades
- Hard to label (which weather caused which failure?)

Synthetic data lets us create perfect correlations for training.

CORRELATION STRATEGY:
---------------------
1. Base patterns: Realistic but random (weather cycles, terrain)
2. Stress indicators: Increase with grid stress level
3. Precursor signals: Appear 10-15 timesteps before failures
4. Failure signatures: Heat/smoke appear when equipment fails
5. Spatial propagation: Threats spread to nearby nodes

Author: Kraftgene AI Inc. (R&D)
Date: October 2025
"""

import numpy as np
from typing import List, Tuple
from scipy.ndimage import gaussian_filter


class EnvironmentalDataGenerator:
    """
    Generates synthetic environmental data correlated with grid failures.
    
    This class creates three types of environmental data:
    1. Satellite imagery (12 channels, 16x16 pixels per node)
    2. Weather sequences (10 timesteps, 8 features per node)
    3. Threat indicators (6 hazard types per node)
    
    All data is correlated with grid stress level and failure events.
    """
    
    def __init__(self, num_nodes: int, positions: np.ndarray, edge_index: np.ndarray):
        """
        Initialize the environmental data generator.
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the power grid
        positions : np.ndarray, shape (num_nodes, 2)
            Geographic positions of nodes (for spatial correlation)
        edge_index : np.ndarray, shape (2, num_edges)
            Edge connectivity (source, destination) pairs
        """
        self.num_nodes = num_nodes
        self.positions = positions
        self.edge_index = edge_index
    
    def generate_satellite_imagery(
        self,
        failed_nodes: List[int],
        timestep: int,
        cascade_start: int,
        stress_level: float,
        precursor_duration: int = 15
    ) -> np.ndarray:
        """
        Generate synthetic satellite imagery correlated with grid state.
        
        SATELLITE IMAGERY (12 channels × 16×16 pixels per node):
        - Channels 0-3: Visible spectrum (RGB + Near-Infrared)
        - Channels 4-7: Vegetation indices (NDVI, etc.)
        - Channels 8-9: Water/moisture content
        - Channels 10-11: Thermal infrared (heat signatures)
        
        Correlation examples:
        - Failed nodes show heat signatures in thermal bands
        - Smoke reduces visibility in RGB bands
        - Storm patterns correlate with high stress
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed (used to add heat signatures)
        timestep : int
            Current timestep (0 to sequence_length-1)
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        stress_level : float
            Overall grid stress (0.0 to 1.0)
        
        Returns:
        --------
        satellite_data : np.ndarray, shape (num_nodes, 12, 16, 16)
            Synthetic satellite imagery for each node (dtype=float16)
        """
        satellite_data = np.zeros((self.num_nodes, 12, 16, 16), dtype=np.float16)
        
        # Generate base patterns for each band
        for node_idx in range(self.num_nodes):
            for band in range(12):
                base_pattern = np.random.randn(16, 16)
                smooth_pattern = gaussian_filter(base_pattern, sigma=2.0)
                satellite_data[node_idx, band] = (
                    (smooth_pattern - smooth_pattern.min()) / 
                    (smooth_pattern.max() - smooth_pattern.min() + 1e-6)
                )
            
            # Scale bands to realistic ranges
            satellite_data[node_idx, 0:4] = 0.3 + 0.3 * satellite_data[node_idx, 0:4]  # Visible
            satellite_data[node_idx, 4:8] = 0.2 + 0.2 * satellite_data[node_idx, 4:8]  # Vegetation
            satellite_data[node_idx, 8:10] = 0.4 + 0.2 * satellite_data[node_idx, 8:10]  # Water
            satellite_data[node_idx, 10:12] = 0.5 + 0.1 * satellite_data[node_idx, 10:12]  # Thermal
        
        # Add precursor signals before cascade
        if timestep >= cascade_start - 15 and cascade_start > 0:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
            precursor_strength = max(0, precursor_strength)
            
            if failed_nodes:
                fire_center = self.positions[failed_nodes[0]]
                
                for node_idx in range(self.num_nodes):
                    distance = np.linalg.norm(self.positions[node_idx] - fire_center)
                    fire_threat = precursor_strength * 0.8 * np.exp(-distance / 25)
                    
                    # Add heat signatures to thermal bands
                    if fire_threat > 0.3:
                        center_x, center_y = 8, 8
                        for x in range(16):
                            for y in range(16):
                                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                heat_signature = fire_threat * np.exp(-dist_from_center / 4)
                                satellite_data[node_idx, 10:12, x, y] += heat_signature
                    
                    # Reduce visibility in RGB bands (smoke)
                    if fire_threat > 0.2:
                        satellite_data[node_idx, 0:4, :, :] *= (1 - fire_threat * 0.3)
        
        # Add failure signatures after cascade starts
        if timestep >= cascade_start and cascade_start > 0 and failed_nodes:
            for node in failed_nodes:
                # Increase thermal signature at failed nodes
                distances = np.linalg.norm(self.positions - self.positions[node], axis=1)
                nearby = np.where(distances < 30)[0]
                
                for nearby_node in nearby:
                    if timestep >= cascade_start - 5:
                        satellite_data[nearby_node, 10:12, :, :] += 0.3
        
        return satellite_data
    
    def generate_weather_sequence(
        self,
        timestep: int,
        stress_level: float
    ) -> np.ndarray:
        """
        Generate synthetic weather time series correlated with grid stress.
        
        WEATHER SEQUENCE (10 timesteps × 8 features per node):
        - Temperature (°C): Diurnal cycle + stress correlation
        - Humidity (%): Inverse correlation with temperature
        - Wind speed (m/s): Higher during stressed scenarios
        - Precipitation (mm/h): Exponential distribution
        - Pressure (hPa): Random walk around 1000 hPa
        - Solar radiation (W/m²): Follows sun angle
        - Cloud cover (%): Inverse of solar radiation
        - Visibility (km): Reduced by precipitation
        
        Parameters:
        -----------
        timestep : int
            Current timestep (used for time-of-day effects)
        stress_level : float
            Overall grid stress (0.0 to 1.0)
        
        Returns:
        --------
        weather_sequence : np.ndarray, shape (num_nodes, 10, 8)
            Weather time series (last 10 timesteps, 8 features, dtype=float16)
        """
        weather_sequence = np.zeros((self.num_nodes, 10, 8), dtype=np.float16)
        
        for node_idx in range(self.num_nodes):
            # Temperature: Diurnal cycle
            hour_of_day = (timestep / 60) * 24
            temp_base = 25 + 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            weather_sequence[node_idx, :, 0] = temp_base + np.random.randn(10) * 2
            
            # Humidity: Inverse correlation with temperature
            weather_sequence[node_idx, :, 1] = (
                70 - (weather_sequence[node_idx, :, 0] - 25) * 1.5 + np.random.randn(10) * 5
            )
            weather_sequence[node_idx, :, 1] = np.clip(weather_sequence[node_idx, :, 1], 20, 95)
            
            # Wind speed: Correlated with stress level
            wind_base = 5 + stress_level * 10
            weather_sequence[node_idx, :, 2] = wind_base + np.random.randn(10) * 2
            weather_sequence[node_idx, :, 2] = np.clip(weather_sequence[node_idx, :, 2], 0, 25)
            
            # Precipitation: Exponential distribution
            precip_prob = (weather_sequence[node_idx, :, 1] - 60) / 40
            weather_sequence[node_idx, :, 3] = np.where(
                np.random.rand(10) < np.clip(precip_prob, 0, 0.3),
                np.random.exponential(5, 10),
                0
            )
            
            # Pressure: Random walk
            weather_sequence[node_idx, :, 4] = 1000 + np.random.randn(10) * 10
            
            # Solar radiation: Follows sun angle
            solar_factor = max(0, np.sin(2 * np.pi * (hour_of_day - 6) / 24))
            weather_sequence[node_idx, :, 5] = 800 * solar_factor + np.random.randn(10) * 50
            weather_sequence[node_idx, :, 5] = np.clip(weather_sequence[node_idx, :, 5], 0, 1000)
            
            # Cloud cover: Inverse of solar radiation
            weather_sequence[node_idx, :, 6] = (
                100 - weather_sequence[node_idx, :, 5] / 10 + np.random.randn(10) * 15
            )
            weather_sequence[node_idx, :, 6] = np.clip(weather_sequence[node_idx, :, 6], 0, 100)
            
            # Visibility: Reduced by precipitation
            weather_sequence[node_idx, :, 7] = (
                20 - weather_sequence[node_idx, :, 3] * 2 - 
                (weather_sequence[node_idx, :, 1] - 50) / 10
            )
            weather_sequence[node_idx, :, 7] = np.clip(weather_sequence[node_idx, :, 7], 0.5, 20)
        
        return weather_sequence
    
    def generate_threat_indicators(
        self,
        failed_nodes: List[int],
        failed_lines: List[int],
        timestep: int,
        cascade_start: int,
        stress_level: float,
        precursor_duration: int = 15
    ) -> np.ndarray:
        """
        Generate threat indicators for various hazard types.
        
        THREAT INDICATORS (6 types per node):
        - [0] Fire/heat threat: High near failed nodes
        - [1] Storm severity: Correlated with stress level
        - [2] Geohazard (landslide/earthquake): Random baseline
        - [3] Flood risk: Correlated with precipitation
        - [4] Ice/snow loading: Seasonal (not implemented)
        - [5] Equipment damage: High on failed lines
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed
        failed_lines : List[int]
            Lines that have failed
        timestep : int
            Current timestep
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        stress_level : float
            Overall grid stress (0.0 to 1.0)
        
        Returns:
        --------
        threat_indicators : np.ndarray, shape (num_nodes, 6)
            Threat levels for 6 hazard types (0.0 to 1.0, dtype=float16)
        """
        threat_indicators = np.zeros((self.num_nodes, 6), dtype=np.float16)
        
        # Base threat level correlated with stress
        base_threat = stress_level * 0.2
        threat_indicators += base_threat
        
        # Add precursor signals before cascade
        if timestep >= cascade_start - 15 and cascade_start > 0:
            precursor_strength = 1.0 - (cascade_start - timestep) / precursor_duration
            precursor_strength = max(0, precursor_strength)
            
            if failed_nodes:
                fire_center = self.positions[failed_nodes[0]]
                
                for node_idx in range(self.num_nodes):
                    distance = np.linalg.norm(self.positions[node_idx] - fire_center)
                    fire_threat = precursor_strength * 0.8 * np.exp(-distance / 25)
                    threat_indicators[node_idx, 0] += fire_threat
        
        # Add failure signatures after cascade starts
        if timestep >= cascade_start and cascade_start > 0 and (failed_nodes or failed_lines):
            # Fire/heat threat at failed nodes
            for node in failed_nodes:
                threat_indicators[node, 0] += 0.6
                
                # Spread to nearby nodes
                distances = np.linalg.norm(self.positions - self.positions[node], axis=1)
                nearby = np.where(distances < 30)[0]
                for nearby_node in nearby:
                    threat_indicators[nearby_node, 0] += 0.3 * np.exp(-distances[nearby_node] / 20)
            
            # Equipment damage on failed lines
            src, dst = self.edge_index
            for line in failed_lines:
                s, d = src[line].item(), dst[line].item()
                threat_indicators[s, 5] += 0.5
                threat_indicators[d, 5] += 0.5
        
        # Clip to valid range
        threat_indicators = np.clip(threat_indicators, 0, 1)
        
        return threat_indicators
    
    def generate_correlated_environmental_data(
        self,
        failed_nodes: List[int],
        failed_lines: List[int],
        timestep: int,
        cascade_start: int,
        stress_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate all environmental data in one call.
        
        This is a convenience method that calls all three generation methods
        and returns the complete environmental data package.
        
        IMPORTANT: Uses a single precursor_duration value for all three data types
        to ensure consistent precursor signal timing across satellite, weather, and
        threat data.
        
        Parameters:
        -----------
        failed_nodes : List[int]
            Nodes that have failed
        failed_lines : List[int]
            Lines that have failed
        timestep : int
            Current timestep (0 to sequence_length-1)
        cascade_start : int
            Timestep when cascade begins (-1 if no cascade)
        stress_level : float
            Overall grid stress (0.0 to 1.0)
        
        Returns:
        --------
        satellite_data : np.ndarray, shape (num_nodes, 12, 16, 16)
            Synthetic satellite imagery
        weather_sequence : np.ndarray, shape (num_nodes, 10, 8)
            Weather time series
        threat_indicators : np.ndarray, shape (num_nodes, 6)
            Threat levels for 6 hazard types
        """
        # Calculate precursor_duration once for consistent timing across all data types
        precursor_duration = np.random.randint(8, 20)
        
        satellite_data = self.generate_satellite_imagery(
            failed_nodes, timestep, cascade_start, stress_level, precursor_duration
        )
        
        weather_sequence = self.generate_weather_sequence(
            timestep, stress_level
        )
        
        threat_indicators = self.generate_threat_indicators(
            failed_nodes, failed_lines, timestep, cascade_start, stress_level, precursor_duration
        )
        
        return satellite_data, weather_sequence, threat_indicators
