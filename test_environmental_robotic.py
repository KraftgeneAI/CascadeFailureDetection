"""
Test Environmental and Robotic Data Generators
==============================================

This script tests the newly modularized environmental and robotic data generators.

Usage:
    python test_environmental_robotic.py
"""

import numpy as np
import torch
from cascade_prediction.data.generator import (
    EnvironmentalDataGenerator,
    RoboticDataGenerator
)


def test_environmental_generator():
    """Test the EnvironmentalDataGenerator module."""
    print("\n" + "="*80)
    print("Testing EnvironmentalDataGenerator")
    print("="*80)
    
    # Setup test parameters
    num_nodes = 118
    positions = np.random.randn(num_nodes, 2) * 20
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Create generator
    env_gen = EnvironmentalDataGenerator(num_nodes, positions, edge_index.numpy())
    print(f"✓ Created EnvironmentalDataGenerator with {num_nodes} nodes")
    
    # Test satellite imagery generation
    print("\n1. Testing satellite imagery generation...")
    failed_nodes = [10, 25, 50]
    timestep = 20
    cascade_start = 15
    stress_level = 0.85
    
    satellite_data = env_gen.generate_satellite_imagery(
        failed_nodes, timestep, cascade_start, stress_level
    )
    print(f"   ✓ Generated satellite data: shape {satellite_data.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 12, 16, 16)")
    print(f"   ✓ Data type: {satellite_data.dtype}")
    print(f"   ✓ Value range: [{satellite_data.min():.3f}, {satellite_data.max():.3f}]")
    
    # Test weather sequence generation
    print("\n2. Testing weather sequence generation...")
    weather_sequence = env_gen.generate_weather_sequence(timestep, stress_level)
    print(f"   ✓ Generated weather data: shape {weather_sequence.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 10, 8)")
    print(f"   ✓ Data type: {weather_sequence.dtype}")
    print(f"   ✓ Temperature range: [{weather_sequence[:, :, 0].min():.1f}, {weather_sequence[:, :, 0].max():.1f}]°C")
    print(f"   ✓ Humidity range: [{weather_sequence[:, :, 1].min():.1f}, {weather_sequence[:, :, 1].max():.1f}]%")
    print(f"   ✓ Wind speed range: [{weather_sequence[:, :, 2].min():.1f}, {weather_sequence[:, :, 2].max():.1f}] m/s")
    
    # Test threat indicators generation
    print("\n3. Testing threat indicators generation...")
    failed_lines = [5, 15, 30]
    threat_indicators = env_gen.generate_threat_indicators(
        failed_nodes, failed_lines, timestep, cascade_start, stress_level
    )
    print(f"   ✓ Generated threat indicators: shape {threat_indicators.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 6)")
    print(f"   ✓ Data type: {threat_indicators.dtype}")
    print(f"   ✓ Value range: [{threat_indicators.min():.3f}, {threat_indicators.max():.3f}]")
    print(f"   ✓ Fire threat (failed nodes): {threat_indicators[failed_nodes, 0]}")
    
    # Test combined generation
    print("\n4. Testing combined environmental data generation...")
    sat_data, weather_data, threat_data = env_gen.generate_correlated_environmental_data(
        failed_nodes, failed_lines, timestep, cascade_start, stress_level
    )
    print(f"   ✓ Satellite data: {sat_data.shape}")
    print(f"   ✓ Weather data: {weather_data.shape}")
    print(f"   ✓ Threat data: {threat_data.shape}")
    
    print("\n✓ All environmental generator tests passed!")
    return True


def test_robotic_generator():
    """Test the RoboticDataGenerator module."""
    print("\n" + "="*80)
    print("Testing RoboticDataGenerator")
    print("="*80)
    
    # Setup test parameters
    num_nodes = 118
    equipment_age = np.random.uniform(0.2, 0.8, num_nodes)
    equipment_condition = np.random.uniform(0.6, 1.0, num_nodes)
    
    # Create generator
    robot_gen = RoboticDataGenerator(num_nodes, equipment_age, equipment_condition)
    print(f"✓ Created RoboticDataGenerator with {num_nodes} nodes")
    
    # Test visual data generation
    print("\n1. Testing visual data generation...")
    failed_nodes = [10, 25, 50]
    timestep = 20
    cascade_start = 15
    
    visual_data = robot_gen.generate_visual_data(failed_nodes, timestep, cascade_start)
    print(f"   ✓ Generated visual data: shape {visual_data.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 3, 32, 32)")
    print(f"   ✓ Data type: {visual_data.dtype}")
    print(f"   ✓ Value range: [{visual_data.min():.3f}, {visual_data.max():.3f}]")
    
    # Test thermal data generation
    print("\n2. Testing thermal data generation...")
    equipment_temps = np.random.uniform(60, 100, num_nodes)
    thermal_data = robot_gen.generate_thermal_data(
        equipment_temps, failed_nodes, timestep, cascade_start
    )
    print(f"   ✓ Generated thermal data: shape {thermal_data.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 1, 32, 32)")
    print(f"   ✓ Data type: {thermal_data.dtype}")
    print(f"   ✓ Temperature range: [{thermal_data.min():.1f}, {thermal_data.max():.1f}]°C")
    
    # Test sensor data generation
    print("\n3. Testing sensor data generation...")
    sensor_data = robot_gen.generate_sensor_data(failed_nodes, timestep, cascade_start)
    print(f"   ✓ Generated sensor data: shape {sensor_data.shape}")
    print(f"   ✓ Expected: ({num_nodes}, 12)")
    print(f"   ✓ Data type: {sensor_data.dtype}")
    print(f"   ✓ Vibration (axis 0): [{sensor_data[:, 0].min():.3f}, {sensor_data[:, 0].max():.3f}]")
    print(f"   ✓ Acoustic (mic 0): [{sensor_data[:, 3].min():.3f}, {sensor_data[:, 3].max():.3f}]")
    print(f"   ✓ Oil quality: [{sensor_data[:, 8].min():.3f}, {sensor_data[:, 8].max():.3f}]")
    print(f"   ✓ Partial discharge: [{sensor_data[:, 11].min():.3f}, {sensor_data[:, 11].max():.3f}]")
    
    # Test combined generation
    print("\n4. Testing combined robotic data generation...")
    failed_lines = [5, 15, 30]
    vis_data, therm_data, sens_data = robot_gen.generate_correlated_robotic_data(
        failed_nodes, failed_lines, timestep, cascade_start, equipment_temps
    )
    print(f"   ✓ Visual data: {vis_data.shape}")
    print(f"   ✓ Thermal data: {therm_data.shape}")
    print(f"   ✓ Sensor data: {sens_data.shape}")
    
    print("\n✓ All robotic generator tests passed!")
    return True


def test_integration():
    """Test integration of environmental and robotic generators."""
    print("\n" + "="*80)
    print("Testing Integration")
    print("="*80)
    
    # Setup shared parameters
    num_nodes = 118
    positions = np.random.randn(num_nodes, 2) * 20
    edge_index = torch.randint(0, num_nodes, (2, 200))
    equipment_age = np.random.uniform(0.2, 0.8, num_nodes)
    equipment_condition = np.random.uniform(0.6, 1.0, num_nodes)
    
    # Create both generators
    env_gen = EnvironmentalDataGenerator(num_nodes, positions, edge_index.numpy())
    robot_gen = RoboticDataGenerator(num_nodes, equipment_age, equipment_condition)
    
    # Simulate a cascade scenario
    print("\n1. Simulating cascade scenario...")
    failed_nodes = [10, 25, 50, 75]
    failed_lines = [5, 15, 30, 45]
    timestep = 25
    cascade_start = 15
    stress_level = 0.92
    equipment_temps = np.random.uniform(80, 110, num_nodes)
    
    # Generate all data
    print("\n2. Generating all multi-modal data...")
    sat_data, weather_data, threat_data = env_gen.generate_correlated_environmental_data(
        failed_nodes, failed_lines, timestep, cascade_start, stress_level
    )
    vis_data, therm_data, sens_data = robot_gen.generate_correlated_robotic_data(
        failed_nodes, failed_lines, timestep, cascade_start, equipment_temps
    )
    
    print(f"   ✓ Environmental data generated:")
    print(f"     - Satellite: {sat_data.shape}")
    print(f"     - Weather: {weather_data.shape}")
    print(f"     - Threats: {threat_data.shape}")
    print(f"   ✓ Robotic data generated:")
    print(f"     - Visual: {vis_data.shape}")
    print(f"     - Thermal: {therm_data.shape}")
    print(f"     - Sensors: {sens_data.shape}")
    
    # Check memory usage
    print("\n3. Checking memory usage...")
    total_size_mb = (
        sat_data.nbytes + weather_data.nbytes + threat_data.nbytes +
        vis_data.nbytes + therm_data.nbytes + sens_data.nbytes
    ) / (1024 * 1024)
    print(f"   ✓ Total data size: {total_size_mb:.2f} MB")
    print(f"   ✓ Per-node average: {total_size_mb / num_nodes:.3f} MB")
    
    print("\n✓ Integration test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENVIRONMENTAL AND ROBOTIC DATA GENERATOR TESTS")
    print("="*80)
    
    try:
        # Run individual tests
        env_success = test_environmental_generator()
        robot_success = test_robotic_generator()
        integration_success = test_integration()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Environmental Generator: {'✓ PASSED' if env_success else '✗ FAILED'}")
        print(f"Robotic Generator: {'✓ PASSED' if robot_success else '✗ FAILED'}")
        print(f"Integration: {'✓ PASSED' if integration_success else '✗ FAILED'}")
        
        if env_success and robot_success and integration_success:
            print("\n✓ ALL TESTS PASSED!")
            print("\nNext steps:")
            print("1. Extract Simulator module from multimodal_data_generator.py")
            print("2. Extract Scenario orchestrator module")
            print("3. Update main generator script to use modular components")
            return 0
        else:
            print("\n✗ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
