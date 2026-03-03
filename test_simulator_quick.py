"""Quick test of simulator - just verify it works"""
from cascade_prediction.data.generator import PhysicsBasedGridSimulator

print("Creating simulator...")
sim = PhysicsBasedGridSimulator(num_nodes=30, seed=42)
print(f"[OK] Created: {sim.num_nodes} nodes, {sim.num_edges} edges")

print("\nGenerating normal scenario...")
scenario = sim.generate_scenario(stress_level=0.5, sequence_length=5)

if scenario:
    print(f"[OK] Scenario generated!")
    print(f"  - Is cascade: {scenario['is_cascade']}")
    print(f"  - Failed nodes: {len(scenario['failed_nodes'])}")
    print(f"  - Temporal shape: {scenario['temporal_sequence'].shape}")
    print(f"  - Satellite shape: {scenario['satellite_data'].shape}")
    print("\n[OK] ALL TESTS PASSED!")
else:
    print("[ERROR] Scenario generation failed")
