"""
Test Scenario Orchestrator
===========================

Quick test to verify the scenario orchestrator works correctly.

Usage:
    python test_scenario_orchestrator.py
"""

from cascade_prediction.data.generator import generate_dataset_from_config

print("="*80)
print("TESTING SCENARIO ORCHESTRATOR")
print("="*80)

print("\nGenerating small test dataset...")
print("  - 3 normal scenarios")
print("  - 2 cascade scenarios")
print("  - 1 stressed scenario")
print("  - 5 timesteps per scenario")
print("  - 30 nodes")

stats = generate_dataset_from_config(
    num_nodes=30,
    num_normal=3,
    num_cascade=2,
    num_stressed=1,
    sequence_length=5,
    output_dir='data_test',
    batch_size=2,
    seed=42
)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print(f"\nGeneration Statistics:")
print(f"  Train: {stats['train']['generated']} generated, {stats['train']['failed']} failed")
print(f"  Val:   {stats['val']['generated']} generated, {stats['val']['failed']} failed")
print(f"  Test:  {stats['test']['generated']} generated, {stats['test']['failed']} failed")

total_generated = sum(s['generated'] for s in stats.values())
total_failed = sum(s['failed'] for s in stats.values())

print(f"\n  Total: {total_generated} generated, {total_failed} failed")

if total_generated > 0:
    print("\n[OK] Scenario orchestrator is working!")
    print("\nYou can now use it to generate larger datasets:")
    print("  python -c \"from cascade_prediction.data.generator import generate_dataset_from_config; \\")
    print("             generate_dataset_from_config(num_normal=100, num_cascade=80, num_stressed=20)\"")
else:
    print("\n[ERROR] No scenarios were generated")
