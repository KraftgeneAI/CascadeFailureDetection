"""
Multi-Modal Cascade Failure Dataset Generator (Refactored)
===========================================================

This is a refactored version of the original multimodal_data_generator.py that uses
the modular components from cascade_prediction.data.generator package.

The refactored version provides the same functionality but with cleaner code organization:
- Uses PhysicsBasedGridSimulator from the refactored modules
- Uses ScenarioOrchestrator for batch generation
- Maintains the same command-line interface
- Produces the same output format

Author: Kraftgene AI Inc. (R&D)
Date: March 2026
"""

import argparse
import pickle
import gc
from pathlib import Path

from cascade_prediction.data.generator import generate_dataset_from_config
from cascade_prediction.data.generator.config import Settings


def main():
    """
    Main entry point for dataset generation using refactored modules.
    
    WORKFLOW:
    ---------
    1. Parse command-line arguments
    2. Call generate_dataset_from_config() which handles:
       - Grid topology creation/loading
       - Scenario generation with retry logic
       - Train/val/test splitting
       - Batch saving
    
    EXAMPLE USAGE:
    --------------
    # Generate 200 scenarios (100 normal, 80 cascade, 20 stressed)
    python multimodal_data_generator_new.py \\
        --normal 100 \\
        --cascade 80 \\
        --stressed 20 \\
        --output-dir data \\
        --topology-file data/grid_topology.pkl
    
    # For production training (10,000 scenarios)
    python multimodal_data_generator_new.py \\
        --normal 5000 \\
        --cascade 4000 \\
        --stressed 1000 \\
        --output-dir data \\
        --topology-file data/grid_topology.pkl
    
    # Parallel generation on multiple machines
    # Machine 1:
    python multimodal_data_generator_new.py --normal 2500 --cascade 2000 \\
        --output-dir data_p1 --start-batch 0 --topology-file shared/grid_topology.pkl
    
    # Machine 2:
    python multimodal_data_generator_new.py --normal 2500 --cascade 2000 \\
        --output-dir data_p2 --start-batch 4500 --topology-file shared/grid_topology.pkl
    
    # Then merge data_p1 and data_p2 folders
    
    OUTPUT STRUCTURE:
    -----------------
    data/
    ├── grid_topology.pkl          # Grid structure (shared by all scenarios)
    ├── train/
    │   ├── scenarios_batch_0.pkl  # Scenarios 0-9 (if batch_size=10)
    │   ├── scenarios_batch_1.pkl  # Scenarios 10-19
    │   └── ...
    ├── val/
    │   ├── scenarios_batch_0.pkl
    │   └── ...
    └── test/
        ├── scenarios_batch_0.pkl
        └── ...
    """
    parser = argparse.ArgumentParser(
        description='Generate multi-modal cascade failure dataset using refactored modules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (200 scenarios)
  python multimodal_data_generator_new.py --normal 100 --cascade 80 --stressed 20
  
  # Production dataset (10,000 scenarios)
  python multimodal_data_generator_new.py --normal 5000 --cascade 4000 --stressed 1000
  
  # Use existing topology
  python multimodal_data_generator_new.py --normal 100 --cascade 80 \\
      --topology-file data/grid_topology.pkl
        """
    )
    
    # Scenario counts
    parser.add_argument(
        '--normal', type=int, default=50,
        help='Number of normal (low-stress) scenarios'
    )
    parser.add_argument(
        '--cascade', type=int, default=50,
        help='Number of cascade scenarios'
    )
    parser.add_argument(
        '--stressed', type=int, default=50,
        help='Number of stressed (high-stress, non-failing) scenarios'
    )
    
    # Grid configuration
    parser.add_argument(
        '--grid-size', type=int, default=118,
        help='Number of nodes in grid'
    )
    parser.add_argument(
        '--sequence-length', type=int, default=60,
        help='Sequence length (timesteps)'
    )
    
    # Output configuration
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Number of scenarios to save in each .pkl file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--topology-file', type=str,
        default=Settings.Dataset.DEFAULT_TOPOLOGY_FILE,
        help='Path to grid topology pickle file'
    )
    
    # Split ratios
    parser.add_argument(
        '--train-ratio', type=float, default=0.70,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.15,
        help='Test set ratio'
    )
    
    # Other options
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--start-batch', type=int, default=0,
        help='Starting batch number for output files (for parallel generation)'
    )

    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Optional path to wildfire video for external stress signal'
    )
    args = parser.parse_args()
    
    # Validate ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train/val/test ratios must sum to 1.0"
    
    # Print configuration
    print("\n" + "="*80)
    print("MULTI-MODAL CASCADE FAILURE DATASET GENERATOR (REFACTORED)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Grid size: {args.grid_size} nodes")
    print(f"  Sequence length: {args.sequence_length} timesteps")
    print(f"  Batch size: {args.batch_size} scenarios/file")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Starting batch: {args.start_batch}")
    
    if args.topology_file:
        print(f"  Topology file: {args.topology_file}")
    else:
        print(f"  Topology file: Will be created in {args.output_dir}/grid_topology.pkl")
    
    print(f"\nScenario counts:")
    print(f"  Normal: {args.normal}")
    print(f"  Stressed: {args.stressed}")
    print(f"  Cascade: {args.cascade}")
    print(f"  Total: {args.normal + args.stressed + args.cascade}")
    
    print(f"\nSplit ratios:")
    print(f"  Train: {args.train_ratio:.1%}")
    print(f"  Val: {args.val_ratio:.1%}")
    print(f"  Test: {args.test_ratio:.1%}")
    
    # Generate dataset using refactored modules
    print("\n" + "="*80)
    print("STARTING DATASET GENERATION")
    print("="*80)
    
    stats = generate_dataset_from_config(
        num_nodes=args.grid_size,
        num_normal=args.normal,
        num_cascade=args.cascade,
        num_stressed=args.stressed,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        topology_file=args.topology_file,
        start_batch=args.start_batch,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    
    if stats:
        total_generated = sum(s['generated'] for s in stats.values())
        total_failed = sum(s['failed'] for s in stats.values())
        
        print(f"\nGeneration Statistics:")
        print(f"  Train: {stats['train']['generated']} generated, {stats['train']['failed']} failed")
        print(f"  Val:   {stats['val']['generated']} generated, {stats['val']['failed']} failed")
        print(f"  Test:  {stats['test']['generated']} generated, {stats['test']['failed']} failed")
        print(f"\n  Total: {total_generated} generated, {total_failed} failed")
        
        if total_generated > 0:
            success_rate = total_generated / (total_generated + total_failed) * 100
            print(f"  Success rate: {success_rate:.1f}%")
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directories:")
    print(f"  Train: {args.output_dir}/train/")
    print(f"  Val:   {args.output_dir}/val/")
    print(f"  Test:  {args.output_dir}/test/")
    
    print(f"\nYou can now train the model using:")
    print(f"  python train_model.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

#%#