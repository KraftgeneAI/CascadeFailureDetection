"""
Data Generation Wrapper Script
==============================
Simplified interface for generating cascade failure training data.

This script wraps the multimodal_data_generator.py with sensible defaults
and provides an easy-to-use interface for the modular system.

Usage:
------
# Quick start with defaults
python generate_data.py

# Custom configuration
python generate_data.py --cascade 1000 --normal 800 --output data/custom

# Large dataset
python generate_data.py --preset large

# Small dataset for testing
python generate_data.py --preset small
"""

import argparse
import subprocess
import sys
from pathlib import Path


PRESETS = {
    'small': {
        'normal': 100,
        'cascade': 80,
        'stressed': 20,
        'description': 'Small dataset for quick testing'
    },
    'medium': {
        'normal': 500,
        'cascade': 400,
        'stressed': 100,
        'description': 'Medium dataset for development'
    },
    'large': {
        'normal': 1000,
        'cascade': 800,
        'stressed': 200,
        'description': 'Large dataset for production training'
    },
    'xlarge': {
        'normal': 2000,
        'cascade': 1600,
        'stressed': 400,
        'description': 'Extra large dataset for best performance'
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate cascade failure training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  small   - 200 scenarios (100 normal, 80 cascade, 20 stressed)
  medium  - 1000 scenarios (500 normal, 400 cascade, 100 stressed)
  large   - 2000 scenarios (1000 normal, 800 cascade, 200 stressed)
  xlarge  - 4000 scenarios (2000 normal, 1600 cascade, 400 stressed)

Examples:
  # Use preset
  python generate_data.py --preset medium
  
  # Custom configuration
  python generate_data.py --cascade 500 --normal 400 --stressed 100
  
  # Custom output directory
  python generate_data.py --preset large --output data/production
        """
    )
    
    # Preset or custom
    parser.add_argument(
        '--preset',
        choices=['small', 'medium', 'large', 'xlarge'],
        help='Use a preset configuration'
    )
    
    # Custom scenario counts
    parser.add_argument(
        '--normal',
        type=int,
        help='Number of normal scenarios (overrides preset)'
    )
    parser.add_argument(
        '--cascade',
        type=int,
        help='Number of cascade scenarios (overrides preset)'
    )
    parser.add_argument(
        '--stressed',
        type=int,
        help='Number of stressed scenarios (overrides preset)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output',
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--topology',
        default='data/grid_topology.pkl',
        help='Path to grid topology file (default: data/grid_topology.pkl)'
    )
    
    # Split configuration
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training data fraction (default: 0.8)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation data fraction (default: 0.1)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Test data fraction (default: 0.1)'
    )
    
    # Advanced options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Scenarios per batch file (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine scenario counts
    if args.preset:
        preset = PRESETS[args.preset]
        normal = args.normal if args.normal is not None else preset['normal']
        cascade = args.cascade if args.cascade is not None else preset['cascade']
        stressed = args.stressed if args.stressed is not None else preset['stressed']
        print(f"Using preset: {args.preset}")
        print(f"  {preset['description']}")
    else:
        # Use custom or defaults
        normal = args.normal if args.normal is not None else 500
        cascade = args.cascade if args.cascade is not None else 400
        stressed = args.stressed if args.stressed is not None else 100
    
    # Validate splits
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 0.01:
        print("Error: train_split + val_split + test_split must equal 1.0")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("DATA GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Normal scenarios:    {normal}")
    print(f"Cascade scenarios:   {cascade}")
    print(f"Stressed scenarios:  {stressed}")
    print(f"Total scenarios:     {normal + cascade + stressed}")
    print(f"\nOutput directory:    {args.output}")
    print(f"Topology file:       {args.topology}")
    print(f"\nData splits:")
    print(f"  Training:   {args.train_split * 100:.1f}%")
    print(f"  Validation: {args.val_split * 100:.1f}%")
    print(f"  Test:       {args.test_split * 100:.1f}%")
    print("=" * 60 + "\n")
    
    # Check if topology file exists
    if not Path(args.topology).exists():
        print(f"Warning: Topology file not found: {args.topology}")
        print("The generator will create a default topology.")
        print()
    
    # Build command
    cmd = [
        sys.executable,
        'multimodal_data_generator.py',
        '--normal', str(normal),
        '--cascade', str(cascade),
        '--stressed', str(stressed),
        '--output-dir', args.output,
        '--topology-file', args.topology,
        '--train-split', str(args.train_split),
        '--val-split', str(args.val_split),
        '--test-split', str(args.test_split),
        '--batch-size', str(args.batch_size),
    ]
    
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    
    if args.verbose:
        cmd.append('--verbose')
    
    # Run generator
    print("Starting data generation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 60)
        print("DATA GENERATION COMPLETE")
        print("=" * 60)
        print(f"\nGenerated data saved to: {args.output}/")
        print(f"  - Training data:   {args.output}/train/")
        print(f"  - Validation data: {args.output}/val/")
        print(f"  - Test data:       {args.output}/test/")
        print("\nNext steps:")
        print("  1. Train a model:")
        print(f"     python train_model_modular.py --train_path {args.output}/train --val_path {args.output}/val")
        print("  2. Run inference:")
        print(f"     python inference_modular.py --data_path {args.output}/test --scenario_idx 0")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Data generation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nData generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
