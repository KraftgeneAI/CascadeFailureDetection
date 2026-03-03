"""
Modular Data Generation Script
==============================
Simplified data generation using modular components.

This script provides a cleaner interface to the data generator by using
modular components from cascade_prediction.data.generator package.

Usage:
------
# Quick start
python generate_data_modular.py --preset medium

# Custom configuration
python generate_data_modular.py \
    --cascade 1000 \
    --normal 800 \
    --stressed 200 \
    --output data/custom

Note: This is a wrapper around the full generator. For advanced features,
use multimodal_data_generator.py directly.
"""

import argparse
import sys
from pathlib import Path

# For now, this wraps the existing generator
# In the future, this will use fully modular components
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Generate cascade failure training data (modular version)'
    )
    
    # Presets
    parser.add_argument(
        '--preset',
        choices=['small', 'medium', 'large', 'xlarge'],
        help='Use a preset configuration'
    )
    
    # Scenario counts
    parser.add_argument('--normal', type=int, help='Number of normal scenarios')
    parser.add_argument('--cascade', type=int, help='Number of cascade scenarios')
    parser.add_argument('--stressed', type=int, help='Number of stressed scenarios')
    
    # Output
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--topology', default='data/grid_topology.pkl', help='Topology file')
    
    # Splits
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--test-split', type=float, default=0.1)
    
    # Advanced
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Map to presets
    presets = {
        'small': {'normal': 100, 'cascade': 80, 'stressed': 20},
        'medium': {'normal': 500, 'cascade': 400, 'stressed': 100},
        'large': {'normal': 1000, 'cascade': 800, 'stressed': 200},
        'xlarge': {'normal': 2000, 'cascade': 1600, 'stressed': 400}
    }
    
    if args.preset:
        preset = presets[args.preset]
        normal = args.normal if args.normal else preset['normal']
        cascade = args.cascade if args.cascade else preset['cascade']
        stressed = args.stressed if args.stressed else preset['stressed']
    else:
        normal = args.normal if args.normal else 500
        cascade = args.cascade if args.cascade else 400
        stressed = args.stressed if args.stressed else 100
    
    print("\n" + "=" * 70)
    print("MODULAR DATA GENERATION")
    print("=" * 70)
    print(f"Normal:    {normal}")
    print(f"Cascade:   {cascade}")
    print(f"Stressed:  {stressed}")
    print(f"Total:     {normal + cascade + stressed}")
    print(f"Output:    {args.output}")
    print("=" * 70 + "\n")
    
    # For now, delegate to existing generator
    # TODO: Replace with modular components
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
    
    if args.seed:
        cmd.extend(['--seed', str(args.seed)])
    if args.verbose:
        cmd.append('--verbose')
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Data generation complete")
        print(f"\nGenerated data in: {args.output}/")
        print("  - train/")
        print("  - val/")
        print("  - test/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
