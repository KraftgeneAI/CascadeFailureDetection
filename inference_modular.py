"""
Cascade Failure Prediction Inference Script (Modular Version)
=============================================================
Uses modular components from cascade_prediction package.
"""

import torch
import json
import argparse
import time
import sys

# Import modular components
try:
    from cascade_prediction.inference import (
        CascadePredictor,
        print_prediction_report
    )
    from cascade_prediction.utils import NumpyEncoder
except ImportError as e:
    print(f"Error: Could not import cascade_prediction modules. {e}")
    print("Make sure the cascade_prediction package is properly installed.")
    sys.exit(1)


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Run cascade failure prediction inference"
    )
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", default="data/test", help="Path to test data directory")
    parser.add_argument("--scenario_idx", type=int, default=0, help="Index of scenario to predict")
    parser.add_argument("--topology_path", default="data/grid_topology.pkl", help="Path to grid topology file")
    parser.add_argument("--output", default="prediction.json", help="Output file for predictions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--window_size", type=int, default=30, help="Sliding window size")
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    
    # Setup device
    dev = torch.device(
        args.device if args.device 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {dev}")
    
    # Create predictor
    predictor = CascadePredictor(
        args.model_path,
        args.topology_path,
        device=dev,
        base_mva=100.0,
        base_freq=60.0
    )
    
    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - PHYSICS-INFORMED INFERENCE")
    print("=" * 80 + "\n")
    
    try:
        # Run prediction
        start_time = time.time()
        results = predictor.predict_scenario(
            args.data_path,
            args.scenario_idx,
            args.window_size,
            args.batch_size
        )
        results['inference_time'] = time.time() - start_time
        
        # Print report
        print_prediction_report(
            results,
            predictor.cascade_threshold,
            predictor.node_threshold
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Full prediction details saved to {args.output}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
