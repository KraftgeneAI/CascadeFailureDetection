"""
Cascade Failure Prediction — CLI Entry Point
============================================
Run inference on a single scenario and print a human-readable report.

Usage:
    python inference.py --model_path checkpoints/best_f1_model.pth \
                        --data_path data/test \
                        --scenario_idx 0
"""

import argparse
import json
import sys
import time

import torch

try:
    from cascade_prediction.inference import CascadePredictor, print_report
    from cascade_prediction.data.generator.config import Settings
except ImportError as e:
    print(f"Error: Could not import from cascade_prediction module: {e}")
    print("Make sure the cascade_prediction package is in your Python path.")
    sys.exit(1)


class NumpyEncoder(json.JSONEncoder):
    import numpy as np
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Cascade failure prediction inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="data/test")
    parser.add_argument("--scenario_idx", type=int, default=0)
    parser.add_argument("--topology_path", default="data/grid_topology.pkl")
    parser.add_argument("--output", default="prediction.json")
    parser.add_argument("--batch_size", type=int, default=Settings.Training.BATCH_SIZE)
    parser.add_argument("--window_size", type=int, default=Settings.Simulation.DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {dev}")

    predictor = CascadePredictor(
        args.model_path,
        args.topology_path,
        device=dev,
        base_mva=Settings.Dataset.BASE_MVA,
        base_freq=Settings.Dataset.BASE_FREQUENCY,
    )

    print("\n" + "=" * 80)
    print("CASCADE FAILURE PREDICTION - PHYSICS-INFORMED INFERENCE")
    print("=" * 80 + "\n")

    try:
        start_time = time.time()
        res = predictor.predict_scenario(args.data_path, args.scenario_idx, args.window_size, args.batch_size)
        res["inference_time"] = time.time() - start_time

        print_report(res, predictor.cascade_threshold, predictor.node_threshold)

        with open(args.output, "w") as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f"Full prediction details saved to {args.output}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
