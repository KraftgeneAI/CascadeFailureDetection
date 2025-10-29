"""
Script to read and inspect pickle files
Usage: python scripts/read_pickle.py <path_to_pickle_file>
"""

import pickle
import sys
import json
import numpy as np
from pathlib import Path


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to serializable format"""
    if isinstance(obj, np.ndarray):
        return {
            'type': 'numpy.ndarray',
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'data': obj.tolist() if obj.size < 1000 else f"<array too large: {obj.shape}>"
        }
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def print_structure(obj, indent=0, max_depth=3):
    """Print the structure of the object recursively"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}...")
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}Dictionary with {len(obj)} keys:")
        for key, value in list(obj.items())[:10]:  # Show first 10 keys
            print(f"{prefix}  '{key}':")
            print_structure(value, indent + 2, max_depth)
        if len(obj) > 10:
            print(f"{prefix}  ... and {len(obj) - 10} more keys")
    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} with {len(obj)} items")
        if len(obj) > 0:
            print(f"{prefix}  First item:")
            print_structure(obj[0], indent + 2, max_depth)
            if len(obj) > 1:
                print(f"{prefix}  ... and {len(obj) - 1} more items")
    
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}numpy.ndarray: shape={obj.shape}, dtype={obj.dtype}")
        if obj.size < 10:
            print(f"{prefix}  Values: {obj}")
    
    else:
        obj_str = str(obj)
        if len(obj_str) > 100:
            obj_str = obj_str[:100] + "..."
        print(f"{prefix}{type(obj).__name__}: {obj_str}")


def read_pickle_file(filepath, save_json=False):
    """Read and display contents of a pickle file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found!")
        return None
    
    print(f"Reading pickle file: {filepath}")
    print("=" * 80)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nSuccessfully loaded pickle file!")
        print(f"Root object type: {type(data).__name__}")
        print("\nStructure:")
        print("-" * 80)
        print_structure(data)
        
        # Save to JSON if requested
        if save_json:
            json_path = filepath.with_suffix('.json')
            print(f"\nConverting to JSON: {json_path}")
            try:
                serializable_data = convert_to_serializable(data)
                with open(json_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                print(f"Successfully saved to {json_path}")
            except Exception as e:
                print(f"Warning: Could not save to JSON: {e}")
        
        return data
    
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_pickle.py <path_to_pickle_file> [--json]")
        print("\nOptions:")
        print("  --json    Also save the contents to a JSON file")
        print("\nExample:")
        print("  python scripts/read_pickle.py data/train_batches/batch_0.pkl")
        print("  python scripts/read_pickle.py data/train_batches/batch_0.pkl --json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    save_json = '--json' in sys.argv
    
    data = read_pickle_file(filepath, save_json=save_json)
    
    if data is not None:
        print("\n" + "=" * 80)
        print("File loaded successfully!")
        print("\nTo access the data in Python:")
        print(f"  import pickle")
        print(f"  with open('{filepath}', 'rb') as f:")
        print(f"      data = pickle.load(f)")


if __name__ == "__main__":
    main()
