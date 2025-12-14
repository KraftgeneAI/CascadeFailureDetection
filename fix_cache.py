import glob
import pickle
import json
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configure your paths
DATA_DIRS = ["data/train", "data/val", "data/test"]

def process_single_file(path):
    """Worker function to process one file."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Logic from your dataset class
        if isinstance(data, list):
            if len(data) == 0: return False
            scenario = data[0]
        else:
            scenario = data
            
        if not isinstance(scenario, dict): return False

        if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
            return scenario['metadata']['is_cascade']
        elif 'sequence' in scenario and len(scenario['sequence']) > 0:
            last_step = scenario['sequence'][-1]
            import numpy as np # Local import for thread safety
            return bool(np.max(last_step.get('node_labels', [0])) > 0.5)
        
        return False
        
    except Exception:
        return False

def create_cache_for_dir_fast(dir_path):
    path = Path(dir_path)
    if not path.exists(): return

    print(f"Scanning {dir_path}...")
    scenario_files = sorted(glob.glob(str(path / "scenario_*.pkl")))
    if not scenario_files:
        scenario_files = sorted(glob.glob(str(path / "scenarios_batch_*.pkl")))
    
    if not scenario_files:
        print(f"  No files found.")
        return

    print(f"  Found {len(scenario_files)} files. Launching 64 threads...")
    
    labels = []
    # Use ThreadPoolExecutor for massive I/O parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        # Submit all tasks
        results = list(tqdm(executor.map(process_single_file, scenario_files), total=len(scenario_files)))
        labels = results

    # Save the cache
    cache_file = path / "metadata_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(labels, f)
    
    print(f"  ✅ Saved cache to {cache_file} ({len(labels)} items)")

if __name__ == "__main__":
    for d in DATA_DIRS:
        create_cache_for_dir_fast(d)