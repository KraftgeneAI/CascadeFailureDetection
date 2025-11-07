import pickle
from pathlib import Path
import glob
import os
import gc
from tqdm import tqdm

def rebatch_data(original_dir, output_dir):
    """
    Loads large batch files one-by-one and saves each scenario
    as its own individual pickle file.
    
    *** MODIFIED: Deletes the original large file after processing
    to save disk space. ***
    """
    original_path = Path(original_dir)
    output_path = Path(output_dir)
    
    if not original_path.exists():
        print(f"Error: Original data directory not found: {original_path}")
        return
        
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Finding batch files in: {original_path}")
    batch_files = sorted(glob.glob(str(original_path / "scenarios_batch_*.pkl")))
    
    if not batch_files:
        print(f"No 'scenarios_batch_*.pkl' files found in {original_path}")
        return

    print(f"Found {len(batch_files)} large batch files. Processing...")
    
    global_scenario_index = 0
    
    for batch_file_path in tqdm(batch_files, desc=f"Processing {original_path.name}"):
        try:
            # 1. This is the step that uses a lot of RAM, but only temporarily
            with open(batch_file_path, 'rb') as f:
                batch_data = pickle.load(f)
            
            # 2. Now, save each scenario individually
            for scenario in batch_data:
                # Use a zero-padded filename for correct sorting
                new_filename = output_path / f"scenario_{global_scenario_index:07d}.pkl"
                
                with open(new_filename, 'wb') as out_f:
                    pickle.dump(scenario, out_f)
                    
                global_scenario_index += 1
            
            # --- START OF MODIFICATION ---
            # 3. Delete the original large batch file *after* successfully saving
            tqdm.write(f"\n   Successfully processed {len(batch_data)} scenarios from {batch_file_path}.")
            tqdm.write(f"   Deleting original file to save disk space...")
            os.remove(batch_file_path)
            tqdm.write(f"   ✓ Deleted {batch_file_path}.")
            # --- END OF MODIFICATION ---

            # 4. Free memory before loading the next giant file
            del batch_data
            gc.collect()
            
        except Exception as e:
            print(f"\nError processing file {batch_file_path}: {e}")
            print("This may be a corrupted file or you are out of memory.")
            print("If it's a memory error, even one file is too big. No fix is possible.")
            return

    print(f"\nSuccessfully re-batched {global_scenario_index} scenarios into {output_path}")

if __name__ == "__main__":
    
    # --- Configuration ---
    ORIGINAL_DATA_ROOT = "data"
    NEW_DATA_ROOT = "data_rebached"
    # ---------------------
    
    print(f"Starting re-batching process...")
    print(f"Original data: {ORIGINAL_DATA_ROOT}")
    print(f"Output data:   {NEW_DATA_ROOT}")
    print("WARNING: This will DELETE the original files in 'data/' after processing.")
    print("Press Enter to continue, or Ctrl+C to cancel.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        exit()

    
    rebatch_data(f"{ORIGINAL_DATA_ROOT}/train", f"{NEW_DATA_ROOT}/train")
    rebatch_data(f"{ORIGINAL_DATA_ROOT}/val", f"{NEW_DATA_ROOT}/val")
    rebatch_data(f"{ORIGINAL_DATA_ROOT}/test", f"{NEW_DATA_ROOT}/test")
    
    print("\n" + "="*80)
    print("RE-BATCHING COMPLETE.")
    print(f"Your data is now in '{NEW_DATA_ROOT}'.")
    print(f"Original files in '{ORIGINAL_DATA_ROOT}' have been deleted.")
    print("Next, update 'cascade_dataset.py' and then run training")
    print(f"using:  python train_model.py --data_dir {NEW_DATA_ROOT}")
    print("="*80)