#################
import numpy as np

from multimodal_data_generator import generate_dataset_from_config
from video_processor import extract_threat_curve


### input video
video_path = "video/wildfire1.mp4"


### extract video stress signal
video_signal = extract_threat_curve(video_path)

if video_signal is None or len(video_signal) == 0:
    raise RuntimeError("Invalid video signal")

print("\n=== Video Signal ===")
print("Shape:", video_signal.shape)
print("Mean:", float(np.mean(video_signal)))


### run dataset generation
stats = generate_dataset_from_config(
    num_nodes=50,
    num_normal=0,
    num_cascade=1,
    num_stressed=0,
    sequence_length=len(video_signal),
    output_dir="debug_data",
    batch_size=1,
    seed=42,
    topology_file=None,
    start_batch=0
)


print("\n=== RESULT ===")
print(stats)
#%#