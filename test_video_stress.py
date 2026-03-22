#################
import numpy as np

from video_processor import extract_threat_curve
from multimodal_data_generator import PhysicsBasedGridSimulator


### define input video path
video_path = "videos/wildfire1.mp4"


### extract stress signal
stress_signal = extract_threat_curve(
    video_path,
    frame_skip=20
)


### print signal summary
print("=== Video Stress Signal ===")
print("Length:", len(stress_signal))
print("Min:", float(np.min(stress_signal)))
print("Max:", float(np.max(stress_signal)))
print("First values:", stress_signal[:10])


### derive stable stress level (IMPORTANT: keep small impact)
video_mean_stress = float(np.mean(stress_signal))

### SAFE mapping (this is the fix)
adjusted_stress_level = 0.5 + 0.2 * video_mean_stress

### clamp to safe range
adjusted_stress_level = min(max(adjusted_stress_level, 0.4), 0.7)

print("\n=== Derived Stress Level ===")
print("Base stress level: 0.5")
print("Adjusted stress level:", adjusted_stress_level)


### initialize simulator
simulator = PhysicsBasedGridSimulator()


### disable dataset mode
if hasattr(simulator, "dataset_mode"):
    simulator.dataset_mode = False

if hasattr(simulator, "num_samples"):
    simulator.num_samples = 1

if hasattr(simulator, "generate_dataset"):
    simulator.generate_dataset = False


### run scenario
print("\n=== Running SINGLE scenario (FAST MODE) ===")

result_no_video = simulator._generate_scenario_data(
    stress_level=0.5,
    sequence_length=10,
    external_stress_signal=None
)

result_with_video = simulator._generate_scenario_data(
    stress_level=adjusted_stress_level,
    sequence_length=10,
    external_stress_signal=None
)


### validate outputs
print("\n=== Simulation Results ===")
print("Without video:", result_no_video is not None)
print("With video:", result_with_video is not None)

if result_no_video is None or result_with_video is None:
    print("\n[ERROR] Simulation failed")
    exit()


### extract failure sets
failed_no_video = set(result_no_video["metadata"]["failed_nodes"])
failed_with_video = set(result_with_video["metadata"]["failed_nodes"])


### print failures
print("\n=== Failed Nodes ===")
print("WITHOUT video:", sorted(list(failed_no_video)))
print("WITH video:", sorted(list(failed_with_video)))


### counts
count_no_video = len(failed_no_video)
count_with_video = len(failed_with_video)

print("\n=== Failure Counts ===")
print("Failures without video:", count_no_video)
print("Failures with video:", count_with_video)


### similarity
intersection = len(failed_no_video & failed_with_video)
union = len(failed_no_video | failed_with_video)
jaccard = intersection / union if union > 0 else 1.0

print("\n=== Cascade Difference Metric ===")
print("Jaccard similarity:", jaccard)


### impact
extra_failures = failed_with_video - failed_no_video

print("\n=== Video Impact ===")
print("Additional failures caused by video stress:")
print(sorted(list(extra_failures)))
print("Additional failure count:", len(extra_failures))

#%#