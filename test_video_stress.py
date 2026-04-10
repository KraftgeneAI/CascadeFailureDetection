########################
import numpy as np

from video_processor import extract_threat_curve
from multimodal_data_generator import PhysicsBasedGridSimulator


### input video
video_path = "videos/wildfire1.mp4"

### extract node-level signal (T, N)
stress_signal = extract_threat_curve(video_path)

print("\n=== Video Signal ===")
print("Shape:", stress_signal.shape)
print("Max:", float(np.max(stress_signal)))


### aggregate + amplify (strong but stable)
global_stress = stress_signal.max(axis=1) * 6

print("\n=== Global Stress ===")
print("Shape:", global_stress.shape)
print("First values:", global_stress[:10])


### simulator
simulator = PhysicsBasedGridSimulator()
simulator.dataset_mode = False
simulator.num_samples = 1
simulator.generate_dataset = False


print("\n=== Running Simulation ===")

### 🔥 lower baseline to avoid saturation
BASE_STRESS = 0.05

### baseline run
result_no_video = simulator._generate_scenario_data(
    stress_level=BASE_STRESS,
    sequence_length=10,
    external_stress_signal=None
)

### video-driven run
result_with_video = simulator._generate_scenario_data(
    stress_level=BASE_STRESS,
    sequence_length=10,
    external_stress_signal=global_stress[:10]
)


### extract failures
failed_no = set(result_no_video["metadata"]["failed_nodes"])
failed_yes = set(result_with_video["metadata"]["failed_nodes"])

print("\n=== Results ===")
print("WITHOUT count:", len(failed_no))
print("WITH count:", len(failed_yes))


### difference
extra = failed_yes - failed_no

print("\n=== Impact ===")
print("Extra failures:", extra)
print("Extra failure count:", len(extra))


### similarity metric
intersection = len(failed_no & failed_yes)
union = len(failed_no | failed_yes)
jaccard = intersection / union if union > 0 else 1.0

print("\n=== Metrics ===")
print("Jaccard similarity:", round(jaccard, 4))


### validation
print("\n=== Validation ===")

if len(extra) > 0:
    print("PASS: video signal changes cascade behavior")
else:
    print("WARNING: no visible impact, increase scaling")


### optional plot
import matplotlib.pyplot as plt

plt.plot(global_stress)
plt.title("Global Fire Stress Signal")
plt.xlabel("Time")
plt.ylabel("Stress")
plt.show()

#%#