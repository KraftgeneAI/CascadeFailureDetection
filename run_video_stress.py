########################
import numpy as np
import matplotlib.pyplot as plt

from video_processor import extract_threat_curve


### debug
print("RUNNING run_video_stress.py")


### input video
video_path = "video/wildfire1.mp4"


### extract stress signal
stress_signal = extract_threat_curve(video_path)

if stress_signal is None or len(stress_signal) == 0:
    raise RuntimeError("Empty stress signal")

stress_signal = np.array(stress_signal)


print("\n=== Video Signal ===")
print("Shape:", stress_signal.shape)
print("Max:", float(np.max(stress_signal)))
print("Min:", float(np.min(stress_signal)))
print("Mean:", float(np.mean(stress_signal)))


# ==============================
### compress saturation
stress_signal = np.clip(stress_signal, 0, 1)

### nonlinear scaling
stress_signal = np.sqrt(stress_signal)


# ==============================
### smoothing
smooth = True
alpha = 0.6

if smooth and len(stress_signal) > 1:
    smoothed = [stress_signal[0]]
    for t in range(1, len(stress_signal)):
        prev = smoothed[-1]
        val = stress_signal[t]
        smoothed.append(alpha * val + (1 - alpha) * prev)
    stress_signal = np.array(smoothed)


# ==============================
### amplify but keep variation
global_stress = 0.3 + 1.2 * stress_signal


print("\n=== Global Stress ===")
print("Shape:", global_stress.shape)
print("First values:", global_stress[:10])
print("Mean:", float(np.mean(global_stress)))


# ==============================
### plot
plt.figure(figsize=(10, 4))
plt.plot(global_stress)
plt.title("Global Fire Stress Signal")
plt.xlabel("Time")
plt.ylabel("Stress")
plt.grid(True)
plt.show()

#%#