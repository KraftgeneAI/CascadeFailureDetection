##################
import cv2
import numpy as np
from typing import Tuple


def extract_threat_curve(
    video_path: str,
    resize: Tuple[int, int] = (224, 224),
    smooth: bool = True,
    smoothing_window: int = 5,
) -> np.ndarray:
    """
    Extract a per-frame normalized threat signal from a video.

    Uses average frame brightness as a simple proxy.
    Output is strictly normalized to [0, 1].
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    threat_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray) / 255.0   # ← 不要乘 5
        threat_values.append(brightness)

    cap.release()

    if len(threat_values) == 0:
        raise ValueError("No frames extracted from video.")

    signal = np.array(threat_values, dtype=np.float32)

    # Optional smoothing
    if smooth and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        signal = np.convolve(signal, kernel, mode="same")

    # Min-max normalization (关键)
    min_val = signal.min()
    max_val = signal.max()

    if max_val > min_val:
        signal = (signal - min_val) / (max_val - min_val)
    else:
        signal = np.zeros_like(signal)

    signal = np.clip(signal, 0.0, 1.0)

    return signal


if __name__ == "__main__":
    video_path = "videos/wildfire1.mp4"

    signal = extract_threat_curve(video_path)

    print("Frames:", len(signal))
    print("Min:", float(signal.min()))
    print("Max:", float(signal.max()))
    print("First 10 values:", signal[:10])

#%#