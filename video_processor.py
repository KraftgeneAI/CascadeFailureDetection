#######################
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO


def compute_node_positions(num_nodes: int):
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)

    x = (np.cos(angles) + 1) / 2
    y = (np.sin(angles) + 1) / 2

    return np.stack([x, y], axis=1)


def extract_threat_curve(
    video_path: str,
    num_nodes: int = 20,
    model_path: str = "best-yolov8n-50epochs.pt",
    resize: Tuple[int, int] = (640, 640),
    confidence_threshold: float = 0.25,
    frame_skip: int = 5,
    smooth: bool = True,
    smoothing_alpha: float = 0.6,
    influence_radius: float = 0.2,
) -> np.ndarray:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    model = YOLO(model_path)
    node_positions = compute_node_positions(num_nodes)

    threat_values = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        if frame_skip > 1 and frame_index % frame_skip != 0:
            continue

        frame = cv2.resize(frame, resize)
        h, w = frame.shape[:2]

        results = model(frame, verbose=False)

        node_stress = np.zeros(num_nodes, dtype=np.float32)
        frame_area = h * w

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                fire_pos = np.array([cx, cy])

                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                area_ratio = box_area / frame_area

                base_score = (conf ** 2) * (area_ratio ** 0.7)

                for i, node_pos in enumerate(node_positions):
                    dist = np.linalg.norm(node_pos - fire_pos)

                    influence = np.exp(
                        -(dist ** 2) / (2 * influence_radius ** 2 + 1e-6)
                    )

                    node_stress[i] += base_score * influence

        threat_values.append(node_stress)

    cap.release()

    signal = np.array(threat_values, dtype=np.float32)

    if len(signal) == 0:
        raise ValueError("No frames processed")

    p95 = np.percentile(signal, 95, axis=0) + 1e-6
    signal = signal / p95
    signal = np.clip(signal, 0.0, 1.0)

    if smooth and len(signal) > 1:
        smoothed = [signal[0]]
        for t in range(1, len(signal)):
            smoothed.append(
                smoothing_alpha * signal[t] + (1 - smoothing_alpha) * smoothed[-1]
            )
        signal = np.array(smoothed, dtype=np.float32)

    print(f"[VideoProcessor] Shape: {signal.shape}")
    print(f"[VideoProcessor] Max stress: {float(signal.max()):.4f}")

    return signal

#%#