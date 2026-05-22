#####################
import cv2
import numpy as np
import os
from typing import Tuple
from ultralytics import YOLO

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best-yolov8n-50epochs.pt")


def extract_threat_curve(
    video_path: str,
    model_path: str = _DEFAULT_MODEL,
    resize: Tuple[int, int] = (640, 640),
    confidence_threshold: float = 0.25,
    frame_skip: int = 5,
    smooth: bool = False,
    smoothing_alpha: float = 0.6,
) -> np.ndarray:

    ### open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ### load YOLO model once
    try:
        model = YOLO(model_path)
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    threat_values = []
    frame_index = 0
    detected_frames = 0

    ### iterate frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        ### skip frames for efficiency
        if frame_skip > 1 and frame_index % frame_skip != 0:
            continue

        frame_resized = cv2.resize(frame, resize)

        ### run detection
        try:
            results = model(frame_resized, verbose=False)
        except Exception:
            threat_values.append(0.0)
            continue

        frame_area = frame_resized.shape[0] * frame_resized.shape[1]

        total_box_area = 0.0

        ### sum all box areas above confidence threshold
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0])

                if conf < confidence_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                total_box_area += max(0.0, (x2 - x1) * (y2 - y1))

        ### fire size = fraction of frame covered by fire
        if total_box_area > 0:
            frame_stress = float(min(total_box_area / frame_area, 1.0))
            detected_frames += 1
        else:
            frame_stress = 0.0

        threat_values.append(frame_stress)

    cap.release()

    if len(threat_values) == 0:
        raise ValueError("No frames processed.")

    signal = np.array(threat_values, dtype=np.float32)

    signal = np.clip(signal, 0.0, 1.0)

    ### optional smoothing
    if smooth and len(signal) > 1:
        smoothed = [signal[0]]
        for val in signal[1:]:
            prev = smoothed[-1]
            smoothed.append(
                smoothing_alpha * val + (1 - smoothing_alpha) * prev
            )
        signal = np.array(smoothed, dtype=np.float32)

    ### debug logs
    print(f"[VideoProcessor] Processed frames: {frame_index}")
    print(f"[VideoProcessor] Detection frames: {detected_frames}")
    print(f"[VideoProcessor] Max stress: {float(signal.max()):.4f}")

    return signal
#%#