#####################
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO


def extract_threat_curve(
    video_path: str,
    model_path: str = "best-yolov8n-50epochs.pt",
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

        scores = []

        ### collect detection scores
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0])

                ### filter low confidence
                if conf < confidence_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                ### normalized object size
                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                area_ratio = box_area / frame_area

                ### balanced scoring (avoid explosion)
                score = conf * np.sqrt(area_ratio)

                scores.append(score)

        ### aggregate per frame (key fix)
        if len(scores) > 0:
            frame_stress = float(np.mean(scores))   # stable aggregation
            detected_frames += 1
        else:
            frame_stress = 0.0

        threat_values.append(frame_stress)

    cap.release()

    if len(threat_values) == 0:
        raise ValueError("No frames processed.")

    signal = np.array(threat_values, dtype=np.float32)

    ### normalize to [0, 1]
    max_val = signal.max()
    min_val = signal.min()

    if max_val > min_val:
        signal = (signal - min_val) / (max_val - min_val)
    else:
        signal = np.zeros_like(signal)

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