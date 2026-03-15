# data/generator.py
import torch
from model.perception import compute_threat, DummyPerceptionModel

class ScenarioGenerator:
    def __init__(self):
        self.perception = DummyPerceptionModel()

    def generate(self, video_frames):
        confidence, area = self.perception.infer(video_frames)
        threat = compute_threat(confidence, area, frame_area=video_frames.shape[-1]**2)

        temperature = 20 + threat * 50
        voltage = 1.0 - 0.3 * threat

        return {
            "video": video_frames,
            "threat": threat,
            "temperature": temperature,
            "voltage": voltage
        }