# model/perception.py
import torch

def compute_threat(confidence, bbox_area, frame_area):
    return confidence * (bbox_area / frame_area)

class DummyPerceptionModel:
    """Stub for YOLO"""
    def infer(self, frames):
        B,T,C,H,W = frames.shape
        confidence = torch.rand(B,T)
        bbox_area = torch.rand(B,T) * (H*W)
        return confidence, bbox_area