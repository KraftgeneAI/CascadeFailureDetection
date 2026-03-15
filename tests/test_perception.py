# tests/test_perception.py
import torch
from model.perception import compute_threat

def test_threat_computation():
    conf = torch.tensor([[0.5]])
    area = torch.tensor([[50.0]])
    frame_area = 100.0

    threat = compute_threat(conf, area, frame_area)
    assert torch.isclose(threat, torch.tensor([[0.25]]))