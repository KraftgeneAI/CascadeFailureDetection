# tests/test_model.py
import torch
from model.fusion_model import FusionModel

def test_model_forward():
    model = FusionModel(scada_dim=10)

    video = torch.randn(4, 8, 3, 64, 64)
    scada = torch.randn(4, 8, 10)

    out = model(video, scada)

    assert out.shape == (4,3)