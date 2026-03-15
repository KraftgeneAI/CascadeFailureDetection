# tests/test_generator.py
import torch
from data.generator import ScenarioGenerator

def test_generator_shapes():
    gen = ScenarioGenerator()
    video = torch.randn(2, 5, 3, 64, 64)

    out = gen.generate(video)

    assert out["threat"].shape == (2,5)
    assert out["temperature"].shape == (2,5)
    assert out["voltage"].shape == (2,5)