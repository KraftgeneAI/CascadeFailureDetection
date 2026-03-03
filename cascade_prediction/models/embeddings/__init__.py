"""
Embedding networks for multi-modal data fusion.
"""

from .environmental import EnvironmentalEmbedding
from .infrastructure import InfrastructureEmbedding
from .robotic import RoboticEmbedding

__all__ = [
    'EnvironmentalEmbedding',
    'InfrastructureEmbedding',
    'RoboticEmbedding',
]
