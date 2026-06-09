"""
Embedding networks for multi-modal data fusion.
"""

from .infrastructure import InfrastructureEmbedding
from .node_mlp import NodeFeatureMLP

__all__ = [
    'InfrastructureEmbedding',
    'NodeFeatureMLP',
]
