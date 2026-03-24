"""
Prediction Heads Module
=======================
Multi-task prediction heads for cascade failure prediction.

This module contains all specialized prediction heads:
- Failure probability prediction
- Voltage prediction (physics-informed)
- Angle prediction (physics-informed)
- Frequency prediction (physics-informed)
- Temperature prediction
- Line flow prediction (active and reactive power)
- Risk assessment (7-dimensional)
- Timing prediction (cascade propagation)
"""

import torch
import torch.nn as nn

from cascade_prediction.data.generator.config import Settings


class FailureProbabilityHead(nn.Module):
    """
    Predicts node failure probability.

    Output: [batch_size, num_nodes, 1] with sigmoid activation (0-1 range)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(FailureProbabilityHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # Increased dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Failure probabilities [batch_size, num_nodes, 1]
        """
        return self.head(x)


class VoltageHead(nn.Module):
    """
    Predicts node voltages (per-unit).

    Physics-informed: Must predict any positive voltage value.
    No hard-coded scaling - learns from data.

    Output: [batch_size, num_nodes, 1] with ReLU activation (positive values)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(VoltageHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Allows prediction of any positive voltage
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Voltages in p.u. [batch_size, num_nodes, 1]
        """
        return self.head(x)


class AngleHead(nn.Module):
    """
    Predicts node voltage angles (radians).

    Physics-informed: Predicts small radian values.
    Tanh activation provides good range [-1, 1] for typical angles.

    Output: [batch_size, num_nodes, 1] with tanh activation
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(AngleHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Good range for radians
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Angles in radians [batch_size, num_nodes, 1]
        """
        return self.head(x)


class FrequencyHead(nn.Module):
    """
    Predicts system frequency (Hz).

    Physics-informed: Must predict any positive frequency value.
    No hard-coded scaling - learns from data.

    Output: [batch_size, 1, 1] with ReLU activation (positive values)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(FrequencyHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Allows prediction of any positive frequency
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Global node embeddings [batch_size, 1, hidden_dim]
        
        Returns:
            Frequency in Hz [batch_size, 1, 1]
        """
        return self.head(x)


class TemperatureHead(nn.Module):
    """
    Predicts node/line temperatures.

    Physics-informed: Temperature must be positive.

    Output: [batch_size, num_nodes, 1] with ReLU activation (positive values)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(TemperatureHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Temperature must be positive
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Temperatures [batch_size, num_nodes, 1]
        """
        return self.head(x)


class LineFlowHead(nn.Module):
    """
    Predicts reactive power flow on transmission lines.

    Physics-informed: Can be positive or negative (bidirectional flow).
    No activation - linear output.

    Output: [batch_size, num_edges, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(LineFlowHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # No activation - allows positive and negative values
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: Concatenated source and destination node embeddings
                          [batch_size, num_edges, hidden_dim * 2]
        
        Returns:
            Reactive power flows [batch_size, num_edges, 1]
        """
        return self.head(edge_features)


class ReactiveFlowHead(nn.Module):
    """
    Predicts reactive power at nodes.

    Physics-informed: Can be positive or negative.
    No activation - linear output.

    Output: [batch_size, num_nodes, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(ReactiveFlowHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
            # No activation - allows positive and negative values
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Reactive power at nodes [batch_size, num_nodes, 1]
        """
        return self.head(x)


class ActivePowerLineFlowHead(nn.Module):
    """
    Predicts active power flow on transmission lines.

    Physics-informed: Can be positive or negative (bidirectional flow).
    No activation - linear output.

    Output: [batch_size, num_edges, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(ActivePowerLineFlowHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # No activation - allows positive and negative values
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: Concatenated source and destination node embeddings
                          [batch_size, num_edges, hidden_dim * 2]
        
        Returns:
            Active power flows [batch_size, num_edges, 1]
        """
        return self.head(edge_features)


class RiskHead(nn.Module):
    """
    Predicts seven-dimensional risk assessment.

    Dimensions:
    1. Threat level
    2. Vulnerability
    3. Impact severity
    4. Cascade probability
    5. Response capability
    6. Safety margin
    7. Urgency

    Output: [batch_size, num_nodes, Settings.Model.RISK_DIM] with sigmoid activation (0-1 range)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(RiskHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, Settings.Model.RISK_DIM),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Risk scores [batch_size, num_nodes, 7]
        """
        return self.head(x)


class TimingHead(nn.Module):
    """
    Predicts cascade failure timing (normalised time-to-failure for each node).

    IMPROVED ARCHITECTURE (v2):
    - Deeper 3-layer MLP with residual skip connection for richer representation
    - Sigmoid output constrained to [0, 1] matching normalised training targets
      (failure_time / sequence_length).  The old linear output had no activation,
      allowing arbitrary negative values that were meaningless for timing.
    - BatchNorm replaced by LayerNorm for stability across variable batch sizes.

    Output: [batch_size, num_nodes, 1] in range (0, 1) — normalised failure time.
    Multiply by sequence_length (or max_time_horizon) at inference time to recover
    the absolute timestep prediction.
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(TimingHead, self).__init__()

        mid = hidden_dim // 2

        # Main path: hidden → mid → mid → 1
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(mid, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(mid, 1)

        # Residual projection: hidden → mid (so we can add to layer2 output)
        self.residual_proj = nn.Linear(hidden_dim, mid)

        # Sigmoid so output is always in (0, 1) — matches normalised targets
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]

        Returns:
            Normalised time-to-failure predictions [batch_size, num_nodes, 1] in (0, 1)
        """
        residual = self.residual_proj(x)   # [B, N, mid]
        h = self.layer1(x)                 # [B, N, mid]
        h = self.layer2(h) + residual      # residual skip
        return self.activation(self.out(h))
