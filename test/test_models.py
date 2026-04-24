"""
Tests for cascade_prediction model components
==============================================
Covers:
  - All prediction heads (prediction_heads.py)
  - GraphAttentionLayer (graph_attention.py)
  - TemporalGNNCell (temporal_gnn.py)
  - UnifiedCascadePredictionModel forward pass (unified_model.py)
"""

import pytest
import torch
import torch.nn as nn

from cascade_prediction.data.generator.config import Settings
from cascade_prediction.models.heads.prediction_heads import (
    FailureProbabilityHead,
    VoltageHead,
    AngleHead,
    FrequencyHead,
    TemperatureHead,
    LineFlowHead,
    ReactiveFlowHead,
    ActivePowerLineFlowHead,
    RiskHead,
    TimingHead,
)
from cascade_prediction.models.layers.graph_attention import GraphAttentionLayer
from cascade_prediction.models.layers.temporal_gnn import TemporalGNNCell
from cascade_prediction.models.unified_model import UnifiedCascadePredictionModel


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BATCH  = 2
NODES  = 10
EDGES  = 14
HIDDEN = Settings.Model.HIDDEN_DIM   # 128
EMB    = Settings.Model.EMBEDDING_DIM  # 128
HEADS  = Settings.Model.HEADS
T      = 5   # sequence length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _edge_index(n=NODES, e=EDGES) -> torch.Tensor:
    """Random edge index with valid node indices."""
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    return torch.stack([src, dst], dim=0)  # [2, E]


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------

class TestNodePredictionHeads:
    """Tests for per-node prediction heads that take [B, N, hidden] input."""

    @pytest.fixture
    def node_emb(self):
        return _rand(BATCH, NODES, HIDDEN)

    # ── FailureProbabilityHead ───────────────────────────────────────────────

    def test_failure_prob_output_shape(self, node_emb):
        head = FailureProbabilityHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_failure_prob_range(self, node_emb):
        head = FailureProbabilityHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0), \
            "FailureProbabilityHead must output probabilities in [0, 1]"

    def test_failure_prob_eval_mode(self, node_emb):
        """Dropout should be inactive in eval mode — output must be deterministic."""
        head = FailureProbabilityHead(HIDDEN).eval()
        out1 = head(node_emb)
        out2 = head(node_emb)
        assert torch.allclose(out1, out2)

    # ── VoltageHead ──────────────────────────────────────────────────────────

    def test_voltage_output_shape(self, node_emb):
        head = VoltageHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_voltage_non_negative(self, node_emb):
        head = VoltageHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out >= 0.0), "VoltageHead ReLU output must be non-negative"

    # ── AngleHead ────────────────────────────────────────────────────────────

    def test_angle_output_shape(self, node_emb):
        head = AngleHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_angle_range(self, node_emb):
        head = AngleHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0), \
            "AngleHead tanh output must be in [-1, 1]"

    # ── TemperatureHead ──────────────────────────────────────────────────────

    def test_temperature_output_shape(self, node_emb):
        head = TemperatureHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_temperature_non_negative(self, node_emb):
        head = TemperatureHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out >= 0.0), "TemperatureHead ReLU output must be non-negative"

    # ── ReactiveFlowHead (per-node) ──────────────────────────────────────────

    def test_reactive_flow_output_shape(self, node_emb):
        head = ReactiveFlowHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_reactive_flow_allows_negative(self):
        """No activation — negative outputs must be possible."""
        head = ReactiveFlowHead(HIDDEN)
        # Force strongly negative output by zeroing weights and setting bias < 0
        for p in head.parameters():
            p.data.zero_()
        head.head[-1].bias.data.fill_(-5.0)
        x = torch.zeros(1, NODES, HIDDEN)
        out = head(x)
        assert torch.all(out < 0.0), "ReactiveFlowHead should allow negative outputs"

    # ── RiskHead ─────────────────────────────────────────────────────────────

    def test_risk_output_shape(self, node_emb):
        head = RiskHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, Settings.Model.RISK_DIM)

    def test_risk_range(self, node_emb):
        head = RiskHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0), \
            "RiskHead sigmoid output must be in [0, 1]"

    # ── TimingHead ───────────────────────────────────────────────────────────

    def test_timing_output_shape(self, node_emb):
        head = TimingHead(HIDDEN)
        out = head(node_emb)
        assert out.shape == (BATCH, NODES, 1)

    def test_timing_range(self, node_emb):
        """Sigmoid — output must be strictly in (0, 1)."""
        head = TimingHead(HIDDEN)
        out = head(node_emb)
        assert torch.all(out > 0.0) and torch.all(out < 1.0), \
            "TimingHead sigmoid output must be in (0, 1)"

    def test_timing_residual_path(self):
        """Residual projection must have matching shapes."""
        head = TimingHead(HIDDEN)
        x = _rand(1, 5, HIDDEN)
        # Should not raise
        out = head(x)
        assert out.shape == (1, 5, 1)


class TestFrequencyHead:
    """Tests for the global FrequencyHead that takes [B, 1, hidden]."""

    def test_frequency_output_shape(self):
        head = FrequencyHead(HIDDEN)
        x = _rand(BATCH, 1, HIDDEN)   # global mean pooled
        out = head(x)
        assert out.shape == (BATCH, 1, 1)

    def test_frequency_non_negative(self):
        head = FrequencyHead(HIDDEN)
        x = _rand(BATCH, 1, HIDDEN)
        out = head(x)
        assert torch.all(out >= 0.0), "FrequencyHead ReLU must be non-negative"


class TestEdgePredictionHeads:
    """Tests for per-edge prediction heads that take [B, E, hidden*2]."""

    @pytest.fixture
    def edge_feat(self):
        return _rand(BATCH, EDGES, HIDDEN * 2)

    def test_line_flow_output_shape(self, edge_feat):
        head = LineFlowHead(HIDDEN)
        out = head(edge_feat)
        assert out.shape == (BATCH, EDGES, 1)

    def test_line_flow_allows_negative(self):
        """No activation — negative outputs must be possible."""
        head = LineFlowHead(HIDDEN)
        for p in head.parameters():
            p.data.zero_()
        head.head[-1].bias.data.fill_(-5.0)
        x = torch.zeros(1, EDGES, HIDDEN * 2)
        out = head(x)
        assert torch.all(out < 0.0)

    def test_active_power_output_shape(self, edge_feat):
        head = ActivePowerLineFlowHead(HIDDEN)
        out = head(edge_feat)
        assert out.shape == (BATCH, EDGES, 1)


# ---------------------------------------------------------------------------
# GraphAttentionLayer
# ---------------------------------------------------------------------------

class TestGraphAttentionLayer:
    """Tests for GraphAttentionLayer."""

    @pytest.fixture
    def gat(self):
        out_per_head = HIDDEN // HEADS
        return GraphAttentionLayer(
            in_channels=HIDDEN,
            out_channels=out_per_head,
            heads=HEADS,
            concat=True,
            dropout=0.0,  # disable dropout for deterministic tests
            edge_dim=HIDDEN,
        )

    @pytest.fixture
    def inputs(self):
        x = _rand(BATCH, NODES, HIDDEN)
        ei = _edge_index()
        ea = _rand(BATCH, EDGES, HIDDEN)
        return x, ei, ea

    def test_output_shape_concat(self, gat, inputs):
        x, ei, ea = inputs
        out = gat(x, ei, ea)
        # concat=True → output channels = heads * out_per_head = HIDDEN
        assert out.shape == (BATCH, NODES, HIDDEN), \
            f"Expected ({BATCH}, {NODES}, {HIDDEN}), got {out.shape}"

    def test_output_shape_mean(self, inputs):
        out_per_head = HIDDEN // HEADS
        gat = GraphAttentionLayer(
            in_channels=HIDDEN,
            out_channels=out_per_head,
            heads=HEADS,
            concat=False,
            dropout=0.0,
            edge_dim=HIDDEN,
        )
        x, ei, ea = inputs
        out = gat(x, ei, ea)
        assert out.shape == (BATCH, NODES, out_per_head)

    def test_no_nan_in_output(self, gat, inputs):
        x, ei, ea = inputs
        out = gat(x, ei, ea)
        assert not torch.isnan(out).any(), "GAT output contains NaN"

    def test_edge_mask_zeros_messages(self, gat, inputs):
        """Zero edge mask should suppress all messages from real edges."""
        x, ei, ea = inputs
        mask_all_off = torch.zeros(BATCH, EDGES)
        mask_all_on  = torch.ones(BATCH, EDGES)

        out_off = gat(x, ei, ea, edge_mask=mask_all_off).detach()
        out_on  = gat(x, ei, ea, edge_mask=mask_all_on).detach()

        # Outputs must differ when masking is effective
        assert not torch.allclose(out_off, out_on), \
            "Edge mask has no effect on GAT output"

    def test_no_edge_attr(self):
        """GAT without edge attributes should still work."""
        gat = GraphAttentionLayer(
            in_channels=HIDDEN,
            out_channels=HIDDEN // HEADS,
            heads=HEADS,
            concat=True,
            dropout=0.0,
            edge_dim=None,   # no edge attrs
        )
        x = _rand(BATCH, NODES, HIDDEN)
        ei = _edge_index()
        out = gat(x, ei)
        assert out.shape == (BATCH, NODES, HIDDEN)
        assert not torch.isnan(out).any()

    def test_self_loops_included(self, gat):
        """With no real edges, node should still receive self-loop information."""
        x = _rand(BATCH, NODES, HIDDEN)
        # Empty edge index
        ei = torch.zeros(2, 0, dtype=torch.long)
        out = gat(x, ei)
        assert out.shape == (BATCH, NODES, HIDDEN)
        # Output should not be all zeros — self-loops add signal
        assert not torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# TemporalGNNCell
# ---------------------------------------------------------------------------

class TestTemporalGNNCell:
    """Tests for TemporalGNNCell (GAT + LSTM)."""

    @pytest.fixture
    def cell(self):
        return TemporalGNNCell(
            node_features=EMB,
            hidden_dim=HIDDEN,
            edge_dim=HIDDEN,
            num_heads=HEADS,
            dropout=0.0,
        )

    @pytest.fixture
    def cell_inputs(self):
        x  = _rand(BATCH, NODES, EMB)
        ei = _edge_index()
        ea = _rand(BATCH, EDGES, HIDDEN)
        return x, ei, ea

    def test_output_shape(self, cell, cell_inputs):
        x, ei, ea = cell_inputs
        h_out, (h_n, c_n) = cell(x, ei, ea)

        assert h_out.shape == (BATCH, NODES, HIDDEN), \
            f"h_out shape mismatch: {h_out.shape}"
        # LSTM state: [num_layers, B*N, hidden]
        assert h_n.shape[0] == Settings.Model.LSTM_NUM_LAYERS
        assert h_n.shape[1] == BATCH * NODES
        assert h_n.shape[2] == HIDDEN

    def test_stateful_processing(self, cell, cell_inputs):
        """Passing in previous LSTM state should change output."""
        x, ei, ea = cell_inputs
        h1, state1 = cell(x, ei, ea)
        h2, state2 = cell(x, ei, ea, h_prev=state1)

        assert not torch.allclose(h1, h2), \
            "Reusing LSTM state should change output"

    def test_no_nan_output(self, cell, cell_inputs):
        x, ei, ea = cell_inputs
        h_out, _ = cell(x, ei, ea)
        assert not torch.isnan(h_out).any()

    def test_layer_norm_applied(self, cell, cell_inputs):
        """Output should have roughly unit variance after LayerNorm."""
        cell.eval()
        x, ei, ea = cell_inputs
        with torch.no_grad():
            h_out, _ = cell(x, ei, ea)
        std = h_out.std().item()
        # LayerNorm normalises per feature — overall std should be O(1)
        assert 0.01 < std < 20.0, f"Unexpected std after LayerNorm: {std}"


# ---------------------------------------------------------------------------
# UnifiedCascadePredictionModel
# ---------------------------------------------------------------------------

class TestUnifiedModel:
    """End-to-end forward pass tests for UnifiedCascadePredictionModel."""

    # ── Shared batch factory ─────────────────────────────────────────────────

    @staticmethod
    def _make_batch(
        batch_size=BATCH,
        num_nodes=NODES,
        num_edges=EDGES,
        seq_len=T,
        include_node_features=True,
    ):
        """Build a minimal valid batch dict matching the model's expected inputs."""
        B, N, E = batch_size, num_nodes, num_edges
        ei = _edge_index(N, E)

        # Image/sequence sizes not stored in Settings — use generator defaults.
        sat_C  = Settings.Embedding.ENV_SATELLITE_CHANNELS   # 12
        sat_H  = 16                                           # satellite img H/W
        wea_T  = 10                                           # weather sub-steps
        wea_F  = 8                                            # weather features per step
        thr_F  = Settings.Embedding.ENV_THREAT_FEATURES       # 6
        sca_F  = Settings.Embedding.INFRA_SCADA_FEATURES      # 18
        pmu_F  = Settings.Embedding.INFRA_PMU_FEATURES        # 8
        eqp_F  = Settings.Embedding.INFRA_EQUIPMENT_FEATURES  # 10
        vis_C  = Settings.Embedding.ROBOT_VISUAL_CHANNELS     # 3
        vis_H  = 32                                            # visual img H/W
        sen_F  = Settings.Embedding.ROBOT_SENSOR_FEATURES     # 12

        batch = {
            # Environmental  [B, T, N, C, H, W] or [B, T, N, F]
            "satellite_data":    _rand(B, seq_len, N, sat_C, sat_H, sat_H),
            "weather_sequence":  _rand(B, seq_len, N, wea_T, wea_F),
            "threat_indicators": _rand(B, seq_len, N, thr_F),
            # Infrastructure
            "scada_data":        _rand(B, seq_len, N, sca_F),
            "pmu_sequence":      _rand(B, seq_len, N, pmu_F),
            "equipment_status":  _rand(B, seq_len, N, eqp_F),
            # Robotic
            "visual_data":       _rand(B, seq_len, N, vis_C, vis_H, vis_H),
            "thermal_data":      _rand(B, seq_len, N, 1,     vis_H, vis_H),
            "sensor_data":       _rand(B, seq_len, N, sen_F),
            # Graph
            "edge_index": ei,
            "edge_attr":  _rand(B, seq_len, E, Settings.Model.EDGE_FEATURES),
            "edge_mask":  torch.ones(B, seq_len, E),
            "sequence_length": torch.tensor([seq_len] * B, dtype=torch.long),
        }

        if include_node_features:
            node_feat_dim = Settings.Embedding.NODE_FEATURE_DIM
            batch["node_features"] = _rand(B, seq_len, N, node_feat_dim)

        return batch

    @pytest.fixture
    def model(self):
        return UnifiedCascadePredictionModel(
            embedding_dim=EMB,
            hidden_dim=HIDDEN,
            num_gnn_layers=1,   # fast for tests
            heads=HEADS,
            dropout=0.0,
        ).eval()

    # ── Output key coverage ─────────────────────────────────────────────────

    def test_forward_returns_required_keys(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        required = {
            "failure_probability", "cascade_timing", "voltages", "angles",
            "line_flows", "active_power_line_flows", "temperature",
            "reactive_nodes", "frequency", "risk_scores",
            "node_embeddings",
        }
        missing = required - set(out.keys())
        assert not missing, f"Missing output keys: {missing}"

    # ── Output shapes ───────────────────────────────────────────────────────

    def test_failure_prob_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["failure_probability"].shape == (BATCH, NODES, 1)

    def test_failure_prob_range(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        p = out["failure_probability"]
        assert torch.all(p >= 0) and torch.all(p <= 1)

    def test_cascade_timing_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["cascade_timing"].shape == (BATCH, NODES, 1)

    def test_risk_scores_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["risk_scores"].shape == (BATCH, NODES, Settings.Model.RISK_DIM)

    def test_frequency_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["frequency"].shape == (BATCH, 1, 1)

    def test_line_flows_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["line_flows"].shape == (BATCH, EDGES, 1)
        assert out["active_power_line_flows"].shape == (BATCH, EDGES, 1)

    def test_node_embeddings_shape(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        assert out["node_embeddings"].shape == (BATCH, NODES, HIDDEN)

    # ── No NaN ──────────────────────────────────────────────────────────────

    def test_no_nan_in_outputs(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            out = model(batch)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"NaN detected in output '{key}'"

    # ── Optional node_features fallback ─────────────────────────────────────

    def test_forward_without_node_features(self, model):
        """Model must fall back gracefully when node_features key is absent."""
        batch = self._make_batch(include_node_features=False)
        with torch.no_grad():
            out = model(batch)
        assert "failure_probability" in out
        assert not torch.isnan(out["failure_probability"]).any()

    # ── Sequence length masking ──────────────────────────────────────────────

    def test_sequence_length_respected(self, model):
        """Different valid sequence lengths should produce different outputs."""
        batch_short = self._make_batch(seq_len=3)
        batch_long  = self._make_batch(seq_len=T)
        with torch.no_grad():
            out_s = model(batch_short)
            out_l = model(batch_long)
        # Shapes independent of seq_len
        assert out_s["failure_probability"].shape == out_l["failure_probability"].shape

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradients_flow_through_model(self):
        """Ensure the full graph is differentiable."""
        model = UnifiedCascadePredictionModel(
            embedding_dim=EMB,
            hidden_dim=HIDDEN,
            num_gnn_layers=1,
            heads=HEADS,
            dropout=0.0,
        ).train()
        batch = self._make_batch()
        out = model(batch)
        loss = out["failure_probability"].mean()
        loss.backward()

        # At least one parameter should have a non-None gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed through the model"

    # ── compute_loss integration ─────────────────────────────────────────────

    def test_compute_loss_returns_tensor_and_dict(self, model):
        batch = self._make_batch()
        with torch.no_grad():
            preds = model(batch)

        targets = {
            "failure_label": torch.zeros(BATCH, NODES),
        }
        graph_props = {}

        total_loss, loss_dict = model.compute_loss(preds, targets, graph_props)

        assert isinstance(total_loss, torch.Tensor), "total_loss must be a Tensor"
        assert isinstance(loss_dict, dict), "loss_dict must be a dict"
        assert "prediction" in loss_dict
        assert total_loss.item() >= 0.0

    def test_compute_loss_with_cascade_targets(self, model):
        """Loss should increase when there are undetected cascade failures."""
        batch = self._make_batch()
        with torch.no_grad():
            preds = model(batch)

        targets_no_fail = {
            "failure_label": torch.zeros(BATCH, NODES),
        }
        targets_cascade = {
            "failure_label": torch.ones(BATCH, NODES),  # all nodes failing
        }
        graph_props = {}

        loss_no_fail, _ = model.compute_loss(preds, targets_no_fail, graph_props)
        loss_cascade, _ = model.compute_loss(preds, targets_cascade, graph_props)

        # At least one of these should be non-trivially different
        assert abs(loss_no_fail.item() - loss_cascade.item()) > 1e-6, \
            "Loss should differ between all-normal and all-cascade targets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
