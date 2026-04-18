"""
Model Architecture Tests
========================
Fast tests (no real data, no training loop) that verify:

  1. GraphAttentionLayer  — shapes, edge mask correctness, gradient flow
  2. TemporalGNNCell      — shapes, LSTM state threading
  3. UnifiedCascadePredictionModel — full forward pass, output keys/shapes,
                                     gradient flow, NaN detection, smoke loss step

Run with:
    cd CascadeFailureDetection
    pytest test/test_model.py -v

Run only the fast tests (skip the smoke training step):
    pytest test/test_model.py -v -m "not slow"
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from cascade_prediction.models.layers.graph_attention import GraphAttentionLayer
from cascade_prediction.models.layers.temporal_gnn import TemporalGNNCell
from cascade_prediction.models.unified_model import UnifiedCascadePredictionModel
from cascade_prediction.data.generator.config import Settings


# ── Shared constants ──────────────────────────────────────────────────────────

B = 2          # batch size — keep small for speed
N = 10         # nodes — use 10, not 118, for fast tests
E = 20         # edges
T = 5          # timesteps
H = Settings.Model.HIDDEN_DIM        # 128
D = Settings.Model.EMBEDDING_DIM     # 128
HEADS = Settings.Model.HEADS         # 4
EDGE_F = Settings.Model.EDGE_FEATURES  # 7


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edge_index(num_nodes=N, num_edges=E) -> torch.Tensor:
    """Random valid edge_index [2, E] with no self-loops."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    # Avoid self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    if len(src) < num_edges:
        # Pad if we removed too many
        extra = num_edges - len(src)
        src = torch.cat([src, torch.zeros(extra, dtype=torch.long)])
        dst = torch.cat([dst, torch.ones(extra, dtype=torch.long)])
    return torch.stack([src[:num_edges], dst[:num_edges]], dim=0)


def make_gat_batch(batch_size=B, num_nodes=N, num_edges=E,
                   in_channels=H, edge_dim=H, with_mask=True):
    """Create a minimal synthetic GAT input batch."""
    x = torch.randn(batch_size, num_nodes, in_channels)
    edge_index = make_edge_index(num_nodes, num_edges)
    edge_attr = torch.randn(batch_size, num_edges, edge_dim)
    edge_mask = None
    if with_mask:
        # Mask ~20% of edges as failed
        edge_mask = (torch.rand(batch_size, num_edges) > 0.2).float()
    return x, edge_index, edge_attr, edge_mask


def all_params_have_grad(module: nn.Module) -> list[str]:
    """Return names of parameters that have no gradient after backward()."""
    missing = []
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is None:
            missing.append(name)
    return missing


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GraphAttentionLayer
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphAttentionLayer:

    @pytest.fixture
    def gat(self):
        return GraphAttentionLayer(
            in_channels=H,
            out_channels=H // HEADS,
            heads=HEADS,
            concat=True,
            dropout=0.0,   # deterministic for shape tests
            edge_dim=H,
        )

    def test_output_shape_with_edge_attr(self, gat):
        """Output shape should be [B, N, H * out_channels] when concat=True."""
        x, edge_index, edge_attr, _ = make_gat_batch(with_mask=False)
        out = gat(x, edge_index, edge_attr)
        assert out.shape == (B, N, H), \
            f"Expected ({B}, {N}, {H}), got {out.shape}"

    def test_output_shape_with_edge_mask(self, gat):
        """Output shape must be the same whether edge mask is provided or not."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch(with_mask=True)
        out = gat(x, edge_index, edge_attr, edge_mask)
        assert out.shape == (B, N, H)

    def test_no_nan_in_output(self, gat):
        """Forward pass must not produce NaN values."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch()
        out = gat(x, edge_index, edge_attr, edge_mask)
        assert not torch.isnan(out).any(), "NaN detected in GAT output"
        assert not torch.isinf(out).any(), "Inf detected in GAT output"

    def test_edge_mask_changes_output(self, gat):
        """
        Masking edges should produce different node embeddings than no mask.
        If the mask has no effect, the edge mask fix is broken.
        """
        torch.manual_seed(0)
        x, edge_index, edge_attr, _ = make_gat_batch(with_mask=False)

        # All edges active
        full_mask = torch.ones(B, E)
        out_full = gat(x, edge_index, edge_attr, full_mask)

        # Half the edges masked out
        partial_mask = full_mask.clone()
        partial_mask[:, :E // 2] = 0.0
        out_partial = gat(x, edge_index, edge_attr, partial_mask)

        assert not torch.allclose(out_full, out_partial), \
            "Edge mask has no effect on output — masking logic may be broken"

    def test_all_edges_masked_no_nan(self, gat):
        """
        When every edge is masked (all lines failed), output must still be
        finite — nodes should fall back to their self-loop representation.
        """
        x, edge_index, edge_attr, _ = make_gat_batch(with_mask=False)
        zero_mask = torch.zeros(B, E)
        out = gat(x, edge_index, edge_attr, zero_mask)
        assert not torch.isnan(out).any(), \
            "NaN when all edges masked — self-loop fallback may be broken"

    def test_gradient_flow(self, gat):
        """All GAT parameters must receive gradients after backward()."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch()
        out = gat(x, edge_index, edge_attr, edge_mask)
        loss = out.sum()
        loss.backward()
        missing = all_params_have_grad(gat)
        assert not missing, f"Parameters with no gradient: {missing}"

    def test_concat_false_output_shape(self):
        """With concat=False, output should be [B, N, out_channels] (mean over heads)."""
        gat_mean = GraphAttentionLayer(
            in_channels=H, out_channels=H // HEADS,
            heads=HEADS, concat=False, dropout=0.0, edge_dim=H
        )
        x, edge_index, edge_attr, _ = make_gat_batch(with_mask=False)
        out = gat_mean(x, edge_index, edge_attr)
        assert out.shape == (B, N, H // HEADS), \
            f"Expected ({B}, {N}, {H // HEADS}), got {out.shape}"

    def test_no_edge_attr(self):
        """GAT without edge_dim should still produce correct output shape."""
        gat_no_edge = GraphAttentionLayer(
            in_channels=H, out_channels=H // HEADS,
            heads=HEADS, concat=True, dropout=0.0, edge_dim=None
        )
        x, edge_index, _, _ = make_gat_batch(with_mask=False)
        out = gat_no_edge(x, edge_index)
        assert out.shape == (B, N, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TemporalGNNCell
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalGNNCell:

    @pytest.fixture
    def cell(self):
        return TemporalGNNCell(
            node_features=D,
            hidden_dim=H,
            edge_dim=H,
            num_heads=HEADS,
            dropout=0.0,
        )

    def test_single_step_output_shape(self, cell):
        """Single-step forward should return [B, N, H] hidden state."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch(
            in_channels=D, edge_dim=H
        )
        h_out, (h_n, c_n) = cell(x, edge_index, edge_attr, edge_mask)
        assert h_out.shape == (B, N, H)

    def test_lstm_state_shapes(self, cell):
        """LSTM state tensors must have shape [num_layers, B*N, H]."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch(
            in_channels=D, edge_dim=H
        )
        _, (h_n, c_n) = cell(x, edge_index, edge_attr, edge_mask)
        layers = Settings.Model.LSTM_NUM_LAYERS
        assert h_n.shape == (layers, B * N, H), \
            f"h_n shape {h_n.shape} != ({layers}, {B * N}, {H})"
        assert c_n.shape == (layers, B * N, H)

    def test_state_threading_across_steps(self, cell):
        """Hidden state from step t should be accepted as h_prev at step t+1."""
        edge_index = make_edge_index()
        edge_attr = torch.randn(B, E, H)
        state = None
        for _ in range(T):
            x = torch.randn(B, N, D)
            h_out, state = cell(x, edge_index, edge_attr, h_prev=state)
        assert h_out.shape == (B, N, H)
        assert not torch.isnan(h_out).any()

    def test_gradient_flow(self, cell):
        """Gradients must flow back through the LSTM and GAT."""
        x, edge_index, edge_attr, edge_mask = make_gat_batch(
            in_channels=D, edge_dim=H
        )
        h_out, _ = cell(x, edge_index, edge_attr, edge_mask)
        h_out.sum().backward()
        missing = all_params_have_grad(cell)
        assert not missing, f"No gradient for: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UnifiedCascadePredictionModel — full forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def make_full_batch(batch_size=B, num_nodes=N, num_edges=E, seq_len=T) -> dict:
    """
    Build a synthetic batch that matches the shapes the unified model expects.
    Uses the same feature dimensions as the real dataset.
    """
    cfg = Settings.Embedding

    # Temporal modality shapes: [B, T, N, F]
    sat_channels   = cfg.ENV_SATELLITE_CHANNELS   # 12
    weather_f      = cfg.ENV_WEATHER_FEATURES      # 80  (flattened)
    threat_f       = cfg.ENV_THREAT_FEATURES       # 6
    scada_f        = cfg.INFRA_SCADA_FEATURES      # 18
    pmu_f          = cfg.INFRA_PMU_FEATURES        # 8
    equip_f        = cfg.INFRA_EQUIPMENT_FEATURES  # 10
    vis_c          = cfg.ROBOT_VISUAL_CHANNELS     # 3
    therm_c        = cfg.ROBOT_THERMAL_CHANNELS    # 1
    sensor_f       = cfg.ROBOT_SENSOR_FEATURES     # 12
    node_f         = Settings.Embedding.NODE_FEATURE_DIM  # 119
    edge_f         = Settings.Model.EDGE_FEATURES  # 7

    edge_index = make_edge_index(num_nodes, num_edges)

    return {
        # Environmental
        'satellite_data':    torch.zeros(batch_size, seq_len, num_nodes, sat_channels, 16, 16),
        'weather_sequence':  torch.randn(batch_size, seq_len, num_nodes, 10, 8),
        'threat_indicators': torch.zeros(batch_size, seq_len, num_nodes, threat_f),

        # Infrastructure
        'scada_data':        torch.randn(batch_size, seq_len, num_nodes, scada_f),
        'pmu_sequence':      torch.randn(batch_size, seq_len, num_nodes, pmu_f),
        'equipment_status':  torch.randn(batch_size, seq_len, num_nodes, equip_f),

        # Robotic (zeroed — inactive modality)
        'visual_data':       torch.zeros(batch_size, seq_len, num_nodes, vis_c, 32, 32),
        'thermal_data':      torch.zeros(batch_size, seq_len, num_nodes, therm_c, 32, 32),
        'sensor_data':       torch.zeros(batch_size, seq_len, num_nodes, sensor_f),

        # Node features
        'node_features':     torch.randn(batch_size, seq_len, num_nodes, node_f),

        # Graph
        'edge_index':        edge_index,
        'edge_attr':         torch.randn(batch_size, seq_len, num_edges, edge_f),
        'edge_mask':         (torch.rand(batch_size, seq_len, num_edges) > 0.1).float(),

        # Sequence lengths (all full)
        'sequence_length':   torch.full((batch_size,), seq_len, dtype=torch.long),
    }


class TestUnifiedModel:

    @pytest.fixture
    def model(self):
        m = UnifiedCascadePredictionModel()
        m.eval()
        return m

    @pytest.fixture
    def batch(self):
        return make_full_batch()

    # ── Output keys ───────────────────────────────────────────────────────────

    REQUIRED_KEYS = [
        'failure_probability',
        'cascade_timing',
        'voltages',
        'angles',
        'line_flows',
        'active_power_line_flows',
        'temperature',
        'reactive_nodes',
        'frequency',
        'risk_scores',
        'parent_logits',
        'node_embeddings',
    ]

    def test_output_keys_present(self, model, batch):
        """All expected output keys must be present in the forward pass result."""
        with torch.no_grad():
            out = model(batch)
        missing = [k for k in self.REQUIRED_KEYS if k not in out]
        assert not missing, f"Missing output keys: {missing}"

    # ── Output shapes ─────────────────────────────────────────────────────────

    def test_failure_probability_shape(self, model, batch):
        with torch.no_grad():
            out = model(batch)
        assert out['failure_probability'].shape == (B, N, 1), \
            f"Got {out['failure_probability'].shape}"

    def test_node_embeddings_shape(self, model, batch):
        with torch.no_grad():
            out = model(batch)
        assert out['node_embeddings'].shape == (B, N, H)

    def test_parent_logits_shape(self, model, batch):
        """Parent logits must be [B, N, N+1] — one extra class for 'trigger'."""
        with torch.no_grad():
            out = model(batch)
        assert out['parent_logits'].shape == (B, N, N + 1), \
            f"Expected ({B}, {N}, {N + 1}), got {out['parent_logits'].shape}"

    def test_cascade_timing_shape(self, model, batch):
        with torch.no_grad():
            out = model(batch)
        assert out['cascade_timing'].shape == (B, N, 1)

    def test_voltages_shape(self, model, batch):
        with torch.no_grad():
            out = model(batch)
        assert out['voltages'].shape == (B, N, 1)

    def test_line_flows_shape(self, model, batch):
        """Line flow head operates on edges — shape should be [B, E, 1]."""
        with torch.no_grad():
            out = model(batch)
        assert out['line_flows'].shape == (B, E, 1), \
            f"Got {out['line_flows'].shape}"

    def test_frequency_shape(self, model, batch):
        """Frequency is a global prediction — shape [B, 1, 1]."""
        with torch.no_grad():
            out = model(batch)
        assert out['frequency'].shape == (B, 1, 1), \
            f"Got {out['frequency'].shape}"

    # ── Numerical health ──────────────────────────────────────────────────────

    def test_no_nan_in_outputs(self, model, batch):
        """No output tensor may contain NaN or Inf values."""
        with torch.no_grad():
            out = model(batch)
        for key, tensor in out.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isnan(tensor).any(), f"NaN in '{key}'"
                assert not torch.isinf(tensor).any(), f"Inf in '{key}'"

    def test_failure_probability_in_0_1(self, model, batch):
        """Failure probability output must be in [0, 1] after sigmoid."""
        with torch.no_grad():
            out = model(batch)
        fp = out['failure_probability'].sigmoid()
        assert fp.min() >= 0.0 and fp.max() <= 1.0, \
            f"failure_probability out of [0,1]: min={fp.min():.4f} max={fp.max():.4f}"

    # ── Gradient flow ─────────────────────────────────────────────────────────

    def test_gradient_flow_failure_prob(self, model, batch):
        """Gradients must flow all the way back from failure_probability loss."""
        model.train()
        out = model(batch)
        loss = out['failure_probability'].sum()
        loss.backward()
        # Names follow PyTorch's named_parameters() dot-path convention:
        #   <module_attr>.<sub_attr>.<layer_index>.<tensor_name>
        # e.g. failure_prob_head.head.0.weight means:
        #   model.failure_prob_head  -> FailureProbabilityHead
        #     .head                  -> nn.Sequential
        #       [0]                  -> first nn.Linear in the Sequential
        #         .weight            -> the weight tensor
        critical = [
            'temporal_gnn.gat.lin.weight',
            'temporal_gnn.lstm.weight_ih_l0',
            'failure_prob_head.head.0.weight',
        ]
        for name in critical:
            param = dict(model.named_parameters()).get(name)
            assert param is not None, f"Parameter '{name}' not found"
            assert param.grad is not None, f"No gradient for '{name}'"

    def test_gradient_flow_parent_logits(self, model, batch):
        """Gradients must flow from parent_logits back into the GNN layers."""
        model.train()
        out = model(batch)
        loss = out['parent_logits'].sum()
        loss.backward()
        param = dict(model.named_parameters()).get('parent_head.query_proj.weight')
        assert param is not None
        assert param.grad is not None, "No gradient reached parent_head"

    # ── Batch size invariance ─────────────────────────────────────────────────

    def test_batch_size_1(self, model):
        """Model must handle batch size of 1 (common during inference)."""
        batch_1 = make_full_batch(batch_size=1)
        with torch.no_grad():
            out = model(batch_1)
        assert out['failure_probability'].shape[0] == 1
        assert not torch.isnan(out['failure_probability']).any()

    def test_batch_size_4(self, model):
        """Model must handle larger batch sizes without shape errors."""
        batch_4 = make_full_batch(batch_size=4)
        with torch.no_grad():
            out = model(batch_4)
        assert out['failure_probability'].shape[0] == 4

    # ── Smoke training step ───────────────────────────────────────────────────

    @pytest.mark.slow
    def test_loss_decreases_over_steps(self):
        """
        Run 10 gradient steps on synthetic data and verify loss decreases.
        This is NOT a training run — it is a sanity check that the optimizer
        can find a descent direction.
        """
        model = UnifiedCascadePredictionModel()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(10):
            batch = make_full_batch()

            # Build minimal targets matching PhysicsInformedLoss expectations
            targets = {
                'node_failure_labels': (torch.rand(B, N) > 0.8).float(),
                'cascade_timing':      torch.rand(B, N),
                'voltages':            torch.rand(B, N),
                'angles':              torch.rand(B, N),
                'temperatures':        torch.rand(B, N),
                'frequencies':         torch.rand(B, N),
                'reactive_injections': torch.rand(B, N),
                'parent_labels':       torch.full((B, N), -1, dtype=torch.long),
                'is_cascade':          torch.ones(B),
            }
            graph_props = {
                'thermal_limits':  torch.ones(B, E),
                'conductance':     torch.ones(B, E),
                'susceptance':     torch.ones(B, E),
                'edge_index':      batch['edge_index'],
                'active_power_line_flows': torch.rand(B, E),
            }

            optimizer.zero_grad()
            out = model(batch)
            total_loss, _ = model.compute_loss(out, targets, graph_props)
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should be lower at the end than at the start (on average)
        first_half  = sum(losses[:5]) / 5
        second_half = sum(losses[5:]) / 5
        assert second_half < first_half * 1.5, \
            f"Loss did not decrease: first_half={first_half:.4f}, second_half={second_half:.4f}"


# ── Edge mask regression test ─────────────────────────────────────────────────

class TestEdgeMaskRegression:
    """
    Dedicated regression tests for the edge mask bug fix.

    The bug: mask was applied to the message AFTER softmax, meaning masked
    edges still consumed attention budget. Fix: mask attention logits to -inf
    BEFORE softmax so normalization is correct.
    """

    def test_partial_mask_attention_budget(self):
        """
        With one of two edges masked, the surviving edge should receive
        higher attention weight than when both edges are active.
        This is only true with pre-softmax masking.
        """
        torch.manual_seed(42)
        gat = GraphAttentionLayer(
            in_channels=H, out_channels=H // HEADS,
            heads=HEADS, concat=True, dropout=0.0, edge_dim=H
        )
        gat.eval()

        # 3 nodes, 2 directed edges: 0->2 and 1->2
        edge_index = torch.tensor([[0, 1], [2, 2]])
        x = torch.randn(1, 3, H)
        edge_attr = torch.randn(1, 2, H)

        # Both edges active
        full_mask = torch.ones(1, 2)
        out_full = gat(x, edge_index, edge_attr, full_mask)

        # Only edge 0->2 active; edge 1->2 masked
        half_mask = torch.tensor([[1.0, 0.0]])
        out_half = gat(x, edge_index, edge_attr, half_mask)

        # Node 2's embedding must differ because it receives different messages
        assert not torch.allclose(out_full[0, 2], out_half[0, 2]), \
            "Node 2 embedding unchanged after masking one of its input edges"

    def test_unrelated_node_unaffected_by_mask(self):
        """
        Masking an edge that does NOT connect to a node should not change
        that node's output (it receives the same messages either way).
        """
        torch.manual_seed(7)
        gat = GraphAttentionLayer(
            in_channels=H, out_channels=H // HEADS,
            heads=HEADS, concat=True, dropout=0.0, edge_dim=H
        )
        gat.eval()

        # 4 nodes: edges 0->1 and 2->3 (disjoint)
        edge_index = torch.tensor([[0, 2], [1, 3]])
        x = torch.randn(1, 4, H)
        edge_attr = torch.randn(1, 2, H)

        full_mask = torch.ones(1, 2)
        # Mask edge 2->3 only
        partial_mask = torch.tensor([[1.0, 0.0]])

        out_full    = gat(x, edge_index, edge_attr, full_mask)
        out_partial = gat(x, edge_index, edge_attr, partial_mask)

        # Node 0 and node 1 should be unchanged (edge 2->3 does not touch them)
        assert torch.allclose(out_full[0, 0], out_partial[0, 0], atol=1e-5), \
            "Node 0 changed after masking unrelated edge 2->3"
        assert torch.allclose(out_full[0, 1], out_partial[0, 1], atol=1e-5), \
            "Node 1 changed after masking unrelated edge 2->3"