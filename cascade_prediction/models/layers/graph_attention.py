"""
Graph Attention Layer
=====================
Physics-aware graph attention network layer with edge masking support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional

from cascade_prediction.data.generator.config import Settings


class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Network layer with physics-aware message passing.
    Implements Equations 2, 3, 4 from the paper.
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 concat: bool = True, dropout: float = Settings.Model.GAT_DROPOUT, edge_dim: Optional[int] = None):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.register_parameter('lin_edge', None)
            self.register_parameter('att_edge', None)
        
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Buffers used to pass edge-indexed tensors to message() without
        # PyG re-indexing them (PyG gathers node-dim tensors by node index,
        # which corrupts already-edge-indexed data).
        self._edge_attr_buf: Optional[torch.Tensor] = None
        self._edge_mask_buf: Optional[torch.Tensor] = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H, C_out = self.heads, self.out_channels
        
        x_flat = x.reshape(B * N, C)
        x_transformed = self.lin(x_flat).reshape(B * N, H, C_out)
        
        edge_attr_transformed = None
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 3:
                B_e, E, edge_dim = edge_attr.shape
                edge_attr_flat = edge_attr.reshape(B_e * E, edge_dim)
            else:
                E, edge_dim = edge_attr.shape
                edge_attr_flat = edge_attr
            
            edge_attr_transformed = self.lin_edge(edge_attr_flat).reshape(-1, H, C_out)
        
        # Create batched edge_index
        edge_index_batched = []
        for b in range(B):
            edge_index_batched.append(edge_index + b * N)
        edge_index_batched = torch.cat(edge_index_batched, dim=1)
        
        # Build per-edge attribute tensor [B*E, H, C_out]
        if edge_attr is not None and edge_attr_transformed is not None:
            if edge_attr.dim() == 3:
                edge_attr_propagated = edge_attr_transformed.reshape(B * edge_attr.shape[1], H, C_out)
            else:
                edge_attr_propagated = edge_attr_transformed.repeat(B, 1, 1)
        else:
            edge_attr_propagated = None

        # Build per-edge mask tensor [B*E, 1]
        edge_mask_propagated = None
        if edge_mask is not None:
            edge_mask_propagated = edge_mask.reshape(-1, 1)
        
        # Add self-loops to edge_index and extend edge-indexed buffers accordingly
        edge_index_batched, _ = add_self_loops(edge_index_batched, num_nodes=B * N)
        
        if edge_attr_propagated is not None:
            num_self_loops = B * N
            self_loop_attr = torch.zeros(num_self_loops, H, C_out, device=edge_attr_propagated.device)
            edge_attr_propagated = torch.cat([edge_attr_propagated, self_loop_attr], dim=0)

        if edge_mask_propagated is not None:
            mask_self_loops = torch.ones(B * N, 1, device=edge_mask.device)
            edge_mask_propagated = torch.cat([edge_mask_propagated, mask_self_loops], dim=0)
        
        # Store edge-indexed tensors in instance buffers so message() can
        # access them directly without PyG re-indexing them by node index.
        self._edge_attr_buf = edge_attr_propagated   # [B*E + B*N, H, C_out] or None
        self._edge_mask_buf = edge_mask_propagated   # [B*E + B*N, 1] or None
        
        # Only pass node-indexed tensors (x) through propagate so PyG's
        # gather logic works correctly.
        out = self.propagate(
            edge_index_batched,
            x=x_transformed,
            size=(B * N, B * N),
        )
        
        # Clean up buffers
        self._edge_attr_buf = None
        self._edge_mask_buf = None
        
        out = out.reshape(B, N, H, C_out)
        
        if self.concat:
            out = out.reshape(B, N, H * C_out)
        else:
            out = out.mean(dim=2)
        
        out = out + self.bias
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_index_i: torch.Tensor, size_i: Optional[int]) -> torch.Tensor:
        """
        Compute attention-weighted messages.

        x_i, x_j: [num_edges, H, C_out]  — gathered by PyG from node-indexed x
        edge_index_i: [num_edges]         — destination node indices (for softmax)
        
        Edge attributes and mask are read from instance buffers using a running
        counter so each call gets the correct slice (PyG calls message() once
        per propagate call, passing all edges at once).
        """
        alpha_src = (x_j * self.att_src).sum(dim=-1)   # [num_edges, H]
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)   # [num_edges, H]
        alpha = alpha_src + alpha_dst                   # [num_edges, H]
        
        # Edge attribute contribution — buffer is already edge-indexed
        if self._edge_attr_buf is not None and self.att_edge is not None:
            alpha_edge = (self._edge_attr_buf * self.att_edge).sum(dim=-1)  # [num_edges, H]
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, Settings.Model.LEAKY_RELU_SLOPE)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted message
        msg = x_j * alpha.unsqueeze(-1)   # [num_edges, H, C_out]
        
        # Apply edge mask from buffer
        if self._edge_mask_buf is not None:
            msg = msg * self._edge_mask_buf.unsqueeze(1)   # broadcast over H
        
        return msg
