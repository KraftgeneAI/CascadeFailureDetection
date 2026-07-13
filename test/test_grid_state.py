"""
Tests for grid_state.py — topology-embedded scenarios
=====================================================
Covers block validation, connectivity checking, and edit application
(added/removed nodes and edges, client-id mapping, deterministic params,
disconnect rejection). Uses a small synthetic block; no torch required.
"""

import numpy as np
import pytest

from cascade_prediction.data.generator.grid_state import (
    TOPOLOGY_BLOCK_VERSION,
    NODE_PROP_KEYS, THRESHOLD_KEYS, LINE_KEYS,
    validate_topology_block, grid_is_connected,
    edge_index_to_adjacency, apply_topology_edits,
)


def make_block(num_nodes=4):
    """Ring of `num_nodes` nodes stored as directed pairs (2E columns)."""
    src, dst = [], []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src += [i, j]
        dst += [j, i]
    ei = np.array([src, dst], dtype=np.int64)
    e = ei.shape[1]

    rng = np.random.default_rng(0)
    block = {
        'version': TOPOLOGY_BLOCK_VERSION,
        'source': 'generated',
        'seed': 42,
        'num_nodes': num_nodes,
        'edge_index': ei,
        'positions': rng.uniform(0, 100, (num_nodes, 2)),
        'adjacency_matrix': edge_index_to_adjacency(ei, num_nodes),
        'client_id_map': {},
    }
    # node types: node 0 generator, rest loads
    types = np.zeros(num_nodes)
    types[0] = 1
    for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
        block[key] = rng.uniform(0.5, 1.5, num_nodes)
    block['node_types'] = types
    for key in LINE_KEYS:
        block[key] = rng.uniform(1.0, 10.0, e)
    return block


class TestValidation:
    def test_valid_block_passes(self):
        assert validate_topology_block(make_block()) == []

    def test_missing_line_array_reported(self):
        block = make_block()
        del block['thermal_limits']
        assert any('thermal_limits' in p for p in validate_topology_block(block))

    def test_wrong_length_reported(self):
        block = make_block()
        block['gen_capacity'] = block['gen_capacity'][:-1]
        assert any('gen_capacity' in p for p in validate_topology_block(block))

    def test_out_of_range_edge_reported(self):
        block = make_block()
        block['edge_index'] = block['edge_index'].copy()
        block['edge_index'][0, 0] = 99
        assert any('outside' in p for p in validate_topology_block(block))


class TestConnectivity:
    def test_ring_is_connected(self):
        block = make_block()
        assert grid_is_connected(block['edge_index'], block['num_nodes'])

    def test_isolated_node_detected(self):
        ei = np.array([[0, 1], [1, 0]])  # nodes 0-1 linked; node 2 isolated
        assert not grid_is_connected(ei, 3)

    def test_removed_nodes_ignored(self):
        ei = np.array([[0, 1], [1, 0]])
        assert grid_is_connected(ei, 3, removed_nodes={2})


class TestApplyEdits:
    def test_add_node_and_edge(self):
        block = make_block()
        edits = {
            'added_nodes': [{'id': 'node_draft1', 'type': 'generator',
                             'position': [50.0, 50.0], 'p_mw': 100.0}],
            'added_edges': [{'from': 'node_draft1', 'to': 2}],
        }
        new_block, id_map = apply_topology_edits(block, edits)

        assert id_map['node_draft1'] == 4          # next integer index
        assert new_block['num_nodes'] == 5
        assert new_block['source'] == 'edited'
        assert new_block['gen_capacity'][4] == 100.0
        assert new_block['node_types'][4] == 1
        # new edge stored as both directed rows with identical params
        ei = new_block['edge_index']
        assert ((ei[0] == 4) & (ei[1] == 2)).any()
        assert ((ei[0] == 2) & (ei[1] == 4)).any()
        for key in LINE_KEYS:
            assert len(new_block[key]) == ei.shape[1]
        assert validate_topology_block(new_block) == []
        # original untouched
        assert block['num_nodes'] == 4

    def test_deterministic_params(self):
        block = make_block()
        edits = {'added_nodes': [{'id': 'n1', 'type': 'load',
                                  'position': [1.0, 2.0], 'p_mw': 25.0}],
                 'added_edges': [{'from': 'n1', 'to': 0}]}
        b1, _ = apply_topology_edits(block, edits)
        b2, _ = apply_topology_edits(block, edits)
        for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
            assert b1[key][4] == b2[key][4], key
        for key in LINE_KEYS:
            assert b1[key][-1] == b2[key][-1], key

    def test_added_node_without_edge_rejected(self):
        """Save validates the whole result: a draft node must be connected
        by at least one added edge in the same save."""
        block = make_block()
        with pytest.raises(ValueError, match="disconnect"):
            apply_topology_edits(
                block, {'added_nodes': [{'id': 'n1', 'type': 'load',
                                         'position': [1.0, 2.0],
                                         'p_mw': 25.0}]})

    def test_remove_edge_drops_both_rows(self):
        block = make_block(5)  # ring of 5 stays connected after one removal
        e_before = block['edge_index'].shape[1]
        new_block, _ = apply_topology_edits(
            block, {'removed_edges': [[0, 1]]})
        assert new_block['edge_index'].shape[1] == e_before - 2
        ei = new_block['edge_index']
        assert not (((ei[0] == 0) & (ei[1] == 1))
                    | ((ei[0] == 1) & (ei[1] == 0))).any()

    def test_remove_edge_that_disconnects_rejected(self):
        block = make_block(4)  # ring of 4: removing one edge keeps it
        # connected, removing two opposite edges splits it
        with pytest.raises(ValueError, match="disconnect"):
            apply_topology_edits(
                block, {'removed_edges': [[0, 1], [2, 3]]})

    def test_remove_node_keeps_indices_stable(self):
        block = make_block(5)
        new_block, _ = apply_topology_edits(block, {'removed_nodes': [2]})
        assert new_block['num_nodes'] == 5            # no renumbering
        assert new_block['gen_capacity'][2] == 0.0
        assert new_block['base_load'][2] == 0.0
        ei = new_block['edge_index']
        assert not ((ei[0] == 2) | (ei[1] == 2)).any()
        assert 2 in new_block['removed_nodes']

    def test_duplicate_edge_rejected(self):
        block = make_block()
        with pytest.raises(ValueError, match="already exists"):
            apply_topology_edits(
                block, {'added_edges': [{'from': 0, 'to': 1}]})

    def test_unknown_reference_rejected(self):
        block = make_block()
        with pytest.raises(ValueError, match="unknown node reference"):
            apply_topology_edits(
                block, {'added_edges': [{'from': 'ghost', 'to': 1}]})

    def test_self_loop_rejected(self):
        block = make_block()
        with pytest.raises(ValueError, match="self-loop"):
            apply_topology_edits(
                block, {'added_edges': [{'from': 1, 'to': 1}]})
