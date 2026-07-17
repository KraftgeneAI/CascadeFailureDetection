"""
Grid State Module
=================
Topology-embedded scenario support (see topology_embedded_scenarios_plan.md).

A scenario file may carry a versioned ``topology`` block containing the FULL
grid parameterization — geometry, node physics, thresholds, and line
parameters — so the scenario is self-contained and edited grids are
reproducible without re-deriving anything from a seed. Loaders fall back to
the default topology file when the block is absent (legacy scenarios).

This module is deliberately torch-free (numpy only) so backend services and
tests can use it without the ML stack.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

TOPOLOGY_BLOCK_VERSION = 1

# Node-level array keys stored in the block (all shape [N])
NODE_PROP_KEYS = [
    'node_types', 'base_load', 'gen_capacity',
    'equipment_age', 'equipment_condition',
    'thermal_capacity', 'cooling_effectiveness', 'thermal_time_constant',
    # Phase 3 reactive-support devices (static sizing/setpoints; switching
    # STATE is dynamic runtime data and not in the block). Legacy blocks are
    # backfilled by ensure_rating_arrays().
    'shunt_cap_mvar', 'shunt_reactor_mvar', 'svc_qmax_mvar', 'avr_setpoint',
]
THRESHOLD_KEYS = [
    'loading_failure_threshold', 'loading_damage_threshold',
    'voltage_failure_threshold', 'voltage_damage_threshold',
    'temperature_failure_threshold', 'temperature_damage_threshold',
    'frequency_failure_threshold', 'frequency_damage_threshold',
]
# Edge-level array keys (all shape [E], one entry per directed edge column)
LINE_KEYS = [
    'line_reactance', 'line_resistance', 'line_susceptance',
    'line_conductance', 'thermal_limits',
    # Phase 1 rating tiers + de-rate — REQUIRED by LineTripMonitor and
    # edge_attr packaging; from_grid_state crashes without them. Legacy
    # blocks saved before Phase 1 are backfilled via ensure_rating_arrays().
    'emergency_limits', 'load_dump_limits', 'rating_derate',
    # Phase 2 branch typing (0=line, 1=LTC transformer, 2=PAR) — static per
    # edge; tap POSITIONS are dynamic runtime state and are NOT in the block.
    'branch_type',
]
# Phase 4 RAS scheme definitions — per-SCHEME arrays (variable length, NOT
# per-node/per-edge, so they skip the length validation). Arming STATUS is
# dynamic runtime state and stays out of the block.
RAS_KEYS = [
    'ras_monitored_edge', 'ras_action_type', 'ras_action_target',
]


def ensure_rating_arrays(block: Dict) -> Dict:
    """
    Backfill Phase 1 per-edge rating arrays on legacy topology blocks
    (embedded before the multi-rating change). Mid-range tier factors match
    the dataset.py backfill; derate defaults to no-op.
    """
    e = int(np.asarray(block['edge_index']).shape[1])
    tl = np.asarray(block.get('thermal_limits', np.ones(e)), dtype=np.float64)
    if 'emergency_limits' not in block:
        block['emergency_limits'] = tl * 1.22
    if 'load_dump_limits' not in block:
        block['load_dump_limits'] = tl * 1.47
    if 'rating_derate' not in block:
        block['rating_derate'] = np.ones(e)
    # Phase 2: legacy blocks predate branch typing — all plain lines.
    if 'branch_type' not in block:
        block['branch_type'] = np.zeros(e, dtype=np.int64)
    # Phase 3: legacy blocks predate reactive support — no devices; AVR
    # setpoints default to 1.0 p.u. on generator buses (0 elsewhere).
    n = int(block['num_nodes'])
    node_types = np.asarray(block.get('node_types', np.zeros(n)))
    for key in ('shunt_cap_mvar', 'shunt_reactor_mvar', 'svc_qmax_mvar'):
        if key not in block:
            block[key] = np.zeros(n)
    if 'avr_setpoint' not in block:
        block['avr_setpoint'] = np.where(node_types == 1, 1.0, 0.0)
    # Phase 4: legacy blocks predate RAS — zero schemes (engine no-ops).
    for key in RAS_KEYS:
        if key not in block:
            block[key] = np.array([], dtype=np.int64)
    return block


def build_topology_block(sim) -> Dict:
    """
    Extract the full grid parameterization from a PhysicsBasedGridSimulator
    into a plain-numpy topology block for embedding in a scenario file.
    """
    edge_index = sim.edge_index
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()

    block = {
        'version': TOPOLOGY_BLOCK_VERSION,
        'source': 'generated',
        'seed': getattr(sim, 'seed', None),
        'num_nodes': int(sim.num_nodes),
        'edge_index': np.asarray(edge_index, dtype=np.int64).copy(),
        'positions': np.asarray(sim.positions, dtype=np.float64).copy(),
        'adjacency_matrix': np.asarray(sim.adjacency_matrix).copy(),
        'client_id_map': {},
    }
    for key in NODE_PROP_KEYS + THRESHOLD_KEYS + LINE_KEYS:
        block[key] = np.asarray(getattr(sim, key)).copy()
    for key in RAS_KEYS:
        block[key] = np.asarray(
            getattr(sim, key, np.array([], dtype=np.int64))).copy()
    return block


def validate_topology_block(block: Dict) -> List[str]:
    """
    Structural validation. Returns a list of problems (empty = valid).
    """
    problems: List[str] = []
    for key in ['version', 'num_nodes', 'edge_index', 'positions']:
        if key not in block:
            problems.append(f"missing key: {key}")
    if problems:
        return problems

    n = int(block['num_nodes'])
    ei = np.asarray(block['edge_index'])
    if ei.ndim != 2 or ei.shape[0] != 2:
        problems.append(f"edge_index must be [2, E], got {ei.shape}")
        return problems
    e = ei.shape[1]

    if np.asarray(block['positions']).shape[0] != n:
        problems.append("positions length != num_nodes")
    if e > 0 and (ei.max() >= n or ei.min() < 0):
        problems.append("edge_index references nodes outside [0, num_nodes)")

    for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
        arr = block.get(key)
        if arr is None:
            problems.append(f"missing node array: {key}")
        elif np.asarray(arr).shape[0] != n:
            problems.append(f"{key} length != num_nodes")

    for key in LINE_KEYS:
        arr = block.get(key)
        if arr is None:
            problems.append(f"missing line array: {key}")
        elif np.asarray(arr).shape[0] != e:
            problems.append(f"{key} length != num_edges")

    return problems


def edge_index_to_adjacency(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Rebuild a dense adjacency matrix from a directed edge_index."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def grid_is_connected(edge_index: np.ndarray, num_nodes: int,
                      removed_nodes: Optional[set] = None) -> bool:
    """BFS connectivity over the undirected graph, ignoring removed nodes."""
    removed = removed_nodes or set()
    alive = [n for n in range(num_nodes) if n not in removed]
    if not alive:
        return False
    adj: Dict[int, List[int]] = {n: [] for n in alive}
    for s, d in zip(edge_index[0], edge_index[1]):
        s, d = int(s), int(d)
        if s in removed or d in removed:
            continue
        adj[s].append(d)
        adj[d].append(s)
    seen = {alive[0]}
    stack = [alive[0]]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(alive)


def _rng_for(client_id: str, base_seed: Optional[int]) -> np.random.Generator:
    """Deterministic RNG per draft element: same edit → same physics."""
    mix = abs(hash((str(client_id), int(base_seed or 0)))) % (2**32)
    return np.random.default_rng(mix)


def _new_node_params(block: Dict, node_type: int, p_mw: float,
                     rng: np.random.Generator) -> Dict[str, float]:
    """
    Physics parameters for a draft node: sample a donor node of the same type
    from the existing grid and copy its params (keeps values inside the
    generated distributions without duplicating NodeConfig logic), then set
    capacity/load from the user's p_mw.
    """
    types = np.asarray(block['node_types'])
    donors = np.nonzero(types == node_type)[0]
    if len(donors) == 0:
        donors = np.arange(len(types))
    donor = int(rng.choice(donors))

    params = {key: float(np.asarray(block[key])[donor])
              for key in NODE_PROP_KEYS + THRESHOLD_KEYS
              if key != 'node_types'}
    params['node_types'] = float(node_type)
    if node_type == 1:      # generator
        params['gen_capacity'] = float(p_mw)
        params['base_load'] = 0.0
    elif node_type == 0:    # load
        params['gen_capacity'] = 0.0
        params['base_load'] = float(p_mw)
    return params


def _new_edge_params(block: Dict, s: int, d: int,
                     rng: np.random.Generator) -> Dict[str, float]:
    """
    Line parameters for a draft edge: copy from the existing edge whose
    endpoint distance is closest to the new edge's distance (impedance and
    ratings correlate with length in the generator).
    """
    pos = np.asarray(block['positions'])
    ei = np.asarray(block['edge_index'])
    new_dist = float(np.linalg.norm(pos[s] - pos[d]))
    dists = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)
    donor = int(np.argmin(np.abs(dists - new_dist)))
    params = {key: float(np.asarray(block[key])[donor]) for key in LINE_KEYS}
    # User-drawn edges are always plain lines — never clone the donor's
    # transformer/PAR typing (Phase 2).
    params['branch_type'] = 0.0
    return params


def apply_topology_edits(block: Dict, edits: Dict) -> Tuple[Dict, Dict[str, int]]:
    """
    Apply frontend edit deltas to a topology block.

    Args:
        block: existing topology block (not mutated)
        edits: {
          'added_nodes':  [{'id': str, 'type': 'generator'|'load'|'substation',
                            'position': [x, y], 'p_mw': float}, ...],
          'added_edges':  [{'from': int|str, 'to': int|str}, ...],
          'moved_nodes':  [{'id': int|str, 'position': [x, y]}, ...],
          'removed_nodes': [int, ...],
          'removed_edges': [[int, int], ...],   # undirected node pairs
        }

    Returns:
        (new_block, client_id_map) — new_block has source='edited';
        client_id_map maps draft string IDs to assigned integer indices.

    Raises:
        ValueError on validation failure (unknown endpoints, duplicate edges,
        disconnected result).
    """
    TYPE_CODES = {'load': 0, 'generator': 1, 'substation': 2}
    new_block = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in block.items()}
    ensure_rating_arrays(new_block)   # legacy blocks lack Phase 1 arrays
    seed = new_block.get('seed')
    n = int(new_block['num_nodes'])

    # ---- 1. added nodes: assign integer indices, extend node arrays -------
    id_map: Dict[str, int] = dict(new_block.get('client_id_map', {}))
    for spec in edits.get('added_nodes', []):
        cid = str(spec['id'])
        if cid in id_map:
            raise ValueError(f"duplicate draft node id: {cid}")
        node_type = TYPE_CODES.get(str(spec.get('type', 'load')).lower())
        if node_type is None:
            raise ValueError(f"unknown node type: {spec.get('type')}")
        rng = _rng_for(cid, seed)
        params = _new_node_params(new_block, node_type,
                                  float(spec.get('p_mw', 0.0)), rng)
        id_map[cid] = n
        n += 1
        new_block['positions'] = np.vstack([
            new_block['positions'],
            np.asarray(spec['position'], dtype=np.float64)])
        for key in NODE_PROP_KEYS + THRESHOLD_KEYS:
            new_block[key] = np.append(new_block[key], params[key])
    new_block['num_nodes'] = n
    new_block['client_id_map'] = id_map

    def resolve(ref) -> int:
        if isinstance(ref, str):
            if ref not in id_map:
                raise ValueError(f"unknown node reference: {ref}")
            return id_map[ref]
        idx = int(ref)
        if not 0 <= idx < n:
            raise ValueError(f"node index out of range: {idx}")
        return idx

    # ---- 1b. moved nodes: update positions in place -----------------------
    # Positions are display/geographic only in this model (line impedances
    # were drawn at generation time and stay fixed), so a move never affects
    # power flow — it's persisted so all views and future edits agree.
    for spec in edits.get('moved_nodes', []):
        idx = resolve(spec['id'])
        new_block['positions'][idx] = np.asarray(
            spec['position'], dtype=np.float64)

    # ---- 2. removed edges (undirected: drop both directed rows) ----------
    ei = np.asarray(new_block['edge_index'])
    keep = np.ones(ei.shape[1], dtype=bool)
    for pair in edits.get('removed_edges', []):
        a, b = resolve(pair[0]), resolve(pair[1])
        hit = (((ei[0] == a) & (ei[1] == b)) | ((ei[0] == b) & (ei[1] == a)))
        if not hit.any():
            raise ValueError(f"removed edge ({a},{b}) does not exist")
        keep &= ~hit
    ei = ei[:, keep]
    for key in LINE_KEYS:
        new_block[key] = np.asarray(new_block[key])[keep]

    # ---- 3. added edges: both directed rows with identical params --------
    existing = {(int(s), int(d)) for s, d in zip(ei[0], ei[1])}
    for spec in edits.get('added_edges', []):
        s, d = resolve(spec['from']), resolve(spec['to'])
        if s == d:
            raise ValueError("self-loop edges are not allowed")
        if (s, d) in existing or (d, s) in existing:
            raise ValueError(f"edge ({s},{d}) already exists")
        rng = _rng_for(f"edge:{spec['from']}->{spec['to']}", seed)
        params = _new_edge_params(new_block, s, d, rng)
        ei = np.hstack([ei, np.array([[s, d], [d, s]], dtype=ei.dtype).T])
        existing.update({(s, d), (d, s)})
        for key in LINE_KEYS:
            new_block[key] = np.append(
                new_block[key], [params[key], params[key]])
    new_block['edge_index'] = ei

    # ---- 4. removed nodes: keep indices stable, disconnect + zero --------
    # Removing rows would renumber every node and invalidate the sequence,
    # labels, and client references. Instead a removed node is disconnected
    # (its edges dropped) and its injection zeroed — physically equivalent
    # to decommissioning, and index-stable.
    removed = {resolve(r) for r in edits.get('removed_nodes', [])}
    if removed:
        ei = np.asarray(new_block['edge_index'])
        keep = ~(np.isin(ei[0], list(removed)) | np.isin(ei[1], list(removed)))
        new_block['edge_index'] = ei[:, keep]
        for key in LINE_KEYS:
            new_block[key] = np.asarray(new_block[key])[keep]
        for idx in removed:
            new_block['gen_capacity'][idx] = 0.0
            new_block['base_load'][idx] = 0.0
        new_block['removed_nodes'] = sorted(
            set(new_block.get('removed_nodes', [])) | removed)

    # ---- 5. validate result ----------------------------------------------
    if not grid_is_connected(np.asarray(new_block['edge_index']), n,
                             set(new_block.get('removed_nodes', []))):
        raise ValueError("edit would disconnect the grid")
    new_block['adjacency_matrix'] = edge_index_to_adjacency(
        np.asarray(new_block['edge_index']), n)
    problems = validate_topology_block(new_block)
    if problems:
        raise ValueError(f"invalid topology after edit: {problems}")

    new_block['source'] = 'edited'
    return new_block, id_map
