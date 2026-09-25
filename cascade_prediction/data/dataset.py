"""
Cascade Dataset Module
======================
Memory-efficient dataset loader for pre-generated cascade failure data.

This module provides the main CascadeDataset class that:
- Loads scenarios from individual pickle files
- Applies physics-based normalization
- Implements sliding window truncation
- Creates dynamic topology masks
- Handles both cascade and normal scenarios
"""

import torch
from torch.utils.data import Dataset
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import glob
import re
from tqdm import tqdm
import json

from .generator.config import Settings
from .. import contract as _contract

from .preprocessing import (
    normalize_power,
    normalize_frequency,
    calculate_truncation_window,
    apply_truncation,
    create_edge_mask_from_failures,
    to_tensor,
)


class CascadeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for cascade failure scenarios.
    
    Features:
    - One scenario per file loading
    - Physics-based normalization
    - Sliding window truncation
    - Dynamic topology masking
    - Metadata caching for fast initialization
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'last_timestep',
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_frequency: float = Settings.Dataset.BASE_FREQUENCY,
        fixed_window: Optional[Tuple[int, int]] = None,
        contract_mode: str = "strict",
    ):
        """
        Initialize dataset from a directory of scenario files.

        Args:
            data_dir: Directory containing scenario_*.pkl files
            mode: Dataset mode ('last_timestep' or 'full_sequence')
            base_mva: Base MVA for power normalization
            base_frequency: Base frequency for normalization (Hz)
            fixed_window: Optional (start_idx, end_idx) that bypasses random
                sliding-window truncation. Used for deterministic streaming
                inference on an explicit window (e.g. steps 0..t).
            contract_mode: Data-contract policy (spec specs/data-contract,
                decision D3).
                'strict'  (default, training) — every scenario must be
                          stamped with the current layout or certified;
                          anything else raises ContractError (DC-020).
                'migrate' (backend / inference) — legacy layouts are
                          upgraded through contract.migrate_timestep, and
                          each migration is logged (DC-021).
        """
        if contract_mode not in ("strict", "migrate"):
            raise ValueError(f"contract_mode must be 'strict' or 'migrate', got {contract_mode!r}")
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        self.fixed_window = fixed_window
        self.contract_mode = contract_mode
        self._norm_constants = {
            "INTERCHANGE_SCALE_MW": float(Settings.Interchange.INTERCHANGE_SCALE_MW),
        }
        
        # Find all scenario files
        print(f"Indexing scenarios from: {data_dir}")
        self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenario_*.pkl")))
        if not self.scenario_files:
            self.scenario_files = sorted(glob.glob(str(self.data_dir / "scenarios_batch_*.pkl")))
        
        # Define cache path
        cache_file = self.data_dir / "metadata_cache.json"
        self.cascade_labels = []

        # Load from cache if available
        if os.path.exists(cache_file):
            print(f"Loading labels from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                self.cascade_labels = json.load(f)
            if len(self.cascade_labels) != len(self.scenario_files):
                print("Warning: Cache length mismatch. Re-scanning...")
                self.cascade_labels = []

        # Scan files if no cache
        if not self.cascade_labels:
            print(f"Scanning {len(self.scenario_files)} files for metadata (first run)...")
            for scenario_file in tqdm(self.scenario_files):
                try:
                    with open(scenario_file, 'rb') as f:
                        scenario_data = pickle.load(f)
                    scenario = scenario_data[0] if isinstance(scenario_data, list) else scenario_data
                    if not isinstance(scenario, dict):
                        self.cascade_labels.append(False)
                        continue
                    if 'metadata' in scenario and 'is_cascade' in scenario['metadata']:
                        has_cascade = bool(scenario['metadata']['is_cascade'])
                    elif 'sequence' in scenario and len(scenario['sequence']) > 0:
                        last_step = scenario['sequence'][-1]
                        has_cascade = bool(
                            np.max(last_step.get('node_labels', np.zeros(1)))
                            > Settings.Dataset.CASCADE_LABEL_THRESHOLD
                        )
                    else:
                        has_cascade = False
                    self.cascade_labels.append(has_cascade)
                except (IOError, pickle.UnpicklingError, EOFError):
                    print(f"Warning: Skipping corrupted file: {scenario_file}")
                    self.cascade_labels.append(False)

            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.cascade_labels, f)
                print(f"Saved metadata cache to {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")

        # Print statistics
        print(f"Physics normalization: base_mva={base_mva}, base_frequency={base_frequency}")
        print(f"Indexed {len(self.scenario_files)} scenarios.")
        if len(self.cascade_labels) == 0:
            print("  [WARNING] No valid scenarios found!")
        else:
            positive_count = sum(self.cascade_labels)
            total = len(self.cascade_labels)
            print(f"  Cascade scenarios: {positive_count} ({positive_count/total*100:.1f}%)")
            print(f"  Normal scenarios:  {total - positive_count} ({(total-positive_count)/total*100:.1f}%)")
        print(f"Ultra-memory-efficient mode: Loading 1 file per sample.")

        # Fail at construction, not mid-epoch, when training on data that
        # doesn't match the current contract (DC-020).
        if self.contract_mode == "strict":
            for path in self.scenario_files:
                try:
                    scenarios = _contract.load_scenarios(path)
                except Exception:
                    continue          # unreadable files are handled per item
                for scenario in scenarios:
                    _contract.require_current(scenario, path)
                break                 # checking the first readable file is enough here
        print(f"Data contract: mode={self.contract_mode}, schema "
              f"{_contract.SCHEMA_VERSION}, layout {_contract.LAYOUT_HASH}")

    @classmethod
    def item_from_scenario(
        cls,
        scenario: Dict[str, Any],
        *,
        fixed_window: Tuple[int, int],
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_frequency: float = Settings.Dataset.BASE_FREQUENCY,
        contract_mode: str = "strict",
    ) -> Dict[str, Any]:
        """
        Process one in-memory scenario ({'sequence', 'edge_index', 'metadata'})
        exactly as __getitem__ would, without scanning a directory or reading
        pickles. Used for live-stream inference (spec 001, STR-041), where the
        frames were already validated against the telemetry contract.
        """
        if contract_mode not in ("strict", "migrate"):
            raise ValueError(f"contract_mode must be 'strict' or 'migrate', got {contract_mode!r}")
        self = cls.__new__(cls)
        self.data_dir = Path("<in-memory>")
        self.mode = "full_sequence"
        self.base_mva = base_mva
        self.base_frequency = base_frequency
        self.fixed_window = fixed_window
        self.contract_mode = contract_mode
        self._norm_constants = {
            "INTERCHANGE_SCALE_MW": float(Settings.Interchange.INTERCHANGE_SCALE_MW),
        }
        self.scenario_files = []
        self.cascade_labels = []
        if not scenario.get("sequence"):
            raise ValueError("scenario has no sequence")
        return self._process_sequence_format(scenario)

    def __len__(self) -> int:
        return len(self.scenario_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and process a single scenario."""
        scenario_file = self.scenario_files[idx]

        try:
            with open(scenario_file, 'rb') as f:
                scenario_data = pickle.load(f)
            scenario = scenario_data[0] if isinstance(scenario_data, list) else scenario_data
            if not isinstance(scenario, dict):
                return {}

        except Exception as e:
            print(f"Error loading {scenario_file}: {e}. Returning empty dict.")
            return {}

        if self.contract_mode == "strict":
            _contract.require_current(scenario, scenario_file)   # DC-020

        # Process based on format
        if 'sequence' in scenario and 'metadata' in scenario:
            sequence = scenario['sequence']
            
            if len(sequence) == 0 and 'failed_nodes' in scenario['metadata']:
                return self._process_metadata_format(scenario)
            elif len(sequence) > 0:
                return self._process_sequence_format(scenario)
            else:
                return {}
        else:
            return {}
    
    def _process_sequence_format(self, scenario: Dict) -> Dict[str, Any]:
        """
        Process sequence format with normalization and truncation.
        
        This method:
        1. Applies sliding window truncation
        2. Normalizes power and frequency values
        3. Creates dynamic edge masks
        4. Extracts graph properties
        
        Args:
            scenario: Scenario dictionary with 'sequence' and 'metadata'
        
        Returns:
            Processed data dictionary
        """
        sequence_original = scenario['sequence']
        edge_index = scenario['edge_index']
        metadata = scenario.get('metadata', {})
        
        # Apply truncation
        cascade_start_time = metadata.get('cascade_start_time', -1)
        is_cascade = metadata.get('is_cascade', False)

        if self.fixed_window is not None:
            # Deterministic explicit window (streaming inference) — bypass
            # random truncation entirely.
            start_idx, end_idx = self.fixed_window
            start_idx = max(0, int(start_idx))
            end_idx = min(len(sequence_original), int(end_idx))
        else:
            start_idx, end_idx = calculate_truncation_window(
                len(sequence_original),
                cascade_start_time,
                is_cascade
            )

        sequence = apply_truncation(sequence_original, start_idx, end_idx)
        
        # Initialize data containers
        data_arrays = {
            'scada_data': [],
            'pmu_sequence': [],
            'equipment_status': [],
            'edge_mask': [],
            'edge_attr': [],       # per-timestep: cols 5-6 (line flows) change each step
            'ras_status': [],      # Phase 4: [N_RAS, 3] per timestep (RAS arming telemetry)
        }
        
        last_step = sequence[-1]
        
        num_nodes = last_step.get('scada_data', np.zeros((Settings.Dataset.DEFAULT_NUM_NODES, 12))).shape[0]
        num_edges = edge_index.shape[1]
        timing_shape = last_step.get('cascade_timing', np.zeros(num_nodes)).shape

        # Process each timestep. Column positions, legacy migration and the
        # per-column normalization policy all come from the data contract
        # (specs/data-contract, DC-021/DC-022) — no hard-coded indices here.
        C = _contract
        _masked_scada = [C.SCADA.idx(n) for n in C.MODEL_MASKED_SCADA]
        _e_thermal = C.EDGE_ATTR.idx("thermal_limit")
        _e_pflow = C.EDGE_ATTR.idx("p_flow")
        for i, ts in enumerate(sequence):
            if self.contract_mode == "migrate":
                ts = C.migrate_timestep(
                    ts, n_ras_schemes=Settings.RAS.N_SCHEMES,
                    context=os.path.basename(str(self.data_dir)))
            for key in (C.SCADA.key, C.PMU.key, C.EQUIPMENT.key, C.EDGE_ATTR.key, C.RAS_STATUS.key):
                if key not in ts:
                    raise C.ContractError(f"timestep {i} is missing '{key}'")

            # SCADA: normalize by policy, then hide absolute-time / scenario-
            # level columns from the model (time_ratio, stress_level).
            scada_data = C.normalize_array(
                C.SCADA, ts[C.SCADA.key], base_mva=self.base_mva,
                base_frequency=self.base_frequency, constants=self._norm_constants)
            scada_data[:, _masked_scada] = 0
            data_arrays['scada_data'].append(to_tensor(scada_data))

            pmu_data = C.normalize_array(
                C.PMU, ts[C.PMU.key], base_mva=self.base_mva,
                base_frequency=self.base_frequency, constants=self._norm_constants)
            data_arrays['pmu_sequence'].append(to_tensor(pmu_data))

            equip = C.normalize_array(
                C.EQUIPMENT, ts[C.EQUIPMENT.key], base_mva=self.base_mva,
                base_frequency=self.base_frequency, constants=self._norm_constants)
            data_arrays['equipment_status'].append(to_tensor(equip))

            # Per-timestep edge attributes (flows, branch status and taps are
            # dynamic, so they are collected every step).
            ea = C.normalize_array(
                C.EDGE_ATTR, ts[C.EDGE_ATTR.key], base_mva=self.base_mva,
                base_frequency=self.base_frequency, constants=self._norm_constants)
            data_arrays['edge_attr'].append(to_tensor(ea))

            data_arrays['ras_status'].append(
                to_tensor(np.asarray(ts[C.RAS_STATUS.key], dtype=np.float32)))

            # Edge stress mask: 1 - |P flow| / normal rating (both normalized
            # on the same base, so the ratio is scale-free).
            thermal = np.abs(ea[:, _e_thermal]) + 1e-6
            loading_ratio = np.abs(ea[:, _e_pflow]) / thermal
            edge_mask = to_tensor(np.clip(1.0 - loading_ratio, 0.0, 1.0).astype(np.float32))
            data_arrays['edge_mask'].append(edge_mask)
        
        # edge_attr is now collected per-timestep inside the loop above → [T, E, 7]
        edge_attr = torch.stack(data_arrays['edge_attr'])  # [T, E, 7]
        
        # Ground truth labels and timing
        final_labels = to_tensor(sequence_original[-1].get('node_labels', np.zeros(num_nodes)))
        
        # ── IMPROVED TIMING TARGETS (v2) ────────────────────────────────────
        # Root-cause fix: previous code stored raw shifted timestep indices
        # (e.g. 21, 23, 25) which produced a very large raw MSE (~8.4) and
        # caused dynamic calibration to shrink lambda_timing to ~0.013,
        # essentially killing the gradient signal.  The new approach:
        #
        #   target = (failure_timestep - start_idx) / total_original_seq_len
        #
        # This maps every target into [0, 1], matching the new Sigmoid output
        # of the improved TimingHead.  The normalisation denominator uses the
        # FULL original sequence length (not the truncated window length) so
        # that the same node always receives a consistent target regardless
        # of which sliding-window crop is sampled.
        #
        # At inference time: multiply the raw Sigmoid output by
        # Settings.Simulation.DEFAULT_SEQUENCE_LENGTH to recover the absolute
        # timestep, then multiply by ThermalConfig.DT_MINUTES (2.0) for minutes.
        # ─────────────────────────────────────────────────────────────────────
        original_cascade_start_time = metadata.get('cascade_start_time', -1)
        # Use per-scenario actual length as the normalization denominator.
        # Each scenario has its own number of timesteps (cascade scenarios may
        # resolve early, normal scenarios may run longer), so a fixed constant
        # like DEFAULT_SEQUENCE_LENGTH is incorrect.
        #
        # The key fix vs the previous version is removing the "- start_idx" shift:
        #   OLD (buggy): (failure_time - start_idx) / seq_len
        #     → target was window-relative, but inference decoded without adding
        #       back start_idx × DT_MINUTES, causing systematic early predictions.
        #   NEW (correct): failure_time / len(sequence_original)
        #     → target is ABSOLUTE (fraction of the actual scenario length).
        #     → inference decode = pred_normed × len(sequence_original) × DT_MINUTES.
        #
        # Example: failure at t=24, start_idx=3, seq_len=32 →
        #   old: (24-3)/32 = 0.656 → decoded 42 min  (error 6 min)
        #   new:    24/32  = 0.750 → decoded 48 min  ✓
        timing_norm_denom = float(max(len(sequence_original), 1))

        if is_cascade and 0 <= original_cascade_start_time < len(sequence_original):
            correct_timing_tensor = torch.full(timing_shape, -1.0, dtype=torch.float32)

            failed_nodes_list = metadata.get('failed_nodes', [])
            failure_times_list = metadata.get('failure_times', [])

            # Build boolean mask over nodes
            mask_failure = torch.zeros(num_nodes, dtype=torch.bool)
            if failed_nodes_list:
                mask_failure[failed_nodes_list] = True

            # Assign ABSOLUTE-NORMALISED timing for failing nodes.
            # failure_times_list contains absolute timestep indices from the
            # *original* (un-truncated) simulation.  Dividing by the fixed
            # sequence length gives a stable target in [0, 1] that is
            # identical regardless of which sliding-window crop is sampled.
            if failed_nodes_list:
                normalised = torch.clamp(
                    torch.tensor(failure_times_list, dtype=torch.float32) / timing_norm_denom,
                    0.0, 1.0
                )
                correct_timing_tensor[mask_failure] = normalised
        else:
            correct_timing_tensor = to_tensor(np.full(num_nodes, -1, dtype=np.float32))

        # Store normalisation denominator so inference can decode predictions
        # (attached to graph_properties for easy propagation to the model)
        _timing_norm_denom = timing_norm_denom

        # ── Parent labels for causal parent prediction head ───────────────────
        # parent_labels[i]:
        #   -1          → node i did not fail (masked out in loss)
        #   num_nodes   → node i is a trigger (no causal parent)
        #   j (0..N-1)  → node j caused node i to fail
        parent_labels = torch.full((num_nodes,), -1, dtype=torch.long)
        failed_nodes_list  = metadata.get('failed_nodes', [])
        failure_parents_list = scenario.get('failure_parents', [])
        if failed_nodes_list and failure_parents_list:
            for node, parent in zip(failed_nodes_list, failure_parents_list):
                node = int(node)
                if 0 <= node < num_nodes:
                    if parent is None:
                        parent_labels[node] = num_nodes   # trigger class
                    else:
                        p = int(parent)
                        if 0 <= p < num_nodes:
                            parent_labels[node] = p
        elif failed_nodes_list:
            # Older data without failure_parents: treat all as triggers
            for node in failed_nodes_list:
                node = int(node)
                if 0 <= node < num_nodes:
                    parent_labels[node] = num_nodes
        
        # ── node-feature tensor (layout: contract.NODE_FEATURE_BLOCKS) ─────────
        # Build (T, N, 115) from the already-collected per-timestep arrays so
        # the NodeFeatureMLP in the unified model can consume it directly.
        # Collect raw edge attributes per timestep (numpy arrays before tensor conversion)
        raw_edge_attrs = [
            np.asarray(seq_step[_contract.EDGE_ATTR.key], dtype=np.float32)
            for seq_step in sequence
        ]
        node_features = self._build_node_features(
            scada_list=data_arrays['scada_data'],       # list[T] of (N, SCADA.width) tensors
            pmu_list=data_arrays['pmu_sequence'],       # list[T] of (N, 8) tensors
            equip_list=data_arrays['equipment_status'], # list[T] of (N,10) tensors
            sequence=sequence,                          # raw dicts for injections
            num_nodes=num_nodes,
            start_idx=start_idx,                        # absolute window start for t_pos
            seq_len_for_norm=len(sequence_original),    # actual scenario length for t_pos normalisation
            edge_index=edge_index,                      # (2, E) for cascade susceptibility features
            edge_attr_list=raw_edge_attrs,              # list[T] of RAW (E, EDGE_ATTR.width) arrays
        )
        # Hide absolute-time features from the model BY NAME (spec D7). This
        # used to be `node_features[:,:,114]=0`, which targeted t_pos in the
        # original 38-wide base layout and silently drifted onto the
        # cap-bank delta2 column when SCADA was widened.
        for _blk in _contract.MODEL_MASKED_NODE_FEATURES:
            node_features[:, :, _contract.node_feature_slice(_blk)] = 0
        # Data augmentation (training only)
        is_training = 'train' in str(self.data_dir)
        scada_tensor = torch.stack(data_arrays['scada_data'])

        if is_training:
            noise = torch.randn_like(scada_tensor) * Settings.Dataset.AUGMENTATION_NOISE_STD
            scada_tensor = scada_tensor + noise

        return {
            'scada_data': scada_tensor,
            'pmu_sequence': torch.stack(data_arrays['pmu_sequence']),
            'equipment_status': torch.stack(data_arrays['equipment_status']),
            # node-feature tensor [T, N, contract.NODE_FEATURE_DIM]
            'node_features': node_features,
            'edge_index': to_tensor(edge_index).long(),
            'edge_attr': edge_attr,           # [T, E, 13] — per-timestep, dynamic flows included
            'edge_mask': torch.stack(data_arrays['edge_mask']),
            # Phase 4 — RAS arming telemetry [T, N_RAS, 3]
            'ras_status': torch.stack(data_arrays['ras_status']),
            'node_failure_labels': final_labels,
            'cascade_timing': correct_timing_tensor,
            'parent_labels': parent_labels,           # [N] long — causal parent per node
            'ground_truth_risk': to_tensor(metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))),
            'graph_properties': self._extract_graph_properties(last_step, metadata, edge_attr[-1]),
            'temporal_sequence': scada_tensor,
            'sequence_length': len(sequence),
            # Actual (un-truncated) scenario length — needed to decode normalised
            # timing predictions back to absolute minutes at inference time.
            'original_sequence_length': len(sequence_original),
        }
    
    @staticmethod
    def _build_node_features(
        scada_list:       List[torch.Tensor],
        pmu_list:         List[torch.Tensor],
        equip_list:       List[torch.Tensor],
        sequence:         List[Dict],
        num_nodes:        int,
        start_idx:        int = 0,
        seq_len_for_norm: int = 30,   # actual un-truncated scenario length
        edge_index:       Optional[np.ndarray] = None,   # (2, E) numpy int array
        edge_attr_list:   Optional[List[np.ndarray]] = None,  # list[T] of RAW edge_attr arrays
    ) -> torch.Tensor:
        """Build the (T, N, contract.NODE_FEATURE_DIM) node-feature tensor.

        Layout is defined by name in the data contract (NODE_BASE_PARTS and
        NODE_FEATURE_BLOCKS in cascade_prediction/contract.py; generated
        reference in specs/data-contract/data_contract.md):
          base            = [scada | pmu | equipment | power_inj | reactive_inj]
          delta1, delta2  = 1- and 2-step temporal differences of base
          t_pos           = absolute timestep position (masked by the caller)
          ttf             = 4 physics time-to-failure estimators
          susceptibility  = 5 cascade-susceptibility features

        Args:
            scada_list:     T tensors of shape (N, 22), already normalised.
            pmu_list:       T tensors of shape (N,  8), already normalised.
            equip_list:     T tensors of shape (N, 10), already normalised.
            sequence:       Raw per-timestep dicts (for power/reactive injection).
            num_nodes:      Number of nodes N.
            edge_index:     Graph connectivity (2, E) numpy int array.
            edge_attr_list: Per-timestep RAW edge attributes, list[T] of (E, width).

        Returns:
            Tensor of shape (T, N, contract.NODE_FEATURE_DIM), float32.
        """
        T = len(scada_list)

        # ── base features (T, N, 38) ─────────────────────────────────────────
        base_parts = []
        for t in range(T):
            ts = sequence[t]

            p_inj = ts.get('power_injection',    np.zeros((num_nodes, 1), dtype=np.float32))
            q_inj = ts.get('reactive_injection', np.zeros((num_nodes, 1), dtype=np.float32))

            p_inj = torch.from_numpy(
                np.array(p_inj, dtype=np.float32).reshape(num_nodes, 1))
            q_inj = torch.from_numpy(
                np.array(q_inj, dtype=np.float32).reshape(num_nodes, 1))

            base_t = torch.cat(
                [scada_list[t], pmu_list[t], equip_list[t], p_inj, q_inj],
                dim=1,
            )                                    # (N, 38)
            base_parts.append(base_t)

        base = torch.stack(base_parts, dim=0)    # (T, N, 38)

        # ── temporal deltas ───────────────────────────────────────────────────
        delta1 = torch.zeros_like(base)
        delta2 = torch.zeros_like(base)
        delta1[1:]  = base[1:]  - base[:-1]
        delta2[2:]  = base[2:]  - base[:-2]

        # ── timestep position (ABSOLUTE, per-scenario) ───────────────────────
        # FIX: Use absolute position (start_idx + step) / len(sequence_original)
        # rather than relative position [0, 1] within the truncated window.
        # Uses seq_len_for_norm (= len(sequence_original)) passed in from the
        # caller — NOT a fixed DEFAULT_SEQUENCE_LENGTH constant — because each
        # scenario has its own actual number of timesteps.
        T_total = float(max(seq_len_for_norm, 1))
        t_start_frac = start_idx / T_total
        t_end_frac   = (start_idx + T - 1) / T_total
        t_pos = torch.linspace(t_start_frac, t_end_frac, T)  # (T,) absolute
        t_pos = t_pos.view(T, 1, 1).expand(T, num_nodes, 1)  # (T, N, 1)

        # ── IMPROVED: physics-based time-to-failure (TTF) estimators ─────────
        # SCADA features relevant to failure proximity (from simulator.py):
        #   [14]: voltage_ratio     (> 1 = safe, < 1 = danger)
        #   [15]: temp_ratio        (< 1 = safe, > 1 = danger)
        #   [16]: freq_ratio        (> 1 = safe, < 1 = danger)
        #   [17]: loading_ratio     (< 1 = safe, > 1 = danger)
        #
        # For each ratio r and its 1-step velocity v = delta1[t, :, idx]:
        #   estimated_steps = (threshold - r) / v  (clamped to [0, 1] normalised)
        #
        # These explicit "estimated steps until threshold crossing" features give
        # the model a direct physics-based signal for WHEN failure will occur,
        # complementing the raw ratio values and their deltas.
        #
        # Feature layout adds 4 TTF channels → (T, N, 4):
        #   [0]: ttf_voltage  (steps to voltage < 1.0, normalised by T)
        #   [1]: ttf_temp     (steps to temp   > 1.0, normalised by T)
        #   [2]: ttf_freq     (steps to freq   < 1.0, normalised by T)
        #   [3]: ttf_loading  (steps to loading > 1.0, normalised by T)
        # ─────────────────────────────────────────────────────────────────────
        EPS = 1e-6
        T_float = float(max(T, 1))

        # Failure-proximity ratios, looked up by name in the base layout.
        _i_volt = _contract.base_index("scada", "voltage_ratio")
        _i_temp = _contract.base_index("scada", "temp_ratio")
        _i_freq = _contract.base_index("scada", "freq_ratio")
        _i_load = _contract.base_index("scada", "loading_ratio")
        r_volt    = base[:, :, _i_volt]   # (T, N) — safe >1
        r_temp    = base[:, :, _i_temp]   # (T, N) — safe <1
        r_freq    = base[:, :, _i_freq]   # (T, N) — safe >1
        r_loading = base[:, :, _i_load]   # (T, N) — safe <1

        # Velocities from 1-step delta (same indices)
        v_volt    = delta1[:, :, _i_volt]  # rate of change per step
        v_temp    = delta1[:, :, _i_temp]
        v_freq    = delta1[:, :, _i_freq]
        v_loading = delta1[:, :, _i_load]

        def ttf(current, threshold, velocity, safe_if_above: bool):
            """Estimated normalised steps to threshold crossing.
            Returns 1.0 (far future) if already safe or velocity is moving away.
            Returns 0.0 if already breached.
            """
            # Gap = remaining margin (positive = still safe)
            if safe_if_above:
                gap = current - threshold        # positive while safe
                approaching = (velocity < -EPS)  # decreasing toward threshold
            else:
                gap = threshold - current        # positive while safe
                approaching = (velocity > EPS)   # increasing toward threshold

            # Already breached
            breached = (gap <= 0)
            # Approaching: steps_left = gap / |velocity|
            abs_vel = velocity.abs().clamp(min=EPS)
            raw_steps = gap / abs_vel            # may be negative or huge

            # Clamp: 0 if breached, cap at T if not approaching / far away
            result = torch.where(breached,
                                 torch.zeros_like(gap),
                                 torch.where(approaching,
                                             raw_steps.clamp(0.0, T_float),
                                             torch.full_like(gap, T_float)))
            return (result / T_float).clamp(0.0, 1.0)   # normalise to [0,1]

        ttf_volt    = ttf(r_volt,    1.0, v_volt,    safe_if_above=True)   # (T,N)
        ttf_temp    = ttf(r_temp,    1.0, v_temp,    safe_if_above=False)
        ttf_freq    = ttf(r_freq,    1.0, v_freq,    safe_if_above=True)
        ttf_loading = ttf(r_loading, 1.0, v_loading, safe_if_above=False)

        ttf_features = torch.stack(
            [ttf_volt, ttf_temp, ttf_freq, ttf_loading], dim=2
        ).float()  # (T, N, 4)

        # ── Cascade susceptibility features (5 new channels) ─────────────────
        # These topology-informed features give the model pre-cascade signals
        # about which nodes are susceptible to cascade failure via propagation.
        # All features are computed from pre-cascade power flow data and topology.
        # ─────────────────────────────────────────────────────────────────────
        if edge_index is not None and edge_attr_list is not None and len(edge_attr_list) == T:
            src_nodes = edge_index[0]   # (E,)  — source of each edge
            dst_nodes = edge_index[1]   # (E,)  — destination of each edge

            # Pre-compute static adjacency lists: neighbors[i] = list of neighbor nodes
            neighbors = [[] for _ in range(num_nodes)]
            adj_edges  = [[] for _ in range(num_nodes)]  # adj_edges[i] = edge indices where dst==i
            for e_idx, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
                neighbors[d].append(s)
                adj_edges[d].append(e_idx)

            susc_parts = []
            for t in range(T):
                ea  = edge_attr_list[t]   # (E, edge width) RAW
                loading_ratio = base[t, :, _i_load].numpy()  # SCADA loading_ratio

                # |P flow| / normal rating — a ratio, so raw MW/MVA is fine.
                flows   = np.abs(ea[:, _contract.EDGE_ATTR.idx("p_flow")]).astype(np.float32)
                thermal = np.abs(ea[:, _contract.EDGE_ATTR.idx("thermal_limit")]).astype(np.float32) + 1e-6
                line_loading = np.clip(flows / thermal, 0.0, 2.0)     # |flow|/limit (E,)

                # Power injection magnitude (N,), by name. This read base col
                # 36 until the contract: after SCADA grew to 28 columns, 36 is
                # equipment_age, so F2 was computed from equipment age.
                p_inj = np.abs(base[t, :, _contract.base_index("power_injection")].numpy()).astype(np.float32)

                # 5 cascade susceptibility features per node:
                # F1: mean_adjacent_line_loading  (+0.074 discrimination, pos_frac=1.00)
                # F2: cascade_initiation_risk      (+0.028 discrimination)
                # F3: cascade_reception_risk       (+0.067 discrimination, pos_frac=1.00)
                # F4: max_adjacent_line_loading    (+0.232 discrimination, pos_frac=1.00) ★ best
                # F5: loading_ratio × f4 (interact)(+0.219 discrimination, pos_frac=1.00)
                f1 = np.zeros(num_nodes, dtype=np.float32)  # mean_adj_line_loading
                f2 = np.zeros(num_nodes, dtype=np.float32)  # cascade_initiation_risk
                f3 = np.zeros(num_nodes, dtype=np.float32)  # cascade_reception_risk
                f4 = np.zeros(num_nodes, dtype=np.float32)  # max_adj_line_loading

                for i in range(num_nodes):
                    nbrs   = neighbors[i]
                    e_idxs = adj_edges[i]

                    if e_idxs:
                        # F1: mean line loading across adjacent edges
                        f1[i] = float(np.mean([line_loading[e] for e in e_idxs]))

                        # F4: max line loading across adjacent edges
                        f4[i] = float(max(line_loading[e] for e in e_idxs))

                        # F2: cascade initiation risk = |P_inj[i]| / sum(thermal limits)
                        sum_thermal = float(sum(thermal[e] for e in e_idxs))
                        f2[i] = float(p_inj[i]) / (sum_thermal + 1e-6)

                        # F3: cascade reception risk — weighted line stress from neighbors
                        #   For each neighbor j→i: line_loading[j→i] × loading_ratio[j]
                        reception = float(sum(line_loading[e] * float(loading_ratio[nbr])
                                              for nbr, e in zip(nbrs, e_idxs)))
                        f3[i] = reception / float(len(e_idxs))

                # Clip to reasonable range
                f2 = np.clip(f2, 0.0, 2.0)
                f3 = np.clip(f3, 0.0, 2.0)

                # F5: interaction term — internal loading × topological exposure
                f5 = np.clip(loading_ratio.astype(np.float32) * f4, 0.0, 2.0)

                susc_t = torch.from_numpy(
                    np.stack([f1, f2, f3, f4, f5], axis=1)
                ).float()  # (N, 5)
                susc_parts.append(susc_t)

            cascade_susceptibility = torch.stack(susc_parts, dim=0)  # (T, N, 5)
        else:
            # Fallback: zero features when edge data is unavailable
            cascade_susceptibility = torch.zeros(T, num_nodes, 5, dtype=torch.float32)

        # Final tensor: 115 + 4 + 5 = 124 features
        return torch.cat(
            [base, delta1, delta2, t_pos, ttf_features, cascade_susceptibility], dim=2
        ).float()  # (T, N, contract.NODE_FEATURE_DIM)

    def _process_metadata_format(self, scenario: Dict) -> Dict[str, Any]:
        """
        Process metadata-only format (fallback for empty sequences).
        
        Args:
            scenario: Scenario dictionary with metadata but no sequence
        
        Returns:
            Processed data dictionary with synthetic data
        """
        # Constitution rule 10: this path fabricates random SCADA/PMU/edge
        # arrays (with a 13-column SCADA that matches no layout). It must never
        # feed training. Non-training consumers keep it for backwards
        # compatibility only.
        if self.contract_mode == "strict":
            raise _contract.ContractError(
                "scenario has an empty sequence; the synthetic metadata-only "
                "fallback is not allowed in strict (training) mode")
        metadata = scenario['metadata']
        edge_index = scenario['edge_index']

        num_nodes = metadata.get('num_nodes', Settings.Dataset.DEFAULT_NUM_NODES)
        num_edges = metadata.get('num_edges', edge_index.shape[1] if hasattr(edge_index, 'shape') else 186)
        
        # Create node failure labels
        node_failure_labels = np.zeros(num_nodes, dtype=np.float32)
        if 'failed_nodes' in metadata and len(metadata['failed_nodes']) > 0:
            for node_idx in metadata['failed_nodes']:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        node_failure_labels[node_idx_int] = 1.0
                except (ValueError, TypeError):
                    continue
        
        # Create synthetic data
        T = 1
        scada_data = torch.randn(T, num_nodes, 13)
        pmu_sequence = torch.randn(T, num_nodes, 8)
        equipment_status = torch.randn(T, num_nodes, 10)
        edge_attr = torch.randn(num_edges, Settings.Model.EDGE_FEATURES)
        edge_mask = torch.ones(T, num_edges)
        
        # Apply normalization
        edge_attr[:, 1] = normalize_power(edge_attr[:, 1], self.base_mva)
        scada_data[..., 2:6] = normalize_power(scada_data[..., 2:6], self.base_mva)
        pmu_sequence[..., 5] = normalize_frequency(pmu_sequence[..., 5], self.base_frequency)
        
        graph_props = self._extract_graph_properties_from_metadata(metadata, num_edges)
        ground_truth_risk = metadata.get('ground_truth_risk', np.zeros(7, dtype=np.float32))
        
        item = {
            'scada_data': scada_data[0],
            'pmu_sequence': pmu_sequence[0],
            'equipment_status': equipment_status[0],
            'edge_index': to_tensor(edge_index).long(),
            'edge_attr': edge_attr,
            'edge_mask': edge_mask,
            'node_failure_labels': to_tensor(node_failure_labels),
            'cascade_timing': torch.zeros(num_nodes),
            'ground_truth_risk': to_tensor(ground_truth_risk),
            'graph_properties': graph_props
        }
        
        if self.mode == 'full_sequence':
            for key in ['scada_data', 'pmu_sequence', 'equipment_status', 'edge_mask']:
                item[key] = item[key].unsqueeze(0)
            item['temporal_sequence'] = item['scada_data']
            item['sequence_length'] = 1
        
        return item
    
    def _extract_graph_properties(
        self,
        timestep_data: Dict,
        metadata: Dict,
        edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract graph properties from timestep data."""
        graph_props = {}
        
        # Conductance and susceptance
        if 'conductance' in timestep_data:
            graph_props['conductance'] = torch.from_numpy(timestep_data['conductance']).float()
        else:
            graph_props['conductance'] = edge_attr[:, _contract.EDGE_ATTR.idx("conductance")]
        
        if 'susceptance' in timestep_data:
            graph_props['susceptance'] = torch.from_numpy(timestep_data['susceptance']).float()
        else:
            graph_props['susceptance'] = edge_attr[:, _contract.EDGE_ATTR.idx("susceptance")]
        
        graph_props['thermal_limits'] = edge_attr[:, _contract.EDGE_ATTR.idx("thermal_limit")]
        
        # Power injection
        if 'power_injection' in timestep_data:
            power_injection_raw = torch.from_numpy(timestep_data['power_injection']).float()
            graph_props['power_injection'] = normalize_power(power_injection_raw, self.base_mva)
        else:
            # Fallback from RAW SCADA, normalized like the primary path.
            scada = timestep_data.get('scada_data', None)
            if scada is not None:
                gen = scada[:, _contract.SCADA.idx("generation")]
                load = scada[:, _contract.SCADA.idx("load")]
                graph_props['power_injection'] = normalize_power(
                    torch.as_tensor(gen - load, dtype=torch.float32), self.base_mva)
        
        # Reactive injection
        if 'reactive_injection' in timestep_data:
            reactive_injection_raw = torch.from_numpy(timestep_data['reactive_injection']).float()
            graph_props['reactive_injection'] = normalize_power(reactive_injection_raw, self.base_mva)
        else:
            # Fallback: the generator writes reactive_injection = node_reactive.
            # (Pre-contract this subtracted SCADA col 5, equipment temperature.)
            scada = timestep_data.get('scada_data', None)
            if scada is not None:
                graph_props['reactive_injection'] = normalize_power(
                    torch.as_tensor(scada[:, _contract.SCADA.idx("node_reactive")], dtype=torch.float32),
                    self.base_mva)
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        # Ground truth temperature (index 5 in scada_data)
        # CRITICAL: scada_data[:, 5] is equipment_temps, NOT frequency!
        # scada_data[:, 6] is frequency (around 60 Hz) - DO NOT USE for temperature!
        if 'scada_data' in timestep_data:
            scada_data = timestep_data['scada_data']
            if scada_data.shape[1] > _contract.SCADA.idx("equipment_temp"):
                graph_props['ground_truth_temperature'] = torch.from_numpy(
                    np.asarray(scada_data[:, _contract.SCADA.idx("equipment_temp")])).float()
            else:
                graph_props['ground_truth_temperature'] = torch.zeros(scada_data.shape[0])
        
        return graph_props
    
    def _extract_graph_properties_from_metadata(
        self,
        metadata: Dict,
        num_edges: int
    ) -> Dict[str, torch.Tensor]:
        """Extract graph properties from metadata (synthetic)."""
        graph_props = {}
        num_nodes = metadata.get('num_nodes', Settings.Dataset.DEFAULT_NUM_NODES)
        
        # Thermal limits
        thermal_limits_raw = torch.rand(num_edges) * 40.0 + 10.0
        graph_props['thermal_limits'] = normalize_power(thermal_limits_raw, self.base_mva)
        
        # Conductance and susceptance
        reactance = torch.rand(num_edges) * 0.3 + 0.05
        resistance = reactance * 0.1
        impedance_sq = resistance**2 + reactance**2
        graph_props['conductance'] = resistance / impedance_sq
        graph_props['susceptance'] = -reactance / impedance_sq
        
        # Power injection
        is_cascade = metadata.get('is_cascade', False)
        failed_nodes = metadata.get('failed_nodes', [])
        
        if is_cascade and len(failed_nodes) > 0:
            power_injection_raw = torch.randn(num_nodes) * 50.0
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        power_injection_raw[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            power_injection_raw = torch.randn(num_nodes) * 5.0
        
        graph_props['power_injection'] = normalize_power(power_injection_raw, self.base_mva)
        
        # Reactive injection
        if is_cascade and len(failed_nodes) > 0:
            reactive_injection_raw = torch.randn(num_nodes) * 30.0
            for node_idx in failed_nodes:
                try:
                    node_idx_int = int(node_idx)
                    if 0 <= node_idx_int < num_nodes:
                        reactive_injection_raw[node_idx_int] = 0.0
                except (ValueError, TypeError):
                    continue
        else:
            reactive_injection_raw = torch.randn(num_nodes) * 3.0
        
        graph_props['reactive_injection'] = normalize_power(reactive_injection_raw, self.base_mva)
        
        if 'base_mva' in metadata:
            graph_props['base_mva'] = torch.tensor(metadata['base_mva'])
        
        return graph_props
    
    def get_cascade_label(self, idx: int) -> bool:
        """Get cascade label without loading full data."""
        return self.cascade_labels[idx]
