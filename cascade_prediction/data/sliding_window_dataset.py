"""
Sliding Window Dataset for GridStateForecaster
===============================================
Each scenario of length T produces T - window_size training samples.

Sample at index i returns window_size + 1 consecutive steps:
  steps [0 : window_size]  → model input  (passed as batch[:, :-1])
  step  [window_size]      → prediction target (extract_next_step_targets uses [:, -1])

All node_features are computed on-the-fly from the infrastructure tensors —
no separate pre-computed files required.
"""

import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .generator.config import Settings, NodeConfig
from .preprocessing import normalize_power, normalize_frequency, to_tensor

_TEMP_NORM = NodeConfig.TEMP_FAILURE_MAX_C   # 130.0 °C — normalises temperature to ~[0, 1]

WINDOW_SIZE = 10  # number of input timesteps per sample (returns WINDOW_SIZE + 1 total)

# Keys in the returned dict that have a leading time dimension
TEMPORAL_KEYS = {
    'scada_data', 'pmu_sequence', 'equipment_status',
    'node_features', 'edge_attr', 'edge_mask', 'temporal_sequence',
}


class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset for GridStateForecaster training.

    Loads each scenario file once, caches the processed tensors in memory, then
    generates (window_size + 1)-step slices via a flat index.

    Output keys per sample:
        scada_data       [W+1, N, 18]
        pmu_sequence     [W+1, N,  8]
        equipment_status [W+1, N, 10]
        node_features    [W+1, N, 119]  — computed on-the-fly
        edge_index       [2,   E]
        edge_attr        [W+1, E,  7]
        edge_mask        [W+1, E]
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        base_mva: float = Settings.Dataset.BASE_MVA,
        base_frequency: float = Settings.Dataset.BASE_FREQUENCY,
    ):
        self.window_size   = window_size
        self.base_mva      = base_mva
        self.base_frequency = base_frequency
        self.is_training   = 'train' in str(data_dir)

        files = sorted(glob.glob(str(Path(data_dir) / 'scenario_*.pkl')))
        if not files:
            files = sorted(glob.glob(str(Path(data_dir) / 'scenarios_batch_*.pkl')))

        self._cascade_labels: List[bool] = []
        self._cache: Dict[int, Dict[str, object]] = {}  # file_idx → tensors dict
        self._index: List[Tuple[int, int]] = []          # (file_idx, start_t)

        print(f'SlidingWindowDataset: loading {len(files)} files from {data_dir}')
        cache_idx = 0
        for path in files:
            tensors, label = self._load(path)
            if tensors is None:
                continue
            T = tensors['scada'].shape[0]
            n_windows = T - window_size
            if n_windows <= 0:
                continue
            self._cascade_labels.append(label)
            self._cache[cache_idx] = tensors
            for start_t in range(n_windows):
                self._index.append((cache_idx, start_t))
            cache_idx += 1

        total = len(self._cascade_labels)
        pos   = sum(self._cascade_labels)
        print(f'  {len(self._index)} windows from {total} scenarios '
              f'({pos} cascade / {total - pos} normal)')

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_t = self._index[idx]
        t = self._cache[file_idx]
        s, e = start_t, start_t + self.window_size + 1

        scada = t['scada'][s:e].clone()   # clone so noise doesn't corrupt cache
        pmu   = t['pmu'][s:e]
        equip = t['equip'][s:e]
        p_inj = t['p_inj'][s:e]
        q_inj = t['q_inj'][s:e]

        if self.is_training:
            scada += torch.randn_like(scada) * Settings.Dataset.AUGMENTATION_NOISE_STD

        node_features = _build_node_features(
            scada, pmu, equip, p_inj, q_inj,
            start_t=start_t,
            seq_len=t['seq_len'],
        )

        return {
            'scada_data':       scada,
            'pmu_sequence':     pmu,
            'equipment_status': equip,
            'node_features':    node_features,
            'edge_index':       t['edge_index'],
            'edge_attr':        t['edge_attr'][s:e],
            'edge_mask':        t['edge_mask'][s:e],
        }

    def get_cascade_label(self, idx: int) -> bool:
        """Return the cascade label of the scenario this window belongs to."""
        file_idx, _ = self._index[idx]
        return self._cascade_labels[file_idx]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, path: str) -> Tuple[Optional[Dict], bool]:
        """Load and normalise one pkl file. Returns (tensors, cascade_label)."""
        try:
            with open(path, 'rb') as f:
                raw = pickle.load(f)
        except Exception:
            return None, False

        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if not isinstance(raw, dict) or not raw.get('sequence'):
            return None, False

        sequence   = raw['sequence']
        edge_index = raw['edge_index']
        metadata   = raw.get('metadata', {})
        T          = len(sequence)

        last_step = sequence[-1]
        num_nodes = last_step.get(
            'scada_data', np.zeros((Settings.Dataset.DEFAULT_NUM_NODES, 18))
        ).shape[0]
        num_edges = edge_index.shape[1]

        scada_l, pmu_l, equip_l = [], [], []
        p_inj_l, q_inj_l       = [], []
        ea_l, mask_l            = [], []

        for ts in sequence:
            # SCADA
            s = ts.get('scada_data', np.zeros((num_nodes, 18), dtype=np.float32)).astype(np.float32)
            if s.shape[1] >= 7:
                s[:, 2] = normalize_power(s[:, 2], self.base_mva)                        # generation
                s[:, 3] = normalize_power(s[:, 3], self.base_mva)                        # reactive
                s[:, 4] = normalize_power(s[:, 4], self.base_mva)                        # load
                s[:, 5] = s[:, 5] / _TEMP_NORM                     # temperature °C
                s[:, 6] = normalize_frequency(s[:, 6], self.base_frequency)              # frequency Hz
            if s.shape[1] > 13:
                s[:, 12] = 0.0   # zero time_ratio  (consistent with CascadeDataset)
                s[:, 13] = 0.0   # zero stress_level
            scada_l.append(to_tensor(s))

            # PMU: [0]=voltage_pu [1]=angle [2]=generation [3]=load [4]=temp
            #      [5]=frequency  [6]=loading_ratio [7]=node_reactive
            p = ts.get('pmu_sequence', np.zeros((num_nodes, 8), dtype=np.float32)).astype(np.float32)
            if p.shape[1] >= 8:
                p[:, 2] = normalize_power(p[:, 2], self.base_mva)                        # generation MW
                p[:, 3] = normalize_power(p[:, 3], self.base_mva)                        # load MW
                p[:, 4] = p[:, 4] / _TEMP_NORM                     # temperature °C
                p[:, 5] = normalize_frequency(p[:, 5], self.base_frequency)              # frequency Hz
                p[:, 7] = normalize_power(p[:, 7], self.base_mva)                        # reactive MVAr
            pmu_l.append(to_tensor(p))

            eq = ts.get('equipment_status', np.zeros((num_nodes, 10), dtype=np.float32)).astype(np.float32)
            if eq.shape[1] > 2:
                eq[:, 2] = eq[:, 2] / _TEMP_NORM   # equipment_temps °C (EQUIP_VAR_IDX[0])
            equip_l.append(to_tensor(eq))

            # Power / reactive injections (node-level physics features)
            p_inj_l.append(torch.from_numpy(
                np.array(ts.get('power_injection',    np.zeros((num_nodes, 1), dtype=np.float32)),
                         dtype=np.float32).reshape(num_nodes, 1)
            ))
            q_inj_l.append(torch.from_numpy(
                np.array(ts.get('reactive_injection', np.zeros((num_nodes, 1), dtype=np.float32)),
                         dtype=np.float32).reshape(num_nodes, 1)
            ))

            # Edge attributes
            ea = ts.get('edge_attr', np.zeros((num_edges, 7), dtype=np.float32)).astype(np.float32)
            if ea.shape[1] >= 7:
                ea[:, 1] = normalize_power(ea[:, 1], self.base_mva)
                ea[:, 5] = normalize_power(ea[:, 5], self.base_mva)
                ea[:, 6] = normalize_power(ea[:, 6], self.base_mva)
            ea_l.append(to_tensor(ea))

            # Edge stress mask
            if ea.shape[1] >= 6:
                thermal      = np.abs(ea[:, 1]) + 1e-6
                loading      = np.abs(ea[:, 5]) / thermal
                mask_l.append(to_tensor(np.clip(1.0 - loading, 0.0, 1.0).astype(np.float32)))
            else:
                mask_l.append(torch.ones(num_edges, dtype=torch.float32))

        cascade_label = bool(metadata.get('is_cascade', False))

        tensors = {
            'scada':      torch.stack(scada_l),           # [T, N, 18]
            'pmu':        torch.stack(pmu_l),             # [T, N,  8]
            'equip':      torch.stack(equip_l),           # [T, N, 10]
            'p_inj':      torch.stack(p_inj_l),           # [T, N,  1]
            'q_inj':      torch.stack(q_inj_l),           # [T, N,  1]
            'edge_index': to_tensor(edge_index).long(),   # [2,  E]
            'edge_attr':  torch.stack(ea_l),              # [T,  E, 7]
            'edge_mask':  torch.stack(mask_l),            # [T,  E]
            'seq_len':    T,
        }
        return tensors, cascade_label


# ---------------------------------------------------------------------------
# Node feature builder (module-level so it can be reused at inference)
# ---------------------------------------------------------------------------

def _build_node_features(
    scada: torch.Tensor,   # [T, N, 18]
    pmu:   torch.Tensor,   # [T, N,  8]
    equip: torch.Tensor,   # [T, N, 10]
    p_inj: torch.Tensor,   # [T, N,  1]
    q_inj: torch.Tensor,   # [T, N,  1]
    start_t: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Build [T, N, 119] node-feature tensor from raw infrastructure tensors.

    Layout mirrors CascadeDataset._build_node_features:
      [0:38]   base (scada + pmu + equip + p_inj + q_inj)
      [38:76]  1-step temporal deltas
      [76:114] 2-step temporal deltas
      [114]    normalised absolute timestep position
      [115:119] TTF physics features (voltage, temp, freq, loading)
    """
    T, N, _ = scada.shape

    # Base [T, N, 38]
    base   = torch.cat([scada, pmu, equip, p_inj, q_inj], dim=2)
    delta1 = torch.zeros_like(base)
    delta2 = torch.zeros_like(base)
    delta1[1:] = base[1:] - base[:-1]
    delta2[2:] = base[2:] - base[:-2]

    # Absolute timestep position
    T_total = float(max(seq_len, 1))
    t_pos = torch.linspace(
        start_t / T_total,
        (start_t + T - 1) / T_total,
        T,
    ).view(T, 1, 1).expand(T, N, 1)

    # TTF features — estimated normalised steps to each failure threshold
    EPS = 1e-6
    T_f = float(max(T, 1))

    def ttf(cur, thr, vel, safe_above: bool) -> torch.Tensor:
        gap         = (cur - thr) if safe_above else (thr - cur)
        approaching = (vel < -EPS) if safe_above else (vel > EPS)
        raw_steps   = gap / vel.abs().clamp(min=EPS)
        result      = torch.where(
            gap <= 0,
            torch.zeros_like(gap),
            torch.where(approaching, raw_steps.clamp(0.0, T_f), torch.full_like(gap, T_f)),
        )
        return (result / T_f).clamp(0.0, 1.0)

    # SCADA ratio columns (indices 14-17 in base, same position as in CascadeDataset)
    r_volt, r_temp = base[:, :, 14], base[:, :, 15]
    r_freq, r_load = base[:, :, 16], base[:, :, 17]
    v_volt, v_temp = delta1[:, :, 14], delta1[:, :, 15]
    v_freq, v_load = delta1[:, :, 16], delta1[:, :, 17]

    ttf_feats = torch.stack([
        ttf(r_volt, 1.0, v_volt, True),
        ttf(r_temp, 1.0, v_temp, False),
        ttf(r_freq, 1.0, v_freq, True),
        ttf(r_load, 1.0, v_load, False),
    ], dim=2)   # [T, N, 4]

    return torch.cat([base, delta1, delta2, t_pos, ttf_feats], dim=2).float()  # [T, N, 119]
