"""
Scenario Data Contract
======================
Single source of truth for the layout of every array a generated scenario
carries, how each column is normalized, what value range it is expected to
stay in, and how the model's node-feature vector is assembled from them.

Spec: ``specs/data-contract/spec.md`` (acceptance criteria DC-xxx).

Rules (constitution §2):
  * Consumers look columns up BY NAME (``SCADA.idx("frequency")``), never by
    a hard-coded number.
  * This module imports nothing from ``cascade_prediction`` so that
    ``config.py`` can derive its widths from it without an import cycle.
  * Changing any array layout, column meaning or normalization is a MAJOR
    ``SCHEMA_VERSION`` bump and changes ``LAYOUT_HASH``; checkpoints stamped
    with another hash refuse to load.

CLI::

    python -m cascade_prediction.contract export   <out_dir>
    python -m cascade_prediction.contract validate <data_dir> [--fail-on-ranges]
    python -m cascade_prediction.contract certify  <data_dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_VERSION = "1.0.0"

log = logging.getLogger("cascade_prediction.contract")


class ContractError(ValueError):
    """Raised when data or an artifact violates the data contract."""


# ---------------------------------------------------------------------------
# Normalization policies
# ---------------------------------------------------------------------------
# A policy is a small tuple so it hashes/serializes deterministically:
#   ("raw",)                      leave as generated
#   ("base_mva",)                 x / base_mva          (MW / MVAr / MVA -> p.u.)
#   ("base_frequency",)           x / base_frequency    (Hz -> p.u.)
#   ("divide", <float | name>)    x / constant; a str names a constant the
#                                 caller supplies (e.g. INTERCHANGE_SCALE_MW)
RAW: Tuple = ("raw",)
BASE_MVA: Tuple = ("base_mva",)
BASE_FREQUENCY: Tuple = ("base_frequency",)


def divide(constant: Union[float, str]) -> Tuple:
    return ("divide", constant)


Range = Optional[Tuple[Optional[float], Optional[float]]]


@dataclass(frozen=True)
class Column:
    name: str
    unit: str
    description: str
    norm: Tuple = RAW
    valid_range: Range = None                 # expected raw-value range (inclusive)
    categories: Optional[Tuple[float, ...]] = None  # allowed discrete values
    scope: str = "local"                      # "local" | "broadcast" (same on every row)
    since: str = "base"                       # generator phase that introduced it
    spec_ref: str = ""                        # external requirement reference


@dataclass(frozen=True)
class ArraySpec:
    """A 2-D per-timestep array: rows x ordered named columns."""
    key: str
    rows: str                                  # "N" nodes | "E" edges | "S" RAS schemes
    columns: Tuple[Column, ...]
    description: str = ""
    legacy_widths: Tuple[int, ...] = ()        # older layouts migrate_array() can upgrade
    _index: Dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ContractError(f"{self.key}: duplicate column names")
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(names)})

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.columns]

    def idx(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise ContractError(f"{self.key} has no column '{name}'. "
                                f"Known: {', '.join(self.names)}") from None

    def column(self, name: str) -> Column:
        return self.columns[self.idx(name)]

    def span(self, first: str, last: str) -> slice:
        """Inclusive name range as a slice, e.g. span('actual_interchange', 'ace')."""
        return slice(self.idx(first), self.idx(last) + 1)


@dataclass(frozen=True)
class VectorSpec:
    """A 1-D per-timestep vector with one entry per row."""
    key: str
    rows: str
    unit: str
    description: str
    valid_range: Range = None
    categories: Optional[Tuple[float, ...]] = None


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------
_MISO = "MISO RTO-SPEC-006"

SCADA = ArraySpec(
    key="scada_data", rows="N", legacy_widths=(18, 22),
    description="Per-node SCADA measurements. Broadcast columns hold one "
                "system-wide value repeated on every node.",
    columns=(
        Column("voltage", "p.u.", "Bus voltage magnitude (0 = de-energized)", RAW, (0.0, 1.5)),
        Column("angle", "rad", "Bus voltage angle", RAW, (-math.pi, math.pi)),
        Column("generation", "MW", "Active generation incl. storage injection", BASE_MVA, (-1000.0, 20000.0)),
        Column("node_reactive", "MVAr", "Net reactive power at the bus", BASE_MVA, (-10000.0, 10000.0)),
        Column("load", "MW", "Active load", BASE_MVA, (0.0, 10000.0)),
        Column("equipment_temp", "degC", "Equipment temperature (physics temperature target)", RAW, (-40.0, 250.0)),
        Column("frequency", "Hz", "System frequency (raw Hz; not normalized in SCADA)", RAW, (57.0, 61.8),
               scope="broadcast"),
        Column("equipment_age", "years", "Equipment age", RAW, (0.0, 100.0)),
        Column("equipment_condition", "fraction", "Condition index, 1 = new", RAW, (0.0, 1.0)),
        Column("gen_capacity", "MW", "Installed generation capacity", RAW, (0.0, 50000.0)),
        Column("base_load", "MW", "Nominal (base) load", RAW, (0.0, 10000.0)),
        Column("node_type", "category", "0 = load, 1 = generator, 2 = substation", RAW,
               categories=(0.0, 1.0, 2.0)),
        Column("time_ratio", "fraction", "t / sequence_length (masked in model input)", RAW, (0.0, 1.0),
               scope="broadcast"),
        Column("stress_level", "fraction", "Scenario stress level (masked in model input)", RAW, (0.0, 1.0),
               scope="broadcast"),
        Column("voltage_ratio", "ratio", "voltage / voltage_failure_threshold (<1 = danger)", RAW, (0.0, 2.0)),
        Column("temp_ratio", "ratio", "temp / temperature_failure_threshold (>1 = danger)", RAW, (0.0, 3.0)),
        Column("freq_ratio", "ratio", "frequency / frequency_failure_threshold (<1 = danger)", RAW, (0.9, 1.1)),
        Column("loading_ratio", "ratio", "load / base_load / loading_failure_threshold (>1 = danger)", RAW,
               (0.0, 5.0)),
        Column("cap_bank_status", "binary", "Capacitor bank in service", RAW, categories=(0.0, 1.0),
               since="phase3", spec_ref=f"{_MISO}: voltage-regulation equipment status"),
        Column("reactor_status", "binary", "Shunt reactor in service", RAW, categories=(0.0, 1.0),
               since="phase3", spec_ref=f"{_MISO}: voltage-regulation equipment status"),
        Column("svc_q", "MVAr", "SVC reactive output", BASE_MVA, (-1000.0, 1000.0),
               since="phase3", spec_ref=f"{_MISO}: voltage-regulation equipment status"),
        Column("avr_setpoint", "p.u.", "AVR voltage setpoint (0 on non-generator buses)", RAW, (0.0, 1.2),
               since="phase3", spec_ref=f"{_MISO}: voltage-regulation equipment status"),
        Column("storage_soc", "fraction", "Storage state of charge (0 on non-storage buses)", RAW, (0.0, 1.0),
               since="phase5", spec_ref=f"{_MISO}: energy storage resources"),
        Column("storage_mw", "MW", "Storage active power, + = discharging", BASE_MVA, (-500.0, 500.0),
               since="phase5", spec_ref=f"{_MISO}: energy storage resources"),
        Column("storage_mvar", "MVAr", "Storage reactive power", BASE_MVA, (-500.0, 500.0),
               since="phase5", spec_ref=f"{_MISO}: energy storage resources"),
        Column("actual_interchange", "MW", "Net actual interchange (sum of signed tie flows)",
               divide("INTERCHANGE_SCALE_MW"), (-20000.0, 20000.0), scope="broadcast",
               since="phase5", spec_ref=f"{_MISO}: tie-line flows & net actual interchange"),
        Column("scheduled_interchange", "MW", "Scheduled interchange",
               divide("INTERCHANGE_SCALE_MW"), (-20000.0, 20000.0), scope="broadcast",
               since="phase5", spec_ref=f"{_MISO}: dynamic schedules"),
        Column("ace", "MW", "Area control error = schedule - actual",
               divide("INTERCHANGE_SCALE_MW"), (-20000.0, 20000.0), scope="broadcast",
               since="phase5", spec_ref=f"{_MISO}: tie-line flows & net actual interchange"),
    ),
)

PMU = ArraySpec(
    key="pmu_sequence", rows="N",
    description="Per-node PMU measurements.",
    columns=(
        Column("voltage", "p.u.", "Bus voltage magnitude", RAW, (0.0, 1.5)),
        Column("angle", "rad", "Bus voltage angle", RAW, (-math.pi, math.pi)),
        Column("generation", "MW", "Active generation", RAW, (-1000.0, 20000.0)),
        Column("load", "MW", "Active load", RAW, (0.0, 10000.0)),
        Column("equipment_temp", "degC", "Equipment temperature", RAW, (-40.0, 250.0)),
        Column("frequency", "Hz", "System frequency", BASE_FREQUENCY, (57.0, 61.8), scope="broadcast"),
        Column("mean_line_loading", "ratio", "Mean line loading ratio across the grid", RAW, (0.0, 5.0),
               scope="broadcast"),
        Column("node_reactive", "MVAr", "Net reactive power at the bus", RAW, (-10000.0, 10000.0)),
    ),
)

EQUIPMENT = ArraySpec(
    key="equipment_status", rows="N",
    description="Per-node equipment health and thermal state.",
    columns=(
        Column("equipment_age", "years", "Equipment age", RAW, (0.0, 100.0)),
        Column("equipment_condition", "fraction", "Condition index", RAW, (0.0, 1.0)),
        Column("equipment_temp", "degC", "Equipment temperature", RAW, (-40.0, 250.0)),
        Column("thermal_capacity", "ratio", "Thermal capacity factor", RAW, (0.0, 5.0)),
        Column("cooling_effectiveness", "fraction", "Cooling effectiveness", RAW, (0.0, 1.0)),
        Column("thermal_time_constant", "x30 min", "Thermal time constant / 30", RAW, (0.0, 10.0)),
        Column("temp_ratio", "ratio", "temp / temperature_failure_threshold", RAW, (0.0, 3.0)),
        Column("node_type", "category", "0 = load, 1 = generator, 2 = substation", RAW,
               categories=(0.0, 1.0, 2.0)),
        Column("gen_capacity_fraction", "fraction", "gen_capacity / max gen_capacity", RAW, (0.0, 1.0)),
        Column("load_ratio", "ratio", "load / base_load", RAW, (0.0, 5.0)),
    ),
)

EDGE_ATTR = ArraySpec(
    key="edge_attr", rows="E", legacy_widths=(7, 10, 13),
    description="Per-branch attributes. Mirrored rows of one undirected branch share values.",
    columns=(
        Column("reactance", "ohm", "Series reactance", RAW, (0.0, 1000.0)),
        Column("thermal_limit", "MVA", "Normal rating, derated for the step", BASE_MVA, (0.0, 100000.0)),
        Column("resistance", "ohm", "Series resistance", RAW, (0.0, 1000.0)),
        Column("susceptance", "S", "Shunt susceptance", RAW, (-1000.0, 1000.0)),
        Column("conductance", "S", "Shunt conductance", RAW, (-1000.0, 1000.0)),
        Column("p_flow", "MW", "Active power flow (physics flow target)", BASE_MVA, (-100000.0, 100000.0)),
        Column("q_flow", "MVAr", "Reactive power flow", BASE_MVA, (-100000.0, 100000.0)),
        Column("branch_status", "category", "1 = closed, 0 = forced open, -1 = planned open", RAW,
               categories=(-1.0, 0.0, 1.0), since="phase1",
               spec_ref=f"{_MISO}: switching-device open/closed status"),
        Column("emergency_limit", "MVA", "Emergency (LEP) rating, derated", BASE_MVA, (0.0, 100000.0),
               since="phase1", spec_ref=f"{_MISO}: line ratings"),
        Column("load_dump_limit", "MVA", "Load-dump rating, derated", BASE_MVA, (0.0, 100000.0),
               since="phase1", spec_ref=f"{_MISO}: line ratings"),
        Column("branch_type", "category", "0 = line, 1 = LTC transformer, 2 = phase shifter", RAW,
               categories=(0.0, 1.0, 2.0), since="phase2",
               spec_ref=f"{_MISO}: transformer / PAR flows and tap positions"),
        Column("tap_position", "steps", "LTC tap position", divide(16.0), (-16.0, 16.0),
               since="phase2", spec_ref=f"{_MISO}: LTC tap positions"),
        Column("tap_angle", "deg", "Phase-shifter angle", divide(30.0), (-30.0, 30.0),
               since="phase2", spec_ref=f"{_MISO}: PAR tap positions"),
        Column("is_tie_line", "binary", "Branch crosses the boundary to the neighbouring RC", RAW,
               categories=(0.0, 1.0), since="phase5",
               spec_ref=f"{_MISO}: tie-line flows & net actual interchange"),
    ),
)

RAS_STATUS = ArraySpec(
    key="ras_status", rows="S",
    description="Per-scheme Remedial Action Scheme telemetry.",
    columns=(
        Column("arming_status", "category", "0 = available, 1 = armed, 2 = triggered, 3 = operated", RAW,
               categories=(0.0, 1.0, 2.0, 3.0), since="phase4", spec_ref=f"{_MISO}: RAS arming status"),
        Column("monitored_loading", "ratio", "Loading of the scheme's monitored line", RAW, (0.0, 5.0),
               since="phase4"),
        Column("action_type", "category", "0 = trip generator, 1 = shed load, 2 = open line", RAW,
               categories=(0.0, 1.0, 2.0), since="phase4"),
    ),
)

ARRAYS: Tuple[ArraySpec, ...] = (SCADA, PMU, EQUIPMENT, EDGE_ATTR, RAS_STATUS)

NODE_LABELS = VectorSpec("node_labels", "N", "binary", "1 if the node has failed by this step",
                         categories=(0.0, 1.0))
CASCADE_TIMING = VectorSpec("cascade_timing", "N", "steps",
                            "Failure time relative to cascade start; -1 = does not fail",
                            valid_range=(-1.0, None))
POWER_INJECTION = VectorSpec("power_injection", "N", "MW", "generation - load")
REACTIVE_INJECTION = VectorSpec("reactive_injection", "N", "MVAr", "Net reactive injection")
CONDUCTANCE = VectorSpec("conductance", "E", "S", "Branch conductance (mirror of edge_attr)")
SUSCEPTANCE = VectorSpec("susceptance", "E", "S", "Branch susceptance (mirror of edge_attr)")
THERMAL_LIMITS = VectorSpec("thermal_limits", "E", "MVA", "Static normal rating (not derated)",
                            valid_range=(0.0, None))

VECTORS: Tuple[VectorSpec, ...] = (
    NODE_LABELS, CASCADE_TIMING, POWER_INJECTION, REACTIVE_INJECTION,
    CONDUCTANCE, SUSCEPTANCE, THERMAL_LIMITS,
)

REQUIRED_METADATA: Tuple[str, ...] = ("is_cascade", "num_nodes", "num_edges", "base_mva")

# Legacy-migration factors (pre-phase-1 edge data had only the normal rating).
LEGACY_EMERGENCY_FACTOR = 1.22
LEGACY_LOAD_DUMP_FACTOR = 1.47


# ---------------------------------------------------------------------------
# Model node-feature layout
# ---------------------------------------------------------------------------
# base  = [scada | pmu | equipment | power_injection | reactive_injection]
# final = [base | delta1(base) | delta2(base) | t_pos | ttf | susceptibility]
NODE_BASE_PARTS: Tuple[Tuple[str, int], ...] = (
    ("scada", SCADA.width),
    ("pmu", PMU.width),
    ("equipment", EQUIPMENT.width),
    ("power_injection", 1),
    ("reactive_injection", 1),
)
NODE_FEATURE_BASE_DIM = sum(w for _, w in NODE_BASE_PARTS)

TTF_FEATURES: Tuple[str, ...] = ("ttf_voltage", "ttf_temp", "ttf_freq", "ttf_loading")
SUSCEPTIBILITY_FEATURES: Tuple[str, ...] = (
    "mean_adj_line_loading", "cascade_initiation_risk", "cascade_reception_risk",
    "max_adj_line_loading", "loading_x_max_adj_loading",
)
NODE_FEATURE_BLOCKS: Tuple[Tuple[str, int], ...] = (
    ("base", NODE_FEATURE_BASE_DIM),
    ("delta1", NODE_FEATURE_BASE_DIM),
    ("delta2", NODE_FEATURE_BASE_DIM),
    ("t_pos", 1),
    ("ttf", len(TTF_FEATURES)),
    ("susceptibility", len(SUSCEPTIBILITY_FEATURES)),
)
NODE_FEATURE_DIM = sum(w for _, w in NODE_FEATURE_BLOCKS)

# Inputs hidden from the model (zeroed) — absolute time / scenario-level info.
MODEL_MASKED_SCADA: Tuple[str, ...] = ("time_ratio", "stress_level")
MODEL_MASKED_NODE_FEATURES: Tuple[str, ...] = ("t_pos",)

_PART_SPECS = {"scada": SCADA, "pmu": PMU, "equipment": EQUIPMENT}


def _offsets(parts: Sequence[Tuple[str, int]]) -> Dict[str, Tuple[int, int]]:
    out, pos = {}, 0
    for name, width in parts:
        out[name] = (pos, width)
        pos += width
    return out


_BASE_OFFSETS = _offsets(NODE_BASE_PARTS)
_BLOCK_OFFSETS = _offsets(NODE_FEATURE_BLOCKS)


def base_index(part: str, column: Optional[str] = None) -> int:
    """Index into the base node-feature vector.

    ``base_index("scada", "loading_ratio")`` -> 17,
    ``base_index("power_injection")`` -> 46.
    """
    if part not in _BASE_OFFSETS:
        raise ContractError(f"unknown base part '{part}'")
    start, width = _BASE_OFFSETS[part]
    if column is None:
        if width != 1:
            raise ContractError(f"base part '{part}' has {width} columns; name one")
        return start
    if part not in _PART_SPECS:
        raise ContractError(f"base part '{part}' has no named columns")
    return start + _PART_SPECS[part].idx(column)


def node_feature_slice(block: str) -> slice:
    if block not in _BLOCK_OFFSETS:
        raise ContractError(f"unknown node-feature block '{block}'")
    start, width = _BLOCK_OFFSETS[block]
    return slice(start, start + width)


def node_feature_index(block: str, offset: int = 0) -> int:
    s = node_feature_slice(block)
    if not 0 <= offset < s.stop - s.start:
        raise ContractError(f"offset {offset} outside block '{block}'")
    return s.start + offset


# ---------------------------------------------------------------------------
# Layout hash + stamps
# ---------------------------------------------------------------------------
def _layout_document() -> Dict[str, Any]:
    """Everything that reaches the model. Ranges and prose are excluded."""
    return {
        "arrays": [
            {"key": a.key, "rows": a.rows,
             "columns": [[c.name, c.unit, list(c.norm)] for c in a.columns]}
            for a in ARRAYS
        ],
        "vectors": [[v.key, v.rows, v.unit] for v in VECTORS],
        "node_base_parts": [list(p) for p in NODE_BASE_PARTS],
        "node_feature_blocks": [list(b) for b in NODE_FEATURE_BLOCKS],
        "masked_scada": list(MODEL_MASKED_SCADA),
        "masked_node_features": list(MODEL_MASKED_NODE_FEATURES),
    }


LAYOUT_HASH = hashlib.sha256(
    json.dumps(_layout_document(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]


def contract_stamp() -> Dict[str, str]:
    """Fields to embed in scenario metadata and checkpoints."""
    return {"schema_version": SCHEMA_VERSION, "layout_hash": LAYOUT_HASH}


def check_checkpoint(checkpoint: Mapping[str, Any], source: str = "checkpoint") -> None:
    """DC-031: refuse a checkpoint built for a different layout; warn if unstamped."""
    stamped = checkpoint.get("layout_hash") if isinstance(checkpoint, Mapping) else None
    if stamped is None:
        log.warning("%s carries no data-contract stamp; cannot verify it matches "
                    "layout %s (schema %s). Retrain to get a stamped checkpoint.",
                    source, LAYOUT_HASH, SCHEMA_VERSION)
        return
    if stamped != LAYOUT_HASH:
        raise ContractError(
            f"{source} was trained on data layout {stamped} "
            f"(schema {checkpoint.get('schema_version', '?')}), but the current contract "
            f"is layout {LAYOUT_HASH} (schema {SCHEMA_VERSION}). Retrain on current data.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    range_violations: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def range_violation_count(self) -> int:
        return sum(self.range_violations.values())

    def add_range(self, where: str, count: int) -> None:
        if count:
            self.range_violations[where] = self.range_violations.get(where, 0) + int(count)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        self.errors.extend(other.errors)
        for k, v in other.range_violations.items():
            self.add_range(k, v)
        return self

    def summary(self, limit: int = 10) -> str:
        parts = []
        if self.errors:
            shown = "; ".join(self.errors[:limit])
            more = f" (+{len(self.errors) - limit} more)" if len(self.errors) > limit else ""
            parts.append(f"{len(self.errors)} structural error(s): {shown}{more}")
        if self.range_violations:
            rv = ", ".join(f"{k}={v}" for k, v in sorted(self.range_violations.items()))
            parts.append(f"{self.range_violation_count} range violation(s): {rv}")
        return " | ".join(parts) or "ok"

    def raise_if_errors(self, context: str = "") -> None:
        if self.errors:
            raise ContractError(f"{context + ': ' if context else ''}{self.summary()}")


_range_patterns_logged: Dict[frozenset, int] = {}


def log_range_warnings(report: "ValidationReport", context: str = "") -> None:
    """Warn about range violations; repeats of the same pattern drop to DEBUG
    (a systematic generator issue would otherwise log once per scenario)."""
    if not report.range_violations:
        return
    pattern = frozenset(report.range_violations)
    seen = _range_patterns_logged.get(pattern, 0)
    _range_patterns_logged[pattern] = seen + 1
    level = logging.WARNING if seen == 0 else logging.DEBUG
    suffix = "" if seen else " (further scenarios with the same pattern logged at DEBUG)"
    log.log(level, "contract range check%s: %s%s",
            f" [{context}]" if context else "", report.summary(), suffix)


def _count_out_of_range(values: np.ndarray, valid_range: Range,
                        categories: Optional[Tuple[float, ...]]) -> int:
    if categories is not None:
        return int(np.count_nonzero(~np.isin(values, np.asarray(categories, dtype=values.dtype))))
    if valid_range is None:
        return 0
    lo, hi = valid_range
    bad = np.zeros(values.shape, dtype=bool)
    if lo is not None:
        bad |= values < lo
    if hi is not None:
        bad |= values > hi
    return int(np.count_nonzero(bad))


def _rows_expected(rows: str, n: Optional[int], e: Optional[int]) -> Optional[int]:
    return {"N": n, "E": e}.get(rows)


def validate_timestep(ts: Mapping[str, Any], num_nodes: Optional[int] = None,
                      num_edges: Optional[int] = None, *, check_ranges: bool = True,
                      where: str = "",
                      vectors: Optional[Sequence[VectorSpec]] = None) -> ValidationReport:
    """Structural (+ optional range) validation of one timestep dict.

    ``vectors`` restricts which per-row vectors are required (default: all of
    VECTORS). Live telemetry passes the observable subset, since ground-truth
    labels such as ``cascade_timing`` do not exist for a live grid.
    """
    rep = ValidationReport()
    pre = f"{where}." if where else ""
    if not isinstance(ts, Mapping):
        rep.errors.append(f"{where or 'timestep'} is not a dict")
        return rep

    for spec in ARRAYS:
        if spec.key not in ts:
            rep.errors.append(f"{pre}{spec.key} missing")
            continue
        arr = np.asarray(ts[spec.key])
        if arr.ndim != 2:
            rep.errors.append(f"{pre}{spec.key} has ndim {arr.ndim}, expected 2")
            continue
        if arr.shape[1] != spec.width:
            legacy = " (legacy layout; use migrate mode)" if arr.shape[1] in spec.legacy_widths else ""
            rep.errors.append(f"{pre}{spec.key} has {arr.shape[1]} columns, "
                              f"contract {SCHEMA_VERSION} requires {spec.width}{legacy}")
            continue
        expected = _rows_expected(spec.rows, num_nodes, num_edges)
        if expected is not None and arr.shape[0] != expected:
            rep.errors.append(f"{pre}{spec.key} has {arr.shape[0]} rows, expected {expected}")
            continue
        if not np.issubdtype(arr.dtype, np.floating):
            rep.errors.append(f"{pre}{spec.key} dtype {arr.dtype} is not floating point")
            continue
        if not np.all(np.isfinite(arr)):
            rep.errors.append(f"{pre}{spec.key} contains NaN/Inf")
            continue
        if check_ranges:
            for j, col in enumerate(spec.columns):
                rep.add_range(f"{spec.key}.{col.name}",
                              _count_out_of_range(arr[:, j], col.valid_range, col.categories))

    for vec in (VECTORS if vectors is None else vectors):
        if vec.key not in ts:
            rep.errors.append(f"{pre}{vec.key} missing")
            continue
        arr = np.asarray(ts[vec.key])
        if arr.ndim != 1:
            rep.errors.append(f"{pre}{vec.key} has ndim {arr.ndim}, expected 1")
            continue
        expected = _rows_expected(vec.rows, num_nodes, num_edges)
        if expected is not None and arr.shape[0] != expected:
            rep.errors.append(f"{pre}{vec.key} has length {arr.shape[0]}, expected {expected}")
            continue
        if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            rep.errors.append(f"{pre}{vec.key} is non-numeric or contains NaN/Inf")
            continue
        if check_ranges:
            rep.add_range(vec.key, _count_out_of_range(arr, vec.valid_range, vec.categories))
    return rep


def validate_scenario(scenario: Mapping[str, Any], *, check_ranges: bool = True,
                      require_stamp: bool = False) -> ValidationReport:
    """Validate the scenario envelope and every timestep."""
    rep = ValidationReport()
    if not isinstance(scenario, Mapping):
        rep.errors.append("scenario is not a dict")
        return rep
    for key in ("sequence", "edge_index", "metadata"):
        if key not in scenario:
            rep.errors.append(f"envelope: '{key}' missing")
    if rep.errors:
        return rep

    seq, meta = scenario["sequence"], scenario["metadata"]
    if not isinstance(seq, (list, tuple)) or not seq:
        rep.errors.append("envelope: sequence is empty or not a list")
        return rep
    if not isinstance(meta, Mapping):
        rep.errors.append("envelope: metadata is not a dict")
        return rep
    for key in REQUIRED_METADATA:
        if key not in meta:
            rep.errors.append(f"metadata: '{key}' missing")

    ei = np.asarray(scenario["edge_index"])
    if ei.ndim != 2 or ei.shape[0] != 2 or not np.issubdtype(ei.dtype, np.integer):
        rep.errors.append(f"envelope: edge_index must be an integer (2, E) array, got "
                          f"{ei.dtype} {ei.shape}")
        num_edges = None
    else:
        num_edges = int(ei.shape[1])

    num_nodes = meta.get("num_nodes")
    if num_nodes is None and isinstance(seq[0], Mapping) and "scada_data" in seq[0]:
        num_nodes = int(np.asarray(seq[0]["scada_data"]).shape[0])
    if num_edges is not None and meta.get("num_edges") not in (None, num_edges):
        rep.errors.append(f"metadata: num_edges={meta.get('num_edges')} but edge_index has {num_edges}")
    if num_nodes is not None and num_edges is not None and ei.size and int(ei.max()) >= int(num_nodes):
        rep.errors.append("envelope: edge_index references a node >= num_nodes")

    if require_stamp and meta.get("layout_hash") != LAYOUT_HASH:
        rep.errors.append(f"metadata: layout_hash={meta.get('layout_hash')!r}, "
                          f"current contract is {LAYOUT_HASH}")

    for t, ts in enumerate(seq):
        rep.merge(validate_timestep(ts, num_nodes, num_edges,
                                    check_ranges=check_ranges, where=f"t{t}"))
    return rep


# ---------------------------------------------------------------------------
# Legacy migration (non-training consumers only — spec D3)
# ---------------------------------------------------------------------------
_migration_logged: set = set()


def _log_migration(key: str, old: int, new: int, context: str) -> None:
    tag = (key, old)
    level = logging.WARNING if tag not in _migration_logged else logging.DEBUG
    _migration_logged.add(tag)
    log.log(level, "contract: migrating legacy %s from %d to %d columns%s",
            key, old, new, f" ({context})" if context else "")


def migrate_array(spec: ArraySpec, arr: np.ndarray, *, context: str = "") -> np.ndarray:
    """Upgrade a RAW (not yet normalized) legacy array to the current layout.

    Values match the pre-contract padding in dataset.py exactly (DC-021).
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ContractError(f"{spec.key}: cannot migrate array with ndim {a.ndim}")
    if a.shape[1] == spec.width:
        return a
    if a.shape[1] not in spec.legacy_widths:
        raise ContractError(f"{spec.key}: no migration from width {a.shape[1]} "
                            f"(known legacy widths {spec.legacy_widths}, current {spec.width})")
    start = a.shape[1]
    rows = a.shape[0]
    if spec is SCADA:
        if a.shape[1] == 18:   # pre-phase-3: no reactive-support devices
            is_gen = (a[:, SCADA.idx("node_type")] == 1).astype(np.float32)
            a = np.concatenate([a, np.zeros((rows, 3), np.float32), is_gen[:, None]], axis=1)
        if a.shape[1] == 22:   # pre-phase-5: no storage, no tie lines -> interchange 0
            a = np.concatenate([a, np.zeros((rows, 6), np.float32)], axis=1)
    elif spec is EDGE_ATTR:
        if a.shape[1] == 7:    # pre-phase-1: breaker closed, rating tiers from normal rating
            thermal = a[:, EDGE_ATTR.idx("thermal_limit")]
            a = np.concatenate([a, np.column_stack([
                np.ones(rows, np.float32),
                thermal * LEGACY_EMERGENCY_FACTOR,
                thermal * LEGACY_LOAD_DUMP_FACTOR,
            ]).astype(np.float32)], axis=1)
        if a.shape[1] == 10:   # pre-phase-2: plain lines, neutral taps
            a = np.concatenate([a, np.zeros((rows, 3), np.float32)], axis=1)
        if a.shape[1] == 13:   # pre-phase-5: no tie lines
            a = np.concatenate([a, np.zeros((rows, 1), np.float32)], axis=1)
    if a.shape[1] != spec.width:  # pragma: no cover - guards future edits
        raise ContractError(f"{spec.key}: migration ended at width {a.shape[1]}")
    _log_migration(spec.key, start, spec.width, context)
    return a


def migrate_timestep(ts: Mapping[str, Any], *, n_ras_schemes: int = 6,
                     context: str = "") -> Dict[str, Any]:
    """Return a copy of ``ts`` with every array upgraded to the current layout."""
    out = dict(ts)
    for spec in (SCADA, EDGE_ATTR):
        if spec.key in out:
            out[spec.key] = migrate_array(spec, out[spec.key], context=context)
    if "ras_status" not in out:
        _log_migration("ras_status", 0, RAS_STATUS.width, context)
        out["ras_status"] = np.zeros((n_ras_schemes, RAS_STATUS.width), np.float32)
    return out


# ---------------------------------------------------------------------------
# Normalization (DC-022)
# ---------------------------------------------------------------------------
def normalize_array(spec: ArraySpec, arr: np.ndarray, *, base_mva: float,
                    base_frequency: float = 60.0,
                    constants: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Apply each column's normalization policy to a current-layout RAW array."""
    a = np.array(arr, dtype=np.float32, copy=True)
    if a.ndim != 2 or a.shape[1] != spec.width:
        raise ContractError(f"{spec.key}: normalize expects (rows, {spec.width}), got {a.shape}")
    constants = constants or {}
    for j, col in enumerate(spec.columns):
        kind = col.norm[0]
        if kind == "raw":
            continue
        if kind == "base_mva":
            a[:, j] = a[:, j] / base_mva
        elif kind == "base_frequency":
            a[:, j] = a[:, j] / base_frequency
        elif kind == "divide":
            c = col.norm[1]
            if isinstance(c, str):
                if c not in constants:
                    raise ContractError(f"{spec.key}.{col.name}: constant '{c}' not supplied")
                c = constants[c]
            a[:, j] = a[:, j] / float(c)
        else:  # pragma: no cover
            raise ContractError(f"{spec.key}.{col.name}: unknown norm {col.norm}")
    return a


# ---------------------------------------------------------------------------
# Files: load, certify, strict acceptance (D5 / DC-020 / DC-042)
# ---------------------------------------------------------------------------
MANIFEST_NAME = "contract_manifest.json"


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return list(data) if isinstance(data, (list, tuple)) else [data]


def scenario_files(directory: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(directory, "scenario_*.pkl")))
    return files or sorted(glob.glob(os.path.join(directory, "scenarios_batch_*.pkl")))


def _file_sig(path: str) -> Dict[str, int]:
    st = os.stat(path)
    return {"size": int(st.st_size), "mtime": int(st.st_mtime)}


_manifest_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}


def _read_manifest(directory: str) -> Optional[Dict[str, Any]]:
    mpath = os.path.join(directory, MANIFEST_NAME)
    try:
        mtime = os.stat(mpath).st_mtime
    except OSError:
        return None
    cached = _manifest_cache.get(mpath)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        manifest = None
    _manifest_cache[mpath] = (mtime, manifest)
    return manifest


def is_certified(path: str) -> bool:
    """True if the file is listed, unchanged, in its directory's manifest for this layout."""
    manifest = _read_manifest(os.path.dirname(os.path.abspath(path)))
    if not manifest or manifest.get("layout_hash") != LAYOUT_HASH:
        return False
    entry = manifest.get("files", {}).get(os.path.basename(path))
    return bool(entry) and entry == _file_sig(path)


def is_current(scenario: Mapping[str, Any]) -> bool:
    meta = scenario.get("metadata", {}) if isinstance(scenario, Mapping) else {}
    return isinstance(meta, Mapping) and meta.get("layout_hash") == LAYOUT_HASH


def require_current(scenario: Mapping[str, Any], path: str) -> None:
    """DC-020: strict (training) acceptance check."""
    if is_current(scenario) or is_certified(path):
        return
    meta = scenario.get("metadata", {}) if isinstance(scenario, Mapping) else {}
    found = meta.get("layout_hash") if isinstance(meta, Mapping) else None
    raise ContractError(
        f"{path}: scenario is not stamped with the current data contract "
        f"(layout {LAYOUT_HASH}, schema {SCHEMA_VERSION}; found {found!r}) and is not "
        f"certified. Regenerate the data, or run `python -m cascade_prediction.contract "
        f"certify {os.path.dirname(path) or '.'}` if it already matches the current layout.")


def validate_directory(directory: str, *, check_ranges: bool = True) -> Tuple[ValidationReport, Dict[str, ValidationReport]]:
    total, per_file = ValidationReport(), {}
    files = scenario_files(directory)
    if not files:
        total.errors.append(f"{directory}: no scenario files found")
    for path in files:
        rep = ValidationReport()
        try:
            for i, sc in enumerate(load_scenarios(path)):
                sub = validate_scenario(sc, check_ranges=check_ranges)
                sub.errors = [f"[{i}] {e}" for e in sub.errors]
                rep.merge(sub)
        except Exception as exc:  # unreadable file is a structural error
            rep.errors.append(f"unreadable: {exc}")
        per_file[os.path.basename(path)] = rep
        total.merge(ValidationReport(
            errors=[f"{os.path.basename(path)}: {e}" for e in rep.errors],
            range_violations=dict(rep.range_violations)))
    return total, per_file


def certify_directory(directory: str) -> ValidationReport:
    """Write a manifest only if every file passes structural validation."""
    total, _ = validate_directory(directory, check_ranges=True)
    if not total.ok:
        return total
    manifest = {
        **contract_stamp(),
        "certified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "range_violations": total.range_violations,
        "files": {os.path.basename(p): _file_sig(p) for p in scenario_files(directory)},
    }
    with open(os.path.join(directory, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return total


# ---------------------------------------------------------------------------
# Export (DC-040)
# ---------------------------------------------------------------------------
def _col_doc(i: int, c: Column) -> Dict[str, Any]:
    return {"index": i, "name": c.name, "unit": c.unit, "norm": list(c.norm),
            "valid_range": list(c.valid_range) if c.valid_range else None,
            "categories": list(c.categories) if c.categories else None,
            "scope": c.scope, "since": c.since, "spec_ref": c.spec_ref,
            "description": c.description}


def to_document() -> Dict[str, Any]:
    return {
        "title": "Cascade scenario data contract",
        "generated_from": "cascade_prediction/contract.py",
        **contract_stamp(),
        "arrays": {a.key: {"rows": a.rows, "width": a.width, "legacy_widths": list(a.legacy_widths),
                           "description": a.description,
                           "columns": [_col_doc(i, c) for i, c in enumerate(a.columns)]}
                   for a in ARRAYS},
        "vectors": {v.key: {"rows": v.rows, "unit": v.unit, "description": v.description,
                            "valid_range": list(v.valid_range) if v.valid_range else None,
                            "categories": list(v.categories) if v.categories else None}
                    for v in VECTORS},
        "required_metadata": list(REQUIRED_METADATA),
        "node_features": {
            "base_parts": [{"name": n, "offset": _BASE_OFFSETS[n][0], "width": w} for n, w in NODE_BASE_PARTS],
            "blocks": [{"name": n, "offset": _BLOCK_OFFSETS[n][0], "width": w} for n, w in NODE_FEATURE_BLOCKS],
            "base_dim": NODE_FEATURE_BASE_DIM,
            "dim": NODE_FEATURE_DIM,
            "ttf": list(TTF_FEATURES),
            "susceptibility": list(SUSCEPTIBILITY_FEATURES),
            "masked_scada": list(MODEL_MASKED_SCADA),
            "masked_node_features": list(MODEL_MASKED_NODE_FEATURES),
        },
    }


def _fmt_range(c: Column) -> str:
    if c.categories:
        return "{" + ", ".join(f"{v:g}" for v in c.categories) + "}"
    if not c.valid_range:
        return ""
    lo, hi = c.valid_range
    return f"[{'-inf' if lo is None else f'{lo:g}'}, {'inf' if hi is None else f'{hi:g}'}]"


def to_markdown() -> str:
    lines = [
        "# Data contract reference",
        "",
        f"Generated from `cascade_prediction/contract.py`. Do not edit by hand. "
        f"Schema **{SCHEMA_VERSION}**, layout hash `{LAYOUT_HASH}`.",
        "",
    ]
    for a in ARRAYS:
        lines += [f"## `{a.key}` ({a.rows} x {a.width})", "", a.description, ""]
        if a.legacy_widths:
            lines += [f"Legacy widths migratable outside training: {', '.join(map(str, a.legacy_widths))}.", ""]
        lines += ["| # | name | unit | norm | expected | scope | since | description |",
                  "|---|------|------|------|----------|-------|-------|-------------|"]
        for i, c in enumerate(a.columns):
            norm = ":".join(str(x) for x in c.norm)
            lines.append(f"| {i} | `{c.name}` | {c.unit} | {norm} | {_fmt_range(c)} | "
                         f"{c.scope} | {c.since} | {c.description} |")
        lines.append("")
    lines += ["## Per-timestep vectors", "", "| key | rows | unit | description |", "|---|---|---|---|"]
    lines += [f"| `{v.key}` | {v.rows} | {v.unit} | {v.description} |" for v in VECTORS]
    lines += ["", "## Model node-feature layout", "",
              f"Base dim {NODE_FEATURE_BASE_DIM}, total dim {NODE_FEATURE_DIM}.", "",
              "| block | offset | width |", "|---|---|---|"]
    lines += [f"| {n} | {_BLOCK_OFFSETS[n][0]} | {w} |" for n, w in NODE_FEATURE_BLOCKS]
    lines += ["", "Base parts: " + ", ".join(f"{n}@{_BASE_OFFSETS[n][0]}" for n, _ in NODE_BASE_PARTS) + ".",
              "", "Masked in model input: SCADA " + ", ".join(f"`{m}`" for m in MODEL_MASKED_SCADA)
              + "; node features " + ", ".join(f"`{m}`" for m in MODEL_MASKED_NODE_FEATURES) + ".", ""]
    return "\n".join(lines)


def export(out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "data_contract.json")
    mpath = os.path.join(out_dir, "data_contract.md")
    with open(jpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(to_document(), indent=2) + "\n")
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_markdown())
    return jpath, mpath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cascade_prediction.contract",
                                description="Scenario data contract tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write data_contract.json + data_contract.md")
    e.add_argument("out_dir")
    v = sub.add_parser("validate", help="validate every scenario file in a directory")
    v.add_argument("data_dir")
    v.add_argument("--fail-on-ranges", action="store_true", help="exit 1 on any range violation")
    c = sub.add_parser("certify", help="write contract_manifest.json if structurally valid")
    c.add_argument("data_dir")
    args = p.parse_args(argv)

    if args.cmd == "export":
        for path in export(args.out_dir):
            print(f"wrote {path}")
        return 0
    if args.cmd == "validate":
        total, per_file = validate_directory(args.data_dir)
        for name, rep in per_file.items():
            if not rep.ok or rep.range_violations:
                print(f"{name}: {rep.summary()}")
        print(f"\n{len(per_file)} file(s) - {total.summary()}")
        if not total.ok or (args.fail_on_ranges and total.range_violations):
            return 1
        return 0
    if args.cmd == "certify":
        total = certify_directory(args.data_dir)
        if not total.ok:
            print(f"NOT certified - {total.summary()}")
            return 1
        print(f"certified {args.data_dir} for layout {LAYOUT_HASH} (schema {SCHEMA_VERSION})"
              + (f"; range warnings: {total.summary()}" if total.range_violations else ""))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
