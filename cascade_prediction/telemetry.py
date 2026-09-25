"""
Live Telemetry Contract
=======================
Single source of truth for the live frame the physics emitter publishes and
the backend ingests (constitution rule 4; spec specs/001-live-grid-streaming,
contracts/telemetry-frame.md is a readable summary of this module).

A frame is one generator timestep wrapped in an envelope that identifies the
stream (one episode) and the frame's position in it. Arrays are keyed by the
data-contract ``key`` and validated by ``contract.validate_timestep``; this
module never pads, defaults or truncates data (rule 10) — it reports errors.

    python -m cascade_prediction.telemetry export <schema.json>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import contract as C

TELEMETRY_VERSION = "1.0.0"

# AWS IoT Core caps a message at 128 KB; keep 25% headroom (research R2).
MAX_FRAME_BYTES = 96 * 1024
DEFAULT_PRECISION = 2

STATUS_OK = "ok"
STATUS_DROPPED = "dropped"

# Rejection reasons surfaced as `gap` events by the backend.
REASON_INVALID = "invalid_frame"
REASON_MISMATCH = "contract_mismatch"

# Observable per-row vectors carried by a live frame. Ground-truth labels
# (cascade_timing) and edge_attr mirrors (conductance, susceptance) are not.
LIVE_VECTORS: Sequence[C.VectorSpec] = (
    C.NODE_LABELS, C.POWER_INJECTION, C.REACTIVE_INJECTION, C.THERMAL_LIMITS,
)
FRAME_ARRAY_KEYS: Sequence[str] = (
    tuple(spec.key for spec in C.ARRAYS) + tuple(vec.key for vec in LIVE_VECTORS)
)

# Envelope fields: name -> (allowed python types, required on ok, required on dropped)
_INT = (int,)
_NUM = (int, float)
ENVELOPE_FIELDS: Dict[str, tuple] = {
    "telemetry_version": ((str,), True, True),
    "schema_version":    ((str,), True, True),
    "layout_hash":       ((str,), True, True),
    "stream_id":         ((str,), True, True),
    "episode":           (_INT, True, True),
    "seq":               (_INT, True, True),
    "is_last":           ((bool,), True, True),
    "emitted_at":        (_NUM, True, True),
    "grid_time_minutes": (_NUM, True, True),
    "status":            ((str,), True, True),
    "drop_reason":       ((str,), False, True),
    "stream_info":       ((dict,), True, False),
    "arrays":            ((dict,), True, False),
    "cascade_events":    ((list,), True, False),
}

# stream_info travels on every ok frame so a consumer that joins mid-episode
# (or loses seq 0) can still render and validate the stream.
STREAM_INFO_FIELDS: Dict[str, tuple] = {
    "num_nodes":      _INT,
    "num_edges":      _INT,
    "episode_length": _INT,
    "dt_minutes":     _NUM,
    "stress_level":   _NUM,
    "edge_index":     (list,),
    "positions":      (list,),
    "node_types":     (list,),
}


def _major(version: str) -> str:
    return str(version).split(".", 1)[0]


def _round_array(value: Any, precision: int) -> List:
    arr = np.asarray(value, dtype=np.float64)
    return np.round(arr, precision).tolist()


def _clean_events(events: Any, precision: int) -> List[Dict[str, Any]]:
    out = []
    for ev in events or []:
        parent = ev.get("parent")
        out.append({
            "node": int(ev["node"]),
            "t_offset": round(float(ev.get("t_offset", 0.0)), precision),
            "reason": str(ev.get("reason", "")),
            "parent": int(parent) if parent is not None else None,
        })
    return out


def build_stream_info(topology: Mapping[str, Any], *, episode_length: int,
                      dt_minutes: float, stress_level: float) -> Dict[str, Any]:
    """Stream description from a generator topology block."""
    edge_index = np.asarray(topology["edge_index"], dtype=np.int64)
    positions = np.asarray(topology["positions"], dtype=np.float64)
    num_nodes = int(topology.get("num_nodes", positions.shape[0]))
    node_types = np.asarray(topology.get("node_types", np.zeros(num_nodes)), dtype=np.int64)
    return {
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "episode_length": int(episode_length),
        "dt_minutes": float(dt_minutes),
        "stress_level": round(float(stress_level), 3),
        "edge_index": edge_index.tolist(),
        "positions": np.round(positions, 2).tolist(),
        "node_types": node_types.tolist(),
    }


def _envelope(*, stream_id: str, episode: int, seq: int, is_last: bool,
              grid_time_minutes: float, emitted_at: Optional[float], status: str) -> Dict[str, Any]:
    return {
        "telemetry_version": TELEMETRY_VERSION,
        "schema_version": C.SCHEMA_VERSION,
        "layout_hash": C.LAYOUT_HASH,
        "stream_id": str(stream_id),
        "episode": int(episode),
        "seq": int(seq),
        "is_last": bool(is_last),
        "emitted_at": round(float(time.time() if emitted_at is None else emitted_at), 3),
        "grid_time_minutes": float(grid_time_minutes),
        "status": status,
    }


def build_frame(timestep: Mapping[str, Any], *, stream_id: str, episode: int, seq: int,
                is_last: bool, stream_info: Mapping[str, Any],
                emitted_at: Optional[float] = None,
                precision: int = DEFAULT_PRECISION) -> Dict[str, Any]:
    """Wrap one generator timestep as a telemetry frame (used by the emitter)."""
    frame = _envelope(stream_id=stream_id, episode=episode, seq=seq, is_last=is_last,
                      grid_time_minutes=seq * float(stream_info["dt_minutes"]),
                      emitted_at=emitted_at, status=STATUS_OK)
    frame["stream_info"] = dict(stream_info)
    frame["arrays"] = {key: _round_array(timestep[key], precision) for key in FRAME_ARRAY_KEYS}
    frame["cascade_events"] = _clean_events(timestep.get("cascade_events"), precision)
    return frame


def build_dropped_marker(*, stream_id: str, episode: int, seq: int, is_last: bool,
                         dt_minutes: float, drop_reason: str,
                         emitted_at: Optional[float] = None) -> Dict[str, Any]:
    """Stand-in for a frame that could not be published in full (keeps seq contiguous)."""
    marker = _envelope(stream_id=stream_id, episode=episode, seq=seq, is_last=is_last,
                       grid_time_minutes=seq * float(dt_minutes),
                       emitted_at=emitted_at, status=STATUS_DROPPED)
    marker["drop_reason"] = str(drop_reason)
    return marker


def encode_frame(frame: Mapping[str, Any]) -> bytes:
    """Compact UTF-8 JSON, the wire form on every transport."""
    return json.dumps(frame, separators=(",", ":"), allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class FrameCheck:
    """Outcome of validate_frame. ``reason`` is set only when not ok."""
    errors: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    stream_id: Optional[str] = None
    seq: Optional[int] = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, reason: str, message: str) -> "FrameCheck":
        # contract_mismatch outranks invalid_frame: it names the root cause.
        if self.reason != REASON_MISMATCH:
            self.reason = reason
        self.errors.append(message)
        return self


def _type_ok(value: Any, types: tuple) -> bool:
    if isinstance(value, bool) and bool not in types:
        return False          # bool is an int subclass; don't accept it as a number
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return isinstance(value, types)


def validate_frame(obj: Any, *, num_nodes: Optional[int] = None,
                   num_edges: Optional[int] = None) -> FrameCheck:
    """
    Check one received frame against this contract.

    ``num_nodes``/``num_edges`` default to the frame's own stream_info. The
    returned FrameCheck carries ``stream_id``/``seq`` whenever they were
    readable, so a rejected frame can still be reported as a gap in its slot.
    """
    chk = FrameCheck()
    if not isinstance(obj, Mapping):
        return chk.fail(REASON_INVALID, "frame is not a JSON object")

    if isinstance(obj.get("stream_id"), str):
        chk.stream_id = obj["stream_id"]
    if _type_ok(obj.get("seq"), _INT):
        chk.seq = obj["seq"]

    status = obj.get("status")
    if status not in (STATUS_OK, STATUS_DROPPED):
        chk.fail(REASON_INVALID, f"status must be '{STATUS_OK}' or '{STATUS_DROPPED}', got {status!r}")
    dropped = status == STATUS_DROPPED

    for name, (types, req_ok, req_dropped) in ENVELOPE_FIELDS.items():
        required = req_dropped if dropped else req_ok
        if name not in obj:
            if required:
                chk.fail(REASON_INVALID, f"{name} missing")
            continue
        if not _type_ok(obj[name], types):
            chk.fail(REASON_INVALID, f"{name} has type {type(obj[name]).__name__}")

    if _major(obj.get("telemetry_version", "")) != _major(TELEMETRY_VERSION):
        chk.fail(REASON_MISMATCH, f"telemetry_version {obj.get('telemetry_version')!r} "
                                  f"is incompatible with {TELEMETRY_VERSION}")
    if obj.get("schema_version") != C.SCHEMA_VERSION or obj.get("layout_hash") != C.LAYOUT_HASH:
        chk.fail(REASON_MISMATCH, f"data contract {obj.get('schema_version')}/{obj.get('layout_hash')} "
                                  f"!= {C.SCHEMA_VERSION}/{C.LAYOUT_HASH}")
    if chk.seq is not None and chk.seq < 0:
        chk.fail(REASON_INVALID, "seq is negative")

    if dropped or not chk.ok:
        return chk

    info = obj["stream_info"]
    for name, types in STREAM_INFO_FIELDS.items():
        if name not in info:
            chk.fail(REASON_INVALID, f"stream_info.{name} missing")
        elif not _type_ok(info[name], types):
            chk.fail(REASON_INVALID, f"stream_info.{name} has type {type(info[name]).__name__}")
    if not chk.ok:
        return chk
    n = info["num_nodes"] if num_nodes is None else num_nodes
    e = info["num_edges"] if num_edges is None else num_edges
    if info["num_nodes"] != n or info["num_edges"] != e:
        chk.fail(REASON_INVALID, f"stream_info size {info['num_nodes']}x{info['num_edges']} "
                                 f"differs from the stream's {n}x{e}")
    if chk.seq is not None and chk.seq >= info["episode_length"]:
        chk.fail(REASON_INVALID, f"seq {chk.seq} >= episode_length {info['episode_length']}")

    ts = decode_arrays(obj["arrays"], chk)
    if ts is not None:
        report = C.validate_timestep(ts, n, e, check_ranges=False, where="arrays",
                                     vectors=LIVE_VECTORS)
        for err in report.errors:
            chk.fail(REASON_INVALID, err)
        extra = sorted(set(obj["arrays"]) - set(FRAME_ARRAY_KEYS))
        if extra:
            chk.fail(REASON_INVALID, f"arrays has unknown keys: {', '.join(extra)}")

    for i, ev in enumerate(obj["cascade_events"]):
        if not isinstance(ev, Mapping) or not _type_ok(ev.get("node"), _INT) \
                or not _type_ok(ev.get("t_offset"), _NUM) or not isinstance(ev.get("reason"), str) \
                or not (ev.get("parent") is None or _type_ok(ev.get("parent"), _INT)):
            chk.fail(REASON_INVALID, f"cascade_events[{i}] is malformed")
    return chk


def decode_arrays(arrays: Mapping[str, Any], chk: Optional[FrameCheck] = None
                  ) -> Optional[Dict[str, np.ndarray]]:
    """JSON lists -> float arrays keyed by contract key. Errors go to ``chk``."""
    out: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        try:
            out[key] = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            if chk is None:
                raise
            chk.fail(REASON_INVALID, f"arrays.{key} is not numeric")
            return None
    return out


# ---------------------------------------------------------------------------
# JSON Schema export (contracts/telemetry-frame.schema.json)
# ---------------------------------------------------------------------------
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               dict: "object", list: "array"}


def _schema_type(types: tuple) -> Any:
    names = sorted({_JSON_TYPES[t] for t in types})
    if names == ["integer", "number"]:
        return "number"
    return names[0] if len(names) == 1 else names


def export_json_schema() -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) generated from the definitions above."""
    arrays = {}
    for spec in C.ARRAYS:
        arrays[spec.key] = {
            "type": "array", "description": f"{spec.rows} rows x {spec.width} columns: "
                                            + ", ".join(spec.names),
            "items": {"type": "array", "minItems": spec.width, "maxItems": spec.width,
                      "items": {"type": "number"}},
        }
    for vec in LIVE_VECTORS:
        arrays[vec.key] = {"type": "array", "description": f"{vec.rows} rows ({vec.unit})",
                           "items": {"type": "number"}}
    envelope = {name: {"type": _schema_type(types)}
                for name, (types, _, _) in ENVELOPE_FIELDS.items()}
    envelope["status"]["enum"] = [STATUS_OK, STATUS_DROPPED]
    envelope["stream_info"] = {
        "type": "object",
        "required": list(STREAM_INFO_FIELDS),
        "properties": {n: {"type": _schema_type(t)} for n, t in STREAM_INFO_FIELDS.items()},
    }
    envelope["arrays"] = {"type": "object", "required": list(FRAME_ARRAY_KEYS),
                          "additionalProperties": False, "properties": arrays}
    envelope["cascade_events"] = {"type": "array", "items": {
        "type": "object", "required": ["node", "t_offset", "reason", "parent"],
        "properties": {"node": {"type": "integer"}, "t_offset": {"type": "number"},
                       "reason": {"type": "string"},
                       "parent": {"type": ["integer", "null"]}}}}
    ok_required = [n for n, (_, req, _) in ENVELOPE_FIELDS.items() if req]
    dropped_required = [n for n, (_, _, req) in ENVELOPE_FIELDS.items() if req]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Live telemetry frame",
        "description": f"telemetry {TELEMETRY_VERSION}, data contract {C.SCHEMA_VERSION} "
                       f"({C.LAYOUT_HASH}). Generated by cascade_prediction.telemetry; do not edit.",
        "type": "object",
        "properties": envelope,
        "oneOf": [
            {"properties": {"status": {"const": STATUS_OK}}, "required": ok_required},
            {"properties": {"status": {"const": STATUS_DROPPED}}, "required": dropped_required},
        ],
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cascade_prediction.telemetry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("export", help="write the frame JSON Schema")
    exp.add_argument("path")
    args = parser.parse_args(argv)
    if args.cmd == "export":
        path = Path(args.path)
        path.write_text(json.dumps(export_json_schema(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
