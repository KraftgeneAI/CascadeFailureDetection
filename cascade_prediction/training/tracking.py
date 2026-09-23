"""
Experiment Tracking (MLflow) — Stage-1 MLOps instrumentation
============================================================

`RunTracker` is a thin, dependency-OPTIONAL wrapper over MLflow. It is a
complete no-op unless MLflow is installed AND tracking is enabled, so training
runs unchanged on a machine that has never heard of MLflow.

The same instrumentation logs to whichever backend the environment points at:

  * a local ``./mlruns`` file store            (default — no configuration)
  * an Azure ML workspace                      (set ``MLFLOW_TRACKING_URI`` to the
                                                workspace tracking URI, obtained via
                                                ``az ml workspace show --query mlflow_tracking_uri -o tsv``)

So promoting from "runs on my laptop" to "tracked in Azure ML" is a
configuration change, not a code change. When the script runs *inside* an
Azure ML job, the job already opens an MLflow run; ``RunTracker`` detects that
and reuses it instead of trying to start a second one.

Disable entirely with ``MLFLOW_DISABLE=1``.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["RunTracker", "data_fingerprint", "git_sha"]

_TRUTHY = {"1", "true", "yes", "on"}


def data_fingerprint(*paths: str) -> str:
    """Cheap, order-stable fingerprint of a dataset.

    Hashes each file's relative path + byte size + mtime (NOT the file bytes),
    so a run can record *exactly which data* it trained on without copying or
    reading gigabytes of pickles. A changed, added, or removed scenario file
    changes the fingerprint — which is what lets you tie a model back to its
    data version and detect "the data moved under me" situations.
    """
    h = hashlib.sha256()
    for root in sorted(str(p) for p in paths):
        base = Path(root)
        if not base.exists():
            h.update(f"{root}:MISSING".encode())
            continue
        files = sorted(base.rglob("*")) if base.is_dir() else [base]
        for f in files:
            if f.is_file():
                st = f.stat()
                rel = f.relative_to(base) if base.is_dir() else Path(f.name)
                h.update(f"{rel.as_posix()}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """Short git commit of the working tree, or None if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd or os.getcwd(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _short(value: Any, limit: int = 250) -> str:
    """MLflow caps param VALUES at 250/500 chars depending on version; truncate."""
    s = str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


class RunTracker:
    """No-op-safe MLflow run wrapper.

    Usage as a context manager::

        with RunTracker(experiment="cascade-prediction", tags={...}) as run:
            run.log_params(vars(args))
            run.log_metrics({"val_node_f1": 0.71}, step=epoch)
            run.log_artifact("checkpoints/best_model.pth")

    or explicitly with ``.start()`` / ``.end()``. Every method is guarded, so a
    misconfigured or absent MLflow never breaks training.
    """

    def __init__(
        self,
        experiment: str = "cascade-prediction",
        run_name: Optional[str] = None,
        enabled: bool = True,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tags = dict(tags or {})
        self._mlflow = None
        self._active = False
        self._owns_run = False

        if not enabled or os.getenv("MLFLOW_DISABLE", "").lower() in _TRUTHY:
            return
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
        except Exception:
            # MLflow not installed — stay a silent no-op.
            self._mlflow = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active and self._mlflow is not None

    def start(self) -> "RunTracker":
        if self._mlflow is None:
            return self
        try:
            # Recent MLflow raises on the default ./mlruns file store unless
            # opted in. When the user hasn't pointed at a real backend, keep the
            # zero-config local store working. (For a durable local setup, set
            # MLFLOW_TRACKING_URI=sqlite:///mlflow.db instead.)
            if not os.getenv("MLFLOW_TRACKING_URI"):
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            # Reuse an already-open run (e.g. one Azure ML started for the job).
            if self._mlflow.active_run() is None:
                self._mlflow.set_experiment(self.experiment)
                self._mlflow.start_run(run_name=self.run_name)
                self._owns_run = True
            else:
                self._owns_run = False
            if self.tags:
                self._mlflow.set_tags(self.tags)
            self._active = True
            # Guarantee the run closes even if training raises later.
            atexit.register(self._safe_end)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[tracking] MLflow disabled ({exc}); continuing without it.")
            self._mlflow = None
        return self

    def end(self, status: str = "FINISHED") -> None:
        if not self.active:
            self._active = False
            return
        try:
            if self._owns_run:
                self._mlflow.end_run(status=status)
        except Exception:
            pass
        self._active = False

    def _safe_end(self) -> None:
        try:
            self.end()
        except Exception:
            pass

    # -- logging -----------------------------------------------------------
    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            self._mlflow.log_params(
                {k: _short(v) for k, v in params.items() if v is not None}
            )
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.active:
            return
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception:
            pass

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            self._mlflow.set_tags({k: _short(v) for k, v in tags.items()})
        except Exception:
            pass

    def log_artifact(self, path: str) -> None:
        if not self.active or not path or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path)
        except Exception:
            pass

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "RunTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end("FAILED" if exc_type else "FINISHED")
        return False
