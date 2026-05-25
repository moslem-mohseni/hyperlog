"""Thin MLflow wrapper used by the trainer for archiving runs.

The wrapper supports two modes:

- **Live MLflow** when the ``mlflow`` package is importable and
  ``track=True`` — the canonical mode used during real GPU training.
- **No-op** otherwise — used in CPU-only architectural tests and on
  machines without an MLflow daemon.

Both modes share the same public surface so calling code never branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any


class _NoOpRun:
    """No-op context manager. Same surface as an MLflow run."""

    def __enter__(self) -> _NoOpRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None

    def log_artifact(self, local_path: str | Path) -> None:
        return None


@dataclass(slots=True)
class MLflowLogger:
    """Adapter around ``mlflow.start_run``.

    Usage::

        logger = MLflowLogger(experiment="phase2-logllm-hdfs")
        with logger.start_run(run_name="seed=42"):
            logger.log_params({"lr_1": 5e-4})
            logger.log_metrics({"train_loss": 0.1}, step=10)
    """

    experiment: str = "hylog-default"
    tracking_uri: str | None = None
    track: bool = True
    _active: bool = field(default=False, init=False)

    def start_run(self, run_name: str | None = None) -> Any:
        if not self.track or not _mlflow_available():
            return _NoOpRun()

        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        self._active = True
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._active or not _mlflow_available():
            return
        import mlflow

        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if not self._active or not _mlflow_available():
            return
        import mlflow

        mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        if not self._active or not _mlflow_available():
            return
        import mlflow

        mlflow.log_artifact(str(local_path))


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = ["MLflowLogger"]
