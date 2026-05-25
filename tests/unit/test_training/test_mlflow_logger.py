"""Tests for the MLflow logger wrapper."""

from __future__ import annotations

from pathlib import Path

from hylog.training.mlflow_logger import MLflowLogger


def test_noop_logger_does_not_raise() -> None:
    logger = MLflowLogger(experiment="test", track=False)
    with logger.start_run(run_name="r1"):
        logger.log_params({"lr": 1e-3})
        logger.log_metrics({"loss": 0.5}, step=1)


def test_noop_log_artifact_accepts_path(tmp_path: Path) -> None:
    logger = MLflowLogger(track=False)
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with logger.start_run():
        logger.log_artifact(f)
