"""Smoke tests for the hylog-calibrate CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from hylog.cli.calibrate import main


def _make_predictions_jsonl(path: Path, n: int = 400, seed: int = 0) -> None:
    """Build a synthetic predictions.jsonl with realistic miscalibration."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # 70 % accurate, 95 % confident -> mis-calibrated.
    correct = rng.random(n) < 0.70
    y_pred = np.where(correct, y_true, 1 - y_true)
    # p_anomaly is the predicted probability of class 1.
    p_anom = np.where(y_pred == 1, rng.uniform(0.90, 0.99, size=n), rng.uniform(0.01, 0.10, size=n))
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "group_id": f"g_{i}",
                        "y_true": int(y_true[i]),
                        "y_pred": int(y_pred[i]),
                        "p_anomaly": float(p_anom[i]),
                    }
                )
                + "\n"
            )


def test_calibrate_cli_end_to_end(tmp_path: Path) -> None:
    preds = tmp_path / "predictions.jsonl"
    _make_predictions_jsonl(preds)
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--predictions",
            str(preds),
            "--out-dir",
            str(out),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["method"] in ("temperature_scaling", "platt_scaling")
    assert "ece_after" in payload
    assert "aurc" in payload
    # Artefacts on disk.
    assert (out / "calibration.json").exists()
    assert (out / "aurc.json").exists()
    assert (out / "tau.json").exists()
    assert (out / "reliability.csv").exists()
    assert (out / "reliability_uncalibrated.csv").exists()


def test_calibrate_cli_too_few_rows(tmp_path: Path) -> None:
    preds = tmp_path / "predictions.jsonl"
    with preds.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(3):
            fh.write(
                json.dumps(
                    {"group_id": f"g{i}", "y_true": i % 2, "y_pred": i % 2, "p_anomaly": 0.5}
                )
                + "\n"
            )
    runner = CliRunner()
    result = runner.invoke(main, ["--predictions", str(preds), "--out-dir", str(tmp_path / "x")])
    assert result.exit_code == 2
