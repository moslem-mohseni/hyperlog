"""Smoke tests for the hylog-loso CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from hylog.cli.loso import main

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_loso_cli_mock_run_succeeds(tmp_path: Path) -> None:
    cfg = REPO_ROOT / "configs" / "experiments" / "loso_hdfs_held.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(cfg),
            "--out-dir",
            str(tmp_path),
            "--mock",
            "--bootstrap-n",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["held_out"] == "hdfs"
    assert "bgl" in payload["train_sources"]
    summary_path = Path(payload["summary_json"])
    assert summary_path.exists()
    parsed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "macro" in parsed
    assert "macro_bootstrap" in parsed


def test_loso_cli_without_mock_returns_zero_with_message() -> None:
    """The real-trainer path is not yet wired; the CLI must exit 0 cleanly
    with an informative message."""
    cfg = REPO_ROOT / "configs" / "experiments" / "loso_hdfs_held.yaml"
    runner = CliRunner()
    result = runner.invoke(main, ["--config", str(cfg)])
    assert result.exit_code == 0
    assert "Phase 5+" in result.output or "--mock" in result.output
