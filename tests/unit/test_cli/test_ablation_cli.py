"""Smoke tests for the hylog-ablation CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from click.testing import CliRunner

from hylog.cli.ablation import main

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_single_axis_run_end_to_end(tmp_path: Path) -> None:
    cfg = REPO_ROOT / "configs" / "ablation" / "a2_lora_rank.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--axis", str(cfg), "--out-dir", str(tmp_path), "--mock"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_axes"] == 1
    # Per-axis artefacts.
    axis_dir = tmp_path / "A2_lora_rank"
    assert (axis_dir / "A2_lora_rank.csv").exists()
    assert (axis_dir / "A2_lora_rank.md").exists()
    assert (axis_dir / "A2_lora_rank_raw.json").exists()
    # Global ablation matrix.
    assert (tmp_path / "ablation_matrix.csv").exists()


def test_all_axes_run_writes_global_matrix(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--all-axes",
            str(REPO_ROOT / "configs" / "ablation"),
            "--out-dir",
            str(tmp_path),
            "--mock",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_axes"] == 8
    matrix = tmp_path / "ablation_matrix.csv"
    rows = list(csv.reader(matrix.read_text(encoding="utf-8").splitlines()))
    # Header + at least one non-baseline comparison per axis (8 axes -> >= 8 rows).
    assert len(rows) >= 1 + 8


def test_without_mock_exits_with_message() -> None:
    cfg = REPO_ROOT / "configs" / "ablation" / "a1_hybrid_vs_standalone.yaml"
    runner = CliRunner()
    result = runner.invoke(main, ["--axis", str(cfg)])
    assert result.exit_code == 0
    assert "GPU-bound" in result.output or "--mock" in result.output


def test_no_args_exits_two() -> None:
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2


def test_mutually_exclusive_args_exits_two(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--axis",
            str(REPO_ROOT / "configs" / "ablation" / "a1_hybrid_vs_standalone.yaml"),
            "--all-axes",
            str(REPO_ROOT / "configs" / "ablation"),
            "--mock",
        ],
    )
    assert result.exit_code == 2
