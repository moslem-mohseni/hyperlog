"""Smoke tests for CLI entry points."""

from __future__ import annotations

from click.testing import CliRunner

from hylog.cli.predict import main as predict_main
from hylog.cli.train import main as train_main


def test_train_dry_run_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(train_main, ["--dry-run"])
    assert result.exit_code == 0
    assert "dry-run OK" in result.output


def test_train_no_args_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(train_main, [])
    assert result.exit_code == 0


def test_predict_no_args_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(predict_main, [])
    assert result.exit_code == 0
