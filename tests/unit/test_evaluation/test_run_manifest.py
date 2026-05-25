"""Tests for the RunManifest helper."""

from __future__ import annotations

import json
from pathlib import Path

from hylog.evaluation.run_manifest import RunManifest


def test_manifest_has_expected_top_level_keys(tmp_path: Path) -> None:
    m = RunManifest(run_name="test-run")
    m.stop()
    payload = m.to_dict()
    for key in (
        "run_name",
        "started_at_utc",
        "finished_at_utc",
        "wallclock_seconds",
        "git",
        "env",
        "package",
    ):
        assert key in payload


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = RunManifest(run_name="r1", config={"key": "value"})
    m.stop()
    path = m.write(tmp_path / "manifest.json")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["run_name"] == "r1"
    assert parsed["config"]["key"] == "value"
    assert parsed["wallclock_seconds"] is not None


def test_manifest_includes_splits_hashes_when_dir_given(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "a.json").write_text('{"x": 1}', encoding="utf-8", newline="\n")
    (splits / "b.json").write_text('{"y": 2}', encoding="utf-8", newline="\n")
    m = RunManifest(run_name="r2", splits_dir=splits)
    m.stop()
    payload = m.to_dict()
    assert "splits_hashes" in payload
    assert set(payload["splits_hashes"]) == {"a.json", "b.json"}


def test_manifest_env_includes_python_version() -> None:
    m = RunManifest(run_name="r3")
    m.stop()
    env = m.to_dict()["env"]
    assert env["python_version"].count(".") >= 1
    assert isinstance(env["cpu_count"], int)


def test_wallclock_none_before_stop() -> None:
    m = RunManifest(run_name="r4")
    assert m.wallclock_seconds is None
