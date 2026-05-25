"""Tests for the LICENSES.txt emitter."""

from __future__ import annotations

from pathlib import Path

from hylog.data.licenses import emit, render_notice


def test_render_notice_contains_all_datasets() -> None:
    spec = {
        "datasets": [
            {"name": "HDFS", "upstream": "https://example.com/hdfs"},
            {"name": "BGL", "citation": "Oliner 2007"},
        ]
    }
    body = render_notice(spec)
    assert "## HDFS" in body
    assert "## BGL" in body
    assert "https://example.com/hdfs" in body
    assert "Oliner 2007" in body


def test_emit_from_repo_yaml(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    yaml_path = repo_root / "data" / "licenses.yaml"
    out = emit(yaml_path, tmp_path / "LICENSES.txt")
    text = out.read_text(encoding="utf-8")
    assert "## HDFS" in text
    assert "## BGL" in text
    assert "## Thunderbird" in text
    assert "## OpenStack" in text
    assert "## Loghub-2.0" in text


def test_emit_is_deterministic(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent.parent.parent
    yaml_path = repo_root / "data" / "licenses.yaml"
    a = emit(yaml_path, tmp_path / "a.txt")
    b = emit(yaml_path, tmp_path / "b.txt")
    assert a.read_bytes() == b.read_bytes()
