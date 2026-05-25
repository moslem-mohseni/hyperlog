"""Tests for the cross-system leakage audit.

The audit is the load-bearing methodological safeguard for novelty
claim N2 (zero-target-label cross-system protocol). These tests
verify that:

- A clean (disjoint) train/test pair passes.
- A *planted* leak — a single line copied from train into test — is
  detected with a deterministic sample.
- A group_id collision is detected even if every line is unique.
- ``assert_clean`` raises ``LeakageError`` on leaky reports.
- The JSON serialisation is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hylog.data.schema import LogSequence
from hylog.evaluation.leakage_audit import (
    LEAKAGE_SAMPLE_CAP,
    LeakageError,
    assert_clean,
    audit_leakage,
    write_report,
)


def _seq(lines: tuple[str, ...], label: int, group_id: str, source: str) -> LogSequence:
    return LogSequence(lines=lines, label=label, group_id=group_id, source=source)


def test_disjoint_inputs_are_clean() -> None:
    train = [_seq(("alpha event <NUM>",), 0, "g0", "a")]
    test = [_seq(("beta event <NUM>",), 0, "g1", "b")]
    report = audit_leakage(train=train, test=test)
    assert report.is_clean
    assert report.verdict == "clean"
    assert report.line_intersection == 0
    assert report.group_intersection == 0
    assert report.train_lines == 1
    assert report.test_lines == 1


def test_planted_line_leak_is_detected() -> None:
    leaked = "FATAL kernel panic at <HEX>"
    train = [_seq(("normal idle <NUM>", leaked), 0, "g0", "a")]
    test = [_seq((leaked, "different unique line <NUM>"), 1, "g1", "b")]
    report = audit_leakage(train=train, test=test)
    assert not report.is_clean
    assert report.verdict == "leakage"
    assert report.line_intersection == 1
    assert leaked in report.leaked_line_samples


def test_group_id_collision_is_detected() -> None:
    train = [_seq(("unique line 1",), 0, "g42", "a")]
    test = [_seq(("unique line 2",), 1, "g42", "b")]
    report = audit_leakage(train=train, test=test)
    assert report.group_intersection == 1
    assert "g42" in report.leaked_group_samples
    assert not report.is_clean


def test_assert_clean_raises_on_leakage() -> None:
    train = [_seq(("same line",), 0, "g0", "a")]
    test = [_seq(("same line",), 0, "g1", "b")]
    with pytest.raises(LeakageError) as ei:
        assert_clean(audit_leakage(train=train, test=test))
    assert "leakage detected" in str(ei.value)


def test_assert_clean_passes_on_clean_report() -> None:
    train = [_seq(("alpha",), 0, "g0", "a")]
    test = [_seq(("beta",), 0, "g1", "b")]
    assert_clean(audit_leakage(train=train, test=test))  # must not raise


def test_leaked_samples_capped() -> None:
    """Sample list never exceeds LEAKAGE_SAMPLE_CAP regardless of leak size."""
    leaked_lines = tuple(f"leaked line {i}" for i in range(LEAKAGE_SAMPLE_CAP + 10))
    train = [_seq(leaked_lines, 0, "gA", "a")]
    test = [_seq(leaked_lines, 0, "gB", "b")]
    report = audit_leakage(train=train, test=test)
    assert report.line_intersection == LEAKAGE_SAMPLE_CAP + 10
    assert len(report.leaked_line_samples) <= LEAKAGE_SAMPLE_CAP


def test_report_json_is_deterministic(tmp_path: Path) -> None:
    train = [_seq(("a",), 0, "g0", "x")]
    test = [_seq(("a",), 1, "g1", "y")]
    r1 = audit_leakage(train=train, test=test)
    r2 = audit_leakage(train=train, test=test)
    assert r1.to_json() == r2.to_json()
    # Round-trips through json.dumps.
    payload = json.loads(r1.to_json())
    assert payload["verdict"] == "leakage"
    assert payload["line_intersection"] == 1


def test_write_report_round_trip(tmp_path: Path) -> None:
    train = [_seq(("alpha",), 0, "g0", "a")]
    test = [_seq(("beta",), 0, "g1", "b")]
    report = audit_leakage(train=train, test=test)
    path = write_report(report, tmp_path / "leak.json")
    assert path.exists()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["verdict"] == "clean"
    assert parsed["line_intersection"] == 0


def test_audit_handles_empty_test() -> None:
    train = [_seq(("a",), 0, "g0", "x")]
    report = audit_leakage(train=train, test=[])
    assert report.test_lines == 0
    assert report.is_clean
