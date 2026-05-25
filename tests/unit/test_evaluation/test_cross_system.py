"""End-to-end tests for the LOSO orchestrator on synthetic datasets.

The orchestrator is exercised with a *mock* trainer so the tests are
GPU-free and deterministic. The mock returns predictions derived
deterministically from the target labels (with a configurable error
rate) so we can assert specific metric values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hylog.data.dataset import LogDataset
from hylog.data.schema import LogSequence
from hylog.evaluation.cross_system import (
    LosoFold,
    build_folds,
    run_loso,
    strip_labels,
)
from hylog.evaluation.leakage_audit import LeakageError


def _make_dataset(source: str, *, n_normal: int = 6, n_anomaly: int = 4) -> LogDataset:
    items: list[LogSequence] = []
    for i in range(n_normal):
        items.append(
            LogSequence(
                lines=(f"{source} normal event <NUM>", f"{source} pad <NUM>"),
                label=0,
                group_id=f"{source}_n_{i}",
                source=source,
            )
        )
    for i in range(n_anomaly):
        items.append(
            LogSequence(
                lines=(f"{source} FATAL <HEX>", f"{source} pad <NUM>"),
                label=1,
                group_id=f"{source}_a_{i}",
                source=source,
            )
        )
    return LogDataset(items, source=source)


def _perfect_trainer(*, train, test, target_system):
    """Mock trainer returning perfect predictions for the test set."""
    y_true = np.asarray([s.label for s in test], dtype=np.int64)
    return y_true.copy(), y_true.astype(np.float64)


def _noisy_trainer(*, train, test, target_system):
    """Mock trainer with a fixed error rate (flip every 5th label)."""
    y_true = np.asarray([s.label for s in test], dtype=np.int64)
    y_pred = y_true.copy()
    for i in range(0, len(y_pred), 5):
        y_pred[i] = 1 - y_pred[i]
    scores = y_pred.astype(np.float64) * 0.9 + (1 - y_pred.astype(np.float64)) * 0.1
    return y_pred, scores


def test_build_folds_rotates_over_all_systems() -> None:
    datasets = {
        "a": _make_dataset("a"),
        "b": _make_dataset("b"),
        "c": _make_dataset("c"),
    }
    folds = build_folds(datasets=datasets)
    assert len(folds) == 3
    held = {f.target_system for f in folds}
    assert held == {"a", "b", "c"}
    for fold in folds:
        assert fold.target_system not in fold.train_sources
        assert set(fold.train_sources) == set(datasets) - {fold.target_system}


def test_build_folds_respects_include_list() -> None:
    datasets = {n: _make_dataset(n) for n in ("a", "b", "c", "d")}
    folds = build_folds(datasets=datasets, include=["a", "b"])
    assert {f.target_system for f in folds} == {"a", "b"}


def test_strip_labels_sets_all_to_zero() -> None:
    ds = _make_dataset("x", n_normal=2, n_anomaly=3)
    stripped = strip_labels(list(ds))
    assert all(s.label == 0 for s in stripped)
    # group_ids and lines preserved.
    assert [s.group_id for s in stripped] == [s.group_id for s in ds]


def test_run_loso_perfect_trainer_produces_perfect_metrics(tmp_path: Path) -> None:
    datasets = {n: _make_dataset(n) for n in ("hdfs", "bgl", "thunderbird")}
    summary = run_loso(
        datasets=datasets,
        trainer=_perfect_trainer,
        out_dir=tmp_path,
    )
    assert summary["n_folds"] == 3
    macro = summary["macro"]
    # Perfect trainer -> F1 = 1.0 on every fold.
    assert macro["f1"]["mean"] == pytest.approx(1.0)
    assert macro["f1"]["std"] == pytest.approx(0.0, abs=1e-9)
    for fold in summary["folds"]:
        assert fold["leakage_clean"] is True
        # Per-fold confusion + leakage artefacts on disk.
        ours = Path(fold["artefacts"]["csv"])
        assert ours.exists()
        leak = Path(fold["artefacts"]["leakage_json"])
        assert leak.exists()


def test_run_loso_writes_summary_json(tmp_path: Path) -> None:
    datasets = {n: _make_dataset(n) for n in ("a", "b", "c")}
    run_loso(datasets=datasets, trainer=_noisy_trainer, out_dir=tmp_path)
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "folds" in payload
    assert "macro" in payload


def test_run_loso_emits_predictions_jsonl(tmp_path: Path) -> None:
    datasets = {n: _make_dataset(n) for n in ("a", "b")}
    run_loso(datasets=datasets, trainer=_perfect_trainer, out_dir=tmp_path)
    preds = (tmp_path / "a" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in preds]
    assert all({"group_id", "y_true", "y_pred", "p_anomaly"} == set(p) for p in parsed)


def test_run_loso_detects_planted_leak(tmp_path: Path) -> None:
    """Planting a verbatim line copy from BGL into the HDFS test set must
    be caught by the orchestrator's leakage audit."""
    # Make HDFS test share a line with a BGL training sequence.
    shared_line = "leaked verbatim line <NUM>"
    bgl_items = list(_make_dataset("bgl"))
    bgl_items[0] = LogSequence(
        lines=(shared_line, "harmless pad"),
        label=0,
        group_id="bgl_leak_src",
        source="bgl",
    )
    hdfs_items = list(_make_dataset("hdfs"))
    hdfs_items[0] = LogSequence(
        lines=(shared_line, "another"),
        label=1,
        group_id="hdfs_leak_tgt",
        source="hdfs",
    )
    datasets = {
        "hdfs": LogDataset(hdfs_items, source="hdfs"),
        "bgl": LogDataset(bgl_items, source="bgl"),
    }
    with pytest.raises(LeakageError):
        run_loso(
            datasets=datasets,
            trainer=_perfect_trainer,
            out_dir=tmp_path,
            include=["hdfs"],
        )


def test_run_loso_non_strict_does_not_raise(tmp_path: Path) -> None:
    """When ``leakage_strict=False`` the run continues but the audit is
    archived."""
    shared = "leaked"
    a = LogDataset(
        [
            LogSequence(lines=(shared,), label=0, group_id="ga0", source="a"),
            LogSequence(lines=("unique-a",), label=1, group_id="ga1", source="a"),
        ],
        source="a",
    )
    b = LogDataset(
        [
            LogSequence(lines=(shared,), label=0, group_id="gb0", source="b"),
            LogSequence(lines=("unique-b",), label=1, group_id="gb1", source="b"),
        ],
        source="b",
    )
    summary = run_loso(
        datasets={"a": a, "b": b},
        trainer=_perfect_trainer,
        out_dir=tmp_path,
        leakage_strict=False,
    )
    # Both folds run; both archived a leakage report flagged as leakage.
    for fold in summary["folds"]:
        assert fold["leakage_clean"] is False


def test_loso_fold_dataclass_invariants() -> None:
    f = LosoFold(
        target_system="x",
        train_sources=("y", "z"),
        n_train_sequences=10,
        n_test_sequences=4,
    )
    assert f.target_system == "x"
    assert f.train_sources == ("y", "z")
