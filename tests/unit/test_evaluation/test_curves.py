"""Tests for ROC + PR curves and archive."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from hylog.evaluation.curves import (
    archive_curves,
    compute_pr,
    compute_roc,
    write_pr_csv,
    write_roc_csv,
)


def test_perfect_roc_auc_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    roc = compute_roc(y_true, y_score)
    assert roc.auc() == pytest.approx(1.0)


def test_perfect_pr_ap_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    pr = compute_pr(y_true, y_score)
    assert pr.average_precision() == pytest.approx(1.0)


def test_roc_pr_handle_single_class() -> None:
    y_true = np.zeros(5, dtype=int)
    y_score = np.ones(5) * 0.5
    roc = compute_roc(y_true, y_score)
    pr = compute_pr(y_true, y_score)
    assert roc.thresholds.size == 0
    assert pr.thresholds.size == 0


def test_write_roc_csv_columns(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.8])
    roc = compute_roc(y_true, y_score)
    path = write_roc_csv(roc, tmp_path / "roc.csv")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["threshold", "fpr", "tpr"]
    assert len(rows) > 1


def test_write_pr_csv_columns(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.4, 0.8])
    pr = compute_pr(y_true, y_score)
    path = write_pr_csv(pr, tmp_path / "pr.csv")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["threshold", "recall", "precision"]
    assert len(rows) > 1


def test_archive_curves_writes_csvs(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.3, 0.7, 0.8, 0.2])
    artefacts = archive_curves(y_true=y_true, y_score=y_score, out_dir=tmp_path, fold_name="held")
    assert artefacts["roc_csv"].exists()
    assert artefacts["pr_csv"].exists()
