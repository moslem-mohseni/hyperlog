"""Tests for the reliability-diagram archive helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hylog.calibration.ece import compute_reliability_bins
from hylog.calibration.reliability import archive_all, write_csv


def _toy_report(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    confs = rng.uniform(0.5, 1.0, size=n)
    probs = np.stack([1.0 - confs, confs], axis=1)
    labels = (rng.random(n) < confs).astype(np.int64)
    return compute_reliability_bins(probs, labels, n_bins=10)


def test_write_csv_header_and_row_count(tmp_path: Path) -> None:
    report = _toy_report()
    path = write_csv(report, tmp_path / "rel.csv")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["lower", "upper", "count", "confidence_mean", "accuracy"]
    assert len(rows) == 1 + 10  # header + 10 bins


def test_archive_all_writes_csv(tmp_path: Path) -> None:
    report = _toy_report()
    artefacts = archive_all(report, out_dir=tmp_path, name="rel")
    assert artefacts["csv"].exists()
    if "png" in artefacts:
        assert artefacts["png"].exists()


def test_csv_records_nan_as_empty_string(tmp_path: Path) -> None:
    """An empty bin should render its acc/conf cells as empty (CSV-friendly)."""
    n = 50
    probs = np.full((n, 2), 0.5)
    probs[:, 1] = 0.9
    probs[:, 0] = 0.1
    labels = np.ones(n, dtype=np.int64)
    report = compute_reliability_bins(probs, labels, n_bins=10)
    path = write_csv(report, tmp_path / "rel.csv")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    # At least one bin will have count=0; verify that row has empty cells.
    empty_rows = [r for r in rows[1:] if r[2] == "0"]
    assert empty_rows
    for r in empty_rows:
        assert r[3] == ""
        assert r[4] == ""
