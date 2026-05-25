"""Tests for the confusion-matrix renderer."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hylog.evaluation.confusion_renderer import (
    CLASS_LABELS,
    ConfusionMatrix,
    archive_all,
    render_text,
    write_csv,
)
from hylog.evaluation.metrics import ConfusionCounts


def test_from_counts_canonical_layout() -> None:
    c = ConfusionCounts(tp=10, fp=2, tn=80, fn=8)
    m = ConfusionMatrix.from_counts(c)
    # matrix[true][pred] = [[tn, fp], [fn, tp]]
    assert m.matrix == ((80, 2), (8, 10))


def test_from_arrays_matches_counts() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1])
    m = ConfusionMatrix.from_arrays(y_true, y_pred)
    # tp=3, fn=1, fp=1, tn=2 -> [[2,1],[1,3]]
    assert m.matrix == ((2, 1), (1, 3))


def test_row_normalize_sums_to_one_per_row() -> None:
    m = ConfusionMatrix(((4, 1), (2, 8)))
    norm = m.row_normalize()
    assert np.allclose(norm.sum(axis=1), [1.0, 1.0])


def test_render_text_contains_labels() -> None:
    m = ConfusionMatrix(((10, 0), (0, 5)))
    out = render_text(m, title="t-1")
    assert "t-1" in out
    for label in CLASS_LABELS:
        assert label in out
    # Row-normalised display present.
    assert "Row-normalised" in out


def test_write_csv_round_trips(tmp_path: Path) -> None:
    m = ConfusionMatrix(((7, 3), (1, 9)))
    path = write_csv(m, tmp_path / "cm.csv")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    # Header row + 2 data rows.
    assert rows[0] == ["true \\ pred", *CLASS_LABELS]
    assert rows[1] == ["normal", "7", "3"]
    assert rows[2] == ["anomaly", "1", "9"]


def test_archive_all_writes_csv_and_text(tmp_path: Path) -> None:
    m = ConfusionMatrix(((50, 5), (2, 43)))
    artefacts = archive_all(m, out_dir=tmp_path, name="fold_hdfs_held", title="t-2")
    assert "csv" in artefacts and artefacts["csv"].exists()
    assert "text" in artefacts and artefacts["text"].exists()
    # PNG is best-effort.
    if "png" in artefacts:
        assert artefacts["png"].exists()


def test_zero_row_does_not_divide_by_zero() -> None:
    m = ConfusionMatrix(((0, 0), (5, 5)))
    norm = m.row_normalize()
    # First row was all zeros; division-by-zero guard keeps it at zero.
    assert np.allclose(norm[0], [0.0, 0.0])
    assert np.allclose(norm[1], [0.5, 0.5])
