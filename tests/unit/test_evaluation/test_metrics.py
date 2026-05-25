"""Tests for the classification metric panel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hylog.evaluation.metrics import (
    auc_pr,
    auc_roc,
    compute_metric_panel,
    confusion_counts,
    f1,
    false_positive_rate,
    fpr_at_recall,
    matthews_correlation_coefficient,
    precision,
    recall,
)


def test_confusion_counts_simple() -> None:
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    c = confusion_counts(y_true, y_pred)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)


def test_precision_recall_f1() -> None:
    c = confusion_counts(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]))
    assert precision(c) == 1.0
    assert recall(c) == 1.0
    assert f1(c) == 1.0


def test_fpr() -> None:
    c = confusion_counts(np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0]))
    assert false_positive_rate(c) == pytest.approx(0.25)


def test_mcc_perfect_and_random() -> None:
    perfect = confusion_counts(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
    assert matthews_correlation_coefficient(perfect) == pytest.approx(1.0)
    none_correct = confusion_counts(np.array([1, 0]), np.array([0, 1]))
    assert matthews_correlation_coefficient(none_correct) == pytest.approx(-1.0)


def test_auc_roc_known_value() -> None:
    # Perfect ranking: AUC=1.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_roc(y_true, y_score) == pytest.approx(1.0)
    # Inverted: AUC=0.
    assert auc_roc(y_true, -y_score) == pytest.approx(0.0)


def test_auc_pr_perfect() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_pr(y_true, y_score) == pytest.approx(1.0)


def test_fpr_at_recall() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.85, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.1])
    # Recall=1.0 reached at index 3 (after 4 positives). All negatives are
    # ranked below the 4th positive (score 0.6), so threshold = 0.6 gives
    # zero FP -> FPR = 0/5 = 0.
    fpr = fpr_at_recall(y_true, y_score, target_recall=1.0)
    assert fpr == pytest.approx(0.0)


def test_metric_panel_returns_all_fields() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0])
    y_score = np.array([0.9, 0.1, 0.85, 0.6, 0.45])
    panel = compute_metric_panel(y_true, y_pred, y_score)
    d = panel.to_dict()
    for key in (
        "precision",
        "recall",
        "f1",
        "fpr",
        "fpr_at_recall_95",
        "mcc",
        "auc_roc",
        "auc_pr",
    ):
        assert key in d
        assert isinstance(d[key], float)


def test_metric_panel_without_scores() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    panel = compute_metric_panel(y_true, y_pred, y_score=None)
    assert math.isnan(panel.auc_roc)
    assert math.isnan(panel.auc_pr)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        confusion_counts(np.array([0, 1]), np.array([0, 1, 0]))
