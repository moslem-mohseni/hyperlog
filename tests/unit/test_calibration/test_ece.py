"""Tests for ECE / MCE / reliability bins."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hylog.calibration.ece import (
    compute_reliability_bins,
    ece_only,
)


def _perfectly_calibrated(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build perfectly-calibrated probabilities: conf c -> accuracy c."""
    rng = np.random.default_rng(seed)
    confidences = rng.uniform(0.5, 1.0, size=n)
    labels = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(confidences):
        # Each sample is correct with probability == confidence.
        if rng.random() < c:
            labels[i] = 1  # we will predict class 1, so "correct" means label=1
    # Build [N, 2] softmax outputs with class 1 confidence equal to ``c``.
    probs = np.stack([1.0 - confidences, confidences], axis=1)
    return probs, labels


def test_perfect_calibration_has_low_ece() -> None:
    probs, labels = _perfectly_calibrated(n=5000)
    report = compute_reliability_bins(probs, labels, n_bins=15)
    assert report.ece < 0.05  # within the Phase-5 budget


def test_over_confident_classifier_has_high_ece() -> None:
    rng = np.random.default_rng(42)
    n = 1000
    labels = rng.integers(0, 2, size=n)
    correct = rng.random(n) < 0.5  # 50 % accuracy
    probs = np.zeros((n, 2))
    for i in range(n):
        cls = int(labels[i]) if correct[i] else 1 - int(labels[i])
        probs[i, cls] = 0.99
        probs[i, 1 - cls] = 0.01
    # The model is 50 % accurate but 99 % confident -> large ECE.
    report = compute_reliability_bins(probs, labels, n_bins=15)
    assert report.ece > 0.4


def test_ece_only_returns_scalar() -> None:
    probs, labels = _perfectly_calibrated(n=200)
    e = ece_only(probs, labels)
    assert isinstance(e, float)
    assert 0.0 <= e <= 1.0


def test_report_to_dict_has_keys() -> None:
    probs, labels = _perfectly_calibrated(n=100)
    report = compute_reliability_bins(probs, labels, n_bins=10)
    d = report.to_dict()
    for key in ("ece", "mce", "n_samples", "n_bins", "bins"):
        assert key in d
    assert len(d["bins"]) == 10


def test_empty_bins_are_nan() -> None:
    """If a bin has zero predictions, its accuracy/confidence is NaN."""
    # Build probs where all confidences fall in a narrow range -> most bins empty.
    n = 100
    probs = np.full((n, 2), 0.5)
    probs[:, 1] = 0.85  # all confidences 0.85, plus class 1 is argmax
    probs[:, 0] = 0.15
    labels = np.ones(n, dtype=np.int64)
    report = compute_reliability_bins(probs, labels, n_bins=15)
    empty = [b for b in report.bins if b.count == 0]
    assert empty
    for b in empty:
        assert math.isnan(b.confidence_mean)
        assert math.isnan(b.accuracy)


def test_well_calibrated_predicate() -> None:
    probs, labels = _perfectly_calibrated(n=5000)
    report = compute_reliability_bins(probs, labels)
    assert report.is_well_calibrated(threshold=0.10)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_reliability_bins(np.zeros((5, 2)), np.zeros(4, dtype=np.int64))


def test_n_bins_lower_bound() -> None:
    with pytest.raises(ValueError):
        compute_reliability_bins(np.zeros((10, 2)), np.zeros(10, dtype=np.int64), n_bins=1)
