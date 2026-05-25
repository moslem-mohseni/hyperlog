"""Tests for Platt scaling."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.calibration.platt import PlattCalibrator, fit_platt


def test_fit_runs_and_returns_calibrator() -> None:
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    # Scores correlated with labels but mis-scaled.
    scores = labels * 3.0 - 1.5 + rng.normal(scale=0.5, size=n)
    cal = fit_platt(scores, labels)
    assert isinstance(cal, PlattCalibrator)


def test_apply_logit_returns_probabilities_in_range() -> None:
    rng = np.random.default_rng(1)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    scores = labels * 2.0 - 1.0 + rng.normal(scale=0.3, size=n)
    cal = fit_platt(scores, labels)
    p = cal.apply_logit(scores)
    assert p.shape == (n,)
    assert np.all((p >= 0) & (p <= 1))


def test_apply_returns_2_column_softmax() -> None:
    rng = np.random.default_rng(2)
    n = 100
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    scores = labels * 2.0 - 1.0
    cal = fit_platt(scores, labels)
    logits = np.stack([np.zeros(n), scores], axis=1)
    probs = cal.apply(logits)
    assert probs.shape == (n, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_apply_wrong_shape_raises() -> None:
    cal = PlattCalibrator(a=1.0, b=0.0, n_calibration=10, init_nll=0.0, final_nll=0.0)
    with pytest.raises(ValueError):
        cal.apply(np.zeros((5, 3)))


def test_shape_mismatch_in_fit_raises() -> None:
    with pytest.raises(ValueError):
        fit_platt(np.zeros(5), np.zeros(4, dtype=np.float64))


def test_to_dict_has_method_marker() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=50)
    labels = (scores > 0).astype(np.float64)
    cal = fit_platt(scores, labels)
    d = cal.to_dict()
    assert d["method"] == "platt_scaling"
    assert "a" in d and "b" in d
