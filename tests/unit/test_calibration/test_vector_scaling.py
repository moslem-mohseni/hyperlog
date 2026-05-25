"""Tests for vector scaling."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.calibration.vector_scaling import fit_vector_scaling


def test_fit_returns_one_weight_per_class() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 3, size=n)
    z = np.zeros((n, 3))
    for i in range(n):
        z[i, int(labels[i])] = 2.0
    cal = fit_vector_scaling(z, labels)
    assert len(cal.weights) == 3


def test_apply_returns_softmax_probabilities() -> None:
    rng = np.random.default_rng(1)
    n = 100
    labels = rng.integers(0, 2, size=n)
    z = np.zeros((n, 2))
    for i in range(n):
        z[i, int(labels[i])] = 1.5
    cal = fit_vector_scaling(z, labels)
    probs = cal.apply(z)
    assert probs.shape == (n, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((probs >= 0) & (probs <= 1))


def test_logits_not_2d_raises() -> None:
    with pytest.raises(ValueError):
        fit_vector_scaling(np.zeros(5), np.zeros(5, dtype=np.int64))


def test_dimension_mismatch_in_apply_raises() -> None:
    cal_logits = np.zeros((20, 2))
    cal_labels = np.zeros(20, dtype=np.int64)
    cal_labels[10:] = 1
    cal_logits[np.arange(20), cal_labels] = 1.0
    cal = fit_vector_scaling(cal_logits, cal_labels)
    with pytest.raises(ValueError):
        cal.apply(np.zeros((4, 3)))


def test_reduces_nll() -> None:
    rng = np.random.default_rng(2)
    n = 300
    labels = rng.integers(0, 2, size=n)
    z = np.zeros((n, 2))
    for i in range(n):
        # Heavily over-confident logits.
        z[i, int(labels[i])] = 5.0
    cal = fit_vector_scaling(z, labels)
    assert cal.final_nll <= cal.init_nll + 1e-6


def test_to_dict_marker() -> None:
    z = np.zeros((10, 2))
    labels = np.zeros(10, dtype=np.int64)
    labels[5:] = 1
    z[np.arange(10), labels] = 1.0
    cal = fit_vector_scaling(z, labels)
    d = cal.to_dict()
    assert d["method"] == "vector_scaling"
    assert isinstance(d["weights"], list)
