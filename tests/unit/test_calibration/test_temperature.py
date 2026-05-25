"""Tests for temperature scaling (Guo 2017)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hylog.calibration.ece import ece_only
from hylog.calibration.temperature import TemperatureCalibrator, fit_temperature


def _make_overconfident_logits(
    *, n: int = 500, seed: int = 0, scale: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build (logits, labels) where the classifier is correct ~70 % but
    expresses ~95 % confidence. Temperature scaling should sharply
    reduce its ECE."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Logits whose softmax is too peaky -> over-confidence.
    z = np.zeros((n, 2))
    correct = rng.random(n) < 0.70
    for i in range(n):
        true_class = int(labels[i])
        if correct[i]:
            z[i, true_class] = scale
        else:
            z[i, 1 - true_class] = scale
    return z, labels


def test_fit_returns_positive_temperature() -> None:
    logits, labels = _make_overconfident_logits()
    cal = fit_temperature(logits, labels)
    assert isinstance(cal, TemperatureCalibrator)
    assert cal.temperature > 0


def test_fit_reduces_nll() -> None:
    logits, labels = _make_overconfident_logits()
    cal = fit_temperature(logits, labels)
    assert cal.final_nll <= cal.init_nll + 1e-6


def test_temperature_reduces_ece_on_overconfident_model() -> None:
    logits, labels = _make_overconfident_logits(n=1000, seed=0)
    # ECE before any calibration:
    cal_identity = TemperatureCalibrator(
        temperature=1.0, n_calibration=1000, init_nll=0.0, final_nll=0.0, n_iters=0
    )
    ece_before = ece_only(cal_identity.apply(logits), labels)
    cal = fit_temperature(logits, labels)
    ece_after = ece_only(cal.apply(logits), labels)
    # Over-confident classifier -> T > 1 -> sharper drop in ECE.
    assert cal.temperature > 1.0
    assert ece_after < ece_before


def test_apply_is_class_preserving() -> None:
    """Temperature scaling never changes argmax."""
    logits, _ = _make_overconfident_logits(n=200, seed=1)
    cal = fit_temperature(logits, np.zeros(200, dtype=np.int64))
    probs = cal.apply(logits)
    assert np.all(probs.argmax(axis=1) == logits.argmax(axis=1))


def test_apply_returns_valid_probabilities() -> None:
    logits, labels = _make_overconfident_logits()
    cal = fit_temperature(logits, labels)
    probs = cal.apply(logits)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_to_dict_round_trip() -> None:
    logits, labels = _make_overconfident_logits()
    cal = fit_temperature(logits, labels)
    d = cal.to_dict()
    assert d["method"] == "temperature_scaling"
    assert d["temperature"] == pytest.approx(cal.temperature)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((5, 2)), np.zeros(4, dtype=np.int64))


def test_logits_not_2d_raises() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros(5), np.zeros(5, dtype=np.int64))


def test_underconfident_temperature_below_1() -> None:
    """Under-confident logits (small scale, correct often) yield T < 1."""
    rng = np.random.default_rng(2)
    n = 500
    labels = rng.integers(0, 2, size=n)
    z = np.zeros((n, 2))
    for i in range(n):
        # Always correct but with very small margin -> under-confident.
        z[i, int(labels[i])] = 0.2
    cal = fit_temperature(z, labels)
    assert cal.temperature < 1.0


def test_deterministic_for_same_inputs() -> None:
    logits, labels = _make_overconfident_logits()
    a = fit_temperature(logits, labels)
    b = fit_temperature(logits, labels)
    assert math.isclose(a.temperature, b.temperature, rel_tol=1e-6)
