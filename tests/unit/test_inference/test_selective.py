"""Tests for the selective predictor + auto-tau."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.inference.selective import (
    LABEL_NAMES,
    select_one,
    select_tau_for_risk_budget,
)


def test_select_one_high_confidence_emits_class() -> None:
    probs = np.array([[0.05, 0.95], [0.90, 0.10]])
    preds = select_one(probabilities=probs, threshold=0.7)
    assert [p.decision for p in preds] == ["anomaly", "normal"]


def test_select_one_below_threshold_abstains() -> None:
    probs = np.array([[0.55, 0.45]])
    preds = select_one(probabilities=probs, threshold=0.7)
    assert preds[0].decision == "abstain"


def test_select_one_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        select_one(probabilities=np.zeros((5, 3)), threshold=0.7)


def test_select_one_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        select_one(probabilities=np.zeros((1, 2)), threshold=0.4)
    with pytest.raises(ValueError):
        select_one(probabilities=np.zeros((1, 2)), threshold=1.1)


def test_auto_tau_finds_feasible_threshold() -> None:
    """A perfectly accurate classifier with mixed confidences can hit
    any risk budget with τ = 0.5 (all accepted)."""
    n = 200
    y = np.tile([0, 1], n // 2)
    confs = np.linspace(0.51, 0.99, n)
    probs = np.zeros((n, 2))
    for i in range(n):
        probs[i, int(y[i])] = confs[i]
        probs[i, 1 - int(y[i])] = 1 - confs[i]
    result = select_tau_for_risk_budget(probabilities=probs, labels=y, risk_budget=0.05)
    assert result.feasible
    assert result.achieved_risk == pytest.approx(0.0)
    assert result.achieved_coverage > 0.99


def test_auto_tau_infeasible_returns_threshold_one() -> None:
    """A pathologically wrong classifier whose every confident
    prediction is wrong cannot meet any positive risk budget."""
    n = 100
    y = np.zeros(n, dtype=np.int64)
    probs = np.zeros((n, 2))
    # Predict class 1 with high confidence on every class-0 sample -> 100 % error.
    probs[:, 1] = 0.99
    probs[:, 0] = 0.01
    result = select_tau_for_risk_budget(probabilities=probs, labels=y, risk_budget=0.05)
    assert not result.feasible
    assert result.threshold == 1.0


def test_auto_tau_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError):
        select_tau_for_risk_budget(
            probabilities=np.zeros((2, 2)),
            labels=np.zeros(2, dtype=np.int64),
            risk_budget=0.0,
        )
    with pytest.raises(ValueError):
        select_tau_for_risk_budget(
            probabilities=np.zeros((2, 2)),
            labels=np.zeros(2, dtype=np.int64),
            risk_budget=1.0,
        )


def test_label_names_match_class_indices() -> None:
    """LABEL_NAMES[0]=normal, [1]=anomaly — drives the decision strings."""
    assert LABEL_NAMES == ("normal", "anomaly")


def test_select_one_per_sample_threshold_in_output() -> None:
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    preds = select_one(probabilities=probs, threshold=0.65)
    assert preds[0].decision == "abstain"
    assert preds[1].decision == "anomaly"
    for p in preds:
        assert p.threshold == 0.65
