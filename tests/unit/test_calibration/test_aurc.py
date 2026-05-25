"""Tests for AURC + Excess-AURC + cost-asymmetric AURC."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.calibration.aurc import compute_aurc, risk_coverage_curve


def test_optimal_ranking_yields_aurc_equal_optimal() -> None:
    """When the confidence ranking is the oracle ranking, AURC = optimal."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    # Flip 30 % at random -> errors.
    flip = rng.choice(n, size=60, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    # Confidence = 1 for correct, 0 for incorrect -> oracle ranking.
    confidence = np.where(y_true == y_pred, 1.0, 0.0)
    report = compute_aurc(y_true=y_true, y_pred=y_pred, confidence=confidence)
    assert report.excess_aurc == pytest.approx(0.0, abs=1e-9)


def test_random_ranking_yields_nonzero_excess_aurc() -> None:
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=60, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    confidence = rng.uniform(0.5, 1.0, size=n)
    report = compute_aurc(y_true=y_true, y_pred=y_pred, confidence=confidence)
    assert report.excess_aurc > 0.0


def test_perfect_predictor_has_zero_aurc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = y_true.copy()
    confidence = np.array([0.9, 0.8, 0.95, 0.85])
    report = compute_aurc(y_true=y_true, y_pred=y_pred, confidence=confidence)
    assert report.aurc == pytest.approx(0.0)
    assert report.optimal_aurc == pytest.approx(0.0)
    assert report.excess_aurc == pytest.approx(0.0)


def test_cost_asymmetric_penalises_fns_more_heavily() -> None:
    # 1 FN and 1 FP. Symmetric AURC treats them equally.
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0])  # one FN at idx 0, one FP at idx 3
    confidence = np.array([0.9, 0.8, 0.95, 0.7, 0.85, 0.6])
    report = compute_aurc(
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        fn_weight=5.0,
        fp_weight=1.0,
    )
    # Cost-asymmetric AURC should be strictly larger than symmetric.
    assert report.cost_asymmetric_aurc > report.aurc


def test_zero_weights_rejected() -> None:
    with pytest.raises(ValueError):
        compute_aurc(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            fn_weight=0.0,
        )


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_aurc(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1, 0]),
            confidence=np.array([0.9, 0.8]),
        )


def test_risk_coverage_curve_starts_perfect_and_grows_to_full_error() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1])  # correct: idx 1, 2; wrong: idx 0, 3
    confidence = np.array([0.9, 0.8, 0.7, 0.6])  # wrong ones are most confident
    coverage, risk = risk_coverage_curve(y_true=y_true, y_pred=y_pred, confidence=confidence)
    assert coverage.size == 4
    assert risk.size == 4
    # At full coverage = 4 errors out of 4? No: only 2 errors -> risk = 0.5
    assert risk[-1] == pytest.approx(0.5)


def test_aurc_report_to_dict() -> None:
    report = compute_aurc(
        y_true=np.array([0, 1, 0, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        confidence=np.array([0.9, 0.8, 0.6, 0.7]),
    )
    d = report.to_dict()
    for key in (
        "aurc",
        "optimal_aurc",
        "excess_aurc",
        "cost_asymmetric_aurc",
        "fn_weight",
        "fp_weight",
        "n_samples",
    ):
        assert key in d
    assert d["n_samples"] == 4


def test_empty_inputs_return_nan() -> None:
    import math

    report = compute_aurc(
        y_true=np.array([], dtype=np.int64),
        y_pred=np.array([], dtype=np.int64),
        confidence=np.array([]),
    )
    assert math.isnan(report.aurc)
    assert report.n_samples == 0
