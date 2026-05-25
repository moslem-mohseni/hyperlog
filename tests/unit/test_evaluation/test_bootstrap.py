"""Tests for bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.evaluation.bootstrap import (
    BootstrapInterval,
    aggregate_macro,
    bootstrap_metric_panel,
    format_ci,
)


def test_bootstrap_returns_intervals_for_every_metric() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    # Flip 10% to introduce error.
    flip = rng.choice(n, size=20, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_score = y_pred.astype(float) * 0.9 + (1 - y_pred.astype(float)) * 0.1

    intervals = bootstrap_metric_panel(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        n_bootstrap=200,
        seed=42,
    )
    import math

    for name in ("precision", "recall", "f1", "auc_roc", "auc_pr", "mcc"):
        assert name in intervals, f"missing CI for {name}"
        ci = intervals[name]
        assert isinstance(ci, BootstrapInterval)
        if math.isnan(ci.point_estimate):
            assert math.isnan(ci.ci_low) and math.isnan(ci.ci_high)
            continue
        # Bounds are well-ordered. The percentile bootstrap can place the
        # point estimate slightly outside the CI for non-linear functionals
        # (AUC-PR on tied scores is a known case), so we do not assert
        # containment here.
        assert ci.ci_low <= ci.ci_high
        assert -1.0 <= ci.point_estimate <= 1.0 + 1e-9


def test_bootstrap_seed_determinism() -> None:
    n = 100
    y_true = np.tile([0, 1], n // 2)
    y_pred = y_true.copy()
    y_score = y_pred.astype(float)
    a = bootstrap_metric_panel(
        y_true=y_true, y_pred=y_pred, y_score=y_score, n_bootstrap=200, seed=7
    )
    b = bootstrap_metric_panel(
        y_true=y_true, y_pred=y_pred, y_score=y_score, n_bootstrap=200, seed=7
    )
    for k in a:
        assert a[k].ci_low == b[k].ci_low
        assert a[k].ci_high == b[k].ci_high


def test_bootstrap_rejects_small_n() -> None:
    with pytest.raises(ValueError):
        bootstrap_metric_panel(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            n_bootstrap=10,
        )


def test_bootstrap_degenerate_single_class_returns_point_estimate() -> None:
    import math

    n = 50
    y_true = np.zeros(n, dtype=np.int64)
    y_pred = np.zeros(n, dtype=np.int64)
    y_score = np.zeros(n)
    intervals = bootstrap_metric_panel(
        y_true=y_true, y_pred=y_pred, y_score=y_score, n_bootstrap=200
    )
    for ci in intervals.values():
        # CI collapses to the point estimate (which may itself be NaN
        # for metrics like AUC that need both classes).
        if math.isnan(ci.point_estimate):
            assert math.isnan(ci.ci_low) and math.isnan(ci.ci_high)
        else:
            assert ci.ci_low == ci.point_estimate
            assert ci.ci_high == ci.point_estimate
        assert ci.n_bootstrap == 0


def test_format_ci_percent_layout() -> None:
    ci = BootstrapInterval(
        metric="f1",
        point_estimate=0.85,
        ci_low=0.80,
        ci_high=0.90,
        n_bootstrap=1000,
        seed=0,
    )
    s = format_ci(ci, percent=True)
    assert "85.00" in s
    assert "80.00" in s
    assert "90.00" in s


def test_aggregate_macro_across_folds() -> None:
    fold_a = {
        "f1": BootstrapInterval("f1", 0.9, 0.85, 0.95, 1000, 0),
        "recall": BootstrapInterval("recall", 0.92, 0.87, 0.97, 1000, 0),
    }
    fold_b = {
        "f1": BootstrapInterval("f1", 0.8, 0.75, 0.85, 1000, 0),
        "recall": BootstrapInterval("recall", 0.82, 0.77, 0.87, 1000, 0),
    }
    out = aggregate_macro([fold_a, fold_b])
    assert out["f1"]["point_mean"] == pytest.approx(0.85)
    assert out["f1"]["n_folds"] == 2
