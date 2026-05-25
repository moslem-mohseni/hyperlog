"""Tests for the drift monitor."""

from __future__ import annotations

import numpy as np
import pytest

from hylog.inference.drift import (
    DriftMonitor,
    DriftMonitorConfig,
    two_sample_ks,
)


def test_ks_zero_on_identical_distributions() -> None:
    a = np.linspace(0, 1, 200)
    b = np.linspace(0, 1, 200)
    d, p = two_sample_ks(a, b)
    assert d == pytest.approx(0.0)
    # Asymptotic series at D=0 gives p close to 1 but not exactly
    # (Kolmogorov distribution series truncation). 0.05 tolerance.
    assert p == pytest.approx(1.0, abs=0.05)


def test_ks_one_on_disjoint_supports() -> None:
    a = np.zeros(100)
    b = np.ones(100)
    d, p = two_sample_ks(a, b)
    assert d == pytest.approx(1.0)
    assert p < 1e-3


def test_drift_monitor_reports_no_drift_when_observed_matches() -> None:
    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 512)
    monitor = DriftMonitor(reference=ref, config=DriftMonitorConfig(window=256))
    for v in rng.uniform(0, 1, 256):
        monitor.observe(float(v))
    report = monitor.evaluate()
    assert not report.drift_detected
    assert report.ks_statistic < 0.25


def test_drift_monitor_flags_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 1, 512)
    monitor = DriftMonitor(reference=ref, config=DriftMonitorConfig(window=256, ks_threshold=0.05))
    # Observed distribution is sharply concentrated near 1.
    for v in rng.uniform(0.9, 1.0, 256):
        monitor.observe(float(v))
    report = monitor.evaluate()
    assert report.drift_detected
    assert report.ks_statistic > 0.5


def test_observe_many_handles_nan() -> None:
    monitor = DriftMonitor(reference=np.linspace(0, 1, 64))
    monitor.observe_many([0.5, float("nan"), 0.7, float("nan"), 0.9])
    report = monitor.evaluate()
    assert report.n_observed == 3


def test_window_must_be_above_8() -> None:
    with pytest.raises(ValueError):
        DriftMonitor(reference=np.zeros(64), config=DriftMonitorConfig(window=4))


def test_reference_must_be_1d() -> None:
    with pytest.raises(ValueError):
        DriftMonitor(reference=np.zeros((4, 4)))


def test_report_summary_contains_quartiles() -> None:
    monitor = DriftMonitor(reference=np.linspace(0, 1, 64))
    for v in np.linspace(0, 1, 32):
        monitor.observe(float(v))
    report = monitor.evaluate()
    for key in ("min", "p25", "median", "p75", "max", "mean"):
        assert key in report.reference_summary
        assert key in report.observed_summary


def test_report_to_dict_keys() -> None:
    monitor = DriftMonitor(reference=np.linspace(0, 1, 64))
    monitor.observe(0.5)
    payload = monitor.evaluate().to_dict()
    for key in (
        "n_observed",
        "n_reference",
        "ks_statistic",
        "ks_p_value",
        "drift_threshold",
        "drift_detected",
        "reference_summary",
        "observed_summary",
    ):
        assert key in payload
