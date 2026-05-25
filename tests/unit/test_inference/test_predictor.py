"""Tests for the mock predictor."""

from __future__ import annotations

from hylog.inference.predictor import MockPredictor


def test_mock_predictor_returns_one_row_per_sequence() -> None:
    p = MockPredictor()
    rows = p.predict_batch([["a"], ["b", "c"], ["d"]])
    assert len(rows) == 3


def test_mock_predictor_deterministic() -> None:
    p1 = MockPredictor()
    p2 = MockPredictor()
    assert p1.predict_batch([["x", "y"]]) == p2.predict_batch([["x", "y"]])


def test_mock_predictor_raises_anomaly_probability_for_fatal_content() -> None:
    p = MockPredictor()
    benign = p.predict_batch([["INFO routine startup"]])[0]
    fatal = p.predict_batch([["FATAL kernel panic data TLB exception"]])[0]
    assert fatal.p_anomaly > benign.p_anomaly


def test_mock_predictor_probabilities_in_range() -> None:
    p = MockPredictor()
    rows = p.predict_batch([["one"], ["two", "three"]])
    for r in rows:
        assert 0.0 <= r.p_anomaly <= 1.0
        assert 0.0 <= r.p_anomaly_calibrated <= 1.0
        assert 0.0 <= r.confidence <= 1.0


def test_model_version_and_threshold_exposed() -> None:
    p = MockPredictor(threshold=0.55, version="vt-1")
    assert p.model_version() == "vt-1"
    assert p.selective_threshold() == 0.55


def test_calibration_info_is_method_none() -> None:
    p = MockPredictor()
    info = p.calibration_info()
    assert info["method"] == "none"
