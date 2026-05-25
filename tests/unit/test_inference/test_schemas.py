"""Pydantic contract tests for the §11.7 API schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hylog.inference.schemas import (
    MAX_BYTES_PER_LINE,
    MAX_LINES_PER_SEQUENCE,
    MAX_SEQUENCES_PER_REQUEST,
    MAX_TOTAL_LINES,
    CalibrationInfo,
    PredictRequest,
    PredictResponse,
    SequencePrediction,
)


def test_predict_request_accepts_valid_payload() -> None:
    PredictRequest(
        sequences=[
            {"id": "r1", "lines": ["a", "b"]},
            {"id": "r2", "lines": ["c"]},
        ]
    )


def test_predict_request_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError) as ei:
        PredictRequest(
            sequences=[
                {"id": "r1", "lines": ["a"]},
                {"id": "r1", "lines": ["b"]},
            ]
        )
    assert "unique" in str(ei.value).lower()


def test_predict_request_rejects_oversized_line() -> None:
    long_line = "x" * (MAX_BYTES_PER_LINE + 1)
    with pytest.raises(ValidationError):
        PredictRequest(sequences=[{"id": "r1", "lines": [long_line]}])


def test_predict_request_rejects_too_many_sequences() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(
            sequences=[
                {"id": f"r{i}", "lines": ["x"]} for i in range(MAX_SEQUENCES_PER_REQUEST + 1)
            ]
        )


def test_predict_request_rejects_total_lines_overflow() -> None:
    # One sequence with too many lines.
    with pytest.raises(ValidationError):
        PredictRequest(sequences=[{"id": "r1", "lines": ["x"] * (MAX_LINES_PER_SEQUENCE + 1)}])


def test_predict_request_rejects_total_across_sequences() -> None:
    # Per-sequence cap is fine but total is too large.
    sequences = []
    per = MAX_LINES_PER_SEQUENCE
    n_seq = (MAX_TOTAL_LINES // per) + 1
    for i in range(min(n_seq, MAX_SEQUENCES_PER_REQUEST)):
        sequences.append({"id": f"r{i}", "lines": ["x"] * per})
    with pytest.raises(ValidationError):
        PredictRequest(sequences=sequences)


def test_predict_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(
            sequences=[{"id": "r1", "lines": ["a"], "extra": "nope"}]  # type: ignore[arg-type]
        )


def test_sequence_prediction_clamps_probabilities() -> None:
    with pytest.raises(ValidationError):
        SequencePrediction(
            id="r1",
            p_anomaly=1.5,
            p_anomaly_calibrated=0.5,
            decision="anomaly",
            confidence=0.9,
        )
    with pytest.raises(ValidationError):
        SequencePrediction(
            id="r1",
            p_anomaly=0.5,
            p_anomaly_calibrated=-0.1,
            decision="normal",
            confidence=0.9,
        )


def test_calibration_info_method_enum() -> None:
    with pytest.raises(ValidationError):
        CalibrationInfo(method="cosmic-rays", fitted_on="x")  # type: ignore[arg-type]
    # Valid values:
    for m in ("none", "temperature_scaling", "platt_scaling", "vector_scaling"):
        CalibrationInfo(method=m, fitted_on="x")  # type: ignore[arg-type]


def test_predict_response_round_trip() -> None:
    resp = PredictResponse(
        model_version="x",
        sequences=[
            SequencePrediction(
                id="r1",
                p_anomaly=0.1,
                p_anomaly_calibrated=0.08,
                decision="normal",
                confidence=0.92,
            )
        ],
        calibration=CalibrationInfo(
            method="temperature_scaling", fitted_on="source", temperature=1.42
        ),
    )
    body = resp.model_dump()
    assert body["calibration"]["temperature"] == 1.42


def test_decision_must_be_one_of_three() -> None:
    with pytest.raises(ValidationError):
        SequencePrediction(
            id="r1",
            p_anomaly=0.5,
            p_anomaly_calibrated=0.5,
            decision="unknown",  # type: ignore[arg-type]
            confidence=0.7,
        )


def test_request_rejects_empty_sequences_array() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sequences=[])


def test_log_sequence_in_rejects_empty_lines_array() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sequences=[{"id": "r1", "lines": []}])
