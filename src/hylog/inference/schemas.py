"""Request / response Pydantic models for the HyLog inference service.

The schema is the authoritative API contract documented in
``docs/ROADMAP.md §11.7``. A breaking change requires a major version
bump and a deprecation window — these models are the unit of API
compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Phase-8 input caps. Tuned to fit the production SLO (p95 < 50 ms,
# batch=8, 100 lines/sequence). Larger requests are rejected with a
# precise 4xx so the client can chunk.
MAX_SEQUENCES_PER_REQUEST = 64
MAX_LINES_PER_SEQUENCE = 256
MAX_BYTES_PER_LINE = 4096
MAX_TOTAL_LINES = 4096
"""Hard cap across the whole request — prevents an attacker submitting
1 sequence x 1M lines (which would individually pass MAX_LINES_PER_SEQUENCE)."""

Decision = Literal["normal", "anomaly", "abstain"]


class LogSequenceIn(BaseModel):
    """One log sequence in the request body."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Client-supplied identifier; echoed back on response.",
    )
    lines: list[str] = Field(..., min_length=1, max_length=MAX_LINES_PER_SEQUENCE)

    @field_validator("lines")
    @classmethod
    def _validate_lines(cls, value: list[str]) -> list[str]:
        for i, line in enumerate(value):
            if len(line.encode("utf-8")) > MAX_BYTES_PER_LINE:
                raise ValueError(f"line {i} exceeds MAX_BYTES_PER_LINE={MAX_BYTES_PER_LINE}")
        return value


class PredictRequest(BaseModel):
    """Top-level POST /v1/predict payload."""

    model_config = ConfigDict(extra="forbid")

    sequences: list[LogSequenceIn] = Field(..., min_length=1, max_length=MAX_SEQUENCES_PER_REQUEST)

    @field_validator("sequences")
    @classmethod
    def _validate_total_lines(cls, value: list[LogSequenceIn]) -> list[LogSequenceIn]:
        total = sum(len(s.lines) for s in value)
        if total > MAX_TOTAL_LINES:
            raise ValueError(
                f"request total lines {total} exceeds MAX_TOTAL_LINES={MAX_TOTAL_LINES}"
            )
        # IDs must be unique within a single request.
        ids = [s.id for s in value]
        if len(ids) != len(set(ids)):
            raise ValueError("sequence ids must be unique within a single request")
        return value


class SequencePrediction(BaseModel):
    """One sequence's prediction in the response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    p_anomaly: float = Field(..., ge=0.0, le=1.0)
    p_anomaly_calibrated: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    abstain_reason: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class CalibrationInfo(BaseModel):
    """Calibration metadata echoed on every response."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["none", "temperature_scaling", "platt_scaling", "vector_scaling"]
    fitted_on: str = Field(..., description="Free-form identifier of the calibration set.")
    temperature: float | None = None
    platt_a: float | None = None
    platt_b: float | None = None


class PredictResponse(BaseModel):
    """Top-level response. Matches docs/ROADMAP.md §11.7 exactly."""

    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(..., min_length=1)
    sequences: list[SequencePrediction]
    calibration: CalibrationInfo


class ModelInfo(BaseModel):
    """GET /v1/model-info response."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    schema_version: int
    decoder_name: str
    encoder_name: str
    quantize_4bit: bool
    selective_threshold: float
    calibration: CalibrationInfo
    trained_on: list[str]
    intended_use: str
    limitations: str


class DriftSummary(BaseModel):
    """GET /v1/drift response. Reports the Phase-8 drift KS-test status."""

    model_config = ConfigDict(extra="forbid")

    n_observed: int
    n_reference: int
    ks_statistic: float
    ks_p_value: float
    drift_threshold: float
    drift_detected: bool
    reference_summary: dict[str, float]
    observed_summary: dict[str, float]


class HealthResponse(BaseModel):
    """GET /healthz response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"] = "ok"
    model_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """4xx / 5xx body. Deliberately strips any echo of the request to
    prevent log-injection-by-error (Phase-8 §11.9)."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    request_id: str | None = None


__all__ = [
    "MAX_BYTES_PER_LINE",
    "MAX_LINES_PER_SEQUENCE",
    "MAX_SEQUENCES_PER_REQUEST",
    "MAX_TOTAL_LINES",
    "CalibrationInfo",
    "Decision",
    "DriftSummary",
    "ErrorResponse",
    "HealthResponse",
    "LogSequenceIn",
    "ModelInfo",
    "PredictRequest",
    "PredictResponse",
    "SequencePrediction",
]
