"""FastAPI inference service for HyLog.

Endpoints (per ROADMAP §11.7 + Phase-8 deliverables):

  GET  /healthz          — liveness probe, no auth.
  GET  /v1/model-info    — frozen model metadata, no auth.
  POST /v1/predict       — the main inference path. Requires X-API-Key.
  GET  /v1/drift         — drift KS summary. Requires X-API-Key.

Security stack (§11.9):
  - X-API-Key header auth backed by SHA-256-hashed key store.
  - Per-key token-bucket rate limiter.
  - Input caps enforced by Pydantic (sequences, lines, bytes-per-line,
    total-lines).
  - 4xx/5xx bodies never echo request content (Phase-8 §11.9).

Calibration:
  - Returned ``p_anomaly`` is the raw anomaly probability.
  - Returned ``p_anomaly_calibrated`` is the temperature-scaled value.
  - The selective predictor's τ controls the abstain branch.

Drift:
  - Every accepted ``p_anomaly_calibrated`` is observed by the drift
    monitor (Phase-8 §11.8). ``GET /v1/drift`` reports the KS test
    against the frozen reference distribution.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from hylog.inference.auth import (
    HEADER_NAME,
    APIKeyStore,
)
from hylog.inference.drift import DriftMonitor, DriftMonitorConfig
from hylog.inference.predictor import MockPredictor, PredictorProtocol
from hylog.inference.rate_limit import TokenBucketLimiter
from hylog.inference.schemas import (
    CalibrationInfo,
    DriftSummary,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    SequencePrediction,
)


@dataclass(slots=True)
class ServerConfig:
    """Wiring config for the FastAPI app."""

    api_key_store: APIKeyStore = field(default_factory=lambda: APIKeyStore(keys={}))
    rate_limiter: TokenBucketLimiter = field(default_factory=TokenBucketLimiter)
    drift_monitor: DriftMonitor | None = None
    predictor: PredictorProtocol = field(default_factory=MockPredictor)
    decoder_name: str = "qwen2.5-1.5b"
    encoder_name: str = "bert-base-uncased"
    quantize_4bit: bool = True
    trained_on: tuple[str, ...] = ("hdfs", "bgl", "thunderbird")
    intended_use: str = (
        "Detect anomalies in system-operational logs (kernel, scheduler, "
        "distributed storage). Outputs are calibrated probabilities + a "
        "selective abstain channel. NOT to be used as a sole automated "
        "decision-maker in safety-critical incident response."
    )
    limitations: str = (
        "Trained on Loghub-2.0 HDFS/BGL/Thunderbird. Cross-system "
        "generalisation is evaluated under a zero-target-label protocol "
        "but novel target systems may require re-calibration. Not "
        "evaluated on user-keyed personal logs; see model_card.md "
        "§dual-use disclosure."
    )


# ---------------------------------------------------------------------------
# Auth + rate-limit dependency (wired against app.state)
# ---------------------------------------------------------------------------


def _require_api_key(
    request: Request,
    header: str | None = Header(default=None, alias=HEADER_NAME),
) -> str:
    """FastAPI dependency: validate header + apply rate limit."""
    store: APIKeyStore = request.app.state.api_key_store
    limiter: TokenBucketLimiter = request.app.state.rate_limiter

    if store.empty():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server has no API keys configured",
        )
    if not header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": HEADER_NAME},
        )
    client = store.authenticate(header)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": HEADER_NAME},
        )
    if not limiter.allow(client):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": "60"},
        )
    return client


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build a FastAPI app wired against the supplied ``ServerConfig``."""
    cfg = config or ServerConfig()
    app = FastAPI(
        title="HyLog Inference Service",
        version="0.8.0",
        description=(
            "REST inference for the HyLog hybrid SLM log anomaly detector. "
            "Auth: X-API-Key. Body schema documented at /openapi.json."
        ),
    )
    app.state.config = cfg
    app.state.api_key_store = cfg.api_key_store
    app.state.rate_limiter = cfg.rate_limiter
    app.state.drift_monitor = cfg.drift_monitor or DriftMonitor(
        reference=np.linspace(0.0, 1.0, 256),
        config=DriftMonitorConfig(),
    )
    app.state.predictor = cfg.predictor
    app.state.started_at = time.monotonic()

    @app.get("/healthz", response_model=HealthResponse, tags=["meta"])
    def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=request.app.state.predictor.model_version(),
            uptime_seconds=float(time.monotonic() - request.app.state.started_at),
        )

    @app.get("/v1/model-info", response_model=ModelInfo, tags=["meta"])
    def model_info(request: Request) -> ModelInfo:
        c: ServerConfig = request.app.state.config
        cal = c.predictor.calibration_info()
        return ModelInfo(
            model_version=c.predictor.model_version(),
            schema_version=1,
            decoder_name=c.decoder_name,
            encoder_name=c.encoder_name,
            quantize_4bit=c.quantize_4bit,
            selective_threshold=c.predictor.selective_threshold(),
            calibration=CalibrationInfo(**_calibration_payload(cal)),
            trained_on=list(c.trained_on),
            intended_use=c.intended_use,
            limitations=c.limitations,
        )

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    def predict(
        request: Request,
        body: PredictRequest,
        _: str = Depends(_require_api_key),
    ) -> PredictResponse:
        predictor: PredictorProtocol = request.app.state.predictor
        drift: DriftMonitor = request.app.state.drift_monitor

        sequences = [s.lines for s in body.sequences]
        rows = predictor.predict_batch(sequences)
        if len(rows) != len(body.sequences):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="predictor returned wrong batch size",
            )

        tau = predictor.selective_threshold()
        out: list[SequencePrediction] = []
        for seq_in, row in zip(body.sequences, rows, strict=True):
            decision = _decision_from_row(row, tau)
            out.append(
                SequencePrediction(
                    id=seq_in.id,
                    p_anomaly=row.p_anomaly,
                    p_anomaly_calibrated=row.p_anomaly_calibrated,
                    confidence=row.confidence,
                    decision=decision,
                    abstain_reason=(
                        f"confidence={row.confidence:.4f} < tau={tau:.4f}"
                        if decision == "abstain"
                        else None
                    ),
                )
            )
            drift.observe(row.p_anomaly_calibrated)

        return PredictResponse(
            model_version=predictor.model_version(),
            sequences=out,
            calibration=CalibrationInfo(**_calibration_payload(predictor.calibration_info())),
        )

    @app.get(
        "/v1/drift",
        response_model=DriftSummary,
        tags=["meta"],
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def drift_endpoint(
        request: Request,
        _: str = Depends(_require_api_key),
    ) -> DriftSummary:
        monitor: DriftMonitor = request.app.state.drift_monitor
        report = monitor.evaluate()
        return DriftSummary(**report.to_dict())

    # --- Error handlers (§11.9: no echo) ------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Strip the user input from the error body; expose only the
        # high-level reason. Phase-8 §11.9.
        first = exc.errors()[0] if exc.errors() else {}
        reason = first.get("msg", "validation error")
        body = ErrorResponse(
            error="invalid_request",
            detail=str(reason),
            request_id=request.headers.get("x-request-id"),
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # Avoid echoing inputs; the message is the only thing returned.
        body = ErrorResponse(
            error=_error_code_for(exc.status_code),
            detail=str(exc.detail) if isinstance(exc.detail, str) else "error",
            request_id=request.headers.get("x-request-id"),
        )
        headers = exc.headers or {}
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
            headers=headers,
        )

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision_from_row(row: Any, tau: float) -> str:
    if row.confidence < tau:
        return "abstain"
    return "anomaly" if row.p_anomaly_calibrated >= 0.5 else "normal"


def _calibration_payload(info: dict[str, object]) -> dict[str, object]:
    """Map a free-form predictor calibration dict into the schema fields."""
    method_str = str(info.get("method", "none"))
    method = (
        method_str
        if method_str in {"none", "temperature_scaling", "platt_scaling", "vector_scaling"}
        else "none"
    )
    return {
        "method": method,
        "fitted_on": str(info.get("fitted_on", "unknown")),
        "temperature": _to_float_or_none(info.get("temperature")),
        "platt_a": _to_float_or_none(info.get("platt_a") or info.get("a")),
        "platt_b": _to_float_or_none(info.get("platt_b") or info.get("b")),
    }


def _to_float_or_none(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _error_code_for(status_code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "invalid_request",
        429: "rate_limited",
        500: "internal_error",
        503: "service_unavailable",
    }.get(status_code, "error")


__all__ = ["ServerConfig", "create_app"]
