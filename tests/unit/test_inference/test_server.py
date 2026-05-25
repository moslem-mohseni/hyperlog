"""End-to-end FastAPI integration tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hylog.inference.auth import APIKeyStore
from hylog.inference.drift import DriftMonitor, DriftMonitorConfig
from hylog.inference.predictor import MockPredictor
from hylog.inference.rate_limit import TokenBucketLimiter
from hylog.inference.server import ServerConfig, create_app


@pytest.fixture
def client() -> TestClient:
    cfg = ServerConfig(
        api_key_store=APIKeyStore.from_plaintext({"test-key": "test-client"}),
        rate_limiter=TokenBucketLimiter(capacity=100.0, refill_per_second=10.0),
        drift_monitor=DriftMonitor(
            reference=np.linspace(0, 1, 256), config=DriftMonitorConfig(window=64)
        ),
        predictor=MockPredictor(threshold=0.6),
    )
    return TestClient(create_app(cfg))


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"X-API-Key": "test-key"}


# ---- Health + meta endpoints -----------------------------------------------


def test_healthz_no_auth_required(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_model_info_no_auth_required(client: TestClient) -> None:
    r = client.get("/v1/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "model_version",
        "schema_version",
        "decoder_name",
        "encoder_name",
        "quantize_4bit",
        "selective_threshold",
        "calibration",
        "trained_on",
        "intended_use",
        "limitations",
    ):
        assert key in body


# ---- Predict endpoint -------------------------------------------------------


def test_predict_happy_path(client: TestClient, auth_header: dict[str, str]) -> None:
    body = {
        "sequences": [
            {"id": "r1", "lines": ["INFO routine startup", "INFO sync ok"]},
            {"id": "r2", "lines": ["FATAL kernel panic", "data TLB error"]},
        ]
    }
    r = client.post("/v1/predict", json=body, headers=auth_header)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["model_version"]
    assert len(payload["sequences"]) == 2
    ids = {s["id"] for s in payload["sequences"]}
    assert ids == {"r1", "r2"}
    for s in payload["sequences"]:
        assert 0.0 <= s["p_anomaly"] <= 1.0
        assert 0.0 <= s["p_anomaly_calibrated"] <= 1.0
        assert s["decision"] in {"normal", "anomaly", "abstain"}


def test_predict_without_auth_returns_401(client: TestClient) -> None:
    r = client.post("/v1/predict", json={"sequences": [{"id": "r1", "lines": ["a"]}]})
    assert r.status_code == 401
    assert r.headers.get("www-authenticate", "").lower() in ("x-api-key", "X-API-Key".lower())


def test_predict_wrong_api_key_returns_401(client: TestClient) -> None:
    marker = "ZZZ-USER-SECRET-MARKER"
    r = client.post(
        "/v1/predict",
        json={"sequences": [{"id": "r1", "lines": [marker]}]},
        headers={"X-API-Key": "ZZZ-WRONG-KEY-XYZ"},
    )
    assert r.status_code == 401
    body_text = json.dumps(r.json())
    # No-echo: neither the request payload nor the wrong key may appear.
    assert marker not in body_text
    assert "ZZZ-WRONG-KEY-XYZ" not in body_text


def test_predict_rejects_oversized_request(client: TestClient, auth_header: dict[str, str]) -> None:
    # Too many sequences.
    body = {"sequences": [{"id": f"r{i}", "lines": ["x"]} for i in range(100)]}
    r = client.post("/v1/predict", json=body, headers=auth_header)
    assert r.status_code == 422
    err = r.json()
    assert err["error"] == "invalid_request"


def test_predict_emits_abstain_when_confidence_below_tau() -> None:
    cfg = ServerConfig(
        api_key_store=APIKeyStore.from_plaintext({"k": "c"}),
        rate_limiter=TokenBucketLimiter(capacity=100.0, refill_per_second=10.0),
        predictor=MockPredictor(threshold=0.99),  # very strict
    )
    client = TestClient(create_app(cfg))
    r = client.post(
        "/v1/predict",
        json={"sequences": [{"id": "r1", "lines": ["INFO bland"]}]},
        headers={"X-API-Key": "k"},
    )
    body = r.json()
    assert body["sequences"][0]["decision"] == "abstain"
    assert body["sequences"][0]["abstain_reason"]


# ---- Rate limiting ----------------------------------------------------------


def test_rate_limit_returns_429_after_quota_exhausted() -> None:
    cfg = ServerConfig(
        api_key_store=APIKeyStore.from_plaintext({"k": "c"}),
        rate_limiter=TokenBucketLimiter(capacity=2.0, refill_per_second=0.01),
    )
    client = TestClient(create_app(cfg))
    body = {"sequences": [{"id": "r1", "lines": ["a"]}]}
    headers = {"X-API-Key": "k"}
    assert client.post("/v1/predict", json=body, headers=headers).status_code == 200
    assert client.post("/v1/predict", json=body, headers=headers).status_code == 200
    r = client.post("/v1/predict", json=body, headers=headers)
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "60"
    assert r.json()["error"] == "rate_limited"


# ---- Drift endpoint ---------------------------------------------------------


def test_drift_endpoint_requires_auth(client: TestClient) -> None:
    assert client.get("/v1/drift").status_code == 401


def test_drift_endpoint_returns_summary(client: TestClient, auth_header: dict[str, str]) -> None:
    # Send a few predictions so the drift monitor has data.
    body = {"sequences": [{"id": f"r{i}", "lines": ["x"]} for i in range(5)]}
    client.post("/v1/predict", json=body, headers=auth_header)
    r = client.get("/v1/drift", headers=auth_header)
    assert r.status_code == 200
    payload = r.json()
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


# ---- Error contract --------------------------------------------------------


def test_validation_error_has_no_echo(client: TestClient, auth_header: dict[str, str]) -> None:
    """A 422 body must not contain the user-supplied payload verbatim."""
    secret_marker = "ZZZ-SECRET-MARKER-PLEASE-DO-NOT-ECHO"
    body = {"sequences": [{"id": "r1", "lines": [secret_marker], "extra_field": "x"}]}
    r = client.post("/v1/predict", json=body, headers=auth_header)
    assert r.status_code == 422
    assert secret_marker not in r.text


def test_request_id_round_trip(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-Id": "test-rid-123"})
    assert r.headers.get("x-request-id") == "test-rid-123"


# ---- Service unavailable when no keys configured ---------------------------


def test_service_unavailable_when_no_keys() -> None:
    cfg = ServerConfig(api_key_store=APIKeyStore(keys={}))
    client = TestClient(create_app(cfg))
    r = client.post(
        "/v1/predict",
        json={"sequences": [{"id": "r1", "lines": ["a"]}]},
        headers={"X-API-Key": "anything"},
    )
    assert r.status_code == 503
