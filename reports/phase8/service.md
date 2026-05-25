# Phase 8 — Production Inference Service

**Author:** Moslem Mohseni Khah
**Phase:** 8 (Production Inference Service)
**Roadmap reference:** `docs/ROADMAP.md` §Phase 8 + §11.7 (schema) + §11.8 (ethics) + §11.9 (security)
**Release tag:** `v0.8.0-service`

Phase 8 turns HyLog from a research codebase into a **deployable
artefact** — a REST inference service that any operator can run with
one `docker run` line and any integrator can call from any language.

---

## 1. What Phase 8 ships

| Artefact | Purpose |
|---|---|
| `src/hylog/inference/server.py` | FastAPI service. `/healthz`, `/v1/model-info`, `/v1/predict`, `/v1/drift`. |
| `src/hylog/inference/schemas.py` | Pydantic models matching ROADMAP §11.7 exactly. Input caps enforced (sequences/lines/bytes/total). |
| `src/hylog/inference/auth.py` | `X-API-Key` auth with SHA-256-hashed key store + constant-time compare. |
| `src/hylog/inference/rate_limit.py` | Per-key token-bucket rate limiter. |
| `src/hylog/inference/drift.py` | Phase-8 §11.8 drift monitor. Two-sample KS test against the frozen reference distribution. |
| `src/hylog/inference/predictor.py` | `PredictorProtocol` + `MockPredictor` (CPU). The GPU predictor wires in via the same interface. |
| `clients/python/hylog_client.py` | Single-file Python SDK example. |
| `scripts/export_onnx.py` | Partial ONNX export (projector + head + frozen BERT). Decoder stays on PyTorch (4-bit QLoRA is not ONNX-exportable today). |
| `scripts/dump_openapi.py` | Emits `reports/phase8/openapi.json` from the live FastAPI app. |
| `reports/phase8/openapi.json` | Committed OpenAPI spec. |
| `reports/phase8/model_card.md` | HF-template model card authored by Moslem Mohseni Khah. |

---

## 2. Phase 8 checklist status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | End-to-end integration test (HTTP → response w/ calibrated prob + abstain) | ✅ | `test_server.py::test_predict_happy_path` + `test_predict_emits_abstain_when_confidence_below_tau` |
| 2 | Load test: 100 req/s × 5 min, p95 < 50 ms | ⏳ **GPU-deferred** | Token-bucket limiter sized for 100 RPS; the latency target requires the real GPU predictor to validate. Mock predictor's median latency on this commit is ~1 ms. |
| 3 | OpenAPI spec auto-generated and committed | ✅ | `scripts/dump_openapi.py` → `reports/phase8/openapi.json` |
| 4 | Model card includes LOSO matrix, ECE, intended use, dual-use disclosure, "not a sole decision-maker" clause, drift-monitoring guidance | ✅ | `reports/phase8/model_card.md` — every required section explicitly named |
| 5 | Security checklist (input cap, rate limit, no-echo, API-key auth) implemented and unit-tested | ✅ | 4 modules + 27 tests across `test_auth.py`, `test_rate_limit.py`, `test_schemas.py`, `test_server.py` |
| 6 | Response schema matches §11.7 exactly; contract test enforced | ✅ | Pydantic `PredictResponse` mirrors §11.7; 14 contract tests |
| 7 | CLI smoke (`hylog-predict --input fixtures/sample.jsonl`) exits 0 with schema-valid output | ✅ | Existing Phase-0 smoke + Phase-8 contract tests |
| 8 | Tag `v0.8.0-service` pushed | ✅ | this commit |

### Item 2 — GPU-deferred
The load-test SLO contract (p95 < 50 ms, 100 RPS for 5 min) is a property
of the **GPU predictor**, not the FastAPI plumbing. Phase 8's FastAPI
overhead is sub-millisecond. Validating the SLO requires the GPU
predictor + a 5-minute Locust / k6 run, which lands when the GPU
training run produces a model checkpoint.

---

## 3. Security stack (§11.9)

| Control | Implementation | Tested by |
|---|---|---|
| Input length cap (max lines per request) | Pydantic `MAX_LINES_PER_SEQUENCE`, `MAX_SEQUENCES_PER_REQUEST`, `MAX_TOTAL_LINES` | `test_schemas.py` (5 tests) |
| Per-line byte cap | Pydantic field validator `MAX_BYTES_PER_LINE` | `test_schemas.py` |
| Rate limit per API key | Token-bucket in-process limiter | `test_rate_limit.py` (7 tests) + `test_server.py::test_rate_limit_returns_429_after_quota_exhausted` |
| No echo of raw request in error bodies | Custom 422 handler strips request content; `ErrorResponse` model has no echo field | `test_server.py::test_validation_error_has_no_echo` + `test_predict_wrong_api_key_returns_401` |
| API key authentication | `X-API-Key` header + SHA-256-hashed key store + constant-time compare | `test_auth.py` (6 tests) |
| Request ID round-trip | Middleware echoes `X-Request-Id` (generates UUID if absent) | `test_server.py::test_request_id_round_trip` |

The model card explicitly states the service is **not** an LLM chat
surface: there is no prompt-injection attack surface because the
classification head replaces autoregressive generation.

---

## 4. Drift monitor (§11.8)

The drift monitor maintains a rolling window of observed
`p_anomaly_calibrated` values and compares it on demand against a
frozen reference distribution via the two-sample Kolmogorov-Smirnov
test:

```
GET /v1/drift -> DriftSummary {
    n_observed, n_reference,
    ks_statistic, ks_p_value,
    drift_threshold, drift_detected,
    reference_summary {min,p25,median,p75,max,mean},
    observed_summary  {min,p25,median,p75,max,mean},
}
```

`drift_detected = True` requires both KS > threshold (default 0.1) and
p < 0.05. Operators can re-calibrate (via `hylog-calibrate` against a
fresh target slice) without redeploying the model.

---

## 5. Test summary at this tag

| Suite | Tests | Status |
|---|---|---|
| Phase 1-7 (regression) | 359 | ✅ |
| **Phase 8 schemas + auth + rate limit + drift + server + predictor** | **61** | ✅ |
| **Total** | **420** | **✅ all pass** |

Verification on this commit:

```text
ruff check src tests scripts -> clean
ruff format --check         -> clean
mypy --strict src/hylog     -> clean (70 source files; +6 vs Phase 7)
pytest -q                   -> 420 passed in 50 s
```

---

## 6. One-command live demo

```powershell
# Start the server (mock predictor, no GPU needed):
python -c "
from hylog.inference.server import ServerConfig, create_app
from hylog.inference.auth import APIKeyStore
import uvicorn
cfg = ServerConfig(api_key_store=APIKeyStore.from_plaintext({'demo-key': 'demo'}))
uvicorn.run(create_app(cfg), host='127.0.0.1', port=8000)
"

# In another shell:
curl -H "X-API-Key: demo-key" -H "Content-Type: application/json" `
     -d '{"sequences":[{"id":"r1","lines":["FATAL kernel panic"]}]}' `
     http://127.0.0.1:8000/v1/predict
```

Returns a §11.7-compliant `PredictResponse` with a calibrated
`p_anomaly_calibrated` and a `decision ∈ {normal, anomaly, abstain}`.

---

## 7. Reproducibility manifest

| Artefact | Path |
|---|---|
| This report | `reports/phase8/service.md` |
| Model card | `reports/phase8/model_card.md` |
| OpenAPI spec | `reports/phase8/openapi.json` |
| Server | `src/hylog/inference/server.py` |
| Schemas | `src/hylog/inference/schemas.py` |
| Auth + rate limit | `src/hylog/inference/{auth,rate_limit}.py` |
| Drift monitor | `src/hylog/inference/drift.py` |
| Client SDK | `clients/python/hylog_client.py` |
| ONNX export | `scripts/export_onnx.py` |
| OpenAPI dumper | `scripts/dump_openapi.py` |
