"""Minimal Python SDK for the HyLog inference service.

Single-file, dependency-light (requests only). Reviewers can ``curl``
the API instead, but the SDK is the canonical example for integrators.

Usage:

    from clients.python.hylog_client import HyLogClient

    client = HyLogClient(base_url="https://hylog.example/", api_key="...")
    info = client.model_info()
    out = client.predict([
        {"id": "req-1", "lines": ["FATAL kernel panic", "data TLB error"]},
        {"id": "req-2", "lines": ["INFO routine startup", "INFO sync ok"]},
    ])
    for seq in out["sequences"]:
        print(seq["id"], seq["decision"], seq["p_anomaly_calibrated"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


DEFAULT_TIMEOUT = 30.0


@dataclass(slots=True)
class HyLogClient:
    """Minimal client. Thread-safe per ``requests.Session``."""

    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    extra_headers: dict[str, str] = field(default_factory=dict)
    _session: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if requests is None:
            raise ImportError(
                "the requests library is required for HyLogClient; "
                "install it with `pip install requests`"
            )
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        })

    # ---- public methods -------------------------------------------------

    def healthz(self) -> dict[str, Any]:
        return self._get("/healthz", auth=False)

    def model_info(self) -> dict[str, Any]:
        return self._get("/v1/model-info", auth=False)

    def drift(self) -> dict[str, Any]:
        return self._get("/v1/drift", auth=True)

    def predict(self, sequences: list[dict[str, Any]]) -> dict[str, Any]:
        """POST ``/v1/predict`` with a list of {id, lines} dicts."""
        body = {"sequences": sequences}
        return self._post("/v1/predict", body, auth=True)

    # ---- low-level helpers --------------------------------------------

    def _get(self, path: str, *, auth: bool) -> dict[str, Any]:
        headers = self._session.headers if auth else {
            k: v for k, v in self._session.headers.items() if k != "X-API-Key"
        }
        response = self._session.get(self.base_url + path, headers=headers, timeout=self.timeout)
        return self._extract(response)

    def _post(self, path: str, body: dict[str, Any], *, auth: bool) -> dict[str, Any]:
        headers = self._session.headers if auth else {
            k: v for k, v in self._session.headers.items() if k != "X-API-Key"
        }
        response = self._session.post(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            timeout=self.timeout,
        )
        return self._extract(response)

    def _extract(self, response: Any) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise HyLogAPIError(
                status_code=response.status_code,
                detail=f"non-JSON response: {exc}",
                body=response.text,
            ) from exc
        if response.status_code >= 400:
            raise HyLogAPIError(
                status_code=response.status_code,
                detail=str(payload.get("detail", payload)),
                body=payload,
            )
        return payload


class HyLogAPIError(RuntimeError):
    """Raised on 4xx / 5xx responses."""

    def __init__(self, *, status_code: int, detail: str, body: Any) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.body = body


__all__ = ["HyLogAPIError", "HyLogClient"]
