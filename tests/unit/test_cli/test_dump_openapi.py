"""Tests for the OpenAPI dumper + the committed spec file (when present)."""

from __future__ import annotations

import json
from pathlib import Path

from hylog.inference.server import ServerConfig, create_app

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_openapi_spec_includes_required_paths() -> None:
    app = create_app(ServerConfig())
    spec = app.openapi()
    paths = spec.get("paths", {})
    for required in ("/healthz", "/v1/model-info", "/v1/predict", "/v1/drift"):
        assert required in paths, f"missing path {required}"


def test_openapi_predict_has_request_body() -> None:
    app = create_app(ServerConfig())
    spec = app.openapi()
    post = spec["paths"]["/v1/predict"].get("post", {})
    assert "requestBody" in post


def test_openapi_dump_round_trip(tmp_path: Path) -> None:
    from scripts.dump_openapi import main

    target = tmp_path / "spec.json"
    rc = main(["--out", str(target)])
    assert rc == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["info"]["title"] == "HyLog Inference Service"
