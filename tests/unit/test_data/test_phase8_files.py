"""Static-asset checks for Phase 8 production-service deliverables."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_model_card_present_with_yaml_frontmatter() -> None:
    p = REPO_ROOT / "reports" / "phase8" / "model_card.md"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    # HF-template YAML front matter.
    assert body.startswith("---\n")
    assert "license: mit" in body
    assert "library_name: hylog" in body
    # Required disclosure sections.
    for marker in (
        "Intended use",
        "Not intended for",
        "dual-use",
        "Sole-decision-maker",
        "Concept drift",
    ):
        assert marker in body or marker.lower() in body.lower()


def test_python_client_sdk_present() -> None:
    p = REPO_ROOT / "clients" / "python" / "hylog_client.py"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "class HyLogClient" in body
    assert "X-API-Key" in body


def test_export_onnx_script_present() -> None:
    p = REPO_ROOT / "scripts" / "export_onnx.py"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "manifest.json" in body
    assert "projector" in body


def test_dump_openapi_script_present() -> None:
    p = REPO_ROOT / "scripts" / "dump_openapi.py"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "OpenAPI" in body or "openapi" in body
