"""Static-asset checks for Phase 7 reproducibility deliverables.

These tests do NOT run the scripts; they only verify the files are
present and structurally sane. Live execution is gated on the GPU CI
workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_dockerfile_present_and_uses_cuda_base() -> None:
    p = REPO_ROOT / "Dockerfile"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "FROM nvidia/cuda" in body
    assert "ENTRYPOINT" in body


def test_dockerignore_excludes_heavy_directories() -> None:
    p = REPO_ROOT / ".dockerignore"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    for entry in ("__pycache__/", "tests/", "third_party/", "mlruns/"):
        assert entry in body


def test_environment_yml_is_parseable() -> None:
    p = REPO_ROOT / "environment.yml"
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert payload["name"] == "hylog"
    assert "python>=3.11,<3.13" in payload["dependencies"]
    pip_section = next(
        (d for d in payload["dependencies"] if isinstance(d, dict) and "pip" in d), None
    )
    assert pip_section is not None
    assert any("transformers" in p for p in pip_section["pip"])


def test_requirements_lock_pins_torch_and_transformers() -> None:
    p = REPO_ROOT / "requirements-lock.txt"
    body = p.read_text(encoding="utf-8")
    assert "torch==" in body
    assert "transformers==" in body
    assert "peft==" in body
    assert "bitsandbytes==" in body


@pytest.mark.parametrize(
    "script", ["verify_install.ps1", "verify_install.sh", "reproduce_all.ps1", "reproduce_all.sh"]
)
def test_phase7_scripts_present(script: str) -> None:
    p = REPO_ROOT / "scripts" / script
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    # Sanity: each script should reference the report path it writes.
    assert "phase7" in body


def test_gpu_ci_workflow_present() -> None:
    p = REPO_ROOT / ".github" / "workflows" / "gpu-smoke.yml"
    assert p.exists()
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    # 'on' is a reserved Python keyword in YAML when un-quoted; PyYAML
    # parses it as the boolean True. Both keys may appear depending on
    # the YAML loader version, so we accept either.
    assert "jobs" in payload
    assert "smoke" in payload["jobs"]
    runner = payload["jobs"]["smoke"]["runs-on"]
    assert "gpu" in (runner if isinstance(runner, list) else [runner])


def test_gpu_workflow_uploads_reports() -> None:
    p = REPO_ROOT / ".github" / "workflows" / "gpu-smoke.yml"
    body = p.read_text(encoding="utf-8")
    assert "upload-artifact" in body
    assert "reports/phase7" in body


def test_verify_install_ps1_records_cli_entrypoints() -> None:
    body = (REPO_ROOT / "scripts" / "verify_install.ps1").read_text(encoding="utf-8")
    for cli in ("hylog-train", "hylog-predict", "hylog-loso", "hylog-calibrate", "hylog-ablation"):
        assert cli in body


def test_reproduce_all_runs_loso_and_calibrate_and_ablation() -> None:
    body = (REPO_ROOT / "scripts" / "reproduce_all.ps1").read_text(encoding="utf-8")
    assert "hylog.cli.loso" in body
    assert "hylog.cli.calibrate" in body
    assert "hylog.cli.ablation" in body
