"""Tests for the Phase 2A feasibility-check artefact."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_feasibility_scripts_exist() -> None:
    assert (REPO_ROOT / "scripts" / "feasibility_check.ps1").exists()
    assert (REPO_ROOT / "scripts" / "feasibility_check.sh").exists()


def test_feasibility_report_emitted() -> None:
    """Phase 2A: the feasibility-check script must produce a JSON report."""
    path = REPO_ROOT / "reports" / "phase2" / "feasibility.json"
    assert path.exists(), (
        "scripts/feasibility_check.ps1 has not been run yet; "
        "expected report at reports/phase2/feasibility.json"
    )
    # PowerShell's `Set-Content -Encoding utf8` writes a BOM on Windows
    # PowerShell 5.1; `utf-8-sig` transparently strips it.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    assert "verdict" in payload
    assert "steps" in payload
    # The verdict must be one of the three documented values.
    assert payload["verdict"].startswith(("PROCEED", "BLOCK"))


def test_feasibility_upstream_repo_cloned() -> None:
    """Phase 2A checklist: upstream LogLLM repo must be locally clonable."""
    upstream = REPO_ROOT / "third_party" / "LogLLM"
    assert upstream.exists()
    assert (upstream / "model.py").exists()
    assert (upstream / "train.py").exists()
