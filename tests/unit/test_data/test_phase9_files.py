"""Static-asset checks for Phase 9 paper deliverables."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_latex_manuscript_present() -> None:
    p = REPO_ROOT / "paper" / "main.tex"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "\\documentclass" in body
    assert "IEEEtran" in body
    assert "Moslem Mohseni Khah" in body
    # Every novelty claim must be named in the manuscript.
    for marker in (
        "N1 (architectural)",
        "N2 (empirical)",
        "N3 (uncertainty)",
        "N4 (ablation)",
    ):
        assert marker in body, f"manuscript missing claim {marker}"


def test_references_bib_present() -> None:
    p = REPO_ROOT / "paper" / "references.bib"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    for cite_key in (
        "deeplog2017",
        "logbert2021",
        "logfit2024",
        "logllm2024",
        "metalog2024",
        "zerolog2025",
        "guo2017",
        "qlora2023",
        "lora2021",
    ):
        assert "@" in body  # at least one bib entry
        assert cite_key in body, f"references.bib missing {cite_key}"


def test_build_figures_script_present() -> None:
    p = REPO_ROOT / "scripts" / "build_figures.py"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    for func in (
        "build_architecture",
        "build_loso_protocol",
        "build_reliability",
        "build_loso_bars",
        "build_ablation_matrix",
    ):
        assert func in body


def test_build_paper_scripts_present() -> None:
    for script in ("build_paper.ps1", "build_paper.sh"):
        p = REPO_ROOT / "scripts" / script
        assert p.exists()


def test_reproducibility_appendix_present_and_covers_every_table() -> None:
    p = REPO_ROOT / "reports" / "phase9" / "reproducibility_appendix.md"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    # Every paper table must appear in the appendix mapping.
    for table in ("tab:repro", "tab:loso", "tab:headtohead", "tab:calibration", "tab:ablation"):
        assert table in body, f"appendix missing {table}"
    # Every paper figure must appear.
    for fig in ("fig:architecture", "fig:loso", "fig:reliability"):
        assert fig in body, f"appendix missing {fig}"


def test_zenodo_metadata_present_and_valid() -> None:
    p = REPO_ROOT / ".zenodo.json"
    assert p.exists()
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["title"]
    assert payload["version"] == "1.0.0"
    creators = payload["creators"]
    assert creators
    assert creators[0]["name"] == "Mohseni Khah, Moslem"
    assert payload["license"] == "MIT"


def test_citation_cff_at_v1_0_0() -> None:
    p = REPO_ROOT / "CITATION.cff"
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert 'version: "1.0.0"' in body
    assert "Mohseni Khah" in body
    assert "Moslem" in body


def test_pyproject_at_v1_0_0() -> None:
    p = REPO_ROOT / "pyproject.toml"
    body = p.read_text(encoding="utf-8")
    assert 'version = "1.0.0"' in body


def test_readme_has_reproducing_section() -> None:
    p = REPO_ROOT / "README.md"
    body = p.read_text(encoding="utf-8")
    assert "Reproducing the paper" in body


def test_figures_script_produces_pdfs(tmp_path: Path) -> None:
    """``scripts/build_figures.py`` runs cleanly and writes ≥ 5 PDFs."""
    from scripts.build_figures import main

    rc = main(
        [
            "--reports-root",
            str(REPO_ROOT / "reports"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) >= 5
