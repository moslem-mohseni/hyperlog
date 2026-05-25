"""Verify the 8 Phase-6 ablation configs are valid."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent
ABLATION_DIR = REPO_ROOT / "configs" / "ablation"

REQUIRED_AXES = {
    "a1_hybrid_vs_standalone.yaml",
    "a2_lora_rank.yaml",
    "a3_lora_target_modules.yaml",
    "a4_projector_depth.yaml",
    "a5_encoder_lora.yaml",
    "a6_temperature_scaling.yaml",
    "a7_preprocessor.yaml",
    "a8_tau_calibration_shift.yaml",
}


def test_all_eight_ablation_configs_present() -> None:
    present = {p.name for p in ABLATION_DIR.glob("*.yaml")}
    missing = REQUIRED_AXES - present
    assert not missing, f"missing ablation configs: {missing}"


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_AXES))
def test_ablation_config_schema(cfg_name: str) -> None:
    payload = yaml.safe_load((ABLATION_DIR / cfg_name).read_text(encoding="utf-8"))
    axis = payload.get("axis", {})
    for key in ("name", "description", "baseline_condition", "conditions"):
        assert key in axis, f"{cfg_name} missing axis.{key}"
    names = {c["name"] for c in axis["conditions"]}
    assert axis["baseline_condition"] in names
    assert len(axis["conditions"]) >= 2
    run = payload.get("run", {})
    assert "seeds" in run
    assert len(run["seeds"]) == 5  # Phase-6 demands the same 5 seeds
    assert "primary_metric" in run


def test_a1_has_hybrid_baseline_and_standalone_variant() -> None:
    payload = yaml.safe_load(
        (ABLATION_DIR / "a1_hybrid_vs_standalone.yaml").read_text(encoding="utf-8")
    )
    names = {c["name"] for c in payload["axis"]["conditions"]}
    assert "hybrid_hylog_core" in names
    assert "standalone_qlora_decoder" in names


def test_a2_includes_all_four_ranks() -> None:
    payload = yaml.safe_load((ABLATION_DIR / "a2_lora_rank.yaml").read_text(encoding="utf-8"))
    ranks = {c["parameters"]["lora_rank"] for c in payload["axis"]["conditions"]}
    assert ranks == {4, 8, 16, 32}


def test_a3_includes_q_qv_qkvo() -> None:
    payload = yaml.safe_load(
        (ABLATION_DIR / "a3_lora_target_modules.yaml").read_text(encoding="utf-8")
    )
    targets = [c["parameters"]["lora_target"] for c in payload["axis"]["conditions"]]
    lens = sorted(len(t) for t in targets)
    assert lens == [1, 2, 4]


def test_a4_includes_depths_1_2_3() -> None:
    payload = yaml.safe_load((ABLATION_DIR / "a4_projector_depth.yaml").read_text(encoding="utf-8"))
    depths = {c["parameters"]["projector_depth"] for c in payload["axis"]["conditions"]}
    assert depths == {1, 2, 3}


def test_a6_primary_metric_is_ece() -> None:
    """The calibration axis is judged by ECE, not F1."""
    payload = yaml.safe_load(
        (ABLATION_DIR / "a6_temperature_scaling.yaml").read_text(encoding="utf-8")
    )
    assert payload["run"]["primary_metric"] == "ece"
