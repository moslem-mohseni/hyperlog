"""Verify per-decoder Hydra configs are present and well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DECODER_CONFIGS = REPO_ROOT / "configs" / "decoders"
EXPERIMENT_CONFIGS = REPO_ROOT / "configs" / "experiments"


REQUIRED_DECODER_CONFIGS = {
    "qwen25_1_5b.yaml",
    "qwen25_1_5b_instruct.yaml",
    "phi35_mini.yaml",
    "llama32_1b.yaml",
    "llama32_3b.yaml",
    "tinyllama.yaml",
}

REQUIRED_EXPERIMENT_CONFIGS = {
    "hylog_hdfs.yaml",
    "hylog_bgl.yaml",
}


def test_all_required_decoder_configs_present() -> None:
    present = {p.name for p in DECODER_CONFIGS.glob("*.yaml")}
    missing = REQUIRED_DECODER_CONFIGS - present
    assert not missing, f"missing decoder configs: {missing}"


def test_all_required_experiment_configs_present() -> None:
    present = {p.name for p in EXPERIMENT_CONFIGS.glob("*.yaml")}
    missing = REQUIRED_EXPERIMENT_CONFIGS - present
    assert not missing, f"missing experiment configs: {missing}"


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_DECODER_CONFIGS))
def test_decoder_config_keys(cfg_name: str) -> None:
    payload = yaml.safe_load((DECODER_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    for key in (
        "name",
        "hf_path",
        "hidden_size",
        "total_parameters_millions",
        "family",
        "lora_target_modules",
    ):
        assert key in payload, f"{cfg_name} missing {key}"
    assert isinstance(payload["hidden_size"], int)
    assert payload["hidden_size"] > 0
    assert isinstance(payload["lora_target_modules"], list)
    assert len(payload["lora_target_modules"]) > 0


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_EXPERIMENT_CONFIGS))
def test_experiment_config_structure(cfg_name: str) -> None:
    payload = yaml.safe_load((EXPERIMENT_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    for top in ("run", "model", "data", "trainer", "stages", "mlflow", "vram_budget_gib"):
        assert top in payload, f"{cfg_name} missing {top}"
    # Roadmap Phase 3 checklist: 5 seeds.
    assert len(payload["run"]["seeds"]) == 5
    # Phase 3 VRAM budget target.
    assert payload["vram_budget_gib"] <= 24.0


def test_decoder_configs_consistent_with_registry() -> None:
    """Each YAML must match the corresponding DecoderSpec in the registry."""
    from hylog.models.decoder import get_decoder_spec

    for cfg_path in DECODER_CONFIGS.glob("*.yaml"):
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        spec = get_decoder_spec(payload["name"])
        assert spec.hf_path == payload["hf_path"], cfg_path.name
        assert spec.hidden_size == payload["hidden_size"], cfg_path.name
        assert spec.family == payload["family"], cfg_path.name
        assert list(spec.lora_target_modules) == payload["lora_target_modules"], cfg_path.name
