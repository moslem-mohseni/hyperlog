"""Verify Phase 4 LOSO configs are present and well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent
LOSO_CONFIGS = REPO_ROOT / "configs" / "experiments"

REQUIRED_LOSO = {
    "loso_hdfs_held.yaml",
    "loso_bgl_held.yaml",
    "loso_thunderbird_held.yaml",
    "loso_openstack_held.yaml",
}


def test_all_required_loso_configs_present() -> None:
    present = {p.name for p in LOSO_CONFIGS.glob("loso_*.yaml")}
    missing = REQUIRED_LOSO - present
    assert not missing, f"missing LOSO configs: {missing}"


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_LOSO))
def test_loso_config_protocol_section(cfg_name: str) -> None:
    payload = yaml.safe_load((LOSO_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    assert "protocol" in payload, f"{cfg_name} missing protocol section"
    p = payload["protocol"]
    for key in ("held_out_system", "train_sources", "strip_target_labels"):
        assert key in p, f"{cfg_name} missing protocol.{key}"
    assert p["strip_target_labels"] is True, f"{cfg_name} must strip target labels"
    assert isinstance(p["train_sources"], list)
    # Held-out system MUST NOT appear in train sources.
    assert p["held_out_system"] not in p["train_sources"]


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_LOSO))
def test_loso_config_kill_switch_section(cfg_name: str) -> None:
    payload = yaml.safe_load((LOSO_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    assert "kill_switch" in payload
    ks = payload["kill_switch"]
    # Both kill-switches present and default-off.
    assert ks["enable_domain_adversarial"] is False
    assert ks["enable_self_supervised_target"] is False
    assert ks["lambda_domain"] == 0.0
    assert ks["self_sup_lambda"] == 0.0


@pytest.mark.parametrize("cfg_name", sorted(REQUIRED_LOSO))
def test_loso_config_5_seeds(cfg_name: str) -> None:
    payload = yaml.safe_load((LOSO_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    assert len(payload["run"]["seeds"]) == 5


def test_openstack_is_flagged_as_sensitivity() -> None:
    payload = yaml.safe_load(
        (LOSO_CONFIGS / "loso_openstack_held.yaml").read_text(encoding="utf-8")
    )
    assert payload["protocol"].get("sensitivity_fold") is True


def test_core_folds_not_flagged_as_sensitivity() -> None:
    for name in ("loso_hdfs_held.yaml", "loso_bgl_held.yaml", "loso_thunderbird_held.yaml"):
        payload = yaml.safe_load((LOSO_CONFIGS / name).read_text(encoding="utf-8"))
        assert payload["protocol"].get("sensitivity_fold", False) is False
