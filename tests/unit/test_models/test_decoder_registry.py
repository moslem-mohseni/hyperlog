"""Tests for the decoder registry."""

from __future__ import annotations

import pytest

from hylog.models.decoder import (
    DecoderSpec,
    get_decoder_spec,
    list_decoder_specs,
    register_decoder,
)


def test_registry_contains_primary_and_secondary_backbones() -> None:
    specs = list_decoder_specs()
    for required in (
        "qwen2.5-1.5b",
        "qwen2.5-1.5b-instruct",
        "phi-3.5-mini",
        "llama-3.2-1b",
        "llama-3.2-3b",
        "tinyllama-1.1b",
    ):
        assert required in specs, f"missing required decoder: {required}"


def test_lookup_case_insensitive() -> None:
    a = get_decoder_spec("qwen2.5-1.5b")
    b = get_decoder_spec("QWEN2.5-1.5B")
    assert a is b


def test_lookup_unknown_raises_keyerror_with_helpful_message() -> None:
    with pytest.raises(KeyError) as ei:
        get_decoder_spec("nonexistent-model")
    # The message must list the registered names for helpful error reporting.
    msg = str(ei.value).lower()
    assert "qwen" in msg


def test_qwen_spec_has_expected_lora_targets() -> None:
    spec = get_decoder_spec("qwen2.5-1.5b")
    assert spec.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert spec.hidden_size == 1536
    assert spec.family == "qwen2"


def test_phi_uses_fused_qkv_target() -> None:
    spec = get_decoder_spec("phi-3.5-mini")
    # Phi-3 family uses a fused qkv_proj.
    assert "qkv_proj" in spec.lora_target_modules


def test_register_new_decoder_round_trip() -> None:
    new = DecoderSpec(
        name="test-fake-decoder",
        hf_path="org/fake",
        hidden_size=64,
        total_parameters_millions=10.0,
        family="testfamily",
    )
    register_decoder(new)
    got = get_decoder_spec("test-fake-decoder")
    assert got is new


def test_invalid_spec_raises() -> None:
    with pytest.raises(ValueError):
        DecoderSpec(
            name="",
            hf_path="x",
            hidden_size=64,
            total_parameters_millions=10.0,
        )
    with pytest.raises(ValueError):
        DecoderSpec(
            name="x",
            hf_path="x",
            hidden_size=-1,
            total_parameters_millions=10.0,
        )
    with pytest.raises(ValueError):
        DecoderSpec(
            name="x",
            hf_path="x",
            hidden_size=64,
            total_parameters_millions=0.0,
        )
    with pytest.raises(ValueError):
        DecoderSpec(
            name="x",
            hf_path="x",
            hidden_size=64,
            total_parameters_millions=1.0,
            lora_target_modules=(),
        )
