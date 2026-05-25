"""Tests for the VRAM estimator."""

from __future__ import annotations

import pytest

from hylog.training.vram import estimate_vram


def test_qwen_1_5b_4bit_fits_in_24gb() -> None:
    """Roadmap §4.1: Qwen-2.5-1.5B + BERT in 4-bit with L=128 lines fits 24GB."""
    est = estimate_vram(
        decoder="qwen2.5-1.5b",
        max_sequence_lines=128,
        micro_batch_size=4,
        quantize_4bit=True,
    )
    # The Phase 3 checklist target is <= 22 GB (headroom in 24 GB).
    assert est.fits_in(22.0), f"peak={est.peak_gib:.2f}"


def test_unquantized_is_strictly_larger_than_quantized() -> None:
    q = estimate_vram(decoder="qwen2.5-1.5b", quantize_4bit=True)
    bf16 = estimate_vram(decoder="qwen2.5-1.5b", quantize_4bit=False)
    assert bf16.peak_gib > q.peak_gib


def test_phi35_mini_3_8b_4bit_still_fits_24gb() -> None:
    """Phi-3.5-mini (3.8B) is the secondary backbone and must also fit."""
    est = estimate_vram(
        decoder="phi-3.5-mini",
        max_sequence_lines=128,
        micro_batch_size=2,  # smaller micro-batch given the larger model
        quantize_4bit=True,
    )
    assert est.fits_in(24.0)


def test_estimate_uses_registry_spec_for_string_name() -> None:
    est_str = estimate_vram(decoder="qwen2.5-1.5b")
    from hylog.models.decoder import get_decoder_spec

    est_obj = estimate_vram(decoder=get_decoder_spec("qwen2.5-1.5b"))
    assert est_str.to_dict() == est_obj.to_dict()


def test_estimate_dict_contains_all_components() -> None:
    est = estimate_vram(decoder="qwen2.5-1.5b")
    d = est.to_dict()
    for key in (
        "decoder_weights_gib",
        "encoder_weights_gib",
        "activations_gib",
        "lora_state_gib",
        "grad_workspace_gib",
        "headroom_gib",
        "peak_gib",
    ):
        assert key in d
        assert d[key] > 0


def test_peak_is_sum_of_components() -> None:
    est = estimate_vram(decoder="qwen2.5-1.5b")
    parts = (
        est.decoder_weights_gib
        + est.encoder_weights_gib
        + est.activations_gib
        + est.lora_state_gib
        + est.grad_workspace_gib
        + est.headroom_gib
    )
    assert est.peak_gib == pytest.approx(parts)
