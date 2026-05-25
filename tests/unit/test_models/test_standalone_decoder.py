"""Tests for the A1 standalone-decoder baseline."""

from __future__ import annotations

from typing import Any

import torch

from hylog.models.decoder import DecoderSpec
from hylog.models.standalone_decoder import (
    StandaloneDecoderConfig,
    StandaloneDecoderModel,
    TrainableParameterMatch,
    estimate_trainable_params,
    pick_lora_rank_to_match_target,
)


def _tiny_spec(hidden: int = 64) -> DecoderSpec:
    return DecoderSpec(
        name="tiny-standalone",
        hf_path="tiny/x",
        hidden_size=hidden,
        total_parameters_millions=0.1,
        family="qwen2",
    )


def test_estimate_trainable_params_scales_with_rank() -> None:
    spec = _tiny_spec()
    p4 = estimate_trainable_params(spec, rank=4)
    p16 = estimate_trainable_params(spec, rank=16)
    # 4x increase in rank should ~4x the LoRA contribution.
    assert p16 > p4 * 3


def test_pick_rank_within_tolerance() -> None:
    spec = _tiny_spec()
    target = estimate_trainable_params(spec, rank=16)
    match = pick_lora_rank_to_match_target(spec=spec, target_trainable=target)
    assert isinstance(match, TrainableParameterMatch)
    assert match.achieved_lora_rank == 16
    assert match.within_tolerance


def test_pick_rank_far_from_candidates_marks_out_of_tolerance() -> None:
    spec = _tiny_spec()
    # Target is well outside any candidate rank.
    out = pick_lora_rank_to_match_target(spec=spec, target_trainable=1)
    assert not out.within_tolerance or out.achieved > 1


def test_standalone_model_forward_shape(tiny_qwen_decoder: tuple[Any, Any]) -> None:
    dec_model, dec_tok = tiny_qwen_decoder
    spec = DecoderSpec(
        name="tiny-stand",
        hf_path="x/x",
        hidden_size=int(dec_model.config.hidden_size),
        total_parameters_millions=0.1,
        family="qwen2",
    )
    cfg = StandaloneDecoderConfig(
        decoder_name=spec.name,
        quantize_4bit=False,
        lora_rank=4,
    )
    model = StandaloneDecoderModel(
        cfg,
        decoder=dec_model,
        decoder_tokenizer=dec_tok,
        decoder_spec=spec,
    )
    out = model(sequences=[["line a", "line b"], ["only one"]])
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_standalone_model_attaches_lora(tiny_qwen_decoder: tuple[Any, Any]) -> None:
    dec_model, dec_tok = tiny_qwen_decoder
    spec = DecoderSpec(
        name="tiny-stand-lora",
        hf_path="x/x",
        hidden_size=int(dec_model.config.hidden_size),
        total_parameters_millions=0.1,
        family="qwen2",
    )
    cfg = StandaloneDecoderConfig(
        decoder_name=spec.name,
        quantize_4bit=False,
        lora_rank=4,
    )
    model = StandaloneDecoderModel(
        cfg,
        decoder=dec_model,
        decoder_tokenizer=dec_tok,
        decoder_spec=spec,
    )
    lora_params = [n for n, _ in model.decoder.named_parameters() if "lora" in n.lower()]
    assert lora_params


def test_match_target_records_metadata(tiny_qwen_decoder: tuple[Any, Any]) -> None:
    dec_model, dec_tok = tiny_qwen_decoder
    spec = DecoderSpec(
        name="tiny-stand-match",
        hf_path="x/x",
        hidden_size=int(dec_model.config.hidden_size),
        total_parameters_millions=0.1,
        family="qwen2",
    )
    target = estimate_trainable_params(spec, rank=8)
    cfg = StandaloneDecoderConfig(
        decoder_name=spec.name,
        quantize_4bit=False,
        target_trainable_parameters=target,
    )
    model = StandaloneDecoderModel(
        cfg,
        decoder=dec_model,
        decoder_tokenizer=dec_tok,
        decoder_spec=spec,
    )
    assert model.parameter_match is not None
    assert model.parameter_match.achieved_lora_rank == 8
