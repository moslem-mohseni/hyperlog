"""Tests for HyLogCore — the Phase-3 hybrid model."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from hylog.models.decoder import DecoderSpec
from hylog.models.encoder import EncoderConfig, LogLineEncoder
from hylog.models.hylog_core import HyLogCore, HyLogCoreConfig, HyLogLoraConfig


def _build_core(
    tiny_bert_per_test: tuple[Any, Any],
    tiny_qwen_decoder: tuple[Any, Any],
    *,
    projector_depth: int = 2,
    lora_r: int = 16,
) -> HyLogCore:
    bert_model, bert_tok = tiny_bert_per_test
    dec_model, dec_tok = tiny_qwen_decoder
    enc = LogLineEncoder(
        EncoderConfig(max_content_len=32), bert_model=bert_model, tokenizer=bert_tok
    )
    spec = DecoderSpec(
        name="tiny-test-qwen",
        hf_path="tiny/test",
        hidden_size=int(dec_model.config.hidden_size),
        total_parameters_millions=0.1,
        family="qwen2",
    )
    return HyLogCore(
        HyLogCoreConfig(
            decoder_name=spec.name,
            quantize_4bit=False,
            projector_depth=projector_depth,
            lora=HyLogLoraConfig(r=lora_r, alpha=2 * lora_r, dropout=0.0),
            max_sequence_lines=16,
        ),
        encoder=enc,
        decoder=dec_model,
        decoder_tokenizer=dec_tok,
        decoder_spec=spec,
        device="cpu",
    )


def test_construction_attaches_lora_to_decoder(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    lora_params = [n for n, _ in core.decoder.named_parameters() if "lora" in n.lower()]
    assert lora_params, "expected LoRA params attached to decoder"


def test_encoder_frozen_after_construction(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    assert not any(p.requires_grad for p in core.encoder.parameters())


def test_default_training_mode_is_projector_lora_head(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    assert all(p.requires_grad for p in core.projector.parameters())
    assert all(p.requires_grad for p in core.head.parameters())
    lora_active = any(
        p.requires_grad for n, p in core.decoder.named_parameters() if "lora" in n.lower()
    )
    non_lora_active = any(
        p.requires_grad for n, p in core.decoder.named_parameters() if "lora" not in n.lower()
    )
    assert lora_active
    assert not non_lora_active


def test_projector_uses_correct_dimensions(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    # Projector maps bert_hidden -> decoder_hidden via depth-2 MLP.
    layers = list(core.projector.layers.children())
    linear_layers = [m for m in layers if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].in_features == 32  # BERT hidden
    assert linear_layers[-1].out_features == 64  # decoder hidden


def test_trainable_fraction_bounded_in_vivo(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    """Trainable fraction is bounded on the tiny test architecture.

    On the production-scale Qwen-2.5-1.5B (1.54 B params) the same model
    geometry yields ``trainable_fraction`` well under 5 %; see
    ``test_trainable_fraction_under_5_percent_at_production_scale`` for the
    analytical check.
    """
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder, lora_r=2)
    frac = core.trainable_fraction()
    # Tiny architecture: head + projector dominate; just assert it is sane.
    assert 0.0 < frac < 0.5


def test_trainable_fraction_under_5_percent_at_production_scale() -> None:
    """Phase 3 checklist: at production scale (Qwen-2.5-1.5B), the
    trainable budget (projector depth-2 + LoRA r=16 on QKVO + head) is
    < 5 % of the decoder's total parameter count.

    Computed analytically so the test does not need to instantiate the
    real 1.5 B-parameter model.
    """
    bert_hidden = 768
    dec_hidden = 1536
    dec_total_params = int(1.54e9)  # Qwen-2.5-1.5B reported total
    dec_layers = 28  # Qwen-2.5-1.5B reported layer count
    lora_r = 16
    target_modules = 4  # q, k, v, o

    # Projector depth=2 with hidden_multiplier=1: in -> dec_hidden -> dec_hidden.
    projector = (bert_hidden * dec_hidden + dec_hidden) + (dec_hidden * dec_hidden + dec_hidden)
    # LoRA per target module per layer: 2 matrices of shape (dec_hidden, r) each.
    lora_params = dec_layers * target_modules * 2 * (dec_hidden * lora_r)
    # Head: dec_hidden -> 2 logits.
    head = dec_hidden * 2 + 2

    trainable = projector + lora_params + head
    fraction = trainable / dec_total_params
    assert fraction < 0.05, f"production trainable fraction {fraction:.3%} >= 5%"


def test_forward_shapes(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    # Two sequences with 3 and 5 lines respectively.
    sequences = [
        ["line one alpha", "line two beta", "line three gamma"],
        ["a", "b", "c", "d", "e"],
    ]
    flat = [line for seq in sequences for line in seq]
    tokenized = core.encoder.tokenize(flat)
    logits = core(
        line_inputs=tokenized,
        sequence_lengths=[len(s) for s in sequences],
    )
    assert logits.shape == (2, 2)
    # Logits must be finite — basic numerical sanity check.
    assert torch.isfinite(logits).all()


def test_forward_invalid_sequence_lengths(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    tokenized = core.encoder.tokenize(["a", "b", "c"])
    # 3 lines but we say [1, 1] = 2 — mismatch must raise.
    with pytest.raises(ValueError):
        core(line_inputs=tokenized, sequence_lengths=[1, 1])


def test_set_train_projector_only_freezes_lora(
    tiny_bert_per_test: tuple[Any, Any], tiny_qwen_decoder: tuple[Any, Any]
) -> None:
    core = _build_core(tiny_bert_per_test, tiny_qwen_decoder)
    core.set_train_projector_only()
    lora_active = any(
        p.requires_grad for n, p in core.decoder.named_parameters() if "lora" in n.lower()
    )
    assert not lora_active
    assert all(p.requires_grad for p in core.projector.parameters())
    assert all(p.requires_grad for p in core.head.parameters())
