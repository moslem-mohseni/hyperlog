"""Architectural tests for the LogLLM re-implementation.

These tests verify *structural* parity with upstream LogLLM without
requiring a GPU or any 4-bit quantization. They construct a tiny BERT and a
tiny LLaMA from scratch, wire them through ``LogLLMBaseline``, and check:

- LoRA adapters are attached to both encoder and decoder.
- Each of the four training-mode setters produces the expected
  ``requires_grad`` mask.
- ``encode_lines`` returns the correct shape.
- The instruction tokens are prepared.

GPU-resident behaviour (4-bit quantization, full Llama-3-8B) is verified
on real hardware via Phase 2B's reproduction run, which is gated on GPU
availability.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from hylog.models.baselines.logllm import (
    BertLoraConfig,
    DecoderLoraConfig,
    LogLLMBaseline,
    LogLLMConfig,
    count_trainable_parameters,
)
from hylog.models.encoder import EncoderConfig, LogLineEncoder


def _build_tiny_model(tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]) -> LogLLMBaseline:
    bert_model, bert_tok = tiny_bert
    llama_model, llama_tok = tiny_llama
    enc = LogLineEncoder(
        EncoderConfig(max_content_len=32), bert_model=bert_model, tokenizer=bert_tok
    )
    return LogLLMBaseline(
        LogLLMConfig(quantize_4bit=False),
        encoder=enc,
        decoder=llama_model,
        decoder_tokenizer=llama_tok,
        device="cpu",
    )


def test_construction_attaches_lora_to_both(
    tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]
) -> None:
    model = _build_tiny_model(tiny_bert, tiny_llama)
    encoder_lora = [n for n, _ in model.encoder.bert.named_parameters() if "lora" in n.lower()]
    decoder_lora = [n for n, _ in model.decoder.named_parameters() if "lora" in n.lower()]
    assert encoder_lora, "expected LoRA params in encoder"
    assert decoder_lora, "expected LoRA params in decoder"


def test_projector_shapes_match_components(
    tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]
) -> None:
    model = _build_tiny_model(tiny_bert, tiny_llama)
    proj = model.projector.layers[0]
    assert isinstance(proj, torch.nn.Linear)
    assert proj.in_features == model.encoder.hidden_size  # 32
    assert proj.out_features == int(model.decoder.config.hidden_size)  # 48


def test_instruction_tokens_prepared(
    tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]
) -> None:
    model = _build_tiny_model(tiny_bert, tiny_llama)
    assert model.instruction_input_ids is not None
    assert model.instruction_input_ids.dim() == 2
    assert model.instruction_input_ids.shape[0] == 2  # prefix + suffix


def test_encode_lines_shape(tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]) -> None:
    model = _build_tiny_model(tiny_bert, tiny_llama)
    tokenized = model.encoder.tokenize(["log a", "log bb", "third line"])
    out = model.encode_lines(tokenized)
    assert out.shape == (3, int(model.decoder.config.hidden_size))


@pytest.mark.parametrize(
    "switch,expected_groups",
    [
        ("set_train_only_projector", {"projector"}),
        ("set_train_only_decoder", {"decoder_lora"}),
        ("set_train_projector_and_encoder", {"projector", "encoder_lora"}),
        ("set_finetuning_all", {"projector", "encoder_lora", "decoder_lora"}),
    ],
)
def test_training_modes_set_correct_grads(
    tiny_bert: tuple[Any, Any],
    tiny_llama: tuple[Any, Any],
    switch: str,
    expected_groups: set[str],
) -> None:
    model = _build_tiny_model(tiny_bert, tiny_llama)
    getattr(model, switch)()

    def grad_state(module: torch.nn.Module, filter_lora: bool = False) -> bool:
        return any(
            p.requires_grad
            for n, p in module.named_parameters()
            if (not filter_lora) or "lora" in n.lower()
        )

    expected_proj = "projector" in expected_groups
    expected_enc = "encoder_lora" in expected_groups
    expected_dec = "decoder_lora" in expected_groups

    assert grad_state(model.projector) is expected_proj
    enc_lora_active = grad_state(model.encoder, filter_lora=True)
    dec_lora_active = grad_state(model.decoder, filter_lora=True)
    assert enc_lora_active is expected_enc
    assert dec_lora_active is expected_dec

    # Non-LoRA params of encoder/decoder must always be frozen.
    enc_non_lora_active = any(
        p.requires_grad for n, p in model.encoder.bert.named_parameters() if "lora" not in n.lower()
    )
    dec_non_lora_active = any(
        p.requires_grad for n, p in model.decoder.named_parameters() if "lora" not in n.lower()
    )
    assert not enc_non_lora_active
    assert not dec_non_lora_active


def test_trainable_param_helper_matches_upstream_semantics(
    tiny_bert: tuple[Any, Any], tiny_llama: tuple[Any, Any]
) -> None:
    """count_trainable_parameters mirrors upstream train.py:58-67."""
    model = _build_tiny_model(tiny_bert, tiny_llama)
    model.set_train_only_projector()
    n = count_trainable_parameters(model)
    # Only the projector is trainable in this stage: in*out + out params.
    expected = model.encoder.hidden_size * int(model.decoder.config.hidden_size) + int(
        model.decoder.config.hidden_size
    )
    assert n == expected


def test_bert_lora_config_defaults_match_upstream() -> None:
    """Parity with upstream model.py:133-136."""
    cfg = BertLoraConfig()
    assert (cfg.r, cfg.alpha, cfg.dropout) == (4, 32, 0.01)


def test_decoder_lora_config_defaults_match_upstream() -> None:
    """Parity with upstream model.py:139-146."""
    cfg = DecoderLoraConfig()
    assert (cfg.r, cfg.alpha, cfg.dropout) == (8, 16, 0.1)
    assert cfg.target_modules == ("q_proj", "v_proj")
