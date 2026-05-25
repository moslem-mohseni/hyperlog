"""Tests for the BERT-based LogLineEncoder."""

from __future__ import annotations

from typing import Any

import torch

from hylog.models.encoder import EncoderConfig, LogLineEncoder


def test_encoder_hidden_size(tiny_bert: tuple[Any, Any]) -> None:
    model, tok = tiny_bert
    enc = LogLineEncoder(EncoderConfig(max_content_len=32), bert_model=model, tokenizer=tok)
    assert enc.hidden_size == 32


def test_encoder_forward_shape(tiny_bert: tuple[Any, Any]) -> None:
    model, tok = tiny_bert
    enc = LogLineEncoder(EncoderConfig(max_content_len=32), bert_model=model, tokenizer=tok)
    inputs = enc.tokenize(["hello world", "log line two", "third"])
    out = enc(inputs)
    assert out.shape == (3, 32)
    assert out.dtype == torch.float32


def test_encoder_freeze_disables_grads(tiny_bert: tuple[Any, Any]) -> None:
    model, tok = tiny_bert
    enc = LogLineEncoder(EncoderConfig(max_content_len=32), bert_model=model, tokenizer=tok)
    enc.freeze()
    assert not any(p.requires_grad for p in enc.bert.parameters())
