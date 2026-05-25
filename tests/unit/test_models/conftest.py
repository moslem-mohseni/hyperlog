"""Model-specific fixtures. Most fixtures live in tests/conftest.py."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(scope="function")
def tiny_bert(tiny_bert_per_test: tuple[Any, Any]) -> tuple[Any, Any]:
    """Alias used by older LogLLM baseline tests."""
    return tiny_bert_per_test


@pytest.fixture(scope="function")
def tiny_llama() -> tuple[Any, Any]:
    """Slightly smaller LLaMA used by the LogLLM baseline tests."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=96,
        vocab_size=256,
        max_position_embeddings=128,
    )
    from tests.conftest import _make_fake_llama_tokenizer

    return LlamaForCausalLM(cfg), _make_fake_llama_tokenizer(vocab_size=256)
