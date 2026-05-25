"""Shared pytest fixtures (project-wide)."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(scope="session")
def fixed_seed() -> int:
    return 1337


# ----------------------------------------------------------------------
# Tiny model fixtures shared by tests/unit/test_models and test_training.
# Function scope so PEFT does not warn about re-wrapping a previously
# wrapped model.
# ----------------------------------------------------------------------


@pytest.fixture(scope="function")
def tiny_bert_per_test() -> tuple[Any, Any]:
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=200,
        max_position_embeddings=128,
    )
    return BertModel(cfg), _make_fake_bert_tokenizer(vocab_size=200)


@pytest.fixture(scope="function")
def tiny_qwen_decoder() -> tuple[Any, Any]:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(cfg), _make_fake_llama_tokenizer(vocab_size=256)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_fake_bert_tokenizer(vocab_size: int) -> Any:
    import torch

    class _Tok:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2

        def __call__(
            self,
            text: Any,
            padding: bool = True,
            truncation: bool = True,
            max_length: int | None = None,
            return_tensors: str | None = None,
        ) -> dict[str, torch.Tensor]:
            lines = [text] if isinstance(text, str) else list(text)
            length = max_length or 24
            input_ids: list[list[int]] = []
            for ln in lines:
                ids = [(ord(c) % (vocab_size - 5)) + 5 for c in ln[:length]]
                ids = [1, *ids[: length - 2], 2]
                input_ids.append(ids)
            longest = max((len(ids) for ids in input_ids), default=2)
            padded = [ids + [0] * (longest - len(ids)) for ids in input_ids]
            attn = [[1 if t else 0 for t in ids] for ids in padded]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

    return _Tok()


def _make_fake_llama_tokenizer(vocab_size: int) -> Any:
    import torch

    class _Tok:
        pad_token = "<pad>"
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 1

        def __call__(
            self,
            text: Any,
            padding: bool = True,
            return_tensors: str | None = None,
        ) -> dict[str, torch.Tensor]:
            lines = [text] if isinstance(text, str) else list(text)
            input_ids: list[list[int]] = []
            for ln in lines:
                ids = [(ord(c) % (vocab_size - 5)) + 5 for c in ln[:32]]
                input_ids.append(ids if ids else [1])
            longest = max(len(ids) for ids in input_ids)
            padded = [ids + [0] * (longest - len(ids)) for ids in input_ids]
            attn = [[1 if t else 0 for t in ids] for ids in padded]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

    return _Tok()
