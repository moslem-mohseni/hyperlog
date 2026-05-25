"""Shared fixtures: tiny BERT and tiny LLaMA built from scratch (no network).

These fixtures use ``transformers.BertConfig`` / ``LlamaConfig`` directly so
unit tests run on CPU in seconds without touching the HuggingFace Hub.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(scope="session")
def tiny_bert() -> tuple[Any, Any]:
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=200,
        max_position_embeddings=128,
    )
    model = BertModel(cfg)
    # We can build a real tokenizer with bert-base-uncased's vocab only with
    # network. For unit tests we use a minimal in-memory tokenizer mock that
    # mimics the interface BertTokenizerFast exposes to LogLineEncoder.
    return model, _make_fake_bert_tokenizer(vocab_size=200, max_length=32)


@pytest.fixture(scope="session")
def tiny_llama() -> tuple[Any, Any]:
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
    model = LlamaForCausalLM(cfg)
    return model, _make_fake_llama_tokenizer(vocab_size=256)


def _make_fake_bert_tokenizer(vocab_size: int, max_length: int) -> Any:
    """Return an object compatible with the BertTokenizerFast call surface."""
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
            length = max_length or max_length
            # Deterministic toy token ids derived from line length modulo vocab.
            input_ids: list[list[int]] = []
            for ln in lines:
                ids = [(ord(c) % (vocab_size - 5)) + 5 for c in ln[: (length or 16)]]
                ids = [1, *ids[: (length or 16) - 2], 2]
                input_ids.append(ids)
            longest = max((len(ids) for ids in input_ids), default=2)
            padded = [ids + [0] * (longest - len(ids)) for ids in input_ids]
            attn = [[1 if t else 0 for t in ids] for ids in padded]
            ids_t = torch.tensor(padded, dtype=torch.long)
            attn_t = torch.tensor(attn, dtype=torch.long)
            return {"input_ids": ids_t, "attention_mask": attn_t}

    return _Tok()


def _make_fake_llama_tokenizer(vocab_size: int) -> Any:
    """Return an object compatible with the AutoTokenizer call surface used
    by LogLLMBaseline._prepare_instruction_tokens."""
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
