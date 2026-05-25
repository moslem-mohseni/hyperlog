"""BERT encoder wrapper used by both the LogLLM baseline and HyLog.

The encoder consumes a sequence of *log lines*, each tokenized independently,
and returns a per-line semantic vector taken from BERT's pooler output
(``[CLS]`` representation). This matches the upstream LogLLM behaviour at
``third_party/LogLLM/model.py:208`` (``self.Bert_model(**inputs).pooler_output``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Static knobs for the BERT encoder."""

    pretrained_name_or_path: str = "bert-base-uncased"
    max_content_len: int = 100
    """Token budget per log line (LogLLM upstream: train.py:27)."""


class LogLineEncoder(nn.Module):
    """Wraps a HuggingFace ``BertModel`` and exposes per-line pooled vectors.

    The wrapper hides three concerns from the rest of the pipeline:

    1. Tokenization is owned by the encoder so callers pass raw strings.
    2. The pooled output is returned in float32 for numerically stable
       projection (the upstream code casts to float in
       ``model.py:209``).
    3. Loading from a HF hub name and constructing from an in-memory
       ``BertModel`` are both supported — the latter enables CPU-only unit
       testing without network access.
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        *,
        bert_model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EncoderConfig()

        if bert_model is not None and tokenizer is not None:
            self.bert = bert_model
            self.tokenizer = tokenizer
        else:
            # Lazy import: transformers is a heavy import we avoid at module
            # load time in environments that only exercise the dataclass.
            from transformers import BertModel, BertTokenizerFast

            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.config.pretrained_name_or_path, do_lower_case=True
            )
            self.bert = BertModel.from_pretrained(self.config.pretrained_name_or_path)

    @property
    def hidden_size(self) -> int:
        """Embedding dimension of the encoder's pooled output."""
        return int(self.bert.config.hidden_size)

    def tokenize(self, lines: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of log lines to a batched input dict.

        Truncation is at ``max_content_len`` tokens per line; padding is to
        the longest line in the batch.
        """
        encoded = self.tokenizer(
            lines,
            padding=True,
            truncation=True,
            max_length=self.config.max_content_len,
            return_tensors="pt",
        )
        return dict(encoded.items())

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one pooled vector per line (shape: ``[n_lines, hidden]``)."""
        out = self.bert(**inputs)
        pooled = out.pooler_output  # [n_lines, hidden]
        return pooled.float()

    def freeze(self) -> None:
        """Disable gradient on every parameter."""
        for p in self.bert.parameters():
            p.requires_grad = False


__all__ = ["EncoderConfig", "LogLineEncoder"]
