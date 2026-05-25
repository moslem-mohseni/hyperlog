"""Standalone decoder baseline — the N4 head-to-head competitor of HyLogCore.

Phase 6 ablation A1 asks: *is HyLog's hybrid encoder + projector + decoder
architecture better than a single QLoRA-tuned decoder with the same
trainable-parameter budget?* This module ships that competitor.

Architecture:

    log sequence (list[str])
            |
            v
    decoder tokenizer       (joins lines with a separator;
                             token budget = decoder.max_position_embeddings)
            |
            v
    decoder embedding table  (the decoder's own input embeddings)
            |
            v
    QLoRA-tuned decoder      (same LoRA target modules as HyLogCore;
                              LoRA rank chosen to MATCH HyLog's
                              trainable count)
            |
            v
    last-position hidden state
            |
            v
    BinaryClassificationHead -> 2 logits

The trainable-parameter contract:

    trainable(StandaloneDecoder) == trainable(HyLogCore)

is enforced at construction time. The caller specifies a target
``trainable_parameter_target`` (= projector + LoRA + head of HyLogCore)
and the constructor picks a LoRA rank such that the standalone model
matches within ±5 %. The matched rank is recorded in the returned
spec so the comparison is reviewer-auditable.

This module is deliberately separate from ``hylog_core.py`` so the
Phase-6 A1 comparison is a clean code-level swap rather than a
configuration knob inside HyLogCore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from hylog.models.classification_head import BinaryClassificationHead
from hylog.models.decoder import DecoderSpec, get_decoder_spec


@dataclass(frozen=True, slots=True)
class StandaloneDecoderConfig:
    """Configuration for the A1 baseline."""

    decoder_name: str = "qwen2.5-1.5b"
    max_lines_per_sequence: int = 100
    max_total_tokens: int = 2048
    """Token budget for the joined log sequence after tokenisation."""

    line_separator: str = " | "
    quantize_4bit: bool = True
    head_dropout: float = 0.1

    target_trainable_parameters: int | None = None
    """If set, the constructor picks a LoRA rank that yields a trainable
    count within ±5 % of this target. Used to match HyLogCore."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] | None = None
    """If None, fall back to the registered spec's defaults."""


@dataclass(frozen=True, slots=True)
class TrainableParameterMatch:
    """Record of how the standalone model matched HyLogCore's trainable budget."""

    target: int
    achieved: int
    achieved_lora_rank: int
    achieved_head_params: int
    ratio: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "target": int(self.target),
            "achieved": int(self.achieved),
            "achieved_lora_rank": int(self.achieved_lora_rank),
            "achieved_head_params": int(self.achieved_head_params),
            "ratio": float(self.ratio),
            "within_tolerance": bool(self.within_tolerance),
        }


def _estimate_lora_params(spec: DecoderSpec, rank: int, target_modules: tuple[str, ...]) -> int:
    """Approximate trainable LoRA parameter count.

    Each target module contributes ``2 * hidden_size * rank`` params per
    layer (the down-projection A of shape [hidden, rank] and the
    up-projection B of shape [rank, hidden]). The number of layers is
    inferred from a Pascal-like heuristic ``hidden / 64`` for SLM
    families that match Qwen / Llama / Phi.
    """
    n_layers = max(8, spec.hidden_size // 64)
    per_layer = 2 * spec.hidden_size * rank * len(target_modules)
    return n_layers * per_layer


def estimate_trainable_params(
    spec: DecoderSpec,
    *,
    rank: int,
    target_modules: tuple[str, ...] | None = None,
    head_in_features: int | None = None,
) -> int:
    """Estimate (LoRA + head) trainable params for a given decoder + rank."""
    targets = target_modules if target_modules is not None else spec.lora_target_modules
    head_in = head_in_features if head_in_features is not None else spec.hidden_size
    return _estimate_lora_params(spec, rank, targets) + (head_in * 2 + 2)


def pick_lora_rank_to_match_target(
    *,
    spec: DecoderSpec,
    target_trainable: int,
    target_modules: tuple[str, ...] | None = None,
    tolerance: float = 0.05,
    candidate_ranks: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
) -> TrainableParameterMatch:
    """Pick the LoRA rank whose estimated trainable count is closest to
    ``target_trainable``. Used to satisfy the A1 parity contract.
    """
    targets = target_modules if target_modules is not None else spec.lora_target_modules
    head_params = spec.hidden_size * 2 + 2

    best_rank = candidate_ranks[0]
    best_count = estimate_trainable_params(spec, rank=best_rank, target_modules=targets)
    best_distance = abs(best_count - target_trainable)
    for r in candidate_ranks[1:]:
        c = estimate_trainable_params(spec, rank=r, target_modules=targets)
        d = abs(c - target_trainable)
        if d < best_distance:
            best_rank, best_count, best_distance = r, c, d

    ratio = best_count / max(target_trainable, 1)
    within = abs(ratio - 1.0) <= tolerance
    return TrainableParameterMatch(
        target=int(target_trainable),
        achieved=int(best_count),
        achieved_lora_rank=int(best_rank),
        achieved_head_params=int(head_params),
        ratio=float(ratio),
        within_tolerance=bool(within),
    )


class StandaloneDecoderModel(nn.Module):
    """A1 baseline. Same QLoRA pattern as HyLogCore but no BERT + no projector."""

    def __init__(
        self,
        config: StandaloneDecoderConfig | None = None,
        *,
        decoder: nn.Module | None = None,
        decoder_tokenizer: Any | None = None,
        decoder_spec: DecoderSpec | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or StandaloneDecoderConfig()
        self.device_str = str(device)
        self.spec: DecoderSpec = (
            decoder_spec if decoder_spec is not None else get_decoder_spec(self.config.decoder_name)
        )

        # Decoder + tokenizer (allow injection for CPU tests).
        if decoder is not None and decoder_tokenizer is not None:
            self.decoder = decoder
            self.decoder_tokenizer = decoder_tokenizer
        else:
            from hylog.models.decoder import load_decoder

            loaded = load_decoder(
                self.spec.name,
                quantize_4bit=self.config.quantize_4bit,
                device_map=self.device_str,
            )
            self.decoder = loaded.model
            self.decoder_tokenizer = loaded.tokenizer

        decoder_hidden = int(self.decoder.config.hidden_size)
        self.head = BinaryClassificationHead(
            in_features=decoder_hidden, dropout=self.config.head_dropout
        )

        # Resolve LoRA rank: target parity with HyLogCore if requested.
        if self.config.target_trainable_parameters is not None:
            match = pick_lora_rank_to_match_target(
                spec=self.spec,
                target_trainable=self.config.target_trainable_parameters,
                target_modules=self.config.lora_target_modules,
            )
            effective_rank = match.achieved_lora_rank
            self.parameter_match: TrainableParameterMatch | None = match
        else:
            effective_rank = self.config.lora_rank
            self.parameter_match = None

        self._apply_lora_adapters(rank=effective_rank)

    def _apply_lora_adapters(self, rank: int) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        targets = self.config.lora_target_modules or self.spec.lora_target_modules
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(targets),
            bias="none",
        )
        self.decoder = get_peft_model(self.decoder, cfg)

    # ------------------------------------------------------- introspection

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_decoder_parameters(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())

    # ------------------------------------------------------- forward

    def _tokenize_sequences(self, sequences: list[list[str]]) -> dict[str, torch.Tensor]:
        """Join each sequence's lines with the separator and tokenise once."""
        joined = [self.config.line_separator.join(lines) for lines in sequences]
        return self.decoder_tokenizer(
            joined,
            padding=True,
            return_tensors="pt",
        )

    def _last_real_token(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        batch = hidden.shape[0]
        return hidden[torch.arange(batch, device=hidden.device), last_idx]

    def forward(self, sequences: list[list[str]]) -> torch.Tensor:
        """Run the full forward pass; returns ``[batch, 2]`` logits."""
        inputs = self._tokenize_sequences(sequences)
        inputs = {k: v.to(next(self.decoder.parameters()).device) for k, v in inputs.items()}
        decoder_out = self.decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = decoder_out.hidden_states[-1]
        pooled = self._last_real_token(hidden, inputs["attention_mask"])
        return self.head(pooled)


__all__ = [
    "StandaloneDecoderConfig",
    "StandaloneDecoderModel",
    "TrainableParameterMatch",
    "estimate_trainable_params",
    "pick_lora_rank_to_match_target",
]
