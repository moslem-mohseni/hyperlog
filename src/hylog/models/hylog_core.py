"""HyLogCore — the hybrid encoder + projector + QLoRA-decoder + head model.

This is the Phase-3 deliverable: the model that retires LogLLM's Llama-7B
decoder in favour of a compact 1-4 B SLM, replaces autoregressive answer
generation with a deterministic classification head, and exposes the
training-mode toggles required by the three-stage trainer.

Pipeline (one forward pass)::

    log lines           list[str], possibly variable length per sequence
       |
       v
    BERT (frozen)       per-line pooled vectors of size bert.hidden_size
       |
       v
    Projector           per-line vectors of size decoder.hidden_size
       |
       v
    Decoder (QLoRA)     consumed as ``inputs_embeds`` — the decoder treats
                        each line as one position in its input. The
                        attention mask is constructed from sequence
                        lengths so padded positions are ignored.
       |
       v
    Last-token pool     a single ``decoder.hidden_size``-dim vector per
                        sequence taken from the last non-padded position
                        of the decoder's last hidden state.
       |
       v
    Classification head 2 logits: {normal, anomaly}

The design intentionally does *not* invoke the decoder's autoregressive
sampling code path; the model is a fixed-output classifier so that
post-hoc temperature scaling (Phase 5) is well-defined.

Differences from ``LogLLMBaseline``:

- Decoder is a 1-4 B SLM, not Llama-7B.
- Classification head replaces token matching.
- No instruction tokens; the prompt structure that LogLLM used is
  unnecessary for a deterministic classifier.
- Trainable parameter budget is strictly bounded (< 5 % of decoder
  total parameters; tested mechanically).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hylog.models.classification_head import BinaryClassificationHead
from hylog.models.decoder import DecoderSpec, get_decoder_spec
from hylog.models.encoder import EncoderConfig, LogLineEncoder
from hylog.models.projector import Projector, ProjectorConfig


@dataclass(frozen=True, slots=True)
class HyLogLoraConfig:
    """LoRA config applied to the decoder. HyLog default is the canonical
    QKVO set used by all major SLM families (see decoder registry)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None
    """If None, fall back to the registered spec's ``lora_target_modules``."""


@dataclass(frozen=True, slots=True)
class HyLogCoreConfig:
    """Top-level configuration for ``HyLogCore``."""

    decoder_name: str = "qwen2.5-1.5b"
    bert_path: str = "bert-base-uncased"
    max_content_len: int = 64  # roadmap §4.1: 64 sub-word tokens per line
    max_sequence_lines: int = 128
    """Max number of log lines in one sequence (decoder input length)."""

    quantize_4bit: bool = True
    projector_depth: int = 2
    projector_dropout: float = 0.1
    head_dropout: float = 0.1
    lora: HyLogLoraConfig = field(default_factory=HyLogLoraConfig)


class HyLogCore(nn.Module):
    """The HyLog hybrid model."""

    def __init__(
        self,
        config: HyLogCoreConfig | None = None,
        *,
        encoder: LogLineEncoder | None = None,
        decoder: nn.Module | None = None,
        decoder_tokenizer: Any | None = None,
        decoder_spec: DecoderSpec | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or HyLogCoreConfig()
        self.device_str = str(device)
        self.spec: DecoderSpec = (
            decoder_spec if decoder_spec is not None else get_decoder_spec(self.config.decoder_name)
        )

        # ---- Encoder ----
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = LogLineEncoder(
                EncoderConfig(
                    pretrained_name_or_path=self.config.bert_path,
                    max_content_len=self.config.max_content_len,
                )
            )

        # ---- Decoder + tokenizer ----
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

        # Discover the true hidden size from the loaded decoder; this is the
        # authoritative value (the registry's spec is just a hint).
        decoder_hidden_size = int(self.decoder.config.hidden_size)

        # ---- Projector (depth >= 2 by default; depth=1 reproduces LogLLM) ----
        self.projector = Projector(
            ProjectorConfig(
                in_features=self.encoder.hidden_size,
                out_features=decoder_hidden_size,
                depth=self.config.projector_depth,
                dropout=self.config.projector_dropout,
            )
        )

        # ---- LoRA adapters on the decoder ----
        self._apply_lora_adapters()

        # ---- Classification head ----
        self.head = BinaryClassificationHead(
            in_features=decoder_hidden_size, dropout=self.config.head_dropout
        )

        # Default: freeze the entire encoder; only the projector + LoRA + head
        # are trainable in the baseline configuration.
        self.encoder.freeze()
        self.set_train_projector_lora_head()

    # ------------------------------------------------------------------ ctor

    def _apply_lora_adapters(self) -> None:
        """Wrap the decoder in PEFT LoRA adapters.

        The target-modules list is the spec's default unless explicitly
        overridden in ``HyLogLoraConfig``.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        target = self.config.lora.target_modules or self.spec.lora_target_modules
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=list(target),
            bias="none",
        )
        self.decoder = get_peft_model(self.decoder, cfg)

    # ----------------------------------------------------------- training modes

    def _named_lora_params(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.decoder.named_parameters() if "lora" in n.lower()]

    def set_train_projector_only(self) -> None:
        """Stage 1: projector warm-up."""
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def set_train_projector_lora_head(self) -> None:
        """Stage 2: projector + LoRA + head jointly. The default config."""
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for _, p in self._named_lora_params():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def set_train_all_trainable(self) -> None:
        """Stage 3: end-to-end refinement at reduced lr."""
        # In HyLog the encoder is permanently frozen (encoder LoRA is a
        # Phase-6 ablation — A5 — not the default). Stage 3 retrains the
        # projector + decoder LoRA + head together, indistinguishable
        # numerically from Stage 2 except for the smaller learning rate.
        self.set_train_projector_lora_head()

    # ----------------------------------------------------------- introspection

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_decoder_parameters(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())

    def trainable_fraction(self) -> float:
        """Fraction of decoder parameters that are trainable.

        Roadmap Phase 3 checklist: must be < 5 %.
        """
        total = self.num_decoder_parameters()
        if total == 0:
            return 0.0
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable / total

    # -------------------------------------------------------------- forward

    def encode_lines(self, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a flat batch of tokenized log lines into projected vectors."""
        pooled = self.encoder(tokenized)
        return self.projector(pooled)

    def _last_position_hidden_state(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """For each sequence, take the hidden state at the last non-padded position.

        ``hidden`` has shape ``[batch, seq, hidden]``; ``attention_mask`` is
        ``[batch, seq]`` with 1 at real positions.
        """
        # Find the index of the last real position per sequence.
        lengths = attention_mask.sum(dim=1).long()  # [batch]
        last_idx = (lengths - 1).clamp(min=0)
        batch_size = hidden.size(0)
        return hidden[torch.arange(batch_size, device=hidden.device), last_idx]

    def forward(
        self,
        *,
        line_inputs: dict[str, torch.Tensor],
        sequence_lengths: list[int],
    ) -> torch.Tensor:
        """End-to-end forward pass.

        Args:
            line_inputs: Output of ``encoder.tokenize`` over a *flat* list
                of lines concatenated across all sequences in the batch.
            sequence_lengths: Number of lines in each sequence, in the same
                order as the lines appear in ``line_inputs``. Must sum to
                the total number of lines.

        Returns:
            Logits of shape ``[batch, 2]`` over {normal, anomaly}.
        """
        if not sequence_lengths:
            raise ValueError("sequence_lengths must be non-empty")
        total_lines = int(sum(sequence_lengths))
        n_input_lines = int(line_inputs["input_ids"].shape[0])
        if total_lines != n_input_lines:
            raise ValueError(
                f"sum(sequence_lengths)={total_lines} but line_inputs has {n_input_lines} lines"
            )

        # 1. BERT + projector over the flat batch of lines.
        flat_embeds = self.encode_lines(line_inputs)  # [total_lines, hidden]

        # 2. Reassemble into [batch, max_len, hidden] with attention mask.
        batch_size = len(sequence_lengths)
        max_len = max(sequence_lengths)
        max_len = min(max_len, self.config.max_sequence_lines)
        hidden_size = flat_embeds.shape[-1]
        device = flat_embeds.device
        embeds = torch.zeros(
            batch_size, max_len, hidden_size, device=device, dtype=flat_embeds.dtype
        )
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        offset = 0
        for b, seq_len in enumerate(sequence_lengths):
            truncated = min(seq_len, max_len)
            embeds[b, :truncated, :] = flat_embeds[offset : offset + truncated]
            attention_mask[b, :truncated] = 1
            offset += seq_len

        # 3. Decoder consumes the inputs_embeds path. We disable the cache to
        # keep peak VRAM predictable.
        decoder_out = self.decoder(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # PEFT-wrapped models expose hidden_states under the same key.
        hidden_states = decoder_out.hidden_states[-1]  # [batch, max_len, hidden]

        # 4. Pool: last non-padded position per sequence.
        pooled = self._last_position_hidden_state(hidden_states, attention_mask)

        # 5. Classification head -> 2 logits.
        logits = self.head(pooled)
        return logits


__all__ = ["HyLogCore", "HyLogCoreConfig", "HyLogLoraConfig"]
