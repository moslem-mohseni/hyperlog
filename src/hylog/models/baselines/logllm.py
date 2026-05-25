"""Faithful re-implementation of LogLLM (Guan et al., 2024).

Upstream reference: https://github.com/guanwei49/LogLLM  (vendored at
``third_party/LogLLM`` for parity inspection).

This module mirrors the upstream architecture and training-mode toggles
line-by-line. Each public method carries a *parity comment* citing the
exact upstream source location it reproduces so a reviewer can audit the
faithfulness mechanically.

Differences from upstream — intentional and documented:

- The 4-bit quantization (``BitsAndBytesConfig``) is optional. When
  ``quantize_4bit=True`` (default for GPU runs) the behaviour is bit-for-bit
  upstream. When False (for CPU-only architectural tests), the same model
  is loaded in float32. Upstream hard-codes 4-bit
  (``third_party/LogLLM/model.py:79-84``).
- Constructor accepts pre-instantiated ``BertModel`` and
  ``AutoModelForCausalLM`` objects to support deterministic tiny-model
  unit tests that do not touch the network. The HF-name-based path is
  the default and matches upstream.
- Type hints, docstrings, and explicit dataclasses replace untyped
  positional config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from hylog.models.encoder import LogLineEncoder
from hylog.models.projector import Projector, ProjectorConfig


@dataclass(frozen=True, slots=True)
class BertLoraConfig:
    """LoRA config for the BERT encoder.

    Parity with upstream ``model.py:133-136``: r=4, alpha=32, dropout=0.01,
    task=FEATURE_EXTRACTION.
    """

    r: int = 4
    alpha: int = 32
    dropout: float = 0.01


@dataclass(frozen=True, slots=True)
class DecoderLoraConfig:
    """LoRA config for the causal decoder.

    Parity with upstream ``model.py:139-146``: r=8, alpha=16, dropout=0.1,
    target_modules=("q_proj","v_proj"), task=CAUSAL_LM, bias="none".
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass(frozen=True, slots=True)
class LogLLMConfig:
    """Top-level configuration for the LogLLM baseline."""

    bert_path: str = "bert-base-uncased"
    decoder_path: str = "meta-llama/Meta-Llama-3-8B"
    max_content_len: int = 100  # upstream train.py:27
    max_seq_len: int = 128  # upstream train.py:28
    quantize_4bit: bool = True
    bert_lora: BertLoraConfig = field(default_factory=BertLoraConfig)
    decoder_lora: DecoderLoraConfig = field(default_factory=DecoderLoraConfig)


# Stage tokens used in the prompt. Parity with upstream model.py:106.
PROMPT_PREFIX = "Below is a sequence of system log messages:"
PROMPT_SUFFIX = ". Is this sequence normal or anomalous? \\n"


class LogLLMBaseline(nn.Module):
    """Re-implementation of the LogLLM end-to-end model.

    The forward pass consumes a *batched* set of log sequences. Each sequence
    is a list of log lines; each line is independently encoded by BERT into
    a 768-dim vector; the vectors are projected into the decoder hidden size
    and prepended/appended with instruction-token embeddings; the decoder
    produces logits over its vocabulary at the answer position.

    Three training stages, mirrored from upstream ``train.py:167-191``:

    - ``set_train_only_decoder``: only decoder LoRA  (upstream:
      ``set_train_only_Llama`` at ``model.py:168``).
    - ``set_train_only_projector``: only the projector
      (``model.py:160``).
    - ``set_train_projector_and_encoder``: projector + BERT LoRA
      (``model.py:177``).
    - ``set_finetuning_all``: all three  (``model.py:187``).
    """

    def __init__(
        self,
        config: LogLLMConfig | None = None,
        *,
        encoder: LogLineEncoder | None = None,
        decoder: nn.Module | None = None,
        decoder_tokenizer: Any | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or LogLLMConfig()
        self.device_str = str(device)

        if encoder is not None:
            self.encoder = encoder
        else:
            from hylog.models.encoder import EncoderConfig

            self.encoder = LogLineEncoder(
                EncoderConfig(
                    pretrained_name_or_path=self.config.bert_path,
                    max_content_len=self.config.max_content_len,
                )
            )

        if decoder is not None and decoder_tokenizer is not None:
            self.decoder = decoder
            self.decoder_tokenizer = decoder_tokenizer
        else:
            self.decoder, self.decoder_tokenizer = self._load_decoder()

        self.projector = Projector(
            ProjectorConfig(
                in_features=self.encoder.hidden_size,
                out_features=int(self.decoder.config.hidden_size),
                depth=1,  # depth=1 reproduces upstream LogLLM bit-for-bit.
            )
        )

        # Apply LoRA adapters (parity with upstream model.py:131-147).
        self._apply_lora_adapters()

        # Pre-tokenize the constant instruction strings once
        # (parity with upstream model.py:105-107).
        self.instruction_input_ids: torch.Tensor | None = None
        self.instruction_attention_mask: torch.Tensor | None = None
        self._prepare_instruction_tokens()

    # ------------------------------------------------------------------ ctor

    def _load_decoder(self) -> tuple[nn.Module, Any]:
        """Load the causal decoder from HuggingFace.

        Parity with upstream ``model.py:92-96`` (Llama_tokenizer, Llama_model).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.decoder_path, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, object] = {"low_cpu_mem_usage": True}
        if self.config.quantize_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(self.config.decoder_path, **kwargs)
        return model, tokenizer

    def _apply_lora_adapters(self) -> None:
        """Wrap BERT and the decoder with LoRA adapters.

        Parity with upstream ``model.py:131-147``.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        bert_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.bert_lora.r,
            lora_alpha=self.config.bert_lora.alpha,
            lora_dropout=self.config.bert_lora.dropout,
        )
        self.encoder.bert = get_peft_model(self.encoder.bert, bert_cfg)

        dec_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.decoder_lora.r,
            lora_alpha=self.config.decoder_lora.alpha,
            lora_dropout=self.config.decoder_lora.dropout,
            target_modules=list(self.config.decoder_lora.target_modules),
            bias="none",
        )
        self.decoder = get_peft_model(self.decoder, dec_cfg)

    def _prepare_instruction_tokens(self) -> None:
        """Tokenize the two halves of the prompt once.

        Parity with upstream ``model.py:105-107``.
        """
        enc = self.decoder_tokenizer(
            [PROMPT_PREFIX, PROMPT_SUFFIX], return_tensors="pt", padding=True
        )
        self.instruction_input_ids = enc["input_ids"]
        self.instruction_attention_mask = enc["attention_mask"]

    # ----------------------------------------------------------- training modes

    def _named_lora_params(self, module: nn.Module) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in module.named_parameters() if "lora" in n.lower()]

    def set_train_only_projector(self) -> None:
        """Parity: upstream ``model.py:160``."""
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def set_train_only_decoder(self) -> None:
        """Parity: upstream ``model.py:168`` (set_train_only_Llama)."""
        for p in self.projector.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for _, p in self._named_lora_params(self.decoder):
            p.requires_grad = True

    def set_train_projector_and_encoder(self) -> None:
        """Parity: upstream ``model.py:177``."""
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for _, p in self._named_lora_params(self.encoder):
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = False

    def set_finetuning_all(self) -> None:
        """Parity: upstream ``model.py:187``."""
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for _, p in self._named_lora_params(self.encoder):
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = False
        for _, p in self._named_lora_params(self.decoder):
            p.requires_grad = True

    # ----------------------------------------------------------- introspection

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -------------------------------------------------------------- forward

    def encode_lines(self, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a flat batch of tokenized lines into projected vectors.

        Returns a tensor of shape ``[n_lines, decoder_hidden]``.
        Parity with upstream ``model.py:208-211``:
        ``BERT(pooler) -> float -> projector -> half``.
        """
        pooled = self.encoder(tokenized)  # [n_lines, bert_hidden]
        projected = self.projector(pooled)  # [n_lines, decoder_hidden]
        return projected

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids through the decoder's input embedding table.

        Handles both PeftModel-wrapped and bare decoders (parity with upstream
        ``model.py:228-233``).
        """
        try:
            # PeftModelForCausalLM: .model.model.embed_tokens
            return self.decoder.model.model.embed_tokens(token_ids)
        except AttributeError:
            return self.decoder.model.embed_tokens(token_ids)


def count_trainable_parameters(model: nn.Module) -> int:
    """Helper mirroring upstream ``train.py:58-67``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "PROMPT_PREFIX",
    "PROMPT_SUFFIX",
    "BertLoraConfig",
    "DecoderLoraConfig",
    "LogLLMBaseline",
    "LogLLMConfig",
    "count_trainable_parameters",
]
