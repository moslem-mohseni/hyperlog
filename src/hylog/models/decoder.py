"""Decoder registry — pluggable backbone for the HyLog hybrid pipeline.

Each entry encodes everything callers need to load a backbone without
hard-coding model-specific knowledge into the trainer or HyLogCore:

- HuggingFace identifier (the canonical pretrained checkpoint).
- Suitable LoRA target modules for the backbone family.
- A short human-readable name used in MLflow runs, file paths, reports.
- An approximate parameter count for VRAM planning and "trainable
  parameters < 5 % of total" assertions.

The registry is the single source of truth: configs reference decoders by
name, the trainer dispatches by name, the VRAM estimator reads parameter
counts by name. Adding a new backbone is a one-line entry here plus a
config file under ``configs/decoders/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DecoderSpec:
    """Static description of a decoder backbone.

    Attributes:
        name: Short human-readable name (also the registry key).
        hf_path: HuggingFace identifier passed to ``AutoModelForCausalLM``.
        hidden_size: Hidden dimension of the model (used by the projector
            auto-discovery; an authoritative value can be read off the
            instantiated ``config`` but pre-recording it here lets us
            estimate VRAM and document expectations without instantiating
            the model).
        total_parameters: Approximate total parameter count, in millions.
            Used for VRAM planning and the trainable-parameter ratio test.
        lora_target_modules: Module names inside the decoder that the LoRA
            adapter wraps. The canonical "QKVO" set is used by default; the
            roadmap §Phase 6 ablation A3 explores sub-sets.
        instruct_variant: True if the registered checkpoint is the
            instruction-tuned variant (suffix ``-Instruct`` on HF).
        family: Architecture family identifier ("qwen2", "phi3",
            "llama3.2", "tinyllama").
    """

    name: str
    hf_path: str
    hidden_size: int
    total_parameters_millions: float
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    instruct_variant: bool = False
    family: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DecoderSpec.name must be non-empty")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.total_parameters_millions <= 0:
            raise ValueError("total_parameters_millions must be positive")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must be non-empty")


# Hidden sizes and parameter counts were verified against the published
# HuggingFace model cards. Cross-references:
# - Qwen-2.5-1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
# - Phi-3.5-mini : https://huggingface.co/microsoft/Phi-3.5-mini-instruct
# - Llama-3.2-1B : https://huggingface.co/meta-llama/Llama-3.2-1B
# - Llama-3.2-3B : https://huggingface.co/meta-llama/Llama-3.2-3B
# - TinyLlama-1.1B: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
_REGISTERED_DECODERS: dict[str, DecoderSpec] = {
    # ---- Primary backbone (roadmap §1.1) -----------------------------
    "qwen2.5-1.5b": DecoderSpec(
        name="qwen2.5-1.5b",
        hf_path="Qwen/Qwen2.5-1.5B",
        hidden_size=1536,
        total_parameters_millions=1540.0,
        family="qwen2",
    ),
    "qwen2.5-1.5b-instruct": DecoderSpec(
        name="qwen2.5-1.5b-instruct",
        hf_path="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=1536,
        total_parameters_millions=1540.0,
        instruct_variant=True,
        family="qwen2",
    ),
    # ---- Secondary backbones (roadmap §1.1) --------------------------
    "phi-3.5-mini": DecoderSpec(
        name="phi-3.5-mini",
        hf_path="microsoft/Phi-3.5-mini-instruct",
        hidden_size=3072,
        total_parameters_millions=3820.0,
        # Phi-3 attention uses "qkv_proj" as a single fused projection;
        # we list the conventional sub-names to allow PEFT to match either
        # the fused or unfused module names depending on its version.
        lora_target_modules=("qkv_proj", "o_proj"),
        instruct_variant=True,
        family="phi3",
    ),
    "llama-3.2-1b": DecoderSpec(
        name="llama-3.2-1b",
        hf_path="meta-llama/Llama-3.2-1B",
        hidden_size=2048,
        total_parameters_millions=1235.0,
        family="llama3.2",
    ),
    "llama-3.2-3b": DecoderSpec(
        name="llama-3.2-3b",
        hf_path="meta-llama/Llama-3.2-3B",
        hidden_size=3072,
        total_parameters_millions=3210.0,
        family="llama3.2",
    ),
    # ---- Additional point (roadmap §1.2) -----------------------------
    "tinyllama-1.1b": DecoderSpec(
        name="tinyllama-1.1b",
        hf_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hidden_size=2048,
        total_parameters_millions=1100.0,
        family="tinyllama",
    ),
}


def get_decoder_spec(name: str) -> DecoderSpec:
    """Look up a decoder by registry key. Case-insensitive."""
    key = name.lower()
    if key not in _REGISTERED_DECODERS:
        registered = ", ".join(sorted(_REGISTERED_DECODERS))
        raise KeyError(f"unknown decoder {name!r}; registered: {registered}")
    return _REGISTERED_DECODERS[key]


def list_decoder_specs() -> Mapping[str, DecoderSpec]:
    """Return the full registry. Immutable view."""
    return dict(_REGISTERED_DECODERS)


def register_decoder(spec: DecoderSpec) -> None:
    """Add a decoder to the registry. Used for test fixtures only."""
    _REGISTERED_DECODERS[spec.name.lower()] = spec


@dataclass(frozen=True, slots=True)
class LoadedDecoder:
    """Container for a loaded backbone + its tokenizer."""

    spec: DecoderSpec
    model: Any
    tokenizer: Any
    quantize_4bit: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)


def load_decoder(
    name: str,
    *,
    quantize_4bit: bool = True,
    device_map: str | None = None,
) -> LoadedDecoder:
    """Load a registered decoder + tokenizer via transformers + (optional) bnb.

    This is the GPU/production path. CPU-only tests construct decoders
    directly from ``LlamaConfig`` to avoid network access — see
    ``tests/unit/test_models/conftest.py``.
    """
    spec = get_decoder_spec(name)
    # Lazy imports keep module load fast and avoid hard dependency on
    # transformers being installed for callers that only need the spec.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if quantize_4bit:
        import torch
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(spec.hf_path, **kwargs)
    return LoadedDecoder(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        quantize_4bit=quantize_4bit,
        config={"hf_path": spec.hf_path, "device_map": device_map},
    )


__all__ = [
    "DecoderSpec",
    "LoadedDecoder",
    "get_decoder_spec",
    "list_decoder_specs",
    "load_decoder",
    "register_decoder",
]
