"""Pre-flight VRAM estimator for HyLog configurations.

Predicts peak GPU memory of a (decoder, sequence length, batch size, dtype)
configuration so a config can be validated *before* spending an hour on a
training run that OOMs. The model follows the back-of-the-envelope used in
docs/ROADMAP.md §4.1:

  peak ≈ bert_weights + decoder_weights_4bit + activations + lora_state + grad_workspace

Each term has a documented formula; the estimator deliberately overshoots
slightly so a green light is reliable.
"""

from __future__ import annotations

from dataclasses import dataclass

from hylog.models.decoder import DecoderSpec, get_decoder_spec

_BYTES_PER_GIB = 1024.0**3


@dataclass(frozen=True, slots=True)
class VRAMEstimate:
    decoder_weights_gib: float
    encoder_weights_gib: float
    activations_gib: float
    lora_state_gib: float
    grad_workspace_gib: float
    headroom_gib: float = 1.0

    @property
    def peak_gib(self) -> float:
        return (
            self.decoder_weights_gib
            + self.encoder_weights_gib
            + self.activations_gib
            + self.lora_state_gib
            + self.grad_workspace_gib
            + self.headroom_gib
        )

    def fits_in(self, gpu_capacity_gib: float) -> bool:
        return self.peak_gib <= gpu_capacity_gib

    def to_dict(self) -> dict[str, float]:
        return {
            "decoder_weights_gib": self.decoder_weights_gib,
            "encoder_weights_gib": self.encoder_weights_gib,
            "activations_gib": self.activations_gib,
            "lora_state_gib": self.lora_state_gib,
            "grad_workspace_gib": self.grad_workspace_gib,
            "headroom_gib": self.headroom_gib,
            "peak_gib": self.peak_gib,
        }


def estimate_vram(
    *,
    decoder: DecoderSpec | str,
    max_sequence_lines: int = 128,
    micro_batch_size: int = 4,
    bert_parameters_millions: float = 110.0,
    quantize_4bit: bool = True,
    lora_trainable_millions: float = 8.0,
) -> VRAMEstimate:
    """Estimate peak training VRAM for the given configuration."""
    spec = get_decoder_spec(decoder) if isinstance(decoder, str) else decoder

    # Decoder weights: 4-bit nf4 = 0.5 byte/param when quantized,
    # bfloat16 = 2 bytes/param otherwise.
    bytes_per_decoder_param = 0.5 if quantize_4bit else 2.0
    decoder_weights = spec.total_parameters_millions * 1e6 * bytes_per_decoder_param

    # BERT-base frozen in fp16 -> 2 bytes/param.
    encoder_weights = bert_parameters_millions * 1e6 * 2.0

    # Decoder activations: rough estimate for one forward+backward at
    # sequence length L, batch B, hidden h = ``decoder.hidden_size``,
    # number of layers ~ h/64 (a heuristic that matches Qwen/Llama).
    hidden = spec.hidden_size
    n_layers_estimate = max(8, hidden // 64)
    # Per layer: attention scores B*L*L + hidden B*L*h; bytes = 2 (bf16).
    per_layer = (
        micro_batch_size * max_sequence_lines * max_sequence_lines
        + micro_batch_size * max_sequence_lines * hidden
    )
    activations = per_layer * n_layers_estimate * 2.0  # bytes

    # LoRA adapter weights + optimizer state (AdamW = m, v in fp32 = 8 bytes/param).
    lora_state = lora_trainable_millions * 1e6 * (2.0 + 8.0)  # bf16 weights + fp32 m,v

    # Gradient activations workspace -- proportional to activations.
    grad_workspace = activations * 1.5

    return VRAMEstimate(
        decoder_weights_gib=decoder_weights / _BYTES_PER_GIB,
        encoder_weights_gib=encoder_weights / _BYTES_PER_GIB,
        activations_gib=activations / _BYTES_PER_GIB,
        lora_state_gib=lora_state / _BYTES_PER_GIB,
        grad_workspace_gib=grad_workspace / _BYTES_PER_GIB,
    )


__all__ = ["VRAMEstimate", "estimate_vram"]
