"""Projector — maps BERT pooled vectors into the decoder's input space.

Upstream LogLLM uses a single ``nn.Linear`` projector
(``third_party/LogLLM/model.py:102``). HyLog generalizes this to a small MLP
(1, 2, or 3 layers) so the Phase 6 ablation A4 (projector depth) is a
configuration knob rather than a code change. The 1-layer configuration is
exactly the LogLLM projector and is bit-for-bit equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ProjectorConfig:
    in_features: int
    out_features: int
    depth: int = 1
    """Number of nn.Linear layers. 1 = LogLLM. 2/3 = HyLog ablation A4."""
    hidden_multiplier: int = 1
    """Hidden width = ``hidden_multiplier * out_features`` (depth >= 2 only)."""
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.depth not in (1, 2, 3):
            raise ValueError(f"depth must be 1, 2, or 3; got {self.depth}")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if self.hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be >= 1")


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


class Projector(nn.Module):
    """Configurable MLP projector. depth=1 reproduces LogLLM bit-for-bit."""

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        if config.depth == 1:
            # Parity with LogLLM: single Linear, no activation, no dropout.
            # (Upstream: nn.Linear(bert_hidden, llama_hidden) — model.py:102.)
            layers.append(nn.Linear(config.in_features, config.out_features))
        else:
            hidden = config.hidden_multiplier * config.out_features
            layers.append(nn.Linear(config.in_features, hidden))
            for _ in range(config.depth - 2):
                layers.append(_make_activation(config.activation))
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                layers.append(nn.Linear(hidden, hidden))
            layers.append(_make_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(hidden, config.out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["Projector", "ProjectorConfig"]
