"""Masked-line self-supervision over unlabeled target-system lines.

Phase 4 kill-switch option (a): augment training with a small fraction
of target-system **unlabeled** logs via masked self-prediction. The
encoder + projector learn to reconstruct masked positions of the
per-line BERT pooled representation, biasing them toward target-system
distributions without consuming any target labels.

The module ships **disabled by default**. ``UnsupervisedTargetAugmentor``
is the orchestrator; ``MaskedLineLoss`` is the differentiable loss head;
``select_target_lines`` is the sampler that draws a deterministic subset
of target lines per epoch.

Honouring the zero-label constraint: this path uses *only* the raw
target-system lines, never their labels. The audit in
``hylog.evaluation.leakage_audit`` continues to apply because the
augmentor does *not* mix target lines into the supervised mini-batches;
it consumes them in a separate, unlabeled pass.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from hylog.data.schema import LogSequence


@dataclass(frozen=True, slots=True)
class TargetAugmentorConfig:
    """Configuration for the unsupervised target augmentor."""

    mask_ratio: float = 0.15
    """Fraction of positions in a sequence whose pooled vectors are zeroed
    (parity with BERT's masked-LM convention)."""

    sample_fraction: float = 0.05
    """Fraction of target-system lines drawn per epoch."""

    lambda_unsup: float = 0.1
    """Weight on the unsupervised loss term added to the main task loss."""

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.lambda_unsup < 0:
            raise ValueError("lambda_unsup must be non-negative")


class MaskedLineLoss(nn.Module):
    """L2 (mean-squared) loss on masked-position reconstructions.

    We reconstruct the *projected* per-line vectors, not the raw text,
    because the projector is where the system-agnostic alignment lives.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        *,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE only on positions where ``mask == True``."""
        if predicted.shape != target.shape:
            raise ValueError(
                f"shape mismatch: predicted {tuple(predicted.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        mask = mask.to(dtype=predicted.dtype)
        denom = mask.sum().clamp(min=1.0)
        squared = ((predicted - target) ** 2) * mask
        return squared.sum() / denom


def make_mask(
    *,
    shape: tuple[int, ...],
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a boolean mask of given shape with ``ratio`` True positions."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    probs = torch.full(shape, ratio)
    return torch.bernoulli(probs, generator=generator).bool()


def select_target_lines(
    sequences: Sequence[LogSequence],
    *,
    fraction: float,
    seed: int = 42,
) -> list[LogSequence]:
    """Draw a deterministic random subset of target-system sequences.

    Labels are intentionally retained in the returned objects because the
    augmentor is structured to ignore them; this keeps the call surface
    consistent with the supervised data layer.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    n = max(1, round(len(sequences) * fraction))
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    return [sequences[i] for i in indices[:n]]


@dataclass(slots=True)
class UnsupervisedTargetAugmentor:
    """Holds the config and computes the auxiliary loss term.

    The augmentor is invoked by the trainer once per epoch. It is *not*
    a torch.nn.Module — it has no parameters of its own.
    """

    config: TargetAugmentorConfig
    loss: MaskedLineLoss = field(default_factory=MaskedLineLoss)

    def compute_step_loss(
        self,
        *,
        projected_vectors: torch.Tensor,
        reconstructed: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mask ``projected_vectors`` and compute the reconstruction loss.

        Both tensors must have shape ``[n_lines, hidden]``. The mask is
        broadcast across the hidden dimension.
        """
        if projected_vectors.dim() != 2:
            raise ValueError("expected [n_lines, hidden]")
        n_lines = projected_vectors.shape[0]
        # Mask per line (a whole line is masked or not) so reconstruction
        # is at the line-vector granularity, not the feature granularity.
        line_mask = make_mask(shape=(n_lines,), ratio=self.config.mask_ratio, generator=generator)
        if not line_mask.any():
            return torch.zeros((), dtype=projected_vectors.dtype, device=projected_vectors.device)
        expanded = line_mask.unsqueeze(1).expand_as(projected_vectors)
        return self.loss(
            predicted=reconstructed,
            target=projected_vectors,
            mask=expanded,
        )


__all__ = [
    "MaskedLineLoss",
    "TargetAugmentorConfig",
    "UnsupervisedTargetAugmentor",
    "make_mask",
    "select_target_lines",
]
