"""Gradient Reversal Layer + domain classifier — Phase 4 kill-switch.

Roadmap §Phase 4 kill-switch option (b): if cross-system macro-F1 collapses,
augment the loss with a domain-adversarial term that pushes the encoder
to produce system-agnostic representations.

Implementation:

- A ``GradientReverse`` autograd Function whose forward is identity and
  whose backward multiplies the gradient by ``-lambda_``. This is the
  canonical DANN trick (Ganin & Lempitsky, ICML 2015).
- A small ``DomainClassifier`` MLP that consumes the per-sequence pooled
  representation (the same tensor that feeds the binary head) and
  predicts which source system the sequence came from. Training the
  classifier through the reversal layer pushes the upstream
  representations to confuse it — i.e. become system-agnostic.

The module ships **disabled by default**. The combined loss adapter
takes ``lambda_domain=0.0`` as its default, which makes the contribution
vanish and is bit-for-bit equivalent to the Phase 3 baseline.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.autograd import Function


class GradientReverse(Function):
    """Identity on forward, sign-flipped scaled gradient on backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, lambda_)  # type: ignore[no-any-return]


class DomainClassifier(nn.Module):
    """Two-layer MLP predicting the source domain.

    Output is ``[batch, n_domains]`` logits suitable for cross-entropy.
    """

    def __init__(self, in_features: int, n_domains: int, hidden: int | None = None) -> None:
        super().__init__()
        if n_domains < 2:
            raise ValueError("DomainClassifier requires at least 2 domains")
        h = hidden if hidden is not None else max(in_features // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_features, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, n_domains),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(x, lambda_))


def combined_loss(
    *,
    task_loss: torch.Tensor,
    domain_logits: torch.Tensor | None,
    domain_targets: torch.Tensor | None,
    lambda_domain: float = 0.0,
) -> torch.Tensor:
    """Composite loss = task_loss + lambda_domain * domain_loss.

    ``lambda_domain=0`` disables the adversarial term entirely and is the
    default during Phase 3 / Phase 4 core experiments. Phase-4 kill-switch
    paths set it to a positive scalar.
    """
    if (domain_logits is None) ^ (domain_targets is None):
        raise ValueError("domain_logits and domain_targets must be both None or both provided")
    if domain_logits is None or domain_targets is None or lambda_domain == 0.0:
        return task_loss
    dom = nn.functional.cross_entropy(domain_logits, domain_targets)
    return task_loss + lambda_domain * dom


__all__ = ["DomainClassifier", "GradientReverse", "combined_loss", "grad_reverse"]
