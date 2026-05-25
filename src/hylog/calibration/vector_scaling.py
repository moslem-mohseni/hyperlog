"""Vector scaling — per-class temperatures.

Generalisation of temperature scaling that lets each class have its own
``T_k``. Reference: Guo et al. 2017 §4.2.

For ``K`` classes, we learn ``K`` parameters ``(w_k)`` such that the
calibrated logits are ``z_k * w_k``. Vector scaling is *not*
class-preserving (the argmax can change) but is strictly more
expressive than temperature scaling.

HyLog deploys vector scaling as the second-tier Phase-5 kill-switch
when both temperature scaling and Platt scaling fail to bring ECE under
the 0.05 target on a given fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class VectorScalingCalibrator:
    """Per-class temperature weights."""

    weights: tuple[float, ...]
    n_calibration: int
    init_nll: float
    final_nll: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply per-class scaling. Returns softmax probabilities."""
        z = np.asarray(logits, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        if z.shape[-1] != w.shape[0]:
            raise ValueError(f"logit dim {z.shape[-1]} != n_weights {w.shape[0]}")
        z_scaled = z * w
        z_shifted = z_scaled - z_scaled.max(axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def to_dict(self) -> dict[str, list[float] | float | int | str]:
        return {
            "method": "vector_scaling",
            "weights": list(self.weights),
            "n_calibration": int(self.n_calibration),
            "init_nll": float(self.init_nll),
            "final_nll": float(self.final_nll),
        }


def fit_vector_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    lr: float = 0.05,
) -> VectorScalingCalibrator:
    """Fit per-class weights to minimise NLL on (logits, labels)."""
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
    if z.dim() != 2:
        raise ValueError(f"logits must be 2-D, got {tuple(z.shape)}")
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: logits {tuple(z.shape)} vs labels {tuple(y.shape)}")

    n, k = z.shape
    weights = torch.ones(k, requires_grad=True, dtype=torch.float64)
    ce = nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        init_nll = float(ce(z * weights, y).item())

    optimizer = torch.optim.LBFGS([weights], lr=lr, max_iter=max_iter)
    prev_loss = {"v": init_nll}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = ce(z * weights, y)
        loss.backward()
        prev_loss["v"] = float(loss.item())
        return loss

    optimizer.step(closure)

    return VectorScalingCalibrator(
        weights=tuple(float(v) for v in weights.detach().tolist()),
        n_calibration=int(n),
        init_nll=init_nll,
        final_nll=float(prev_loss["v"]),
    )


__all__ = ["VectorScalingCalibrator", "fit_vector_scaling"]
