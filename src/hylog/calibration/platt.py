"""Platt scaling — binary calibration fallback for HyLog.

Reference: Platt, J. (1999). *Probabilistic outputs for support vector
machines.* In *Advances in Large Margin Classifiers*.

For binary classification, Platt scaling fits ``sigmoid(a * f(x) + b)``
to validation scores, where ``f(x)`` is a raw model score (typically a
logit or a margin). Two parameters are learned (``a``, ``b``) via
maximum likelihood.

HyLog uses Platt scaling as the Phase-5 kill-switch: if temperature
scaling fails to bring ECE below the 0.05 target on some folds, the
two-parameter family of Platt gives extra flexibility to fit
asymmetric miscalibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """Fitted sigmoid(a * z + b) parameters."""

    a: float
    b: float
    n_calibration: int
    init_nll: float
    final_nll: float

    def apply_logit(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration to a 1-D array of raw scores.

        ``scores`` is typically ``logits[:, 1] - logits[:, 0]`` for a
        binary classifier, or any monotone function thereof.
        Returns calibrated probability of class 1, shape ``[N]``.
        """
        z = np.asarray(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply to a ``[N, 2]`` logit array, returning ``[N, 2]`` probs."""
        z = np.asarray(logits, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != 2:
            raise ValueError(f"apply() expects [N, 2] logits, got {z.shape}")
        margin = z[:, 1] - z[:, 0]
        p1 = self.apply_logit(margin)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "method": "platt_scaling",
            "a": float(self.a),
            "b": float(self.b),
            "n_calibration": int(self.n_calibration),
            "init_nll": float(self.init_nll),
            "final_nll": float(self.final_nll),
        }


def fit_platt(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    init_a: float = 1.0,
    init_b: float = 0.0,
    max_iter: int = 200,
    lr: float = 0.05,
) -> PlattCalibrator:
    """Fit Platt parameters via LBFGS over ``binary_cross_entropy_with_logits``.

    The targets follow Platt's smoothing recipe (N+1)/(N+2) for positives,
    1/(N-+2) for negatives, which prevents over-confident extremes when
    the calibration set is small.
    """
    z = torch.as_tensor(np.asarray(scores, dtype=np.float64), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float64)
    if z.dim() != 1 or z.shape != y.shape:
        raise ValueError(f"shape mismatch: scores {tuple(z.shape)} vs labels {tuple(y.shape)}")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("labels must be in {0, 1}")

    n = z.shape[0]
    n_pos = int(y.sum().item())
    n_neg = int(n - n_pos)
    # Platt smoothing: shift labels off the hard 0/1 boundary.
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    targets = torch.where(y > 0.5, torch.full_like(y, t_pos), torch.full_like(y, t_neg))

    a = torch.tensor(float(init_a), requires_grad=True, dtype=torch.float64)
    b = torch.tensor(float(init_b), requires_grad=True, dtype=torch.float64)

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
    with torch.no_grad():
        init_nll = float(bce(a * z + b, targets).item())

    optimizer = torch.optim.LBFGS([a, b], lr=lr, max_iter=max_iter)
    prev_loss = {"v": init_nll}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = bce(a * z + b, targets)
        loss.backward()
        prev_loss["v"] = float(loss.item())
        return loss

    optimizer.step(closure)

    return PlattCalibrator(
        a=float(a.detach().item()),
        b=float(b.detach().item()),
        n_calibration=int(n),
        init_nll=init_nll,
        final_nll=float(prev_loss["v"]),
    )


__all__ = ["PlattCalibrator", "fit_platt"]
