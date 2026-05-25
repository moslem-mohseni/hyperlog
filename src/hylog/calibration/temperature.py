"""Temperature scaling (Guo, Pleiss, Sun, Weinberger; ICML 2017).

Reference: https://arxiv.org/abs/1706.04599

Method: given pre-softmax logits ``z`` of shape ``[N, K]`` and integer
labels ``y`` of shape ``[N]`` from a held-out calibration set, find a
scalar temperature ``T > 0`` that minimises the negative log-likelihood
of ``softmax(z / T)`` against ``y``. The optimisation is convex in T
(a single scalar) so LBFGS converges in tens of iterations from
``T = 1.0`` initialisation.

Production-side application is a one-line tensor-divide:
``calibrated_probs = softmax(logits / T_fit)``. The classifier's
predicted class never changes (temperature scaling is class-preserving);
only the *confidences* change. This is exactly the property that makes
the Phase-5 selective-prediction story mathematically clean.

The implementation depends only on ``torch`` and is fully deterministic
for a given (logits, labels, init_t) tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    """A frozen, pickle-safe representation of a fitted temperature.

    Attributes:
        temperature: The fitted ``T`` (scalar, positive).
        n_calibration: Sample size used to fit T.
        init_nll: Negative log-likelihood at T = 1.0.
        final_nll: Negative log-likelihood at the fitted T.
        n_iters: LBFGS iteration count at convergence.
    """

    temperature: float
    n_calibration: int
    init_nll: float
    final_nll: float
    n_iters: int

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted T to fresh logits. Returns softmax probabilities."""
        z = np.asarray(logits, dtype=np.float64)
        z_scaled = z / max(self.temperature, 1e-9)
        # Numerically stable softmax.
        z_shifted = z_scaled - z_scaled.max(axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "method": "temperature_scaling",
            "temperature": float(self.temperature),
            "n_calibration": int(self.n_calibration),
            "init_nll": float(self.init_nll),
            "final_nll": float(self.final_nll),
            "n_iters": int(self.n_iters),
        }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    init_t: float = 1.0,
    max_iter: int = 200,
    lr: float = 0.01,
    tol: float = 1e-7,
) -> TemperatureCalibrator:
    """Fit a single temperature scalar to minimise NLL on (logits, labels).

    Args:
        logits: ``[N, K]`` pre-softmax outputs of the classifier.
        labels: ``[N]`` integer labels in ``[0, K)``.
        init_t: Initial temperature (a good prior is ``1.0``).
        max_iter: LBFGS iteration cap.
        lr: LBFGS learning rate.
        tol: Convergence tolerance on the change in NLL.
    """
    logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)

    if logits_t.dim() != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits_t.shape)}")
    if logits_t.shape[0] != labels_t.shape[0]:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits_t.shape)} vs labels {tuple(labels_t.shape)}"
        )
    if labels_t.min() < 0 or labels_t.max() >= logits_t.shape[1]:
        raise ValueError("labels must lie in [0, n_classes)")

    # Initial NLL at T=1.
    with torch.no_grad():
        init_nll = float(nn.functional.cross_entropy(logits_t / init_t, labels_t).item())

    # Parameterise log T so T is unconstrained-positive.
    log_t = torch.tensor(float(np.log(max(init_t, 1e-9))), requires_grad=True, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, tolerance_grad=tol)

    iters = {"n": 0}
    prev_loss: dict[str, float] = {"v": init_nll}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3)
        loss = nn.functional.cross_entropy(logits_t / t, labels_t)
        loss.backward()
        iters["n"] += 1
        prev_loss["v"] = float(loss.item())
        return loss

    optimizer.step(closure)

    final_t_value = float(torch.exp(log_t).detach().item())
    final_t_value = max(final_t_value, 1e-3)

    return TemperatureCalibrator(
        temperature=final_t_value,
        n_calibration=int(logits_t.shape[0]),
        init_nll=init_nll,
        final_nll=float(prev_loss["v"]),
        n_iters=int(iters["n"]),
    )


__all__ = ["TemperatureCalibrator", "fit_temperature"]
