"""Expected Calibration Error (ECE), MCE, and reliability bins.

Reference: Guo et al. 2017 (https://arxiv.org/abs/1706.04599).

Definition (equal-width binning):

  bin_m  = predictions whose top-class confidence ∈ [m/M, (m+1)/M)
  ECE    = sum_m (|bin_m| / N) * |acc(bin_m) - conf(bin_m)|
  MCE    = max_m |acc(bin_m) - conf(bin_m)|

where acc(bin_m) is the empirical accuracy of predictions in that bin
and conf(bin_m) is the mean predicted confidence.

The Phase-5 target is ECE ≤ 0.05 (matches Guo et al.'s well-calibrated
threshold on CIFAR/SVHN post-temperature-scaling).

This module is *purely* metric computation; the visual reliability
diagram lives in ``hylog.calibration.reliability``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin's worth of reliability statistics."""

    lower: float
    upper: float
    count: int
    confidence_mean: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "count": int(self.count),
            "confidence_mean": float(self.confidence_mean),
            "accuracy": float(self.accuracy),
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Top-level container for ECE / MCE / per-bin breakdown."""

    ece: float
    mce: float
    n_samples: int
    n_bins: int
    bins: tuple[ReliabilityBin, ...]

    def is_well_calibrated(self, threshold: float = 0.05) -> bool:
        """Phase-5 target: ECE ≤ 0.05 by default."""
        return self.ece <= threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "ece": float(self.ece),
            "mce": float(self.mce),
            "n_samples": int(self.n_samples),
            "n_bins": int(self.n_bins),
            "bins": [b.to_dict() for b in self.bins],
        }


def compute_reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationReport:
    """Compute equal-width reliability bins + ECE + MCE.

    Args:
        probabilities: ``[N, K]`` softmax outputs. Each row sums to 1.
        labels: ``[N]`` integer ground-truth labels in ``[0, K)``.
        n_bins: Number of bins (Guo et al. used 15 on CIFAR/SVHN).

    Returns:
        A ``CalibrationReport`` capturing ECE, MCE, and per-bin
        breakdown.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got {probs.shape}")
    if probs.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: probs {probs.shape} vs labels {y.shape}")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    n = int(y.size)

    # Equal-width bins on [0, 1].
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece_acc = 0.0
    mce = 0.0
    for m in range(n_bins):
        lower, upper = float(edges[m]), float(edges[m + 1])
        if m == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    count=0,
                    confidence_mean=float("nan"),
                    accuracy=float("nan"),
                )
            )
            continue
        conf_mean = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        gap = abs(acc - conf_mean)
        ece_acc += (count / n) * gap
        mce = max(mce, gap)
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                confidence_mean=conf_mean,
                accuracy=acc,
            )
        )

    return CalibrationReport(
        ece=ece_acc,
        mce=mce,
        n_samples=n,
        n_bins=n_bins,
        bins=tuple(bins),
    )


def ece_only(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 15) -> float:
    """Shortcut returning just the ECE scalar."""
    return compute_reliability_bins(probabilities, labels, n_bins=n_bins).ece


__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "compute_reliability_bins",
    "ece_only",
]
