"""AURC, Excess-AURC, and cost-asymmetric selective error.

References:

- Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for
  deep neural networks.* NeurIPS.
- Traub et al. (ICML 2025). *A novel characterization of the population
  AURC.*

Definitions (used throughout the Phase-5 / Phase-6 evaluation):

  Given predictions sorted in DESCENDING order of confidence, for each
  coverage level c ∈ [0, 1] (the top ``c x N`` predictions are kept,
  the rest abstained), define:

    risk(c) = error rate over the kept predictions
    AURC    = (1/N) * sum_i risk(i / N)  evaluated at sample boundaries

  An *oracle* selector that ranks predictions perfectly by correctness
  (correct first, then errors) achieves the lowest possible AURC,
  ``optimal_AURC``. The HyLog-relevant quantity is:

    Excess-AURC (E-AURC) = AURC - optimal_AURC

  E-AURC isolates the *ranking quality* of the confidence score from
  the absolute error rate, enabling fair cross-model comparison.

  The cost-asymmetric variant generalises ``risk`` so that
  false negatives (missed anomalies) cost ``fn_weight``x more than
  false positives. The LAD operational default is ``fn_weight = 5``.

The Phase-5 checklist requires AURC + E-AURC + cost-asymmetric for every
fold; all three are computed in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AURCReport:
    """AURC summary at point estimate + cost-asymmetric variant."""

    aurc: float
    optimal_aurc: float
    excess_aurc: float
    cost_asymmetric_aurc: float
    fn_weight: float
    fp_weight: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "aurc": float(self.aurc),
            "optimal_aurc": float(self.optimal_aurc),
            "excess_aurc": float(self.excess_aurc),
            "cost_asymmetric_aurc": float(self.cost_asymmetric_aurc),
            "fn_weight": float(self.fn_weight),
            "fp_weight": float(self.fp_weight),
            "n_samples": int(self.n_samples),
        }


def _cumulative_risk(loss_per_sample_descending: np.ndarray) -> np.ndarray:
    """Cumulative mean of per-sample losses sorted by confidence DESC.

    For element i (zero-indexed), returns the mean loss of the top
    (i+1) predictions. Length N.
    """
    cumsum = np.cumsum(loss_per_sample_descending, dtype=np.float64)
    counts = np.arange(1, loss_per_sample_descending.size + 1, dtype=np.float64)
    return cumsum / counts


def _aurc_from_risk(risk_curve: np.ndarray) -> float:
    """AURC = mean of risk(c) over c = 1/N, 2/N, ..., 1."""
    if risk_curve.size == 0:
        return float("nan")
    return float(risk_curve.mean())


def compute_aurc(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> AURCReport:
    """Compute AURC, optimal AURC, E-AURC, and cost-asymmetric AURC.

    Args:
        y_true: ``[N]`` integer ground-truth labels in {0, 1}.
        y_pred: ``[N]`` integer predicted labels in {0, 1}.
        confidence: ``[N]`` per-sample confidence scores. Higher is more
            confident. For HyLog this is ``max(p_normal, p_anomaly)``.
        fn_weight: Cost weight for false negatives (missed anomalies).
            Operational default for LAD: 5.0.
        fp_weight: Cost weight for false positives (false alarms).
            Default 1.0.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (y_true.shape == y_pred.shape == confidence.shape):
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"confidence {confidence.shape}"
        )
    if fn_weight <= 0 or fp_weight <= 0:
        raise ValueError("weights must be positive")
    n = int(y_true.size)
    if n == 0:
        return AURCReport(
            aurc=float("nan"),
            optimal_aurc=float("nan"),
            excess_aurc=float("nan"),
            cost_asymmetric_aurc=float("nan"),
            fn_weight=fn_weight,
            fp_weight=fp_weight,
            n_samples=0,
        )

    # Per-sample 0/1 loss for the symmetric AURC.
    losses = (y_true != y_pred).astype(np.float64)

    # Per-sample cost-asymmetric loss.
    cost_losses = np.where(
        (y_true == 1) & (y_pred == 0),
        fn_weight,  # FN
        np.where(
            (y_true == 0) & (y_pred == 1),
            fp_weight,  # FP
            0.0,  # correct
        ),
    )

    # Sort indices by DESCENDING confidence with stable tie-breaking on
    # original index so two runs over the same inputs yield identical AURC.
    order = np.lexsort((np.arange(n), -confidence))

    losses_desc = losses[order]
    cost_losses_desc = cost_losses[order]

    aurc = _aurc_from_risk(_cumulative_risk(losses_desc))
    cost_aurc = _aurc_from_risk(_cumulative_risk(cost_losses_desc))

    # Optimal ranking: correct predictions first, then errors.
    optimal_order = np.argsort(losses)  # 0s first, then 1s
    optimal_aurc = _aurc_from_risk(_cumulative_risk(losses[optimal_order]))

    return AURCReport(
        aurc=aurc,
        optimal_aurc=optimal_aurc,
        excess_aurc=aurc - optimal_aurc,
        cost_asymmetric_aurc=cost_aurc,
        fn_weight=fn_weight,
        fp_weight=fp_weight,
        n_samples=n,
    )


def risk_coverage_curve(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coverage, risk)`` arrays for plotting / archiving.

    ``coverage[i] = (i + 1) / N`` and ``risk[i]`` is the error rate over
    the top-(i + 1) most confident predictions.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    n = y_true.size
    if n == 0:
        return np.array([]), np.array([])
    losses = (y_true != y_pred).astype(np.float64)
    order = np.lexsort((np.arange(n), -confidence))
    risk = _cumulative_risk(losses[order])
    coverage = (np.arange(1, n + 1, dtype=np.float64)) / n
    return coverage, risk


__all__ = [
    "AURCReport",
    "compute_aurc",
    "risk_coverage_curve",
]
