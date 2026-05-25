"""Selective prediction — abstain when uncertain.

A selective classifier returns one of three outcomes per input:

  * ``"normal"``   — predicted class 0 with confidence ≥ τ
  * ``"anomaly"``  — predicted class 1 with confidence ≥ τ
  * ``"abstain"``  — confidence below τ; caller routes to a human or
                      a more expensive verification path

The hyperparameter τ ∈ [0.5, 1.0] is selected to satisfy a *risk
budget*: the maximum tolerated error rate on the accepted (non-abstained)
predictions. The selector is fitted on a held-out *calibration* slice
and then frozen for deployment.

This module is calibration-method-agnostic; it consumes calibrated
probabilities from any of:

  * ``hylog.calibration.temperature.TemperatureCalibrator``
  * ``hylog.calibration.platt.PlattCalibrator``
  * ``hylog.calibration.vector_scaling.VectorScalingCalibrator``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Decision = Literal["normal", "anomaly", "abstain"]
LABEL_NAMES: tuple[str, str] = ("normal", "anomaly")


@dataclass(frozen=True, slots=True)
class SelectivePrediction:
    """One sample's selective output."""

    decision: Decision
    p_anomaly: float
    confidence: float
    threshold: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "decision": self.decision,
            "p_anomaly": float(self.p_anomaly),
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class TauSelectionResult:
    """Outcome of the auto-tau search."""

    threshold: float
    achieved_risk: float
    achieved_coverage: float
    risk_budget: float
    n_calibration: int
    feasible: bool
    """``False`` when no τ in [0.5, 1.0] satisfies the budget. In that
    case ``threshold = 1.0`` (= abstain on everything not perfectly
    confident) is returned."""

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "threshold": float(self.threshold),
            "achieved_risk": float(self.achieved_risk),
            "achieved_coverage": float(self.achieved_coverage),
            "risk_budget": float(self.risk_budget),
            "n_calibration": int(self.n_calibration),
            "feasible": bool(self.feasible),
        }


def select_one(
    *,
    probabilities: np.ndarray,
    threshold: float,
) -> list[SelectivePrediction]:
    """Apply a fixed threshold to a batch of calibrated probabilities.

    Args:
        probabilities: ``[N, 2]`` calibrated softmax outputs.
        threshold: ``τ ∈ [0.5, 1.0]``. A prediction is accepted iff its
            max-class probability ≥ τ.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"probabilities must be [N, 2], got {p.shape}")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0.5, 1.0], got {threshold}")

    p_anom = p[:, 1]
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)

    out: list[SelectivePrediction] = []
    for pa, conf, cls in zip(p_anom, confidence, predicted, strict=True):
        if conf < threshold:
            decision: Decision = "abstain"
        else:
            decision = LABEL_NAMES[int(cls)]  # type: ignore[assignment]
        out.append(
            SelectivePrediction(
                decision=decision,
                p_anomaly=float(pa),
                confidence=float(conf),
                threshold=float(threshold),
            )
        )
    return out


def select_tau_for_risk_budget(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    risk_budget: float = 0.05,
    candidates: np.ndarray | None = None,
) -> TauSelectionResult:
    """Search for the smallest τ achieving ``risk(selected) ≤ risk_budget``.

    "Smallest τ" maximises coverage subject to the budget. The candidate
    sweep is the set of empirical confidences in the calibration data,
    plus 0.5 and 1.0 as boundary cases.

    Args:
        probabilities: ``[N, 2]`` calibrated softmax outputs.
        labels: ``[N]`` ground-truth labels in {0, 1}.
        risk_budget: Maximum tolerated error rate on accepted predictions.
        candidates: Optional fixed sweep of τ values. When ``None``, a
            data-driven sweep over the empirical confidences is used.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"probabilities must be [N, 2], got {p.shape}")
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: probs {p.shape} vs labels {y.shape}")
    if not 0.0 < risk_budget < 1.0:
        raise ValueError("risk_budget must be in (0, 1)")

    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    n = int(y.size)

    if candidates is None:
        # Sort unique confidences ascending so we test more permissive
        # thresholds first.
        cand = np.unique(confidences)
        cand = cand[(cand >= 0.5) & (cand <= 1.0)]
        if cand.size == 0:
            cand = np.array([1.0])
        # Ensure boundary candidates are present.
        cand = np.concatenate([[0.5], cand, [1.0]])
        cand = np.unique(np.clip(cand, 0.5, 1.0))
    else:
        cand = np.asarray(candidates, dtype=np.float64)
        cand = np.clip(cand, 0.5, 1.0)

    best: TauSelectionResult | None = None
    for tau in cand:
        accepted = confidences >= tau
        n_accepted = int(accepted.sum())
        if n_accepted == 0:
            risk = 0.0
            coverage = 0.0
            # Trivial-zero-coverage solution is not considered feasible —
            # we want at least one accepted prediction.
            feasible = False
        else:
            risk = 1.0 - float(correct[accepted].mean())
            coverage = n_accepted / n
            feasible = risk <= risk_budget
        result = TauSelectionResult(
            threshold=float(tau),
            achieved_risk=float(risk),
            achieved_coverage=float(coverage),
            risk_budget=float(risk_budget),
            n_calibration=n,
            feasible=feasible,
        )
        # We want the smallest τ that is feasible (maximises coverage).
        if feasible and (best is None or coverage > best.achieved_coverage):
            best = result

    if best is not None:
        return best

    # Nothing feasible — return the infeasible high-confidence fallback.
    return TauSelectionResult(
        threshold=1.0,
        achieved_risk=float("nan"),
        achieved_coverage=0.0,
        risk_budget=float(risk_budget),
        n_calibration=n,
        feasible=False,
    )


__all__ = [
    "LABEL_NAMES",
    "Decision",
    "SelectivePrediction",
    "TauSelectionResult",
    "select_one",
    "select_tau_for_risk_budget",
]
