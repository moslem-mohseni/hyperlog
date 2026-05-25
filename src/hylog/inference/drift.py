"""Concept-drift monitor for the production inference path.

Phase-8 §11.8: track the empirical distribution of calibrated anomaly
probabilities (``p_anomaly_calibrated``). At inference time the monitor
maintains a rolling window of observed values; on demand it compares
the window against a frozen reference distribution (recorded at the
end of Phase 5) via the two-sample Kolmogorov-Smirnov test. A
sufficiently large KS statistic with a small p-value flags drift to
the operator.

The monitor is intentionally cheap (O(window_size) per request) so it
runs in the hot path without dragging the SLO.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_WINDOW = 1024


@dataclass(slots=True)
class DriftMonitorConfig:
    """Configuration knobs."""

    window: int = DEFAULT_WINDOW
    """Number of recent observations retained for the test."""

    ks_threshold: float = 0.1
    """If the empirical KS statistic exceeds this and p < 0.05, drift fires."""

    p_value_threshold: float = 0.05


@dataclass(slots=True)
class DriftReport:
    """Outcome of one drift evaluation."""

    n_observed: int
    n_reference: int
    ks_statistic: float
    ks_p_value: float
    drift_threshold: float
    drift_detected: bool
    reference_summary: dict[str, float]
    observed_summary: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_observed": int(self.n_observed),
            "n_reference": int(self.n_reference),
            "ks_statistic": float(self.ks_statistic),
            "ks_p_value": float(self.ks_p_value),
            "drift_threshold": float(self.drift_threshold),
            "drift_detected": bool(self.drift_detected),
            "reference_summary": dict(self.reference_summary),
            "observed_summary": dict(self.observed_summary),
        }


def _summarise(values: np.ndarray) -> dict[str, float]:
    """5-number summary used in the drift report (cheap, no scipy)."""
    if values.size == 0:
        return {
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def two_sample_ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value.

    Returns ``(D, p_value)`` where ``D = sup_x |F_a(x) - F_b(x)|`` and
    ``p_value`` is the asymptotic two-sided value computed without scipy.
    """
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    combined = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, combined, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, combined, side="right") / b_sorted.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    if d <= 1e-12:
        # The Kolmogorov series degenerates at D=0 (formal limit p=1).
        return 0.0, 1.0
    n, m = a.size, b.size
    # Asymptotic Kolmogorov distribution. Acceptable for n + m > 20.
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    # Sum of the Kolmogorov distribution series.
    p = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j * lam) ** 2)
        p += term
        if abs(term) < 1e-8:
            break
    return d, max(0.0, min(1.0, p))


@dataclass(slots=True)
class DriftMonitor:
    """Holds the reference + a rolling observation window.

    Thread-safe; ``observe`` is the hot-path call. ``evaluate`` is the
    slower endpoint-call.
    """

    reference: np.ndarray
    config: DriftMonitorConfig = field(default_factory=DriftMonitorConfig)
    _window: deque[float] = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.reference.ndim != 1:
            raise ValueError("reference must be 1-D")
        if self.config.window <= 8:
            raise ValueError("window must be > 8")
        self._window = deque(maxlen=self.config.window)

    def observe(self, value: float) -> None:
        if math.isnan(value):
            return
        with self._lock:
            self._window.append(float(value))

    def observe_many(self, values: list[float]) -> None:
        with self._lock:
            for v in values:
                if not math.isnan(v):
                    self._window.append(float(v))

    def evaluate(self) -> DriftReport:
        with self._lock:
            observed = np.asarray(list(self._window), dtype=np.float64)
        d, p = two_sample_ks(observed, self.reference)
        drift = bool(d > self.config.ks_threshold and p < self.config.p_value_threshold)
        return DriftReport(
            n_observed=int(observed.size),
            n_reference=int(self.reference.size),
            ks_statistic=float(d),
            ks_p_value=float(p),
            drift_threshold=float(self.config.ks_threshold),
            drift_detected=drift,
            reference_summary=_summarise(self.reference),
            observed_summary=_summarise(observed),
        )


__all__ = [
    "DEFAULT_WINDOW",
    "DriftMonitor",
    "DriftMonitorConfig",
    "DriftReport",
    "two_sample_ks",
]
