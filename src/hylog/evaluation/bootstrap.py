"""Bootstrap confidence intervals for the metric panel.

A Q1 reviewer reading "F1 = 0.87" needs to know the precision of that
number. We compute 95 % stratified-bootstrap confidence intervals for
every metric in the panel and archive them alongside the point estimate
in the per-fold ``metrics.json``.

Method:

- Stratified resampling: positives and negatives are resampled
  independently so each bootstrap replicate preserves the empirical
  class prevalence. This matters for AUC-PR and FPR@R=0.95 on highly
  imbalanced LAD data, where naive resampling can produce replicates
  with zero positives.
- Deterministic seeding: every call accepts a ``seed`` so the CI is
  exactly reproducible.
- Percentile interval: the (2.5 %, 97.5 %) quantiles of the bootstrap
  distribution. This is the BCa-free interval — appropriate for
  metrics that may be skewed but for which we have no analytical
  expression. The seed manifest is recorded so a reviewer can recompute.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from hylog.evaluation.metrics import MetricPanel, compute_metric_panel


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A 95 % percentile interval around a metric."""

    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }


def _stratified_resample(
    rng: np.random.Generator,
    indices_pos: np.ndarray,
    indices_neg: np.ndarray,
) -> np.ndarray:
    pos = rng.choice(indices_pos, size=indices_pos.size, replace=True)
    neg = rng.choice(indices_neg, size=indices_neg.size, replace=True)
    return np.concatenate([pos, neg])


def bootstrap_metric_panel(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    metric_extractor: Callable[[MetricPanel], dict[str, float]] | None = None,
    alpha: float = 0.05,
) -> dict[str, BootstrapInterval]:
    """Compute percentile bootstrap CIs for every metric in the panel.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Probability scores for the anomaly class. Required for
            AUC-ROC / AUC-PR / FPR@R=0.95 CIs; threshold metrics are
            computed regardless.
        n_bootstrap: Number of bootstrap replicates. Defaults to 1000.
        seed: RNG seed for reproducibility.
        metric_extractor: Custom extractor; defaults to ``panel.to_dict()``.
        alpha: Significance level. ``alpha=0.05`` yields a 95% CI.

    Returns:
        Dict ``metric_name -> BootstrapInterval``.
    """
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be >= 100, got {n_bootstrap}")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_score = None if y_score is None else np.asarray(y_score, dtype=np.float64)

    extractor: Callable[[MetricPanel], dict[str, float]] = (
        metric_extractor if metric_extractor is not None else lambda p: p.to_dict()
    )

    point_panel = compute_metric_panel(y_true, y_pred, y_score)
    point_metrics = extractor(point_panel)

    indices_pos = np.where(y_true == 1)[0]
    indices_neg = np.where(y_true == 0)[0]
    if indices_pos.size == 0 or indices_neg.size == 0:
        # Degenerate panel (single-class test set). Return point estimates
        # with collapsed CIs — caller is responsible for interpretation.
        return {
            name: BootstrapInterval(
                metric=name,
                point_estimate=float(value),
                ci_low=float(value),
                ci_high=float(value),
                n_bootstrap=0,
                seed=seed,
            )
            for name, value in point_metrics.items()
        }

    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {name: [] for name in point_metrics}
    for _ in range(n_bootstrap):
        idx = _stratified_resample(rng, indices_pos, indices_neg)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]
        sample_score = None if y_score is None else y_score[idx]
        panel = compute_metric_panel(sample_true, sample_pred, sample_score)
        for name, value in extractor(panel).items():
            if value != value:  # skip NaN replicates (e.g. degenerate ROC)
                continue
            replicates[name].append(float(value))

    half_alpha = alpha / 2.0
    out: dict[str, BootstrapInterval] = {}
    for name, value in point_metrics.items():
        values = replicates[name]
        if not values:
            ci_low = ci_high = float(value)
        else:
            arr = np.asarray(values, dtype=np.float64)
            ci_low = float(np.quantile(arr, half_alpha))
            ci_high = float(np.quantile(arr, 1.0 - half_alpha))
        out[name] = BootstrapInterval(
            metric=name,
            point_estimate=float(value),
            ci_low=ci_low,
            ci_high=ci_high,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return out


def format_ci(interval: BootstrapInterval, *, percent: bool = True) -> str:
    """Render an interval as ``87.45 [83.20, 91.10]``."""
    if percent:
        return (
            f"{interval.point_estimate * 100:.2f} "
            f"[{interval.ci_low * 100:.2f}, {interval.ci_high * 100:.2f}]"
        )
    return f"{interval.point_estimate:.4f} [{interval.ci_low:.4f}, {interval.ci_high:.4f}]"


def aggregate_macro(
    intervals_per_fold: Sequence[dict[str, BootstrapInterval]],
) -> dict[str, dict[str, float]]:
    """Macro-average bootstrap intervals across folds.

    For each metric, computes mean(point), mean(ci_low), mean(ci_high)
    across folds. This is *not* the same as a pooled bootstrap (which
    would require concatenating raw predictions); it is a clear,
    interpretable macro view that a reviewer can verify by inspection.
    """
    out: dict[str, dict[str, float]] = {}
    metric_names = sorted({m for d in intervals_per_fold for m in d})
    for name in metric_names:
        points: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for d in intervals_per_fold:
            if name not in d:
                continue
            points.append(d[name].point_estimate)
            lows.append(d[name].ci_low)
            highs.append(d[name].ci_high)
        if not points:
            continue
        out[name] = {
            "point_mean": float(np.mean(points)),
            "ci_low_mean": float(np.mean(lows)),
            "ci_high_mean": float(np.mean(highs)),
            "n_folds": len(points),
        }
    return out


__all__ = [
    "BootstrapInterval",
    "aggregate_macro",
    "bootstrap_metric_panel",
    "format_ci",
]
