"""Classification metrics used across HyLog.

The metric panel matches docs/ROADMAP.md §4bis:
precision / recall / F1 / AUC-ROC / AUC-PR / MCC / FPR@R=0.95.
Calibration metrics (ECE, MCE, AURC) live in ``hylog.calibration``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def precision(c: ConfusionCounts) -> float:
    denom = c.tp + c.fp
    return c.tp / denom if denom else 0.0


def recall(c: ConfusionCounts) -> float:
    denom = c.tp + c.fn
    return c.tp / denom if denom else 0.0


def f1(c: ConfusionCounts) -> float:
    p, r = precision(c), recall(c)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def false_positive_rate(c: ConfusionCounts) -> float:
    denom = c.fp + c.tn
    return c.fp / denom if denom else 0.0


def matthews_correlation_coefficient(c: ConfusionCounts) -> float:
    num = (c.tp * c.tn) - (c.fp * c.fn)
    denom_sq = (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    if denom_sq == 0:
        return 0.0
    return num / float(denom_sq) ** 0.5


def fpr_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.95) -> float:
    """Operating-point metric: FPR at a fixed recall target."""
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    if not pos.any() or not neg.any():
        return float("nan")
    # Sort scores descending; sweep thresholds and pick the smallest one that
    # achieves >= target_recall.
    order = np.argsort(-y_score)
    sorted_pos = pos[order]
    cum_tp = np.cumsum(sorted_pos)
    total_pos = int(pos.sum())
    recalls = cum_tp / total_pos
    sufficient = np.where(recalls >= target_recall)[0]
    if sufficient.size == 0:
        return 1.0
    idx = int(sufficient[0])
    # At this index, threshold is y_score[order[idx]]; everything ranked >=
    # is predicted positive.
    cum_fp = np.cumsum(~sorted_pos)
    total_neg = int(neg.sum())
    return float(cum_fp[idx] / total_neg) if total_neg else float("nan")


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve, computed by rank statistics (no sklearn)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U statistic / (n_pos * n_neg) is AUC-ROC.
    ranks = _rank_values(np.concatenate([pos, neg]))
    pos_rank_sum = float(ranks[: pos.size].sum())
    n_p, n_n = pos.size, neg.size
    u = pos_rank_sum - n_p * (n_p + 1) / 2
    return float(u / (n_p * n_n))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average-precision form)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return float("nan")
    recalls = cum_tp / total_pos
    # Standard AP = sum_n (R_n - R_{n-1}) * P_n
    deltas = np.diff(np.concatenate([[0.0], recalls]))
    return float((deltas * precisions).sum())


def _rank_values(values: np.ndarray) -> np.ndarray:
    """Average-rank for ties; 1-indexed."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@dataclass(frozen=True, slots=True)
class MetricPanel:
    precision: float
    recall: float
    f1: float
    fpr: float
    fpr_at_recall_95: float
    mcc: float
    auc_roc: float
    auc_pr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fpr_at_recall_95": self.fpr_at_recall_95,
            "mcc": self.mcc,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
        }


def compute_metric_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> MetricPanel:
    """Compute the full panel. If ``y_score`` is None, ranking metrics return NaN."""
    c = confusion_counts(y_true, y_pred)
    auroc = auc_roc(y_true, y_score) if y_score is not None else float("nan")
    aupr = auc_pr(y_true, y_score) if y_score is not None else float("nan")
    fpr95 = fpr_at_recall(y_true, y_score, 0.95) if y_score is not None else float("nan")
    return MetricPanel(
        precision=precision(c),
        recall=recall(c),
        f1=f1(c),
        fpr=false_positive_rate(c),
        fpr_at_recall_95=fpr95,
        mcc=matthews_correlation_coefficient(c),
        auc_roc=auroc,
        auc_pr=aupr,
    )


__all__ = [
    "ConfusionCounts",
    "MetricPanel",
    "auc_pr",
    "auc_roc",
    "compute_metric_panel",
    "confusion_counts",
    "f1",
    "false_positive_rate",
    "fpr_at_recall",
    "matthews_correlation_coefficient",
    "precision",
    "recall",
]
