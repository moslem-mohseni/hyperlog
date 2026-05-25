"""ROC and Precision-Recall curve generation + per-fold archival.

The metric panel already reports area under each curve; this module
adds the *full curve* (every threshold's (FPR, TPR) and (Recall,
Precision) pair) so a reviewer can re-plot.

Outputs per fold:

- ``roc.csv``  — columns: threshold, fpr, tpr
- ``pr.csv``   — columns: threshold, recall, precision
- ``roc.png``  — best-effort matplotlib render
- ``pr.png``   — best-effort matplotlib render

The CSV is always written; the PNG is best-effort and silently
unavailable on environments without matplotlib.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ROCPoints:
    thresholds: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray

    def auc(self) -> float:
        # Trapezoid rule. np.trapezoid is the canonical name in numpy >= 2.0;
        # numpy 1.x uses np.trapz.
        order = np.argsort(self.fpr)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
        return float(trapezoid(self.tpr[order], self.fpr[order]))


@dataclass(frozen=True, slots=True)
class PRPoints:
    thresholds: np.ndarray
    recall: np.ndarray
    precision: np.ndarray

    def average_precision(self) -> float:
        # Standard AP = sum_n (R_n - R_{n-1}) * P_n
        order = np.argsort(self.recall)
        r = self.recall[order]
        p = self.precision[order]
        dr = np.diff(np.concatenate([[0.0], r]))
        return float((dr * p).sum())


def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> ROCPoints:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    n_pos = int(sorted_true.sum())
    n_neg = int(sorted_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return ROCPoints(thresholds=np.array([]), fpr=np.array([]), tpr=np.array([]))
    cum_tp = np.cumsum(sorted_true)
    cum_fp = np.cumsum(1 - sorted_true)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    return ROCPoints(thresholds=sorted_score, fpr=fpr, tpr=tpr)


def compute_pr(y_true: np.ndarray, y_score: np.ndarray) -> PRPoints:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    n_pos = int(sorted_true.sum())
    if n_pos == 0:
        return PRPoints(thresholds=np.array([]), recall=np.array([]), precision=np.array([]))
    cum_tp = np.cumsum(sorted_true)
    cum_fp = np.cumsum(1 - sorted_true)
    recall = cum_tp / n_pos
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    return PRPoints(thresholds=sorted_score, recall=recall, precision=precision)


def write_roc_csv(roc: ROCPoints, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold", "fpr", "tpr"])
        for t, f, tp in zip(roc.thresholds, roc.fpr, roc.tpr, strict=True):
            writer.writerow([float(t), float(f), float(tp)])
    return p


def write_pr_csv(pr: PRPoints, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold", "recall", "precision"])
        for t, r, pr_val in zip(pr.thresholds, pr.recall, pr.precision, strict=True):
            writer.writerow([float(t), float(r), float(pr_val)])
    return p


def try_render_curve_png(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path | str,
    diagonal: bool = False,
) -> Path | None:
    """Best-effort matplotlib render. Returns ``None`` if matplotlib unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot(x, y, linewidth=1.5)
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def archive_curves(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path | str,
    fold_name: str,
) -> dict[str, Path]:
    """Emit ROC + PR CSVs (+ PNGs if available) for a single fold.

    Returns a dict mapping artefact kind to its path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roc = compute_roc(y_true, y_score)
    pr = compute_pr(y_true, y_score)
    artefacts: dict[str, Path] = {
        "roc_csv": write_roc_csv(roc, out / "roc.csv"),
        "pr_csv": write_pr_csv(pr, out / "pr.csv"),
    }
    roc_png = try_render_curve_png(
        x=roc.fpr,
        y=roc.tpr,
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"ROC — held-out {fold_name} (AUC={roc.auc():.3f})",
        path=out / "roc.png",
        diagonal=True,
    )
    if roc_png is not None:
        artefacts["roc_png"] = roc_png
    pr_png = try_render_curve_png(
        x=pr.recall,
        y=pr.precision,
        xlabel="Recall",
        ylabel="Precision",
        title=f"PR — held-out {fold_name} (AP={pr.average_precision():.3f})",
        path=out / "pr.png",
        diagonal=False,
    )
    if pr_png is not None:
        artefacts["pr_png"] = pr_png
    return artefacts


__all__ = [
    "PRPoints",
    "ROCPoints",
    "archive_curves",
    "compute_pr",
    "compute_roc",
    "try_render_curve_png",
    "write_pr_csv",
    "write_roc_csv",
]
