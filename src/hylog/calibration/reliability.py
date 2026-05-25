"""Reliability diagram artefacts: CSV + PNG.

Phase-5 checklist: "Reliability diagrams archived as PNG + bin CSV per
(backbone, dataset, fold) tuple."

The CSV contains raw bin statistics that a reviewer can re-plot. The
PNG is a best-effort matplotlib rendering with both the bar chart and
the perfect-calibration diagonal overlaid.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hylog.calibration.ece import CalibrationReport


def write_csv(report: CalibrationReport, path: Path | str) -> Path:
    """Write per-bin statistics to a labelled CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lower", "upper", "count", "confidence_mean", "accuracy"])
        for b in report.bins:
            writer.writerow(
                [
                    f"{b.lower:.6f}",
                    f"{b.upper:.6f}",
                    int(b.count),
                    "" if np.isnan(b.confidence_mean) else f"{b.confidence_mean:.6f}",
                    "" if np.isnan(b.accuracy) else f"{b.accuracy:.6f}",
                ]
            )
    return p


def try_render_png(
    report: CalibrationReport,
    path: Path | str,
    *,
    title: str = "Reliability diagram",
) -> Path | None:
    """Best-effort matplotlib rendering. Returns None if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    bin_centres: list[float] = []
    bin_widths: list[float] = []
    accuracies: list[float] = []
    gaps_positive: list[float] = []
    gaps_negative: list[float] = []
    for b in report.bins:
        centre = (b.lower + b.upper) / 2.0
        bin_centres.append(centre)
        bin_widths.append(b.upper - b.lower)
        if b.count == 0 or np.isnan(b.accuracy):
            accuracies.append(0.0)
            gaps_positive.append(0.0)
            gaps_negative.append(0.0)
            continue
        accuracies.append(b.accuracy)
        # Gap is conf - acc (positive => over-confident).
        gap = b.confidence_mean - b.accuracy
        gaps_positive.append(max(gap, 0.0))
        gaps_negative.append(max(-gap, 0.0))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    width = 1.0 / report.n_bins
    ax.bar(
        bin_centres,
        accuracies,
        width=width * 0.95,
        edgecolor="black",
        color="C0",
        alpha=0.7,
        label="accuracy",
    )
    # Plot over-confident gaps (red) above accuracy, under-confident below.
    ax.bar(
        bin_centres,
        gaps_positive,
        bottom=accuracies,
        width=width * 0.95,
        color="C3",
        alpha=0.4,
        label="over-confident gap",
    )
    ax.bar(
        bin_centres,
        [-g for g in gaps_negative],
        bottom=accuracies,
        width=width * 0.95,
        color="C2",
        alpha=0.4,
        label="under-confident gap",
    )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", label="perfect calibration"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {report.ece:.4f} | MCE = {report.mce:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def archive_all(
    report: CalibrationReport,
    *,
    out_dir: Path | str,
    name: str = "reliability",
    title: str | None = None,
) -> dict[str, Path]:
    """Emit CSV + PNG (+ best-effort fallback) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artefacts: dict[str, Path] = {"csv": write_csv(report, out / f"{name}.csv")}
    png = try_render_png(report, out / f"{name}.png", title=title or name)
    if png is not None:
        artefacts["png"] = png
    return artefacts


__all__ = ["archive_all", "try_render_png", "write_csv"]
