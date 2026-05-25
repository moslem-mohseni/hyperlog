"""Confusion-matrix artefacts: CSV, PNG, and a network-free text-art renderer.

Phase 4 checklist: "Per-fold confusion matrices archived as PNG + CSV."

The renderer is dependency-soft. The CSV and text-art paths use only the
standard library plus numpy; the PNG path tries matplotlib and falls back
to a text-art ``ascii.txt`` if matplotlib is unavailable so a deployment
on a headless CI runner still produces *some* artefact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from hylog.evaluation.metrics import ConfusionCounts

CLASS_LABELS: Final[tuple[str, str]] = ("normal", "anomaly")


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Two-class confusion matrix in canonical (true x predicted) layout.

    Rows are true labels (normal, anomaly), columns are predicted.
    """

    matrix: tuple[tuple[int, int], tuple[int, int]]

    @classmethod
    def from_counts(cls, c: ConfusionCounts) -> ConfusionMatrix:
        # Standard convention: matrix[true][pred].
        return cls(
            (
                (c.tn, c.fp),
                (c.fn, c.tp),
            )
        )

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
        from hylog.evaluation.metrics import confusion_counts

        return cls.from_counts(confusion_counts(y_true, y_pred))

    def as_numpy(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.int64)

    def row_normalize(self) -> np.ndarray:
        arr = self.as_numpy().astype(np.float64)
        row_sums = arr.sum(axis=1, keepdims=True)
        return np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)


def write_csv(matrix: ConfusionMatrix, path: Path | str) -> Path:
    """Write a labelled CSV. Header row + index column."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true \\ pred", *CLASS_LABELS])
        for i, row in enumerate(matrix.matrix):
            writer.writerow([CLASS_LABELS[i], *row])
    return p


def render_text(matrix: ConfusionMatrix, *, title: str = "Confusion matrix") -> str:
    """Network-free text-art rendering. Always available."""
    arr = matrix.as_numpy()
    width = max(6, *(len(str(v)) for v in arr.flat))
    cell = "{:>" + str(width) + "}"

    lines = [title, ""]
    header = " " * 10 + " | " + " | ".join(cell.format(c) for c in CLASS_LABELS)
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for i, row in enumerate(arr):
        line = f"{CLASS_LABELS[i]:>10} | " + " | ".join(cell.format(v) for v in row)
        lines.append(line)
    lines.append("")

    # Add per-row %, useful for imbalanced datasets where raw counts mislead.
    norm = matrix.row_normalize()
    lines.append("Row-normalised:")
    for i, row in enumerate(norm):
        pct = " | ".join(f"{v * 100:>6.2f}%" for v in row)
        lines.append(f"{CLASS_LABELS[i]:>10} | {pct}")
    return "\n".join(lines) + "\n"


def write_text(
    matrix: ConfusionMatrix, path: Path | str, *, title: str = "Confusion matrix"
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_text(matrix, title=title), encoding="utf-8", newline="\n")
    return p


def try_render_png(
    matrix: ConfusionMatrix,
    path: Path | str,
    *,
    title: str = "Confusion matrix",
) -> Path | None:
    """Best-effort PNG. Returns the path on success, ``None`` if matplotlib
    is not available (in which case the caller should fall back to text).

    The PNG includes both raw counts and row-normalised percentages so a
    single figure tells the full story even at an extreme class imbalance.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = matrix.as_numpy()
    norm = matrix.row_normalize()

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            text_colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{raw[i, j]}\n({norm[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def archive_all(
    matrix: ConfusionMatrix,
    *,
    out_dir: Path | str,
    name: str,
    title: str | None = None,
) -> dict[str, Path]:
    """Emit CSV + PNG (if available) + text-art into ``out_dir/{name}.*``.

    Returns a dict mapping artefact kind to its written path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(matrix, out / f"{name}.csv")
    text_path = write_text(matrix, out / f"{name}.txt", title=title or name)
    png_path = try_render_png(matrix, out / f"{name}.png", title=title or name)
    artefacts = {"csv": csv_path, "text": text_path}
    if png_path is not None:
        artefacts["png"] = png_path
    return artefacts


__all__ = [
    "CLASS_LABELS",
    "ConfusionMatrix",
    "archive_all",
    "render_text",
    "try_render_png",
    "write_csv",
    "write_text",
]
