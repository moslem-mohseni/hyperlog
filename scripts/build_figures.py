"""Regenerate every paper figure from the committed JSON artefacts.

Outputs PDF files under ``paper/figures/`` ready for ``\\includegraphics``.

The script is *idempotent*: rerunning it overwrites figures in place and
never modifies JSON inputs. When a backing artefact is absent (e.g.
``reports/phase4/runs/<run>/summary.json`` before the GPU run lands),
the corresponding figure is replaced with a placeholder PDF labelled
``TBD: awaits GPU run`` so the LaTeX build never breaks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_or_default(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _placeholder_pdf(plt: Any, out_path: Path, message: str) -> Path:
    """Emit a small PDF whose only content is a labelled placeholder."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round", "facecolor": "#fff5cc", "edgecolor": "#aaa"},
    )
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_architecture(plt: Any, out_dir: Path) -> Path:
    """A schematic of the HyLog architecture. Pure layout; no JSON input."""
    target = out_dir / "architecture.pdf"
    if plt is None:
        target.write_bytes(b"%PDF-1.4\n%placeholder\n%%EOF\n")
        return target

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    boxes = [
        ("frozen BERT\nencoder", 0.5, 5.5, "#cde8ff"),
        ("learned\nprojector", 0.5, 4.0, "#d4f7d4"),
        ("QLoRA-tuned\ncompact decoder", 0.5, 2.5, "#ffe5cc"),
        ("classification\nhead", 0.5, 1.0, "#f7d4f0"),
    ]
    for label, x, y, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 4.0, 1.0, facecolor=color, edgecolor="#222"))
        ax.text(x + 2.0, y + 0.5, label, ha="center", va="center", fontsize=10)
    for y0, y1 in [(5.5, 5.0), (4.0, 3.5), (2.5, 2.0)]:
        ax.annotate(
            "", xy=(2.5, y1), xytext=(2.5, y0), arrowprops={"arrowstyle": "->", "color": "#333"}
        )
    ax.text(5.0, 1.5, "temperature\nscaling -> tau", fontsize=9, color="#555")
    ax.set_xlim(0, 7)
    ax.set_ylim(0.5, 7.0)
    ax.axis("off")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def build_loso_protocol(plt: Any, out_dir: Path) -> Path:
    """Flowchart for the LOSO protocol."""
    target = out_dir / "loso_protocol.pdf"
    if plt is None:
        target.write_bytes(b"%PDF-1.4\n%placeholder\n%%EOF\n")
        return target

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    steps = [
        ("Compose train = union of sources", 0.5, 6.0, "#cde8ff"),
        ("Strip labels of target", 0.5, 5.0, "#cde8ff"),
        ("SHA-256 leakage audit", 0.5, 4.0, "#ffd4d4"),
        ("Three-stage QLoRA training", 0.5, 3.0, "#d4f7d4"),
        ("Predict on target test", 0.5, 2.0, "#d4f7d4"),
        ("Metric panel + bootstrap CIs", 0.5, 1.0, "#f7d4f0"),
    ]
    for label, x, y, color in steps:
        ax.add_patch(plt.Rectangle((x, y), 6.0, 0.7, facecolor=color, edgecolor="#222"))
        ax.text(x + 3.0, y + 0.35, label, ha="center", va="center", fontsize=10)
    for y0 in (6.0, 5.0, 4.0, 3.0, 2.0):
        ax.annotate(
            "",
            xy=(3.5, y0 - 0.3),
            xytext=(3.5, y0),
            arrowprops={"arrowstyle": "->", "color": "#333"},
        )
    ax.text(7.0, 4.35, "abort fold\non non-zero\nintersection", fontsize=8, color="#a00")
    ax.set_xlim(0, 9)
    ax.set_ylim(0.8, 6.9)
    ax.axis("off")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def build_reliability(plt: Any, out_dir: Path, reports_root: Path) -> Path:
    """Reliability diagrams. Backed by phase5 CSVs when present."""
    import csv

    target = out_dir / "reliability.pdf"
    if plt is None:
        target.write_bytes(b"%PDF-1.4\n%placeholder\n%%EOF\n")
        return target

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    found_any = False
    for fold in ("hdfs", "bgl", "thunderbird"):
        cal_csv = reports_root / "phase5" / "runs" / f"{fold}" / "reliability.csv"
        if not cal_csv.exists():
            continue
        found_any = True
        with cal_csv.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        xs = [(float(r["lower"]) + float(r["upper"])) / 2 for r in rows]
        ys = [float(r["accuracy"]) if r["accuracy"] else 0.0 for r in rows]
        ax.plot(xs, ys, marker="o", label=fold)

    if not found_any:
        plt.close(fig)
        return _placeholder_pdf(plt, target, "TBD: reliability\nawaits GPU run")

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability after temperature scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def build_loso_bars(plt: Any, out_dir: Path, reports_root: Path) -> Path:
    """Macro-F1 bar chart per fold. Backed by phase4 summary.json."""
    target = out_dir / "loso_bars.pdf"
    if plt is None:
        target.write_bytes(b"%PDF-1.4\n%placeholder\n%%EOF\n")
        return target

    folds: dict[str, float] = {}
    runs_dir = reports_root / "phase4" / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            summary = _load_or_default(run_dir / "summary.json")
            if not summary:
                continue
            macro = summary.get("macro", {})
            f1 = macro.get("f1", {}) if isinstance(macro, dict) else {}
            if isinstance(f1, dict) and "mean" in f1:
                folds[run_dir.name] = float(f1["mean"])

    if not folds:
        return _placeholder_pdf(plt, target, "TBD: cross-system F1\nawaits GPU run")

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    names = list(folds)
    vals = [folds[n] for n in names]
    ax.bar(names, vals, color="#4a6fb3")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Cross-system LOSO macro-F1 per run")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def build_ablation_matrix(plt: Any, out_dir: Path, reports_root: Path) -> Path:
    """Cliff's δ heat-map across the eight ablation axes."""
    import csv

    target = out_dir / "ablation_matrix.pdf"
    if plt is None:
        target.write_bytes(b"%PDF-1.4\n%placeholder\n%%EOF\n")
        return target

    cells: list[tuple[str, str, float]] = []
    matrix_csv = reports_root / "phase6" / "runs" / "ablation_matrix.csv"
    if matrix_csv.exists():
        with matrix_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    delta = float(row["cliffs_delta"])
                except (TypeError, ValueError):
                    continue
                cells.append((row["axis"], row["variant"], delta))

    if not cells:
        return _placeholder_pdf(plt, target, "TBD: ablation matrix\nawaits GPU run")

    import numpy as np

    axes = sorted({a for a, _, _ in cells})
    variants = sorted({v for _, v, _ in cells})
    grid = np.full((len(axes), len(variants)), np.nan)
    for a, v, d in cells:
        grid[axes.index(a), variants.index(v)] = d

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_yticks(range(len(axes)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticklabels(axes)
    fig.colorbar(im, ax=ax, label="Cliff's δ")
    ax.set_title("Ablation matrix — Cliff's δ vs. baseline")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args(argv)

    plt = _try_matplotlib()
    if plt is None:
        print("[build_figures] matplotlib not available; emitting placeholder PDFs only.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.append(build_architecture(plt, args.out_dir))
    outputs.append(build_loso_protocol(plt, args.out_dir))
    outputs.append(build_reliability(plt, args.out_dir, args.reports_root))
    outputs.append(build_loso_bars(plt, args.out_dir, args.reports_root))
    outputs.append(build_ablation_matrix(plt, args.out_dir, args.reports_root))

    print("[build_figures] wrote:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
