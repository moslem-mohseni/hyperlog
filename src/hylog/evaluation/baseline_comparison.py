"""Head-to-head comparison against published cross-system baselines.

The canonical numbers live in ``reports/phase4/published_numbers.yaml``;
this module loads them and renders the comparison table that goes into
the paper.

Two reports are emitted:

- A *macro-F1* table — the headline number across folds.
- A *per-fold* table — full disclosure of per-system numbers, with
  ``-`` placeholders where the published paper did not report the value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUBLISHED_PATH = _REPO_ROOT / "reports" / "phase4" / "published_numbers.yaml"


@dataclass(frozen=True, slots=True)
class PublishedBaseline:
    """One row of the head-to-head table."""

    method: str
    paper_link: str
    year: int
    protocol: str
    per_fold: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    macro_f1: float | None = None


def load_published_baselines(path: Path | str | None = None) -> list[PublishedBaseline]:
    """Read ``published_numbers.yaml`` and return the parsed baselines."""
    p = Path(path) if path is not None else DEFAULT_PUBLISHED_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[PublishedBaseline] = []
    for entry in raw.get("baselines", []):
        if not isinstance(entry, Mapping):
            continue
        out.append(
            PublishedBaseline(
                method=str(entry.get("method", "")),
                paper_link=str(entry.get("paper_link", "")),
                year=int(entry.get("year", 0)),
                protocol=str(entry.get("protocol", "")).strip(),
                per_fold=dict(entry.get("per_fold", {}) or {}),
                macro_f1=_to_float_or_none(entry.get("macro_f1")),
            )
        )
    return out


def _to_float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, *, percent: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def render_macro_comparison(
    *,
    hylog_macro_f1: float | None,
    baselines: Sequence[PublishedBaseline],
) -> str:
    """Render a macro-F1 head-to-head Markdown table.

    HyLog's row is the first row so the comparison is "us vs published".
    """
    lines = [
        "| Method | Year | Macro-F1 (%) | Protocol |",
        "|---|---|---|---|",
        f"| **HyLog (this work)** | 2026 | **{_fmt(hylog_macro_f1)}** | "
        "Zero-label LOSO over HDFS/BGL/Thunderbird; calibrated; sub-2B SLM. |",
    ]
    for b in baselines:
        protocol = b.protocol.replace("\n", " ").strip()
        lines.append(
            f"| [{b.method}]({b.paper_link}) | {b.year} | {_fmt(b.macro_f1)} | {protocol} |"
        )
    return "\n".join(lines) + "\n"


def render_per_fold_comparison(
    *,
    hylog_per_fold: Mapping[str, float | None],
    baselines: Sequence[PublishedBaseline],
    folds: Sequence[str] = ("HDFS", "BGL", "Thunderbird", "OpenStack"),
) -> str:
    """Render the per-fold F1 head-to-head Markdown table."""
    header = "| Method | " + " | ".join(folds) + " |"
    sep = "|---|" + "---|" * len(folds)
    rows: list[str] = [header, sep]
    hylog_cells = [_fmt(hylog_per_fold.get(f)) for f in folds]
    rows.append("| **HyLog (this work)** | " + " | ".join(hylog_cells) + " |")
    for b in baselines:
        cells: list[str] = []
        for f in folds:
            val = (b.per_fold.get(f) or {}).get("f1")
            cells.append(_fmt(_to_float_or_none(val)))
        rows.append(f"| [{b.method}]({b.paper_link}) | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


__all__ = [
    "DEFAULT_PUBLISHED_PATH",
    "PublishedBaseline",
    "load_published_baselines",
    "render_macro_comparison",
    "render_per_fold_comparison",
]
