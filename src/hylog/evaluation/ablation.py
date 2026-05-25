"""Ablation matrix orchestrator + Q1-grade statistical comparison.

The Phase-6 deliverable: eight ablation axes (A1-A8), each with a
fixed set of conditions, all run on the **same 5 seeds**. The
orchestrator computes per-cell means/stds, paired Wilcoxon p-values,
Cliff's delta effect sizes, and Holm-Bonferroni-corrected p-values
across the family of A1-A8 comparisons. The output is one
``ablation_matrix.csv`` plus a Markdown table per axis.

The orchestrator is *runner-agnostic*: callers supply a callable that
produces per-seed metrics for a given (axis, condition) pair. Phase-6
will wire the real trainer in; this module is fully testable on CPU
with mock runners.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hylog.evaluation.cliffs_delta import CliffsDelta, cliffs_delta
from hylog.evaluation.statistical_tests import (
    TestResult,
    holm_bonferroni,
    wilcoxon_paired,
)

DEFAULT_PRIMARY_METRIC = "f1"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AblationCondition:
    """One condition (= one cell of the ablation matrix)."""

    axis: str
    name: str
    description: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def cell_id(self) -> str:
        return f"{self.axis}::{self.name}"


@dataclass(frozen=True, slots=True)
class AblationAxis:
    """One axis = a family of conditions to compare against each other."""

    name: str
    description: str
    conditions: tuple[AblationCondition, ...]
    baseline_condition: str
    """Name of the condition every other condition is compared against
    (Holm-Bonferroni runs across the non-baseline comparisons)."""

    def __post_init__(self) -> None:
        if self.baseline_condition not in {c.name for c in self.conditions}:
            raise ValueError(
                f"baseline_condition {self.baseline_condition!r} is not among "
                f"conditions {[c.name for c in self.conditions]}"
            )


@dataclass(slots=True)
class CellResult:
    """Per-seed metrics for one condition."""

    condition: AblationCondition
    seeds: tuple[int, ...]
    seed_metrics: dict[int, Mapping[str, float]] = field(default_factory=dict)

    def metric_values(self, metric: str) -> list[float]:
        out: list[float] = []
        for s in self.seeds:
            v = self.seed_metrics.get(s, {}).get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN
                continue
            out.append(fv)
        return out

    def summary(self, metric: str) -> dict[str, float]:
        vals = self.metric_values(metric)
        if not vals:
            return {
                "mean": float("nan"),
                "std": 0.0,
                "n": 0,
                "min": float("nan"),
                "max": float("nan"),
            }
        return {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }


@dataclass(slots=True)
class ComparisonResult:
    """Paired comparison between two conditions on a specific metric."""

    axis: str
    metric: str
    baseline: str
    variant: str
    n: int
    wilcoxon: TestResult | None
    cliffs: CliffsDelta
    delta_mean: float
    """Mean(variant) - mean(baseline) — sign is positive when variant
    improves on the metric (assuming larger-is-better)."""

    p_value_raw: float
    p_value_holm: float | None = None
    significant_at_0_05: bool = False
    significant_under_holm: bool = False
    """``True`` if the Holm-corrected p-value is below 0.05 AND
    |Cliff's δ| > 0.33 (the Phase-6 checklist's joint criterion)."""

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "metric": self.metric,
            "baseline": self.baseline,
            "variant": self.variant,
            "n": self.n,
            "wilcoxon_p": self.p_value_raw,
            "wilcoxon_p_holm": self.p_value_holm,
            "wilcoxon_statistic": (self.wilcoxon.statistic if self.wilcoxon else None),
            "delta_mean": self.delta_mean,
            "cliffs_delta": self.cliffs.delta,
            "cliffs_magnitude": self.cliffs.magnitude,
            "significant_at_0_05": self.significant_at_0_05,
            "significant_under_holm": self.significant_under_holm,
        }


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------


CellRunner = Callable[[AblationCondition, int], Mapping[str, float]]
"""Callable: (condition, seed) -> metric dict (e.g. {'f1': 0.91, ...})."""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_axis(
    axis: AblationAxis,
    *,
    seeds: Sequence[int],
    runner: CellRunner,
) -> dict[str, CellResult]:
    """Execute every condition in an axis for every seed."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    results: dict[str, CellResult] = {}
    for condition in axis.conditions:
        cell = CellResult(condition=condition, seeds=tuple(seeds))
        for seed in seeds:
            cell.seed_metrics[seed] = dict(runner(condition, seed))
        results[condition.name] = cell
    return results


def compare_axis(
    axis: AblationAxis,
    cells: Mapping[str, CellResult],
    *,
    metric: str = DEFAULT_PRIMARY_METRIC,
    alternative: str = "two-sided",
) -> list[ComparisonResult]:
    """Compute per-variant comparisons against the baseline condition.

    Returns a list with one ``ComparisonResult`` per non-baseline
    condition. Holm-Bonferroni correction is applied across these
    comparisons (within the axis).
    """
    if axis.baseline_condition not in cells:
        raise KeyError(f"baseline {axis.baseline_condition!r} not in cells")
    baseline_cell = cells[axis.baseline_condition]
    baseline_values = baseline_cell.metric_values(metric)

    raw_comparisons: list[ComparisonResult] = []
    for cond in axis.conditions:
        if cond.name == axis.baseline_condition:
            continue
        cell = cells[cond.name]
        variant_values = cell.metric_values(metric)
        n = min(len(baseline_values), len(variant_values))
        delta_mean = (sum(variant_values[:n]) / n) - (sum(baseline_values[:n]) / n) if n else 0.0
        if n >= 2:
            wil = wilcoxon_paired(variant_values[:n], baseline_values[:n], alternative=alternative)
            p_raw = wil.p_value
        else:
            wil = None
            p_raw = 1.0
        cliffs = cliffs_delta(variant_values, baseline_values)
        raw_comparisons.append(
            ComparisonResult(
                axis=axis.name,
                metric=metric,
                baseline=axis.baseline_condition,
                variant=cond.name,
                n=n,
                wilcoxon=wil,
                cliffs=cliffs,
                delta_mean=float(delta_mean),
                p_value_raw=float(p_raw),
                p_value_holm=None,
                significant_at_0_05=bool(p_raw < 0.05),
                significant_under_holm=False,
            )
        )

    if raw_comparisons:
        p_values = [c.p_value_raw for c in raw_comparisons]
        rejected = holm_bonferroni(p_values, alpha=0.05)
        # Holm correction emits one decision per hypothesis; reconstruct
        # the corrected p-value as the smallest alpha at which it would
        # be rejected. We compute the canonical Holm-adjusted p-values.
        order = sorted(range(len(p_values)), key=lambda i: p_values[i])
        adjusted = [0.0] * len(p_values)
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, p_values[idx] * (len(p_values) - rank))
            adj = max(adj, prev)  # monotone non-decreasing per Holm
            adjusted[idx] = adj
            prev = adj
        for i, comp in enumerate(raw_comparisons):
            sig_pair = bool(rejected[i] and abs(comp.cliffs.delta) > 0.33)
            raw_comparisons[i] = ComparisonResult(
                axis=comp.axis,
                metric=comp.metric,
                baseline=comp.baseline,
                variant=comp.variant,
                n=comp.n,
                wilcoxon=comp.wilcoxon,
                cliffs=comp.cliffs,
                delta_mean=comp.delta_mean,
                p_value_raw=comp.p_value_raw,
                p_value_holm=float(adjusted[i]),
                significant_at_0_05=comp.significant_at_0_05,
                significant_under_holm=sig_pair,
            )

    return raw_comparisons


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------


def write_axis_csv(
    *,
    axis: AblationAxis,
    cells: Mapping[str, CellResult],
    comparisons: Sequence[ComparisonResult],
    out_path: Path | str,
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> Path:
    """Single-CSV-per-axis emission. The full ablation_matrix.csv across
    every axis is produced by :func:`write_global_csv`.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "axis",
                "condition",
                "is_baseline",
                "metric",
                "mean",
                "std",
                "min",
                "max",
                "n",
                "wilcoxon_p",
                "wilcoxon_p_holm",
                "cliffs_delta",
                "cliffs_magnitude",
                "delta_mean",
                "significant_under_holm",
            ]
        )
        for cond in axis.conditions:
            cell = cells[cond.name]
            summ = cell.summary(metric)
            is_base = cond.name == axis.baseline_condition
            comp = next((c for c in comparisons if c.variant == cond.name), None)
            writer.writerow(
                [
                    axis.name,
                    cond.name,
                    is_base,
                    metric,
                    summ["mean"],
                    summ["std"],
                    summ["min"],
                    summ["max"],
                    summ["n"],
                    "" if comp is None else comp.p_value_raw,
                    "" if comp is None or comp.p_value_holm is None else comp.p_value_holm,
                    "" if comp is None else comp.cliffs.delta,
                    "" if comp is None else comp.cliffs.magnitude,
                    "" if comp is None else comp.delta_mean,
                    "" if comp is None else comp.significant_under_holm,
                ]
            )
    return p


def write_global_csv(
    *,
    per_axis_comparisons: Mapping[str, Sequence[ComparisonResult]],
    out_path: Path | str,
) -> Path:
    """The single ``ablation_matrix.csv`` required by the Phase-6 checklist."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "axis",
                "baseline",
                "variant",
                "metric",
                "n",
                "delta_mean",
                "wilcoxon_p",
                "wilcoxon_p_holm",
                "cliffs_delta",
                "cliffs_magnitude",
                "significant_at_0_05",
                "significant_under_holm",
            ]
        )
        for axis_name in sorted(per_axis_comparisons):
            for comp in per_axis_comparisons[axis_name]:
                writer.writerow(
                    [
                        axis_name,
                        comp.baseline,
                        comp.variant,
                        comp.metric,
                        comp.n,
                        comp.delta_mean,
                        comp.p_value_raw,
                        "" if comp.p_value_holm is None else comp.p_value_holm,
                        comp.cliffs.delta,
                        comp.cliffs.magnitude,
                        comp.significant_at_0_05,
                        comp.significant_under_holm,
                    ]
                )
    return p


def render_axis_markdown(
    *,
    axis: AblationAxis,
    cells: Mapping[str, CellResult],
    comparisons: Sequence[ComparisonResult],
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> str:
    """Per-axis Markdown table with mean ± std + Cliff's δ + p-values."""
    lines = [
        f"### {axis.name} — {axis.description}",
        "",
        f"| Condition | {metric} (mean ± std) | n | Cliff's δ | δ-magnitude | p (raw) | p (Holm) | significant? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond in axis.conditions:
        cell = cells[cond.name]
        summ = cell.summary(metric)
        is_base = cond.name == axis.baseline_condition
        comp = next((c for c in comparisons if c.variant == cond.name), None)
        if is_base:
            line = (
                f"| **{cond.name}** (baseline) | "
                f"{summ['mean']:.4f} ± {summ['std']:.4f} | "
                f"{summ['n']} | — | — | — | — | — |"
            )
        else:
            assert comp is not None
            sig_marker = "✅" if comp.significant_under_holm else "—"
            p_holm = "—" if comp.p_value_holm is None else f"{comp.p_value_holm:.4f}"
            line = (
                f"| {cond.name} | "
                f"{summ['mean']:.4f} ± {summ['std']:.4f} | "
                f"{summ['n']} | "
                f"{comp.cliffs.delta:+.3f} | "
                f"{comp.cliffs.magnitude} | "
                f"{comp.p_value_raw:.4f} | "
                f"{p_holm} | "
                f"{sig_marker} |"
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_run_record(
    *,
    out_dir: Path | str,
    axis: AblationAxis,
    cells: Mapping[str, CellResult],
    comparisons: Sequence[ComparisonResult],
    metric: str = DEFAULT_PRIMARY_METRIC,
) -> dict[str, Path]:
    """Per-axis directory: ``axis.csv``, ``axis.md``, ``raw.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = write_axis_csv(
        axis=axis,
        cells=cells,
        comparisons=comparisons,
        out_path=out / f"{axis.name}.csv",
        metric=metric,
    )
    md = render_axis_markdown(axis=axis, cells=cells, comparisons=comparisons, metric=metric)
    md_path = out / f"{axis.name}.md"
    md_path.write_text(md, encoding="utf-8", newline="\n")
    raw_path = out / f"{axis.name}_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "axis": axis.name,
                "description": axis.description,
                "baseline": axis.baseline_condition,
                "metric": metric,
                "cells": {
                    name: {
                        "description": cell.condition.description,
                        "parameters": dict(cell.condition.parameters),
                        "seed_metrics": {str(s): dict(m) for s, m in cell.seed_metrics.items()},
                    }
                    for name, cell in cells.items()
                },
                "comparisons": [c.to_dict() for c in comparisons],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"csv": csv_path, "md": md_path, "raw": raw_path}


__all__ = [
    "DEFAULT_PRIMARY_METRIC",
    "AblationAxis",
    "AblationCondition",
    "CellResult",
    "CellRunner",
    "ComparisonResult",
    "compare_axis",
    "render_axis_markdown",
    "run_axis",
    "write_axis_csv",
    "write_global_csv",
    "write_run_record",
]
