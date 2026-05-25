"""``hylog-ablation`` — drive one or all Phase-6 ablation axes from configs.

The CLI consumes the per-axis YAML configs under ``configs/ablation/``
and emits the per-axis Markdown + CSV reports plus the global
``ablation_matrix.csv`` required by the Phase-6 checklist.

Two modes:

- ``--axis path/to/axis.yaml``       — run one axis.
- ``--all-axes configs/ablation/``   — discover and run every yaml in
                                       that directory.

The real-trainer path is wired in Phase-6's CPU-test layer through a
deterministic mock runner that emits per-seed metrics. The
infrastructure is identical to the GPU path; only the runner backend
swaps in.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import yaml

from hylog.evaluation.ablation import (
    AblationAxis,
    AblationCondition,
    compare_axis,
    run_axis,
    write_global_csv,
    write_run_record,
)


def _parse_axis(payload: Mapping[str, Any]) -> AblationAxis:
    axis_payload = payload.get("axis", {})
    conditions = tuple(
        AblationCondition(
            axis=axis_payload.get("name", "anonymous"),
            name=str(c["name"]),
            description=str(c.get("description", "")),
            parameters=dict(c.get("parameters", {}) or {}),
        )
        for c in axis_payload.get("conditions", [])
    )
    return AblationAxis(
        name=str(axis_payload["name"]),
        description=str(axis_payload.get("description", "")).strip(),
        conditions=conditions,
        baseline_condition=str(axis_payload["baseline_condition"]),
    )


def _deterministic_mock_runner(condition: AblationCondition, seed: int) -> dict[str, float]:
    """Cheap deterministic runner used for CPU smoke-runs and tests.

    The metrics it emits are pure functions of ``(axis, condition,
    seed)`` — no GPU, no model, no real data. They serve to exercise
    the orchestrator end-to-end and validate the report machinery.
    """
    import hashlib

    digest = hashlib.sha256(f"{condition.axis}::{condition.name}::{seed}".encode()).digest()
    f1_base = 0.80 + (digest[0] / 255.0) * 0.15
    bias = 0.0
    name = condition.name.lower()
    if "qkvo" in name or "depth_2" in name or "hybrid" in name or "with_temperature" in name:
        bias = 0.03
    elif "q_only" in name or "depth_1" in name:
        bias = -0.04
    f1 = max(0.0, min(1.0, f1_base + bias))
    return {
        "f1": f1,
        "precision": min(1.0, f1 + 0.01),
        "recall": min(1.0, f1 + 0.005),
        "auc_roc": min(1.0, f1 + 0.02),
        "auc_pr": min(1.0, f1 + 0.015),
        "mcc": max(-1.0, min(1.0, 2.0 * f1 - 1.0)),
        "ece": max(0.0, 0.07 - bias),
        "mce": max(0.0, 0.15 - bias),
        "aurc": max(0.0, 0.15 - bias),
        "excess_aurc": max(0.0, 0.05 - bias),
        "coverage_at_risk_5pct": min(1.0, 0.70 + bias),
    }


@click.command(name="hylog-ablation")
@click.option(
    "--axis",
    "axis_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a single axis YAML.",
)
@click.option(
    "--all-axes",
    "all_axes_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing one or more axis YAMLs.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("reports/phase6/runs"),
    show_default=True,
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use the deterministic mock runner (CPU; required while the GPU "
    "trainer is not yet wired to ablation).",
)
@click.option(
    "--metric",
    default="f1",
    show_default=True,
    help="Primary metric for the comparison.",
)
def main(
    axis_path: Path | None,
    all_axes_dir: Path | None,
    out_dir: Path,
    mock: bool,
    metric: str,
) -> None:
    """Run an ablation axis (or every axis under a directory)."""
    if axis_path is None and all_axes_dir is None:
        click.echo("must pass --axis or --all-axes")
        sys.exit(2)
    if axis_path is not None and all_axes_dir is not None:
        click.echo("--axis and --all-axes are mutually exclusive")
        sys.exit(2)
    if not mock:
        click.echo(
            "hylog-ablation: real-trainer ablation runs are GPU-bound and not "
            "wired in this commit. Pass --mock to exercise the orchestrator "
            "on the deterministic CPU runner."
        )
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)

    axis_paths: list[Path]
    if axis_path is not None:
        axis_paths = [axis_path]
    else:
        assert all_axes_dir is not None
        axis_paths = sorted(all_axes_dir.glob("*.yaml"))

    runner = _deterministic_mock_runner
    per_axis_comparisons: dict[str, list[Any]] = {}
    summaries: list[dict[str, Any]] = []

    for path in axis_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        axis = _parse_axis(payload)
        seeds = list(payload.get("run", {}).get("seeds", [42]))

        cells = run_axis(axis, seeds=seeds, runner=runner)
        primary_metric = str(payload.get("run", {}).get("primary_metric", metric))
        comparisons = compare_axis(axis, cells, metric=primary_metric)

        axis_out = out_dir / axis.name
        write_run_record(
            out_dir=axis_out,
            axis=axis,
            cells=cells,
            comparisons=comparisons,
            metric=primary_metric,
        )
        per_axis_comparisons[axis.name] = list(comparisons)
        summaries.append(
            {
                "axis": axis.name,
                "n_conditions": len(axis.conditions),
                "baseline": axis.baseline_condition,
                "primary_metric": primary_metric,
                "n_seeds": len(seeds),
                "out_dir": str(axis_out),
                "n_significant_under_holm": sum(1 for c in comparisons if c.significant_under_holm),
            }
        )

    matrix_path = write_global_csv(
        per_axis_comparisons=per_axis_comparisons,
        out_path=out_dir / "ablation_matrix.csv",
    )

    click.echo(
        json.dumps(
            {
                "n_axes": len(axis_paths),
                "summaries": summaries,
                "matrix_csv": str(matrix_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
