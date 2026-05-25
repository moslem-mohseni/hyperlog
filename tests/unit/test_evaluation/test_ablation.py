"""Tests for the ablation orchestrator + statistical comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hylog.evaluation.ablation import (
    AblationAxis,
    AblationCondition,
    compare_axis,
    render_axis_markdown,
    run_axis,
    write_axis_csv,
    write_global_csv,
    write_run_record,
)


def _axis_three_conditions() -> AblationAxis:
    return AblationAxis(
        name="A_test",
        description="Synthetic test axis.",
        conditions=(
            AblationCondition(
                axis="A_test",
                name="baseline",
                description="The reference cell.",
                parameters={"x": 1},
            ),
            AblationCondition(
                axis="A_test",
                name="variant_better",
                description="Should beat the baseline.",
                parameters={"x": 2},
            ),
            AblationCondition(
                axis="A_test",
                name="variant_worse",
                description="Should lose to the baseline.",
                parameters={"x": 3},
            ),
        ),
        baseline_condition="baseline",
    )


def _runner(condition: AblationCondition, seed: int) -> dict[str, float]:
    """Deterministic runner: variant_better wins by +0.05, variant_worse loses by -0.05."""
    base_f1 = 0.85 + (seed % 5) * 0.002
    if condition.name == "variant_better":
        return {"f1": base_f1 + 0.05, "ece": 0.04}
    if condition.name == "variant_worse":
        return {"f1": base_f1 - 0.05, "ece": 0.12}
    return {"f1": base_f1, "ece": 0.08}


def test_run_axis_returns_one_cell_per_condition() -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[42, 1337, 2024, 31415, 27182], runner=_runner)
    assert set(cells) == {"baseline", "variant_better", "variant_worse"}
    for c in cells.values():
        assert len(c.seed_metrics) == 5


def test_run_axis_rejects_duplicate_seeds() -> None:
    axis = _axis_three_conditions()
    with pytest.raises(ValueError):
        run_axis(axis, seeds=[42, 42, 1337], runner=_runner)


def test_compare_axis_signals_winning_variant() -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[42, 1337, 2024, 31415, 27182], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    by_name = {c.variant: c for c in comparisons}
    assert by_name["variant_better"].delta_mean > 0
    assert by_name["variant_worse"].delta_mean < 0
    # The "better" variant should have a positive Cliff's δ and pass the threshold.
    assert by_name["variant_better"].cliffs.delta > 0
    assert by_name["variant_better"].cliffs.is_medium_or_above


def test_compare_axis_applies_holm_correction() -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[42, 1337, 2024, 31415, 27182], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    for c in comparisons:
        assert c.p_value_holm is not None
        assert c.p_value_holm >= c.p_value_raw - 1e-9  # Holm ≥ raw


def test_baseline_missing_raises_at_construction() -> None:
    with pytest.raises(ValueError):
        AblationAxis(
            name="X",
            description="",
            conditions=(AblationCondition(axis="X", name="a", description="", parameters={}),),
            baseline_condition="nonexistent",
        )


def test_write_axis_csv_round_trip(tmp_path: Path) -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[1, 2, 3, 4, 5], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    path = write_axis_csv(
        axis=axis,
        cells=cells,
        comparisons=comparisons,
        out_path=tmp_path / "x.csv",
        metric="f1",
    )
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0][0] == "axis"
    assert len(rows) == 1 + len(axis.conditions)


def test_render_axis_markdown_contains_all_conditions() -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[1, 2, 3, 4, 5], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    md = render_axis_markdown(axis=axis, cells=cells, comparisons=comparisons, metric="f1")
    for cond_name in ("baseline", "variant_better", "variant_worse"):
        assert cond_name in md


def test_write_run_record_emits_csv_md_json(tmp_path: Path) -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[1, 2, 3, 4, 5], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    paths = write_run_record(
        out_dir=tmp_path, axis=axis, cells=cells, comparisons=comparisons, metric="f1"
    )
    for key in ("csv", "md", "raw"):
        assert paths[key].exists()
    payload = json.loads(paths["raw"].read_text(encoding="utf-8"))
    assert payload["axis"] == "A_test"
    assert "cells" in payload


def test_write_global_csv_includes_every_axis(tmp_path: Path) -> None:
    axis = _axis_three_conditions()
    cells = run_axis(axis, seeds=[1, 2, 3, 4, 5], runner=_runner)
    comparisons = compare_axis(axis, cells, metric="f1")
    path = write_global_csv(
        per_axis_comparisons={"A_test": comparisons},
        out_path=tmp_path / "matrix.csv",
    )
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    # Header + 2 non-baseline variants.
    assert rows[0][0] == "axis"
    assert len(rows) == 1 + 2


def test_metric_with_missing_seed_still_summarises() -> None:
    """Cells with a missing seed metric are tolerated (returns NaN-safe summary)."""
    axis = _axis_three_conditions()

    def sparse_runner(cond: AblationCondition, seed: int) -> dict[str, float]:
        if seed == 42:
            return {}  # missing metrics
        return _runner(cond, seed)

    cells = run_axis(axis, seeds=[42, 1337, 2024, 31415, 27182], runner=sparse_runner)
    base = cells["baseline"]
    summ = base.summary("f1")
    assert summ["n"] == 4  # 5 seeds - 1 missing
