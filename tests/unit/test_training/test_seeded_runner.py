"""Tests for the multi-seed driver."""

from __future__ import annotations

import math

import pytest

from hylog.training.seeded_runner import SeedAggregate, run_seeded


def test_run_seeded_aggregates_mean_std() -> None:
    def routine(seed: int) -> dict[str, float]:
        # Make metrics seed-dependent and have non-zero variance.
        return {"f1": 0.9 + 0.01 * (seed % 5)}

    result = run_seeded(seeds=[1, 2, 3, 4, 5], routine=routine)
    agg = result.aggregates["f1"]
    assert agg.mean == pytest.approx((0.91 + 0.92 + 0.93 + 0.94 + 0.90) / 5)
    assert agg.std > 0


def test_seed_aggregate_single_value() -> None:
    a = SeedAggregate(metric="x", values=(0.5,))
    assert a.mean == 0.5
    assert a.std == 0.0
    assert a.minimum == 0.5
    assert a.maximum == 0.5


def test_seed_aggregate_empty() -> None:
    a = SeedAggregate(metric="x", values=())
    assert math.isnan(a.mean)
    assert a.std == 0.0


def test_run_seeded_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError):
        run_seeded(seeds=[], routine=lambda s: {"x": 1.0})


def test_run_seeded_headline_formats_metric() -> None:
    def routine(seed: int) -> dict[str, float]:
        return {"f1": 0.95}

    result = run_seeded(seeds=[1, 2, 3], routine=routine)
    line = result.headline("f1")
    assert "f1" in line
    assert "±" in line
    assert "0.9500" in line


def test_run_seeded_records_each_seed() -> None:
    def routine(seed: int) -> dict[str, float]:
        return {"loss": float(seed)}

    result = run_seeded(seeds=[10, 20, 30], routine=routine)
    assert result.seeds == (10, 20, 30)
    assert [d["loss"] for d in result.per_seed_metrics] == [10.0, 20.0, 30.0]
    assert result.aggregates["loss"].mean == pytest.approx(20.0)
