"""Multi-seed runner — repeats a training routine across N seeds and
aggregates mean / std / min / max of every metric.

Phase 3 checklist: "5-seed runs; standard deviation per metric reported."
The runner is the canonical entry point for any reproducibility-bound
experiment.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from hylog.utils.seeding import set_global_seed


@dataclass(frozen=True, slots=True)
class SeedAggregate:
    metric: str
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        if not self.values:
            return float("nan")
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        n = len(self.values)
        if n <= 1:
            return 0.0
        mu = self.mean
        var = sum((v - mu) ** 2 for v in self.values) / (n - 1)
        return math.sqrt(var)

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else float("nan")

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else float("nan")

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std, "min": self.minimum, "max": self.maximum}


@dataclass(frozen=True, slots=True)
class MultiSeedResult:
    seeds: tuple[int, ...]
    per_seed_metrics: tuple[Mapping[str, float], ...]
    aggregates: Mapping[str, SeedAggregate] = field(default_factory=dict)

    def headline(self, metric: str) -> str:
        agg = self.aggregates.get(metric)
        if agg is None:
            return f"{metric}: n/a"
        return f"{metric}: {agg.mean:.4f} ± {agg.std:.4f} (min={agg.minimum:.4f}, max={agg.maximum:.4f}, n={len(agg.values)})"


def run_seeded(
    *,
    seeds: Sequence[int],
    routine: Callable[[int], Mapping[str, float]],
) -> MultiSeedResult:
    """Run ``routine`` once per seed and aggregate.

    Args:
        seeds: List of integer seeds to iterate over.
        routine: Callable taking a seed and returning a mapping of metric
            names to scalar floats.

    Returns:
        A ``MultiSeedResult`` with per-seed metrics and per-metric aggregates.
    """
    if not seeds:
        raise ValueError("at least one seed must be provided")

    per_seed: list[Mapping[str, float]] = []
    for seed in seeds:
        set_global_seed(int(seed))
        metrics = dict(routine(int(seed)))
        per_seed.append(metrics)

    metric_names: set[str] = set()
    for d in per_seed:
        metric_names.update(d.keys())

    aggregates: dict[str, SeedAggregate] = {}
    for name in sorted(metric_names):
        values = tuple(float(d[name]) for d in per_seed if name in d)
        aggregates[name] = SeedAggregate(metric=name, values=values)

    return MultiSeedResult(
        seeds=tuple(int(s) for s in seeds),
        per_seed_metrics=tuple(per_seed),
        aggregates=aggregates,
    )


__all__ = ["MultiSeedResult", "SeedAggregate", "run_seeded"]
