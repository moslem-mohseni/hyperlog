"""Statistical hypothesis tests for head-to-head comparisons.

Two tests are exposed:

- **Paired Wilcoxon signed-rank** over per-seed metrics — used when we
  have multiple seeds and the published baseline reports a single
  point estimate. We test ``H0: median(HyLog seed F1) == baseline_F1``
  against the two-sided alternative.
- **One-sample Wilcoxon** wrapping ``scipy.stats.wilcoxon`` — falls
  back to a permutation-based exact computation when n is small.

The tests are self-contained and do not require sklearn. SciPy is an
existing dependency (used by the metric panel for AUC) so this module
adds no new install footprint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of a hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    n: int
    alternative: str
    rejected_at_alpha_0_05: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "alternative": self.alternative,
            "rejected_at_alpha_0_05": self.rejected_at_alpha_0_05,
        }


def wilcoxon_one_sample(
    values: Sequence[float],
    *,
    expected: float,
    alternative: str = "two-sided",
) -> TestResult:
    """One-sample Wilcoxon signed-rank test.

    Args:
        values: Observed measurements (e.g. per-seed F1 of HyLog).
        expected: The hypothesized median (e.g. ZeroLog's reported F1).
        alternative: ``"two-sided"``, ``"greater"`` (values > expected),
            or ``"less"``.

    Notes:
        Uses scipy.stats.wilcoxon when available (with method="exact"
        for small samples) and a fallback exact computation otherwise.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unknown alternative: {alternative}")
    arr = np.asarray(values, dtype=np.float64)
    differences = arr - expected
    nonzero = differences[differences != 0]
    n = int(nonzero.size)
    if n == 0:
        return TestResult(
            test_name="wilcoxon_one_sample",
            statistic=0.0,
            p_value=1.0,
            n=0,
            alternative=alternative,
            rejected_at_alpha_0_05=False,
        )

    try:
        from scipy.stats import wilcoxon

        kwargs: dict[str, str] = {"alternative": alternative}
        if n <= 25:
            kwargs["method"] = "exact"
        result = wilcoxon(differences, **kwargs)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ImportError:
        statistic, p_value = _wilcoxon_exact(nonzero, alternative)

    return TestResult(
        test_name="wilcoxon_one_sample",
        statistic=statistic,
        p_value=p_value,
        n=n,
        alternative=alternative,
        rejected_at_alpha_0_05=bool(p_value < 0.05),
    )


def wilcoxon_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> TestResult:
    """Paired Wilcoxon signed-rank: test that ``a - b`` has zero median."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"shape mismatch: a {arr_a.shape} vs b {arr_b.shape}; "
            "paired test requires identical lengths"
        )
    return wilcoxon_one_sample(list(arr_a - arr_b), expected=0.0, alternative=alternative)


def _wilcoxon_exact(differences: np.ndarray, alternative: str) -> tuple[float, float]:
    """Exact two-sided Wilcoxon p-value via signed-rank enumeration.

    Used only when SciPy is unavailable. O(2^n) so capped at n <= 20.
    """
    n = int(differences.size)
    if n > 20:
        # Asymptotic normal approximation.
        ranks = _signed_ranks(differences)
        w_plus = float(ranks[differences > 0].sum())
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w_plus - mu) / sigma if sigma > 0 else 0.0
        if alternative == "two-sided":
            p = 2 * (1 - _normal_cdf(abs(z)))
        elif alternative == "greater":
            p = 1 - _normal_cdf(z)
        else:
            p = _normal_cdf(z)
        return w_plus, float(min(max(p, 0.0), 1.0))

    ranks = _signed_ranks(differences)
    observed_w_plus = float(ranks[differences > 0].sum())
    float(ranks.sum())

    # Enumerate every sign assignment.
    count_le = 0
    count_ge = 0
    total = 1 << n
    abs_ranks = np.abs(_signed_ranks(differences))
    for mask in range(total):
        w_plus = 0.0
        for i in range(n):
            if mask & (1 << i):
                w_plus += abs_ranks[i]
        if w_plus <= observed_w_plus:
            count_le += 1
        if w_plus >= observed_w_plus:
            count_ge += 1
    if alternative == "two-sided":
        p = 2 * min(count_le, count_ge) / total
    elif alternative == "greater":
        p = count_ge / total
    else:
        p = count_le / total
    return observed_w_plus, float(min(max(p, 0.0), 1.0))


def _signed_ranks(values: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(values)
    order = np.argsort(abs_vals)
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < abs_vals.size:
        j = i
        while j + 1 < abs_vals.size and abs_vals[order[j + 1]] == abs_vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni multiple-comparisons correction.

    Returns a list of booleans, ``True`` where the corresponding
    hypothesis is rejected at family-wise error rate ``alpha``.
    """
    if not p_values:
        return []
    n = len(p_values)
    ordered = sorted(range(n), key=lambda i: p_values[i])
    rejected = [False] * n
    for k, idx in enumerate(ordered):
        threshold = alpha / (n - k)
        if p_values[idx] < threshold:
            rejected[idx] = True
        else:
            break  # remaining hypotheses cannot be rejected.
    return rejected


__all__ = [
    "TestResult",
    "holm_bonferroni",
    "wilcoxon_one_sample",
    "wilcoxon_paired",
]
