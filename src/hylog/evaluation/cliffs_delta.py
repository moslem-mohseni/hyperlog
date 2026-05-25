"""Cliff's delta — non-parametric effect-size for paired comparisons.

Reference: Cliff, N. (1993). *Dominance statistics: Ordinal analyses to
answer ordinal questions.* Psychological Bulletin, 114(3), 494-509.

Definition (for two samples A of size ``n_a`` and B of size ``n_b``):

    δ = (#{(a, b) : a > b} - #{(a, b) : a < b}) / (n_a * n_b)

where pairs are taken across the Cartesian product A x B. ``δ`` lies in
``[-1, +1]``:

- ``+1`` — every value in A is greater than every value in B.
- ``0``  — equal distributions.
- ``-1`` — every value in A is less than every value in B.

Magnitude interpretation (Romano et al. 2006):

  |δ| < 0.147   negligible
  0.147 ≤ |δ| < 0.330  small
  0.330 ≤ |δ| < 0.474  medium
  |δ| ≥ 0.474   large

The Phase-6 checklist explicitly requires ``|Cliff's δ| > 0.33`` for A1
to count as a positive result, so the "medium-or-large" threshold is
the operational decision boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliffsDelta:
    """Effect-size record + qualitative magnitude."""

    delta: float
    magnitude: str
    n_a: int
    n_b: int

    @property
    def is_negligible(self) -> bool:
        return self.magnitude == "negligible"

    @property
    def is_small_or_above(self) -> bool:
        return self.magnitude != "negligible"

    @property
    def is_medium_or_above(self) -> bool:
        return self.magnitude in {"medium", "large"}

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "delta": float(self.delta),
            "magnitude": self.magnitude,
            "n_a": int(self.n_a),
            "n_b": int(self.n_b),
        }


def _interpret_magnitude(abs_delta: float) -> str:
    """Romano et al. 2006 thresholds."""
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.330:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> CliffsDelta:
    """Compute Cliff's δ between two samples.

    The implementation is the naive ``O(n_a * n_b)`` form which is
    unproblematic for the seed sweeps in Phase 6 (n = 5 per group).
    """
    a_list = list(a)
    b_list = list(b)
    n_a, n_b = len(a_list), len(b_list)
    if n_a == 0 or n_b == 0:
        return CliffsDelta(delta=0.0, magnitude="negligible", n_a=n_a, n_b=n_b)

    gt = 0
    lt = 0
    for x in a_list:
        for y in b_list:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    denom = n_a * n_b
    delta = (gt - lt) / denom
    return CliffsDelta(
        delta=delta,
        magnitude=_interpret_magnitude(abs(delta)),
        n_a=n_a,
        n_b=n_b,
    )


__all__ = ["CliffsDelta", "cliffs_delta"]
