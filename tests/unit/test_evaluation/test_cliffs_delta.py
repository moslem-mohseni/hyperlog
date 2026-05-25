"""Tests for Cliff's delta + magnitude interpretation."""

from __future__ import annotations

import pytest

from hylog.evaluation.cliffs_delta import CliffsDelta, cliffs_delta


def test_identical_distributions_have_delta_zero() -> None:
    r = cliffs_delta([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert r.delta == pytest.approx(0.0)
    assert r.magnitude == "negligible"


def test_dominated_distribution_has_delta_one() -> None:
    r = cliffs_delta([1, 1, 1], [0, 0, 0])
    assert r.delta == pytest.approx(1.0)
    assert r.magnitude == "large"


def test_dominated_other_way_has_delta_minus_one() -> None:
    r = cliffs_delta([0, 0, 0], [1, 1, 1])
    assert r.delta == pytest.approx(-1.0)
    assert r.magnitude == "large"


def test_negligible_for_tied_distributions() -> None:
    """Identical distributions -> delta=0 -> negligible."""
    a = [0.95, 0.95, 0.95, 0.95, 0.95]
    b = [0.95, 0.95, 0.95, 0.95, 0.95]
    r = cliffs_delta(a, b)
    assert r.is_negligible
    assert abs(r.delta) < 0.147


def test_large_effect_classification() -> None:
    """Strictly dominant distribution -> magnitude is large."""
    a = [3, 4, 5, 6, 7]
    b = [1, 1, 1, 1, 1]
    r = cliffs_delta(a, b)
    assert r.magnitude == "large"
    assert r.delta == 1.0


def test_empty_inputs_default_to_negligible() -> None:
    r = cliffs_delta([], [1, 2, 3])
    assert r.delta == 0.0
    assert r.magnitude == "negligible"


def test_to_dict_round_trip() -> None:
    r = cliffs_delta([1.0, 2.0], [0.5, 1.5])
    d = r.to_dict()
    for key in ("delta", "magnitude", "n_a", "n_b"):
        assert key in d


def test_is_medium_or_above_predicate() -> None:
    cd = CliffsDelta(delta=0.5, magnitude="large", n_a=5, n_b=5)
    assert cd.is_medium_or_above
    cd2 = CliffsDelta(delta=0.2, magnitude="small", n_a=5, n_b=5)
    assert not cd2.is_medium_or_above
    assert cd2.is_small_or_above
