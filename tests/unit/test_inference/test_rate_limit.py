"""Tests for the token-bucket limiter."""

from __future__ import annotations

import pytest

from hylog.inference.rate_limit import TokenBucketLimiter


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_initial_capacity_allows_burst() -> None:
    limiter = TokenBucketLimiter(capacity=5.0, refill_per_second=1.0)
    for _ in range(5):
        assert limiter.allow("alice")
    assert not limiter.allow("alice")


def test_refill_over_time() -> None:
    clock = _FakeClock()
    limiter = TokenBucketLimiter(capacity=2.0, refill_per_second=1.0, clock=clock)
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")
    clock.advance(2.0)
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")


def test_different_keys_independent() -> None:
    limiter = TokenBucketLimiter(capacity=1.0, refill_per_second=1.0)
    assert limiter.allow("alice")
    assert not limiter.allow("alice")
    assert limiter.allow("bob")


def test_reset_clears_state() -> None:
    limiter = TokenBucketLimiter(capacity=1.0, refill_per_second=1.0)
    assert limiter.allow("k")
    assert not limiter.allow("k")
    limiter.reset("k")
    assert limiter.allow("k")


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(capacity=0)


def test_invalid_cost_raises() -> None:
    limiter = TokenBucketLimiter(capacity=10.0)
    with pytest.raises(ValueError):
        limiter.allow("k", cost=0)


def test_state_snapshot_returns_nondestructive_view() -> None:
    limiter = TokenBucketLimiter(capacity=5.0)
    limiter.allow("alice")
    snap = limiter.state_snapshot()
    assert "alice" in snap
    assert snap["alice"]["tokens"] < 5.0
