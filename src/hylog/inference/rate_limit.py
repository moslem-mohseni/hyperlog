"""Token-bucket rate limiter, per API-key.

Phase-8 §11.9 security checklist: limit each client to a configurable
requests-per-minute. The implementation is in-process (no Redis, no
external state store) which is sufficient for the single-pod
deployment topology HyLog targets at v1.0.0.

For multi-pod deployments, the same interface accepts a custom
``Clock`` and ``BucketStore`` so a future Redis-backed adapter is a
swap-in.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], float]


@dataclass(slots=True)
class _Bucket:
    """One client's token bucket."""

    tokens: float
    last_refill: float


@dataclass(slots=True)
class TokenBucketLimiter:
    """Per-key token-bucket rate limiter.

    Attributes:
        capacity: Maximum burst size (tokens).
        refill_per_second: Sustained rate of token replenishment.
        clock: Monotonic-clock callable. Override in tests.
    """

    capacity: float = 100.0
    refill_per_second: float = 100.0 / 60.0  # ~100 requests / minute
    clock: Clock = time.monotonic
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")

    def allow(self, key: str, *, cost: float = 1.0) -> bool:
        """Return True if ``cost`` tokens are available for ``key``."""
        if cost <= 0:
            raise ValueError("cost must be positive")
        now = self.clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, last_refill=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_refill)
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
                bucket.last_refill = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True
            return False

    def reset(self, key: str) -> None:
        """Clear one bucket. Used by tests and admin endpoints."""
        with self._lock:
            self._buckets.pop(key, None)

    def state_snapshot(self) -> dict[str, dict[str, float]]:
        """Diagnostic-only view of every bucket."""
        with self._lock:
            return {
                key: {"tokens": b.tokens, "last_refill": b.last_refill}
                for key, b in self._buckets.items()
            }


__all__ = ["Clock", "TokenBucketLimiter"]
