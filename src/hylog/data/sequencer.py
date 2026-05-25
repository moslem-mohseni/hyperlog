"""Sliding-window and session-window sequencers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """A window into a raw line list.

    Attributes:
        start: Inclusive 0-based start index in the raw line list.
        end: Exclusive 0-based end index.
        group_id: A stable string key (window index, block id, instance id).
    """

    start: int
    end: int
    group_id: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")

    def __len__(self) -> int:
        return self.end - self.start


def slide_windows(
    total: int,
    window: int,
    stride: int,
    *,
    pad_last: bool = False,
    group_prefix: str = "win",
) -> Iterator[WindowSpec]:
    """Yield fixed-stride sliding windows over ``range(total)``.

    Args:
        total: Number of raw lines.
        window: Window size (lines).
        stride: Stride between successive window starts.
        pad_last: If True and ``total`` is not a multiple of stride, emit a
            final window aligned to ``total`` so every line is covered.
        group_prefix: Prefix for the window id (defaults to ``"win"``).

    Yields:
        WindowSpec instances ordered by their ``start`` field.

    Raises:
        ValueError: if ``window`` or ``stride`` is non-positive.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")

    if total == 0:
        return
    if total < window:
        yield WindowSpec(0, total, f"{group_prefix}_0")
        return

    last_start = -1
    idx = 0
    start = 0
    while start + window <= total:
        yield WindowSpec(start, start + window, f"{group_prefix}_{idx}")
        last_start = start
        start += stride
        idx += 1

    if pad_last and last_start + window < total:
        # Align the final window to the end of the stream.
        final_start = total - window
        if final_start != last_start:
            yield WindowSpec(final_start, total, f"{group_prefix}_{idx}")


def coverage_union(windows: Iterable[WindowSpec]) -> set[int]:
    """Return the set of raw indices covered by the given windows."""
    covered: set[int] = set()
    for w in windows:
        covered.update(range(w.start, w.end))
    return covered


__all__ = ["WindowSpec", "coverage_union", "slide_windows"]
