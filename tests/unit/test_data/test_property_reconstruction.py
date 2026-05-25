"""Property test: union of sliding windows reconstructs the raw file up to
the stride boundary.

Phase 1 checklist item: "Property test: union of sliding windows reconstructs
the raw file up to stride."
"""

from __future__ import annotations

from hylog.data.sequencer import coverage_union, slide_windows


def _full_coverage(total: int, window: int, stride: int) -> bool:
    """True iff the union of (non-padded) sliding windows covers the entire
    range [0, total).
    """
    wins = list(slide_windows(total=total, window=window, stride=stride))
    return coverage_union(wins) == set(range(total))


def test_full_coverage_when_total_is_aligned() -> None:
    # total - window is a multiple of stride -> full coverage without padding.
    assert _full_coverage(total=20, window=5, stride=5)
    assert _full_coverage(total=21, window=4, stride=1)


def test_tail_recovered_with_pad_last() -> None:
    for total, window, stride in [(11, 4, 3), (53, 10, 4), (17, 6, 5)]:
        wins = list(slide_windows(total=total, window=window, stride=stride, pad_last=True))
        assert coverage_union(wins) == set(range(total)), (total, window, stride)


def test_no_overlap_when_stride_equals_window() -> None:
    wins = list(slide_windows(total=40, window=8, stride=8))
    indices = sorted(coverage_union(wins))
    assert indices == list(range(40))
    # Windows tile exactly.
    assert len(wins) == 5
    for w in wins:
        assert (w.end - w.start) == 8


def test_overlap_count_when_stride_lt_window() -> None:
    wins = list(slide_windows(total=20, window=10, stride=5))
    # Every interior position should be in exactly 2 windows.
    counts: dict[int, int] = {}
    for w in wins:
        for i in range(w.start, w.end):
            counts[i] = counts.get(i, 0) + 1
    # First 5 and last 5 indices appear once, middle 10 appear twice.
    assert counts[0] == 1
    assert counts[5] == 2
    assert counts[14] == 2
