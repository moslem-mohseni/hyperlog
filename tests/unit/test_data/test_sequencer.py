"""Tests for the sliding-window sequencer."""

from __future__ import annotations

import pytest

from hylog.data.sequencer import coverage_union, slide_windows


def test_basic_window_emission() -> None:
    wins = list(slide_windows(total=10, window=4, stride=2))
    assert [(w.start, w.end) for w in wins] == [
        (0, 4),
        (2, 6),
        (4, 8),
        (6, 10),
    ]


def test_no_windows_when_total_zero() -> None:
    assert list(slide_windows(total=0, window=5, stride=1)) == []


def test_single_window_when_total_lt_window() -> None:
    wins = list(slide_windows(total=3, window=10, stride=2))
    assert len(wins) == 1
    assert (wins[0].start, wins[0].end) == (0, 3)


def test_pad_last_covers_tail() -> None:
    wins = list(slide_windows(total=11, window=4, stride=3, pad_last=True))
    coverage = coverage_union(wins)
    assert coverage == set(range(11))


def test_no_pad_may_drop_tail() -> None:
    wins = list(slide_windows(total=11, window=4, stride=3))
    coverage = coverage_union(wins)
    # 0..3, 3..7, 6..10 — index 10 only included if pad_last True.
    assert 10 not in coverage


def test_group_ids_are_unique_and_ordered() -> None:
    wins = list(slide_windows(total=30, window=5, stride=5))
    ids = [w.group_id for w in wins]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda s: int(s.rsplit("_", 1)[1]))


def test_invalid_window() -> None:
    with pytest.raises(ValueError):
        list(slide_windows(total=10, window=0, stride=1))


def test_invalid_stride() -> None:
    with pytest.raises(ValueError):
        list(slide_windows(total=10, window=4, stride=0))


def test_invalid_total() -> None:
    with pytest.raises(ValueError):
        list(slide_windows(total=-1, window=4, stride=1))


def test_property_union_covers_prefix_up_to_stride() -> None:
    # Property: windows starting at 0, stride, 2*stride, ... cover the prefix
    # up to (n_full * stride + window). The union over all emitted windows
    # equals exactly that prefix.
    total, window, stride = 53, 10, 4
    wins = list(slide_windows(total=total, window=window, stride=stride))
    covered = coverage_union(wins)
    expected_max = wins[-1].end if wins else 0
    assert covered == set(range(expected_max))
