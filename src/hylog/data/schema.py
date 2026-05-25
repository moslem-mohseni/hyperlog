"""Common typed schema for the data layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class LogSequence:
    """A sequence of log lines with a binary label and a group key.

    Attributes:
        lines: Preprocessed log lines (already passed through Preprocessor).
        label: 1 for anomaly, 0 for normal.
        group_id: The grouping key used for split assignment. For HDFS this is
            the block id; for BGL / Thunderbird this is the window index; for
            OpenStack it is the instance id. The group id is the unit of
            split-disjointness checks.
        source: Free-form identifier of the originating dataset (e.g. ``hdfs``).
    """

    lines: tuple[str, ...]
    label: int
    group_id: str
    source: str

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if not self.lines:
            raise ValueError("LogSequence.lines must be non-empty")

    def __len__(self) -> int:
        return len(self.lines)
