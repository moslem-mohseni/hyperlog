"""``LogDataset`` — the per-system tensor/sequence abstraction.

A ``LogDataset`` is constructed by a per-system loader (HDFS, BGL, …) and
behaves as a sized, ordered collection of ``LogSequence`` items. It is the
input substrate for the Phase-3 trainer and for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hylog.data.schema import LogSequence


class LogDataset(Sequence[LogSequence]):
    """An immutable ordered list of ``LogSequence`` items.

    Implements ``collections.abc.Sequence`` so it works with PyTorch's
    ``Dataset`` protocol without inheriting from ``torch.utils.data.Dataset``.
    The Phase-0 / Phase-1 layer intentionally does not import torch — that
    dependency is introduced only at training time.
    """

    __slots__ = ("_items", "_source")

    def __init__(self, items: Sequence[LogSequence], *, source: str) -> None:
        self._items: tuple[LogSequence, ...] = tuple(items)
        self._source = source

    @property
    def source(self) -> str:
        return self._source

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[LogSequence]:
        return iter(self._items)

    def __getitem__(self, index: int) -> LogSequence:  # type: ignore[override]
        return self._items[index]

    def label_distribution(self) -> dict[int, int]:
        """Return a {label: count} histogram."""
        counts = {0: 0, 1: 0}
        for seq in self._items:
            counts[seq.label] += 1
        return counts

    def anomaly_fraction(self) -> float:
        """Fraction of sequences labelled 1 (anomaly)."""
        if not self._items:
            return 0.0
        return self.label_distribution()[1] / len(self._items)

    def group_ids(self) -> tuple[str, ...]:
        """All group ids in dataset order. May contain duplicates if a single
        group spans multiple sequences."""
        return tuple(seq.group_id for seq in self._items)


__all__ = ["LogDataset"]
