"""Shared base for per-system log loaders."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from hylog.data.dataset import LogDataset
from hylog.data.preprocess import Preprocessor, default_preprocessor
from hylog.data.schema import LogSequence


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    """Common loader knobs."""

    window: int = 100
    stride: int = 20
    max_line_chars: int = 4096
    encoding: str = "utf-8"
    encoding_errors: str = "replace"


class BaseLogLoader(abc.ABC):
    """Base class for per-system loaders.

    Subclasses implement :meth:`_iter_raw` (yielding ``(label, raw_line)``
    pairs in arrival order) and :meth:`_build_sequences` (grouping the raw
    stream into ``LogSequence`` items).
    """

    source: str

    def __init__(
        self,
        config: LoaderConfig | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.config = config or LoaderConfig()
        self.preprocessor = preprocessor or default_preprocessor()

    @abc.abstractmethod
    def _iter_raw(self, path: Path) -> Iterator[tuple[int, str]]:
        """Yield ``(label, raw_line)`` in file order. label ∈ {0, 1}."""

    @abc.abstractmethod
    def _build_sequences(self, raw: Iterable[tuple[int, str]]) -> Iterator[LogSequence]:
        """Group raw lines into LogSequence items."""

    def load(self, path: Path | str) -> LogDataset:
        """Load a dataset from a single raw file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        # Materialize raw stream into memory once, capped per-line length so a
        # pathological 10 MB line cannot blow the loader up.
        raw = [(lab, ln[: self.config.max_line_chars]) for lab, ln in self._iter_raw(p)]
        sequences = list(self._build_sequences(raw))
        return LogDataset(sequences, source=self.source)


__all__ = ["BaseLogLoader", "LoaderConfig"]
