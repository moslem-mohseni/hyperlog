"""HDFS log loader.

HDFS logs use session-window grouping by block id. Each raw line has the form::

    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-160... src: ...

Labels are not embedded in the log lines themselves; they live in a sibling
``anomaly_label.csv`` file with ``BlockId,Label`` rows (``Anomaly``/``Normal``).

The loader yields one :class:`LogSequence` per block id, with the lines for
that block in their original arrival order. This matches the LogLLM
convention.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from hylog.data.loaders._base import BaseLogLoader, LoaderConfig
from hylog.data.preprocess import Preprocessor
from hylog.data.schema import LogSequence

_BLOCK_RE = re.compile(r"blk_-?\d+")


class HDFSLoader(BaseLogLoader):
    """Session-windowed loader for HDFS."""

    source = "hdfs"

    def __init__(
        self,
        label_path: Path | str | None = None,
        config: LoaderConfig | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        super().__init__(config, preprocessor)
        self.label_path = Path(label_path) if label_path is not None else None
        self._labels: dict[str, int] | None = None

    def _load_labels(self) -> dict[str, int]:
        """Lazily load the ``anomaly_label.csv`` map."""
        if self._labels is not None:
            return self._labels
        if self.label_path is None or not self.label_path.exists():
            # No labels available — treat everything as normal. Useful for
            # smoke tests and inference-only runs.
            self._labels = {}
            return self._labels

        mapping: dict[str, int] = {}
        with self.label_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header and header[0].lower() == "blockid":
                pass
            else:
                # Header missing — treat the first row as data.
                if header is not None:
                    mapping[header[0]] = 1 if header[1].lower() == "anomaly" else 0
            for row in reader:
                if len(row) < 2:
                    continue
                mapping[row[0]] = 1 if row[1].strip().lower() == "anomaly" else 0
        self._labels = mapping
        return self._labels

    def _iter_raw(self, path: Path) -> Iterator[tuple[int, str]]:
        # HDFS labels live per-block, not per-line, so we emit label=0 for
        # every raw line and resolve the true label in _build_sequences.
        with path.open(encoding=self.config.encoding, errors=self.config.encoding_errors) as fh:
            for line in fh:
                yield 0, line

    def _build_sequences(self, raw: Iterable[tuple[int, str]]) -> Iterator[LogSequence]:
        labels = self._load_labels()
        by_block: dict[str, list[str]] = defaultdict(list)
        for _, line in raw:
            match = _BLOCK_RE.search(line)
            if not match:
                continue
            by_block[match.group(0)].append(self.preprocessor.preprocess(line))

        # Stable, deterministic ordering by block id so the resulting dataset
        # is byte-reproducible even though dict iteration is now insertion
        # ordered (we sort for safety).
        for block_id in sorted(by_block):
            lines = tuple(by_block[block_id])
            if not lines:
                continue
            label = labels.get(block_id, 0)
            yield LogSequence(
                lines=lines,
                label=label,
                group_id=block_id,
                source=self.source,
            )


__all__ = ["HDFSLoader"]
