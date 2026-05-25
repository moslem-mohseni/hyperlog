"""BGL (Blue Gene/L) log loader.

BGL lines start with the label column: ``-`` for normal, anything else for an
anomaly. Example::

    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ... RAS KERNEL INFO ...
    KERNDTLB 1118017028 2005.06.06 R23-M0-N4-C:J04-U01 ... data TLB error ...

The loader groups raw lines into fixed-stride sliding windows. A window is
labelled anomalous if any line inside it is anomalous (LogLLM convention).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from hylog.data.loaders._base import BaseLogLoader
from hylog.data.schema import LogSequence
from hylog.data.sequencer import slide_windows


class BGLLoader(BaseLogLoader):
    """Fixed-stride sliding-window loader for BGL."""

    source = "bgl"

    def _iter_raw(self, path: Path) -> Iterator[tuple[int, str]]:
        with path.open(encoding=self.config.encoding, errors=self.config.encoding_errors) as fh:
            for line in fh:
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                # First whitespace-separated token is the label column.
                first, _, rest = stripped.partition(" ")
                label = 0 if first == "-" else 1
                yield label, rest if rest else stripped

    def _build_sequences(self, raw: Iterable[tuple[int, str]]) -> Iterator[LogSequence]:
        materialized = list(raw)
        if not materialized:
            return
        lines = [self.preprocessor.preprocess(text) for _, text in materialized]
        labels = [lab for lab, _ in materialized]

        for win in slide_windows(
            total=len(lines),
            window=self.config.window,
            stride=self.config.stride,
            pad_last=False,
            group_prefix=f"{self.source}_win",
        ):
            window_lines = tuple(lines[win.start : win.end])
            label = 1 if any(labels[win.start : win.end]) else 0
            yield LogSequence(
                lines=window_lines,
                label=label,
                group_id=win.group_id,
                source=self.source,
            )


class ThunderbirdLoader(BGLLoader):
    """Thunderbird shares BGL's label-prefixed line format."""

    source = "thunderbird"


__all__ = ["BGLLoader", "ThunderbirdLoader"]
