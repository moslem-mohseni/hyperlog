"""OpenStack log loader.

OpenStack log lines reference an instance id in square brackets, e.g.::

    nova-compute.log.2017-05-16_13:53:08 2017-05-16 00:00:04.562 25746 INFO ... [instance: abc-123-def] ...

The loader groups lines by instance id and treats labels as per-instance. If
the optional ``label_path`` (CSV with ``InstanceId,Label`` rows) is missing
all instances are treated as normal.
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

_INSTANCE_RE = re.compile(r"\[instance:\s*([^\]]+)\]")

_MAX_SEQUENCE = 100  # roadmap §3.3: OpenStack instances truncated to 100 lines


class OpenStackLoader(BaseLogLoader):
    """Session-windowed loader for OpenStack instance logs."""

    source = "openstack"

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
        if self._labels is not None:
            return self._labels
        if self.label_path is None or not self.label_path.exists():
            self._labels = {}
            return self._labels

        mapping: dict[str, int] = {}
        with self.label_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            missing_header = not header or header[0].lower() != "instanceid"
            if missing_header and header is not None and len(header) >= 2:
                mapping[header[0].strip()] = 1 if header[1].strip().lower() == "anomaly" else 0
            for row in reader:
                if len(row) < 2:
                    continue
                mapping[row[0].strip()] = 1 if row[1].strip().lower() == "anomaly" else 0
        self._labels = mapping
        return self._labels

    def _iter_raw(self, path: Path) -> Iterator[tuple[int, str]]:
        with path.open(encoding=self.config.encoding, errors=self.config.encoding_errors) as fh:
            for line in fh:
                yield 0, line

    def _build_sequences(self, raw: Iterable[tuple[int, str]]) -> Iterator[LogSequence]:
        labels = self._load_labels()
        by_instance: dict[str, list[str]] = defaultdict(list)
        for _, line in raw:
            match = _INSTANCE_RE.search(line)
            if not match:
                continue
            instance_id = match.group(1).strip()
            by_instance[instance_id].append(self.preprocessor.preprocess(line))

        for instance_id in sorted(by_instance):
            lines = tuple(by_instance[instance_id][:_MAX_SEQUENCE])
            if not lines:
                continue
            label = labels.get(instance_id, 0)
            yield LogSequence(
                lines=lines,
                label=label,
                group_id=instance_id,
                source=self.source,
            )


__all__ = ["OpenStackLoader"]
