"""Deterministic split assignment + manifest serialization.

The split policy honors the project's hygiene rules:

- Splits are by **group id**, not by raw line. ``train``, ``val``, and ``test``
  share no group id (the group-disjointness invariant).
- The mapping group_id -> split is computed by hashing the group id with a
  fixed seed and the dataset name, so re-running the loader produces the
  byte-identical manifest required by Phase-1 checklist item #2.
- The manifest is written with sorted keys and a stable indentation so that
  the SHA-256 over the manifest file is reproducible across runs and OSes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hylog.data.schema import SPLITS, LogSequence, SplitName


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        for name, r in (("train", self.train), ("val", self.val), ("test", self.test)):
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"{name} ratio out of range: {r}")


def _stable_bucket(dataset: str, group_id: str, n_buckets: int = 1_000_000) -> int:
    """Map a (dataset, group_id) pair to a deterministic integer bucket.

    Uses SHA-256 to avoid any reliance on Python's hash randomization.
    """
    h = hashlib.sha256(f"{dataset}::{group_id}".encode()).digest()
    # Use the leading 8 bytes as an unsigned integer.
    val = int.from_bytes(h[:8], byteorder="big", signed=False)
    return val % n_buckets


def assign_split(
    dataset: str,
    group_id: str,
    ratios: SplitRatios | None = None,
    n_buckets: int = 1_000_000,
) -> SplitName:
    """Deterministically assign ``group_id`` to one of train/val/test."""
    if ratios is None:
        ratios = SplitRatios()
    bucket = _stable_bucket(dataset, group_id, n_buckets)
    train_cut = round(ratios.train * n_buckets)
    val_cut = train_cut + round(ratios.val * n_buckets)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def partition_by_group(
    sequences: Sequence[LogSequence],
    *,
    dataset: str,
    ratios: SplitRatios | None = None,
) -> dict[SplitName, list[LogSequence]]:
    """Partition ``sequences`` into train/val/test by group id.

    Sequences sharing a group_id are always co-located in the same split.
    """
    if ratios is None:
        ratios = SplitRatios()
    buckets: dict[SplitName, list[LogSequence]] = {s: [] for s in SPLITS}
    for seq in sequences:
        split = assign_split(dataset, seq.group_id, ratios)
        buckets[split].append(seq)
    return buckets


def _serialize_manifest(
    *,
    dataset: str,
    ratios: SplitRatios,
    group_to_split: Mapping[str, SplitName],
    split_stats: Mapping[SplitName, Mapping[str, float | int]],
) -> str:
    """Produce a deterministic, indented JSON string for the manifest."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset,
        "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
        "groups": {gid: group_to_split[gid] for gid in sorted(group_to_split)},
        "stats": {split: dict(sorted(split_stats[split].items())) for split in SPLITS},
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def compute_split_stats(
    partitioned: Mapping[SplitName, Sequence[LogSequence]],
) -> dict[SplitName, dict[str, float | int]]:
    """Compute per-split count + anomaly fraction."""
    out: dict[SplitName, dict[str, float | int]] = {}
    for split in SPLITS:
        items = partitioned.get(split, ())
        n = len(items)
        anomalies = sum(1 for s in items if s.label == 1)
        frac = (anomalies / n) if n else 0.0
        out[split] = {
            "count": n,
            "anomalies": anomalies,
            "anomaly_fraction": round(frac, 6),
        }
    return out


def write_manifest(
    *,
    dataset: str,
    sequences: Iterable[LogSequence],
    out_dir: Path,
    ratios: SplitRatios | None = None,
) -> Path:
    """Write the split manifest for ``dataset`` to ``out_dir/dataset.json``.

    Returns the path to the written manifest. The function is deterministic:
    calling it twice with the same inputs produces a byte-identical file.
    """
    if ratios is None:
        ratios = SplitRatios()
    seq_list = list(sequences)
    group_to_split: dict[str, SplitName] = {}
    for seq in seq_list:
        split = assign_split(dataset, seq.group_id, ratios)
        # Sanity check: a given group_id must always hash to the same split.
        prev = group_to_split.setdefault(seq.group_id, split)
        if prev != split:
            raise AssertionError(f"group {seq.group_id!r} mapped to both {prev} and {split}")

    partitioned = partition_by_group(seq_list, dataset=dataset, ratios=ratios)
    stats = compute_split_stats(partitioned)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{dataset}.json"
    body = _serialize_manifest(
        dataset=dataset,
        ratios=ratios,
        group_to_split=group_to_split,
        split_stats=stats,
    )
    target.write_text(body, encoding="utf-8", newline="\n")
    return target


def manifest_sha256(path: Path) -> str:
    """SHA-256 of the manifest file, computed over its raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "SplitRatios",
    "assign_split",
    "compute_split_stats",
    "manifest_sha256",
    "partition_by_group",
    "write_manifest",
]
