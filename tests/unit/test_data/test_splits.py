"""Split-manifest tests.

Covers Phase 1 checklist items:
- byte-identical split manifests across two runs (SHA-256);
- group-disjointness invariant: no group_id appears in two splits.
"""

from __future__ import annotations

from pathlib import Path

from hylog.data.dataset import LogDataset
from hylog.data.loaders import BGLLoader, HDFSLoader, LoaderConfig
from hylog.data.schema import LogSequence
from hylog.data.splits import (
    SplitRatios,
    manifest_sha256,
    partition_by_group,
    write_manifest,
)


def _synthetic_dataset(n_groups: int = 64) -> LogDataset:
    items = [
        LogSequence(
            lines=("a <NUM>", "b <NUM>"),
            label=(i % 5 == 0),
            group_id=f"grp_{i:04d}",
            source="synthetic",
        )
        for i in range(n_groups)
    ]
    return LogDataset(items, source="synthetic")


def test_partition_disjoint_groups_synthetic() -> None:
    ds = _synthetic_dataset(128)
    parts = partition_by_group(list(ds), dataset="synthetic")
    train = {s.group_id for s in parts["train"]}
    val = {s.group_id for s in parts["val"]}
    test = {s.group_id for s in parts["test"]}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_split_ratios_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        SplitRatios(train=0.5, val=0.5, test=0.5)


def test_manifest_byte_identical_two_runs(tmp_path: Path) -> None:
    ds = _synthetic_dataset()
    a = write_manifest(dataset="synthetic", sequences=ds, out_dir=tmp_path / "a")
    b = write_manifest(dataset="synthetic", sequences=ds, out_dir=tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()
    assert manifest_sha256(a) == manifest_sha256(b)


def test_manifest_disjoint_invariant_in_payload(tmp_path: Path) -> None:
    import json

    ds = _synthetic_dataset(50)
    path = write_manifest(dataset="synthetic", sequences=ds, out_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, str] = payload["groups"]
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for gid, split in groups.items():
        by_split[split].add(gid)
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_manifest_includes_stats(tmp_path: Path) -> None:
    import json

    ds = _synthetic_dataset(40)
    path = write_manifest(dataset="synthetic", sequences=ds, out_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert {"train", "val", "test"} <= set(payload["stats"])
    for stats in payload["stats"].values():
        assert {"count", "anomalies", "anomaly_fraction"} <= set(stats)


def test_real_loader_manifest_disjoint(
    tmp_path: Path,
    hdfs_paths: tuple[Path, Path],
    bgl_path: Path,
) -> None:
    """Run manifest writer over the real loader output (still synthetic data)."""
    hdfs = HDFSLoader(label_path=hdfs_paths[1]).load(hdfs_paths[0])
    bgl = BGLLoader(config=LoaderConfig(window=20, stride=10)).load(bgl_path)
    for ds in (hdfs, bgl):
        path = write_manifest(dataset=ds.source, sequences=ds, out_dir=tmp_path)
        import json

        groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
        by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for gid, split in groups.items():
            by_split[split].add(gid)
        assert by_split["train"].isdisjoint(by_split["val"])
        assert by_split["train"].isdisjoint(by_split["test"])
        assert by_split["val"].isdisjoint(by_split["test"])
