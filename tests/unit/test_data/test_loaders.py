"""Loader unit tests against the 100-line synthetic fixtures.

Phase 1 checklist item: each loader unit-tested against a 100-line synthetic
fixture — count, label distribution, regex output match.
"""

from __future__ import annotations

from pathlib import Path

from hylog.data.dataset import LogDataset
from hylog.data.loaders import (
    BGLLoader,
    HDFSLoader,
    LoaderConfig,
    OpenStackLoader,
    ThunderbirdLoader,
)


def _assert_no_raw_artifacts(ds: LogDataset) -> None:
    """Every preprocessed line must contain a mask token, not a raw IP/blk_id."""
    for seq in ds:
        for line in seq.lines:
            assert "blk_-" not in line and not _has_raw_ipv4(line), line


def _has_raw_ipv4(text: str) -> bool:
    import re

    return bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))


def test_hdfs_loader_counts(hdfs_paths: tuple[Path, Path]) -> None:
    log_path, label_path = hdfs_paths
    loader = HDFSLoader(label_path=label_path)
    ds = loader.load(log_path)

    # The fixture spans 10 blocks of 10 lines each.
    assert len(ds) == 10
    for seq in ds:
        assert len(seq.lines) == 10
    # 4 anomalies in the fixture.
    assert ds.label_distribution() == {0: 6, 1: 4}
    _assert_no_raw_artifacts(ds)


def test_bgl_loader_counts(bgl_path: Path) -> None:
    cfg = LoaderConfig(window=20, stride=10)
    loader = BGLLoader(config=cfg)
    ds = loader.load(bgl_path)

    # total=100, window=20, stride=10 -> windows at 0,10,20,...,80 -> 9 windows.
    assert len(ds) == 9
    for seq in ds:
        assert len(seq.lines) == 20
    # Each window of 20 contains at least one anomaly token (every 11th line is
    # anomalous: indices 0,11,22,33,44,55,66,77,88,99), so all 9 windows are
    # labelled 1.
    assert ds.label_distribution()[1] >= 8


def test_thunderbird_loader_counts(thunderbird_path: Path) -> None:
    cfg = LoaderConfig(window=25, stride=25)
    loader = ThunderbirdLoader(config=cfg)
    ds = loader.load(thunderbird_path)

    # total=100, window=25, stride=25 -> exactly 4 disjoint windows.
    assert len(ds) == 4
    for seq in ds:
        assert len(seq.lines) == 25


def test_openstack_loader_counts(openstack_paths: tuple[Path, Path]) -> None:
    log_path, label_path = openstack_paths
    loader = OpenStackLoader(label_path=label_path)
    ds = loader.load(log_path)

    assert len(ds) == 5  # 5 instances
    for seq in ds:
        assert len(seq.lines) == 20
    assert ds.label_distribution() == {0: 2, 1: 3}


def test_loader_idempotent(hdfs_paths: tuple[Path, Path]) -> None:
    """Loading twice yields identical sequences."""
    log_path, label_path = hdfs_paths
    a = HDFSLoader(label_path=label_path).load(log_path)
    b = HDFSLoader(label_path=label_path).load(log_path)
    assert list(a) == list(b)
