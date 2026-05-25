"""Fixture file paths shared across data-layer tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture(scope="session")
def hdfs_paths(fixtures_dir: Path) -> tuple[Path, Path]:
    return fixtures_dir / "hdfs" / "HDFS.log", fixtures_dir / "hdfs" / "anomaly_label.csv"


@pytest.fixture(scope="session")
def bgl_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "bgl" / "BGL.log"


@pytest.fixture(scope="session")
def thunderbird_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "thunderbird" / "Thunderbird.log"


@pytest.fixture(scope="session")
def openstack_paths(fixtures_dir: Path) -> tuple[Path, Path]:
    base = fixtures_dir / "openstack"
    return base / "openstack.log", base / "anomaly_label.csv"
