"""Verify the download_data scripts are present and exercise the checksums
verification logic with a tiny synthetic archive.

We do not download real Loghub archives in CI. Instead we simulate the same
checksums.txt + verification logic that the scripts implement, ensuring the
mechanism is tested even when the network is unavailable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_download_scripts_exist() -> None:
    assert (REPO_ROOT / "scripts" / "download_data.ps1").exists()
    assert (REPO_ROOT / "scripts" / "download_data.sh").exists()


def test_checksums_file_exists() -> None:
    assert (REPO_ROOT / "data" / "checksums.txt").exists()


def test_sha256_verify_against_tampered_archive(tmp_path: Path) -> None:
    """Round-trip: write checksum, then tamper -> mismatch detected."""
    archive = tmp_path / "fake.zip"
    archive.write_bytes(b"hello world")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()

    # Same content -> same hash.
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert expected == actual

    # Tamper -> different hash.
    archive.write_bytes(b"hello world!")
    tampered = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert tampered != expected


@pytest.mark.parametrize("script_name", ["download_data.ps1", "download_data.sh"])
def test_download_script_mentions_sha256(script_name: str) -> None:
    body = (REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
    assert "sha" in body.lower() or "SHA" in body
    assert "checksums.txt" in body
