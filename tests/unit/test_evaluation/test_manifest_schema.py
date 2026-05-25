"""Tests for the run_manifest JSON Schema validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hylog.evaluation.manifest_schema import (
    MANIFEST_SCHEMA,
    ManifestValidationError,
    validate,
    validate_file,
)
from hylog.evaluation.run_manifest import RunManifest


def _valid_minimal_payload() -> dict[str, object]:
    return {
        "run_name": "test",
        "started_at_utc": "2026-05-26T10:00:00Z",
        "finished_at_utc": "2026-05-26T10:05:00Z",
        "wallclock_seconds": 300.0,
        "git": {"available": True, "sha": "a" * 40, "branch": "main", "dirty": False},
        "env": {
            "python_version": "3.11.0",
            "platform": "Linux-x86_64",
            "machine": "x86_64",
            "processor": "x86_64",
            "cpu_count": 16,
            "user": "alice",
            "cuda_available": False,
        },
        "package": {"hylog_version": "0.0.1"},
        "config": {},
    }


def test_minimal_payload_validates() -> None:
    validate(_valid_minimal_payload())


def test_runmanifest_output_validates(tmp_path: Path) -> None:
    """Live test: ``RunManifest.write()`` produces schema-valid JSON."""
    m = RunManifest(run_name="phase7-smoke", config={"k": "v"})
    m.stop()
    path = m.write(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate(payload)


def test_missing_required_key_raises() -> None:
    payload = _valid_minimal_payload()
    del payload["run_name"]
    with pytest.raises(ManifestValidationError) as ei:
        validate(payload)
    assert "run_name" in str(ei.value)


def test_wrong_type_for_run_name_raises() -> None:
    payload = _valid_minimal_payload()
    payload["run_name"] = 42
    with pytest.raises(ManifestValidationError):
        validate(payload)


def test_short_run_name_raises() -> None:
    payload = _valid_minimal_payload()
    payload["run_name"] = ""
    with pytest.raises(ManifestValidationError):
        validate(payload)


def test_splits_hashes_must_be_hex() -> None:
    payload = _valid_minimal_payload()
    payload["splits_hashes"] = {"hdfs.json": "not-a-valid-hex"}
    with pytest.raises(ManifestValidationError):
        validate(payload)


def test_splits_hashes_with_valid_hex() -> None:
    payload = _valid_minimal_payload()
    payload["splits_hashes"] = {"hdfs.json": "f" * 64}
    validate(payload)


def test_validate_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_valid_minimal_payload()), encoding="utf-8")
    validate_file(str(path))


def test_schema_marker_exposed() -> None:
    assert isinstance(MANIFEST_SCHEMA, dict)
    assert MANIFEST_SCHEMA["type"] == "object"


def test_bool_not_accepted_as_integer() -> None:
    """JSON Schema's integer type must exclude booleans."""
    payload = _valid_minimal_payload()
    payload["env"]["cpu_count"] = True
    with pytest.raises(ManifestValidationError):
        validate(payload)


def test_finished_at_utc_can_be_null() -> None:
    payload = _valid_minimal_payload()
    payload["finished_at_utc"] = None
    payload["wallclock_seconds"] = None
    validate(payload)
