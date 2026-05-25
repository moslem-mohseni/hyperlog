"""JSON Schema for the run_manifest.json artefact.

The Phase-7 checklist demands that every ``run_manifest.json`` written
by ``RunManifest.write()`` (Phase 4) round-trips through a strict JSON
Schema. The schema lives here in source so the contract is versioned
alongside the code that emits it. The validator is hand-written —
zero external dependencies — so it works on any clean Windows machine
without ``jsonschema`` being installed.

A round-trip:

  - run_manifest.json is written by ``RunManifest.write()``.
  - ``validate(payload)`` either returns silently or raises
    ``ManifestValidationError`` with a path-prefixed message.
  - The CI workflow runs ``validate_file(path)`` over every freshly
    produced manifest as part of the green gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1


# The schema is a Mapping so callers can persist it alongside the
# manifest payload. Keep the dict literal small and human-readable.
MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HyLog run manifest",
    "type": "object",
    "required": [
        "run_name",
        "started_at_utc",
        "env",
        "git",
        "package",
        "config",
    ],
    "properties": {
        "run_name": {"type": "string", "minLength": 1},
        "started_at_utc": {"type": "string", "minLength": 1},
        "finished_at_utc": {"type": ["string", "null"]},
        "wallclock_seconds": {"type": ["number", "null"]},
        "git": {
            "type": "object",
            "required": ["available"],
            "properties": {
                "available": {"type": "boolean"},
                "sha": {"type": ["string", "null"]},
                "branch": {"type": ["string", "null"]},
                "dirty": {"type": ["boolean", "null"]},
            },
        },
        "env": {
            "type": "object",
            "required": ["python_version", "platform"],
            "properties": {
                "python_version": {"type": "string"},
                "platform": {"type": "string"},
                "machine": {"type": "string"},
                "processor": {"type": "string"},
                "cpu_count": {"type": ["integer", "null"]},
                "user": {"type": "string"},
                "cuda_available": {"type": "boolean"},
                "torch_version": {"type": "string"},
                "gpu": {"type": ["object", "null"]},
            },
        },
        "package": {
            "type": "object",
            "properties": {
                "hylog_version": {"type": ["string", "null"]},
            },
        },
        "config": {"type": "object"},
        "splits_hashes": {
            "type": "object",
            "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    },
}


class ManifestValidationError(ValueError):
    """Raised when ``validate`` finds a structural problem."""


# ---------------------------------------------------------------------------
# Hand-rolled validator (subset of JSON Schema)
# ---------------------------------------------------------------------------


_PRIMITIVE_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (Mapping,),
    "array": (list, tuple),
    "null": (type(None),),
}


def _check_type(value: object, expected: str | Sequence[str], path: str) -> None:
    types = [expected] if isinstance(expected, str) else list(expected)
    for t in types:
        if t == "null":
            if value is None:
                return
        else:
            py_t = _PRIMITIVE_TYPES.get(t)
            if py_t is None:
                continue
            if isinstance(value, py_t) and not (t == "integer" and isinstance(value, bool)):
                # ``bool`` is a subclass of ``int`` in Python; the JSON Schema
                # ``integer`` type explicitly excludes booleans.
                return
    raise ManifestValidationError(f"{path}: expected type {expected!r}, got {type(value).__name__}")


def _validate_subschema(value: object, schema: Mapping[str, Any], path: str) -> None:
    if "type" in schema:
        _check_type(value, schema["type"], path)
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        raise ManifestValidationError(
            f"{path}: string shorter than minLength {schema['minLength']}"
        )
    if "pattern" in schema and isinstance(value, str):
        import re

        if not re.search(schema["pattern"], value):
            raise ManifestValidationError(
                f"{path}: string does not match pattern {schema['pattern']!r}"
            )

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ManifestValidationError(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, sub in properties.items():
            if key in value:
                _validate_subschema(value[key], sub, f"{path}.{key}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping):
            for key, v in value.items():
                if key in properties:
                    continue
                _validate_subschema(v, additional, f"{path}.{key}")


def validate(payload: Mapping[str, Any]) -> None:
    """Validate ``payload`` against ``MANIFEST_SCHEMA``.

    Raises ``ManifestValidationError`` with a path-prefixed message on
    failure. Returns ``None`` on success.
    """
    _validate_subschema(payload, MANIFEST_SCHEMA, path="$")


def validate_file(path: str) -> None:
    """Read a JSON file and validate it against the manifest schema."""
    import json
    from pathlib import Path

    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    validate(payload)


__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA_VERSION",
    "ManifestValidationError",
    "validate",
    "validate_file",
]
