"""Run manifest — captures environment + config + timing into a single file.

Phase 4 — but also a general-purpose utility — that records the
following at the start of every LOSO run:

- Wallclock start/end timestamps (UTC) and total seconds.
- Git SHA + branch + dirty flag (best effort; absent on detached
  clones).
- Python version, platform, CPU count, GPU info (best effort).
- The full Hydra config that drove the run (passed in by the caller).
- The SHA-256 of every input split manifest under ``splits/``.
- The HyLog package version (``__version__``).

The manifest is the canonical evidence trail for a reviewer who asks
"how was this number produced?".
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunManifest:
    """A mutable run-state record finalised by ``stop()``."""

    run_name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    splits_dir: Path | None = None
    started_at_utc: float = field(default_factory=time.time)
    finished_at_utc: float | None = None

    def stop(self) -> None:
        self.finished_at_utc = time.time()

    @property
    def wallclock_seconds(self) -> float | None:
        if self.finished_at_utc is None:
            return None
        return float(self.finished_at_utc - self.started_at_utc)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_name": self.run_name,
            "started_at_utc": _utc_iso(self.started_at_utc),
            "finished_at_utc": _utc_iso(self.finished_at_utc) if self.finished_at_utc else None,
            "wallclock_seconds": self.wallclock_seconds,
            "git": _git_info(),
            "env": _env_info(),
            "package": _package_info(),
            "config": dict(self.config),
        }
        if self.splits_dir is not None:
            payload["splits_hashes"] = _hash_splits(self.splits_dir)
        return payload

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return p


def _utc_iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _git_info() -> dict[str, str | bool | None]:
    info: dict[str, str | bool | None] = {
        "available": False,
        "sha": None,
        "branch": None,
        "dirty": None,
    }
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        info["available"] = True
        info["sha"] = sha
        info["branch"] = branch
        info["dirty"] = bool(status)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return info


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "user": _safe_getuser(),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu"] = {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "cuda_runtime": getattr(torch.version, "cuda", None),
            }
    except ImportError:
        pass
    return info


def _safe_getuser() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


def _package_info() -> dict[str, str | None]:
    try:
        import hylog

        return {"hylog_version": getattr(hylog, "__version__", None)}
    except ImportError:
        return {"hylog_version": None}


def _hash_splits(splits_dir: Path) -> dict[str, str]:
    """SHA-256 of every ``.json`` manifest under ``splits_dir``."""
    out: dict[str, str] = {}
    if not splits_dir.exists():
        return out
    for path in sorted(splits_dir.glob("*.json")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


__all__ = ["RunManifest"]
