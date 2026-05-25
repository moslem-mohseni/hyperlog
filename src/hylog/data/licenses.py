"""Emit a deterministic ``LICENSES.txt`` notice from ``data/licenses.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def render_notice(spec: Mapping[str, object]) -> str:
    """Render the canonical LICENSES.txt body from the YAML spec."""
    lines: list[str] = []
    lines.append("HyLog — Dataset License Attribution")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "HyLog does not redistribute raw datasets. The download_data scripts "
        "fetch each archive from its upstream source at run time. The "
        "attributions below are required reading before any redistribution."
    )
    lines.append("")
    datasets = spec.get("datasets", [])
    if not isinstance(datasets, list):
        raise TypeError("licenses.yaml: 'datasets' must be a list")
    for entry in datasets:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "<unknown>"))
        lines.append(f"## {name}")
        for key in ("upstream", "upstream_v2"):
            if key in entry:
                lines.append(f"  {key}: {entry[key]}")
        if "citation" in entry:
            lines.append("  citation:")
            for cl in str(entry["citation"]).rstrip().splitlines():
                lines.append(f"    {cl}")
        if "redistribution" in entry:
            lines.append("  redistribution:")
            for rl in str(entry["redistribution"]).rstrip().splitlines():
                lines.append(f"    {rl}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def emit(yaml_path: Path | str, out_path: Path | str) -> Path:
    """Read ``yaml_path`` and write the rendered notice to ``out_path``."""
    yp = Path(yaml_path)
    op = Path(out_path)
    spec = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
    body = render_notice(spec)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(body, encoding="utf-8", newline="\n")
    return op


__all__ = ["emit", "render_notice"]
