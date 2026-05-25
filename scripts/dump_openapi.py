"""Dump the FastAPI OpenAPI spec to a committed JSON file.

Phase-8 checklist: ``OpenAPI spec auto-generated and committed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hylog.inference.server import ServerConfig, create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/phase8/openapi.json"),
    )
    args = parser.parse_args(argv)

    app = create_app(ServerConfig())
    spec = app.openapi()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"OpenAPI spec written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
