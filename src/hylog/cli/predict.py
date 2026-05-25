"""``hylog-predict`` entry point.

Phase 0 stub. Real implementation lands in Phase 8.
"""

from __future__ import annotations

import sys

import click


@click.command(name="hylog-predict")
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=False, dir_okay=True),
    required=False,
    help="Path to a trained HyLog model directory.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=False, dir_okay=False),
    required=False,
    help="JSONL file of log sequences to score.",
)
def main(model: str | None, input_path: str | None) -> None:
    """Score log sequences with a trained HyLog model. Stub during Phase 0."""
    click.echo(
        "hylog-predict: inference is not yet implemented (Phase 8 deliverable). "
        f"Received model={model!r}, input={input_path!r}."
    )
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
