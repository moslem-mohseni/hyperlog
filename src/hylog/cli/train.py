"""``hylog-train`` entry point.

Phase 0 stub. Real implementation lands in Phase 3.
"""

from __future__ import annotations

import sys

import click


@click.command(name="hylog-train")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Path to a Hydra/OmegaConf training config.",
)
@click.option("--dry-run", is_flag=True, help="Validate the config and exit without training.")
def main(config: str | None, dry_run: bool) -> None:
    """Train a HyLog model. Stub during Phase 0."""
    if dry_run:
        click.echo("hylog-train: dry-run OK (Phase 0 stub).")
        return
    click.echo(
        "hylog-train: training is not yet implemented (Phase 3 deliverable). "
        f"Received config={config!r}."
    )
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
