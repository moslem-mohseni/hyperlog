"""``hylog-loso`` — run the LOSO protocol from a Hydra config.

This CLI is the single-command entry point a reviewer (or a future
maintainer) uses to reproduce HyLog's cross-system numbers.

Phase 4 mechanics path (CPU): the CLI accepts ``--mock`` so the LOSO
orchestrator can be exercised end-to-end on the synthetic fixtures
without a GPU. The real-data path requires the GPU stack and is
delegated to Phase 3's trainer (Phase 5 will wire it in).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import yaml

from hylog.data.dataset import LogDataset
from hylog.data.loaders import (
    BGLLoader,
    HDFSLoader,
    LoaderConfig,
    OpenStackLoader,
    ThunderbirdLoader,
)
from hylog.evaluation.cross_system import run_loso

_FIXTURE_BASE = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def _load_fixture_dataset(name: str) -> LogDataset:
    """Convenience loader for the CPU mock path."""
    cfg_small = LoaderConfig(window=20, stride=20)
    if name == "hdfs":
        return HDFSLoader(label_path=_FIXTURE_BASE / "hdfs" / "anomaly_label.csv").load(
            _FIXTURE_BASE / "hdfs" / "HDFS.log"
        )
    if name == "bgl":
        return BGLLoader(config=cfg_small).load(_FIXTURE_BASE / "bgl" / "BGL.log")
    if name == "thunderbird":
        return ThunderbirdLoader(config=cfg_small).load(
            _FIXTURE_BASE / "thunderbird" / "Thunderbird.log"
        )
    if name == "openstack":
        return OpenStackLoader(label_path=_FIXTURE_BASE / "openstack" / "anomaly_label.csv").load(
            _FIXTURE_BASE / "openstack" / "openstack.log"
        )
    raise click.BadParameter(f"unknown dataset {name!r}")


def _mock_trainer(*, train, test, target_system):
    """A deterministic mock trainer for CLI smoke runs.

    It predicts the majority class of the training set and assigns a
    fixed score of 0.5 to every test sequence. Useful for verifying the
    LOSO machinery end-to-end without a GPU.
    """
    train_labels = [s.label for s in train]
    majority = 1 if sum(train_labels) > len(train_labels) / 2 else 0
    y_pred = np.full(len(test), majority, dtype=np.int64)
    y_score = np.full(len(test), 0.5, dtype=np.float64)
    return y_pred, y_score


@click.command(name="hylog-loso")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Hydra YAML config (e.g. configs/experiments/loso_hdfs_held.yaml).",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("reports/phase4/runs"),
    show_default=True,
    help="Where to archive the run.",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use the deterministic mock trainer on the synthetic fixtures "
    "(CPU-only). When omitted, the real trainer is required (Phase 5+).",
)
@click.option(
    "--leakage-strict/--no-leakage-strict",
    default=True,
    help="Abort on any leakage (default) or continue with audit archived.",
)
@click.option(
    "--bootstrap-n",
    type=int,
    default=1000,
    show_default=True,
    help="Number of bootstrap resamples per fold (>= 100).",
)
def main(
    config: Path,
    out_dir: Path,
    mock: bool,
    leakage_strict: bool,
    bootstrap_n: int,
) -> None:
    """Run a LOSO experiment driven by a Hydra config."""
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    protocol = payload.get("protocol", {})
    held_out = protocol.get("held_out_system")
    train_sources = protocol.get("train_sources", [])
    run_name = payload.get("run", {}).get("name", config.stem)

    if not held_out or not train_sources:
        click.echo("config is missing protocol.held_out_system or protocol.train_sources")
        sys.exit(2)

    if not mock:
        click.echo(
            "hylog-loso: the real-trainer path is implemented in Phase 5+; "
            "rerun with --mock for the CPU smoke run."
        )
        sys.exit(0)

    needed = {held_out, *train_sources}
    datasets = {name: _load_fixture_dataset(name) for name in sorted(needed)}

    fold_out = out_dir / run_name
    fold_out.mkdir(parents=True, exist_ok=True)

    summary = run_loso(
        datasets=datasets,
        trainer=_mock_trainer,
        out_dir=fold_out,
        include=[held_out],
        leakage_strict=leakage_strict,
        bootstrap_n=bootstrap_n,
        run_name=run_name,
        splits_dir=Path("splits"),
        config_snapshot=payload,
    )
    summary_path = fold_out / "summary.json"
    click.echo(
        json.dumps(
            {
                "run_name": run_name,
                "held_out": held_out,
                "train_sources": train_sources,
                "summary_json": str(summary_path),
                "macro": summary.get("macro", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
