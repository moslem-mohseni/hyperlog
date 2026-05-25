"""Leave-One-System-Out (LOSO) cross-system evaluation orchestrator.

This is the canonical Phase-4 entry point. Given a set of registered
datasets, the orchestrator builds N folds, each holding one system out:

  fold i: train on union of (datasets - {target_i});
          drop every label from the target_i train+val splits;
          test on target_i's test split (labels used only for metrics).

The orchestrator does *not* train a model. Training is delegated to a
``TrainerProtocol`` callable so the LogLLM baseline, the HyLog core,
and any future model share the same LOSO machinery without a
trainer-level if/else.

A single run produces, per fold:

- ``reports/phase4/runs/{run_name}/{target}/metrics.json``  — full panel
- ``reports/phase4/runs/{run_name}/{target}/confusion.{csv,png,txt}``
- ``reports/phase4/runs/{run_name}/{target}/leakage.json`` — audit
- ``reports/phase4/runs/{run_name}/{target}/predictions.jsonl`` — raw

And one aggregated artefact:

- ``reports/phase4/runs/{run_name}/summary.json`` — macro mean ± std.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hylog.data.dataset import LogDataset
from hylog.data.schema import LogSequence
from hylog.evaluation.bootstrap import BootstrapInterval, bootstrap_metric_panel
from hylog.evaluation.confusion_renderer import ConfusionMatrix, archive_all
from hylog.evaluation.curves import archive_curves
from hylog.evaluation.leakage_audit import (
    LeakageError,
    assert_clean,
    audit_leakage,
)
from hylog.evaluation.leakage_audit import (
    write_report as write_leakage_report,
)
from hylog.evaluation.metrics import MetricPanel, compute_metric_panel
from hylog.evaluation.ood_distance import OODDistanceReport, ood_distance
from hylog.evaluation.run_manifest import RunManifest


@dataclass(frozen=True, slots=True)
class LosoFold:
    """A single LOSO fold specification."""

    target_system: str
    train_sources: tuple[str, ...]
    n_train_sequences: int
    n_test_sequences: int


@dataclass(slots=True)
class FoldResult:
    """Per-fold output of the orchestrator."""

    fold: LosoFold
    panel: MetricPanel
    confusion: ConfusionMatrix
    leakage_clean: bool
    n_predictions: int
    artefacts: dict[str, Path] = field(default_factory=dict)
    bootstrap_intervals: dict[str, BootstrapInterval] = field(default_factory=dict)
    ood_distances: list[OODDistanceReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_system": self.fold.target_system,
            "train_sources": list(self.fold.train_sources),
            "n_train_sequences": self.fold.n_train_sequences,
            "n_test_sequences": self.fold.n_test_sequences,
            "metrics": self.panel.to_dict(),
            "confusion": {
                "matrix": self.confusion.matrix,
                "rows_are_true": True,
                "labels": ["normal", "anomaly"],
            },
            "leakage_clean": self.leakage_clean,
            "n_predictions": self.n_predictions,
            "artefacts": {k: str(v) for k, v in self.artefacts.items()},
            "bootstrap_intervals": {
                name: ci.to_dict() for name, ci in self.bootstrap_intervals.items()
            },
            "ood_distances": [d.to_dict() for d in self.ood_distances],
        }


class TrainerProtocol(Protocol):
    """Callable that trains and predicts for one LOSO fold.

    Returning shape: ``(y_pred, y_score)`` where ``y_pred`` is the
    hard-decision per test sequence (1 = anomaly, 0 = normal) and
    ``y_score`` is the per-sequence probability of the anomaly class.
    """

    def __call__(
        self,
        *,
        train: Sequence[LogSequence],
        test: Sequence[LogSequence],
        target_system: str,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def build_folds(
    *,
    datasets: Mapping[str, LogDataset],
    include: Sequence[str] | None = None,
) -> list[LosoFold]:
    """Build N folds rotating over the registered datasets.

    Args:
        datasets: Map system name -> ``LogDataset``. *All* registered
            datasets are available as training sources regardless of the
            ``include`` filter; ``include`` only restricts which systems
            are *held out as targets*.
        include: Optional restriction on which systems become held-out
            targets. ``None`` rotates over every registered dataset.
    """
    all_names = sorted(datasets)
    held_out_names = all_names if include is None else list(include)
    for n in held_out_names:
        if n not in datasets:
            raise KeyError(f"dataset {n!r} not registered")

    folds: list[LosoFold] = []
    for target in held_out_names:
        sources = tuple(n for n in all_names if n != target)
        n_train = sum(len(datasets[s]) for s in sources)
        n_test = len(datasets[target])
        folds.append(
            LosoFold(
                target_system=target,
                train_sources=sources,
                n_train_sequences=n_train,
                n_test_sequences=n_test,
            )
        )
    return folds


def strip_labels(sequences: Sequence[LogSequence]) -> list[LogSequence]:
    """Return copies of ``sequences`` with every label set to 0.

    This is the mechanical realisation of the "zero target labels"
    promise — the function is called on every sequence the orchestrator
    feeds into training when those sequences come from the target system.
    Phase-4 augmentation modes (self-supervised, domain-adversarial)
    consume these stripped sequences.
    """
    return [
        LogSequence(lines=s.lines, label=0, group_id=s.group_id, source=s.source) for s in sequences
    ]


def _aggregate_summary(results: Sequence[FoldResult]) -> dict[str, object]:
    """Compute macro mean/std/min/max per metric across folds."""
    metric_keys = sorted({k for r in results for k in r.panel.to_dict()})
    summary: dict[str, object] = {
        "n_folds": len(results),
        "folds": [r.to_dict() for r in results],
        "macro": {},
    }
    macro: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = []
        for r in results:
            v = r.panel.to_dict().get(key)
            if v is None or (isinstance(v, float) and (v != v)):  # skip NaN
                continue
            values.append(float(v))
        if not values:
            macro[key] = {
                "mean": float("nan"),
                "std": 0.0,
                "min": float("nan"),
                "max": float("nan"),
            }
            continue
        arr = np.asarray(values, dtype=np.float64)
        macro[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
    summary["macro"] = macro
    return summary


def run_loso(
    *,
    datasets: Mapping[str, LogDataset],
    trainer: TrainerProtocol,
    out_dir: Path | str,
    include: Sequence[str] | None = None,
    leakage_strict: bool = True,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
    ood_ngram: int = 2,
    run_name: str = "loso-run",
    splits_dir: Path | str | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Run the full LOSO protocol and write per-fold + summary artefacts.

    Args:
        datasets: Map system name -> ``LogDataset``. Each dataset is
            consumed as a whole; the orchestrator does not re-split.
        trainer: Callable that trains on the source union and predicts
            on the target test set.
        out_dir: Directory under which to archive every fold's artefacts.
        include: Optional restriction. ``None`` = all registered systems.
        leakage_strict: If True (default), a leakage audit failure
            raises ``LeakageError`` and aborts the run. If False, the
            audit still runs and its report is archived but the fold
            continues — used during development to surface and fix
            leakage interactively.

    Returns:
        A summary dictionary identical in structure to the persisted
        ``summary.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        run_name=run_name,
        config=dict(config_snapshot or {}),
        splits_dir=Path(splits_dir) if splits_dir is not None else None,
    )

    folds = build_folds(datasets=datasets, include=include)
    results: list[FoldResult] = []

    for fold in folds:
        fold_dir = out / fold.target_system
        fold_dir.mkdir(parents=True, exist_ok=True)

        train: list[LogSequence] = []
        for src in fold.train_sources:
            train.extend(datasets[src])
        test = list(datasets[fold.target_system])

        # ---- Mandatory leakage audit ----
        report = audit_leakage(train=train, test=test)
        write_leakage_report(report, fold_dir / "leakage.json")
        if not report.is_clean and leakage_strict:
            assert_clean(report)  # raises LeakageError
        leakage_clean = report.is_clean

        # ---- Trainer callback ----
        y_pred, y_score = trainer(train=train, test=test, target_system=fold.target_system)
        y_true = np.asarray([s.label for s in test], dtype=np.int64)
        if y_pred.shape != y_true.shape:
            raise RuntimeError(
                f"trainer returned predictions of shape {y_pred.shape}; expected {y_true.shape}"
            )

        panel = compute_metric_panel(y_true=y_true, y_pred=y_pred, y_score=y_score)
        confusion = ConfusionMatrix.from_arrays(y_true, y_pred)
        artefacts = archive_all(
            confusion,
            out_dir=fold_dir,
            name="confusion",
            title=f"Confusion — held-out {fold.target_system}",
        )
        artefacts["leakage_json"] = fold_dir / "leakage.json"

        # ---- Curves (ROC + PR) ----
        curve_artefacts = archive_curves(
            y_true=y_true,
            y_score=y_score,
            out_dir=fold_dir,
            fold_name=fold.target_system,
        )
        artefacts.update(curve_artefacts)

        # ---- Bootstrap CIs ----
        intervals = bootstrap_metric_panel(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            n_bootstrap=bootstrap_n,
            seed=bootstrap_seed,
        )
        (fold_dir / "bootstrap.json").write_text(
            json.dumps(
                {name: ci.to_dict() for name, ci in intervals.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        artefacts["bootstrap_json"] = fold_dir / "bootstrap.json"

        # ---- OOD distance: source mixture vs target ----
        ood_reports: list[OODDistanceReport] = []
        for src_name in fold.train_sources:
            report_d = ood_distance(
                source_sequences=datasets[src_name],
                target_sequences=test,
                source_system=src_name,
                target_system=fold.target_system,
                n=ood_ngram,
            )
            ood_reports.append(report_d)
        (fold_dir / "ood_distance.json").write_text(
            json.dumps([r.to_dict() for r in ood_reports], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        artefacts["ood_json"] = fold_dir / "ood_distance.json"

        # ---- Per-fold artefacts ----
        (fold_dir / "metrics.json").write_text(
            json.dumps(panel.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (fold_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for seq, pred, score in zip(test, y_pred.tolist(), y_score.tolist(), strict=True):
                fh.write(
                    json.dumps(
                        {
                            "group_id": seq.group_id,
                            "y_true": int(seq.label),
                            "y_pred": int(pred),
                            "p_anomaly": float(score),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        results.append(
            FoldResult(
                fold=fold,
                panel=panel,
                confusion=confusion,
                leakage_clean=leakage_clean,
                n_predictions=int(y_true.size),
                artefacts=artefacts,
                bootstrap_intervals=intervals,
                ood_distances=ood_reports,
            )
        )

    summary = _aggregate_summary(results)
    # Macro-aggregated bootstrap intervals (cross-fold).
    summary["macro_bootstrap"] = _aggregate_bootstrap(results)
    manifest.stop()
    manifest_path = manifest.write(out / "run_manifest.json")
    summary["run_manifest_path"] = str(manifest_path)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _aggregate_bootstrap(
    results: Sequence[FoldResult],
) -> dict[str, dict[str, float]]:
    from hylog.evaluation.bootstrap import aggregate_macro

    return aggregate_macro([r.bootstrap_intervals for r in results])


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _check_leakage_error_chain() -> type[LeakageError]:
    """Re-export for callers that want to catch the error type."""
    return LeakageError


__all__ = [
    "FoldResult",
    "LosoFold",
    "TrainerProtocol",
    "build_folds",
    "run_loso",
    "strip_labels",
]
