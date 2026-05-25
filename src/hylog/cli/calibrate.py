"""``hylog-calibrate`` — fit a calibrator to a LOSO run's predictions.

The CLI is the canonical Phase-5 entry point. It consumes a
``predictions.jsonl`` written by ``hylog-loso`` (each line has
``y_true``, ``y_pred``, ``p_anomaly``) and:

  1. reconstructs the implied 2-class logits via the inverse sigmoid;
  2. fits a temperature calibrator (Guo 2017) on the supplied
     calibration slice;
  3. optionally falls back to Platt scaling if temperature scaling
     fails to reach the ECE budget;
  4. computes ECE / MCE before and after calibration;
  5. computes AURC + E-AURC + cost-asymmetric AURC on the calibrated
     scores;
  6. emits a complete artefact bundle under ``out_dir``.

The CLI is fully deterministic given the input file and the seed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import numpy as np

from hylog.calibration.aurc import compute_aurc
from hylog.calibration.ece import compute_reliability_bins
from hylog.calibration.platt import fit_platt
from hylog.calibration.reliability import archive_all as archive_reliability
from hylog.calibration.temperature import fit_temperature
from hylog.inference.selective import select_tau_for_risk_budget


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read predictions.jsonl into ``(y_true, y_pred, p_anomaly)`` arrays."""
    y_true: list[int] = []
    y_pred: list[int] = []
    p_anom: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
            p_anom.append(float(row["p_anomaly"]))
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(p_anom, dtype=np.float64),
    )


def _probs_to_logits(p_anomaly: np.ndarray) -> np.ndarray:
    """Convert ``p_anomaly`` to a 2-class logit matrix via inverse sigmoid.

    Clipping at [1e-7, 1 - 1e-7] guards against ``log(0)`` for the
    Phase-3 trainer's saturated outputs.
    """
    p = np.clip(p_anomaly, 1e-7, 1.0 - 1e-7)
    log_odds = np.log(p / (1.0 - p))
    # The implied 2-class logits are (0, log_odds) up to a constant shift;
    # softmax recovers the same probabilities.
    return np.stack([np.zeros_like(log_odds), log_odds], axis=1)


def _split_calibration_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_anom: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Deterministic stratified split into (calibration, held-out)."""
    rng = np.random.default_rng(seed)
    n = y_true.size
    indices = np.arange(n)
    # Stratify on y_true so both splits have the same anomaly rate.
    pos = indices[y_true == 1]
    neg = indices[y_true == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_cal = max(1, round(pos.size * calibration_fraction))
    n_neg_cal = max(1, round(neg.size * calibration_fraction))
    cal_idx = np.sort(np.concatenate([pos[:n_pos_cal], neg[:n_neg_cal]]))
    held_idx = np.sort(np.concatenate([pos[n_pos_cal:], neg[n_neg_cal:]]))
    return (
        (y_true[cal_idx], y_pred[cal_idx], p_anom[cal_idx]),
        (y_true[held_idx], y_pred[held_idx], p_anom[held_idx]),
    )


@click.command(name="hylog-calibrate")
@click.option(
    "--predictions",
    "-p",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="A predictions.jsonl file produced by hylog-loso.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory under which to write the calibration artefacts.",
)
@click.option(
    "--calibration-fraction",
    type=float,
    default=0.5,
    show_default=True,
    help="Fraction of predictions to use as the held-out calibration slice.",
)
@click.option(
    "--ece-budget",
    type=float,
    default=0.05,
    show_default=True,
    help="ECE target. When temperature scaling cannot meet it, fall back to Platt.",
)
@click.option(
    "--risk-budget",
    type=float,
    default=0.05,
    show_default=True,
    help="Selective-prediction risk budget for the τ search.",
)
@click.option("--seed", type=int, default=42, show_default=True)
def main(
    predictions: Path,
    out_dir: Path,
    calibration_fraction: float,
    ece_budget: float,
    risk_budget: float,
    seed: int,
) -> None:
    """Fit + evaluate calibration for one LOSO fold."""
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, p_anom = _load_predictions(predictions)
    if y_true.size < 4:
        click.echo("predictions file has too few rows to calibrate (need >= 4).")
        sys.exit(2)

    _probs_to_logits(p_anom)

    # Calibration / held-out split.
    (cal_true, _cal_pred, cal_panom), (held_true, _held_pred, held_panom) = _split_calibration_test(
        y_true,
        y_pred,
        p_anom,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    cal_logits = _probs_to_logits(cal_panom)
    held_logits = _probs_to_logits(held_panom)

    # Pre-calibration reliability on the held-out set.
    base_probs = _softmax(held_logits)
    base_report = compute_reliability_bins(base_probs, held_true)

    # ---- Temperature scaling ----
    temp_cal = fit_temperature(cal_logits, cal_true)
    temp_probs = temp_cal.apply(held_logits)
    temp_report = compute_reliability_bins(temp_probs, held_true)

    used_method: str = "temperature_scaling"
    final_probs = temp_probs
    final_report = temp_report
    method_dict: dict[str, object] = dict(temp_cal.to_dict())

    # ---- Kill-switch: Platt scaling if ECE still too high ----
    if temp_report.ece > ece_budget:
        try:
            margin = cal_logits[:, 1] - cal_logits[:, 0]
            platt_cal = fit_platt(margin, cal_true.astype(np.float64))
            platt_probs = platt_cal.apply(held_logits)
            platt_report = compute_reliability_bins(platt_probs, held_true)
            if platt_report.ece < temp_report.ece:
                used_method = "platt_scaling"
                final_probs = platt_probs
                final_report = platt_report
                method_dict = dict(platt_cal.to_dict())
        except (ValueError, RuntimeError) as exc:  # pragma: no cover
            click.echo(f"platt fallback failed: {exc}; keeping temperature.", err=True)

    # ---- AURC + selective-prediction τ ----
    final_pred = final_probs.argmax(axis=1)
    final_conf = final_probs.max(axis=1)
    aurc_report = compute_aurc(
        y_true=held_true,
        y_pred=final_pred,
        confidence=final_conf,
    )
    tau = select_tau_for_risk_budget(
        probabilities=final_probs,
        labels=held_true,
        risk_budget=risk_budget,
    )

    # ---- Archive ----
    archive_reliability(
        final_report,
        out_dir=out_dir,
        name="reliability",
        title=f"Reliability ({used_method})",
    )
    archive_reliability(
        base_report,
        out_dir=out_dir,
        name="reliability_uncalibrated",
        title="Reliability (uncalibrated)",
    )
    (out_dir / "calibration.json").write_text(
        json.dumps(
            {
                "method": used_method,
                "params": method_dict,
                "ece_before": base_report.ece,
                "ece_after": final_report.ece,
                "mce_before": base_report.mce,
                "mce_after": final_report.mce,
                "n_calibration": int(cal_true.size),
                "n_held": int(held_true.size),
                "ece_budget": ece_budget,
                "well_calibrated": bool(final_report.ece <= ece_budget),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "aurc.json").write_text(
        json.dumps(aurc_report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "tau.json").write_text(
        json.dumps(tau.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    click.echo(
        json.dumps(
            {
                "method": used_method,
                "ece_before": round(base_report.ece, 6),
                "ece_after": round(final_report.ece, 6),
                "well_calibrated": bool(final_report.ece <= ece_budget),
                "aurc": round(aurc_report.aurc, 6),
                "excess_aurc": round(aurc_report.excess_aurc, 6),
                "tau": round(tau.threshold, 4),
                "tau_coverage": round(tau.achieved_coverage, 4),
                "out_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# Math import lazily; only used by `_probs_to_logits`'s docstring example.
del math


if __name__ == "__main__":  # pragma: no cover
    main()
