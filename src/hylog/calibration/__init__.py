"""Post-hoc calibration: temperature scaling, ECE/MCE, AURC.

The package exposes a uniform interface that any calibrator can satisfy:

  fit_*(logits, labels)        -> *Calibrator
  calibrator.apply(logits)     -> calibrated probabilities

Plus the metric helpers:

  compute_reliability_bins     -> ECE / MCE / per-bin
  compute_aurc                  -> AURC / E-AURC / cost-asymmetric
  risk_coverage_curve           -> (coverage, risk) arrays for plotting
"""

from __future__ import annotations

from hylog.calibration.aurc import AURCReport, compute_aurc, risk_coverage_curve
from hylog.calibration.ece import (
    CalibrationReport,
    ReliabilityBin,
    compute_reliability_bins,
    ece_only,
)
from hylog.calibration.platt import PlattCalibrator, fit_platt
from hylog.calibration.temperature import TemperatureCalibrator, fit_temperature
from hylog.calibration.vector_scaling import VectorScalingCalibrator, fit_vector_scaling

__all__ = [
    "AURCReport",
    "CalibrationReport",
    "PlattCalibrator",
    "ReliabilityBin",
    "TemperatureCalibrator",
    "VectorScalingCalibrator",
    "compute_aurc",
    "compute_reliability_bins",
    "ece_only",
    "fit_platt",
    "fit_temperature",
    "fit_vector_scaling",
    "risk_coverage_curve",
]
