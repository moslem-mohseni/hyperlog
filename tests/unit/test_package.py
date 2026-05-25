"""Smoke tests verifying the package is importable and self-consistent."""

from __future__ import annotations

import importlib


def test_root_import() -> None:
    import hylog

    assert hylog.__version__
    assert hylog.__author__ == "Moslem Mohseni Khah"


def test_subpackages_importable() -> None:
    for sub in (
        "hylog.data",
        "hylog.data.loaders",
        "hylog.models",
        "hylog.models.baselines",
        "hylog.training",
        "hylog.evaluation",
        "hylog.calibration",
        "hylog.inference",
        "hylog.cli",
        "hylog.utils",
    ):
        importlib.import_module(sub)
