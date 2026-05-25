"""Tests for hylog.utils.seeding."""

from __future__ import annotations

import os
import random

import pytest

from hylog.utils.seeding import set_global_seed


def test_set_global_seed_python_rng_deterministic(fixed_seed: int) -> None:
    set_global_seed(fixed_seed)
    first = [random.random() for _ in range(5)]
    set_global_seed(fixed_seed)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seed_sets_pythonhashseed(fixed_seed: int) -> None:
    set_global_seed(fixed_seed)
    assert os.environ["PYTHONHASHSEED"] == str(fixed_seed)


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)
