"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def fixed_seed() -> int:
    return 1337
