"""Deterministic seeding across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    When ``deterministic_torch`` is True, cuDNN is set to deterministic mode,
    which is required for reproducibility but may slow some kernels.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
