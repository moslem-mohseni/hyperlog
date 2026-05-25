"""Tests for the HyLog binary classification head."""

from __future__ import annotations

import pytest
import torch

from hylog.models.classification_head import BinaryClassificationHead


def test_output_shape() -> None:
    head = BinaryClassificationHead(in_features=32)
    out = head(torch.randn(8, 32))
    assert out.shape == (8, 2)


def test_dropout_present() -> None:
    head = BinaryClassificationHead(in_features=8, dropout=0.3)
    head.eval()
    out = head(torch.randn(4, 8))
    assert out.shape == (4, 2)


def test_invalid_features() -> None:
    with pytest.raises(ValueError):
        BinaryClassificationHead(in_features=0)
