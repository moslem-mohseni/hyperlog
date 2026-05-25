"""Tests for the projector module."""

from __future__ import annotations

import pytest
import torch

from hylog.models.projector import Projector, ProjectorConfig


def test_depth_1_is_pure_linear() -> None:
    p = Projector(ProjectorConfig(in_features=8, out_features=16, depth=1))
    # depth=1 reproduces LogLLM bit-for-bit: exactly one Linear, no activation.
    modules = list(p.layers.children())
    assert len(modules) == 1
    assert isinstance(modules[0], torch.nn.Linear)


def test_depth_2_has_activation() -> None:
    p = Projector(ProjectorConfig(in_features=8, out_features=16, depth=2))
    modules = list(p.layers.children())
    assert any(isinstance(m, (torch.nn.GELU, torch.nn.ReLU, torch.nn.SiLU)) for m in modules)


def test_forward_shape() -> None:
    p = Projector(ProjectorConfig(in_features=8, out_features=16))
    x = torch.randn(4, 8)
    assert p(x).shape == (4, 16)


def test_invalid_depth() -> None:
    with pytest.raises(ValueError):
        Projector(ProjectorConfig(in_features=8, out_features=16, depth=4))


def test_invalid_features() -> None:
    with pytest.raises(ValueError):
        Projector(ProjectorConfig(in_features=0, out_features=16))


def test_parameter_count_depth_1() -> None:
    """LogLLM bit-for-bit: single Linear has in*out + out params."""
    p = Projector(ProjectorConfig(in_features=32, out_features=64, depth=1))
    expected = 32 * 64 + 64
    assert p.num_trainable_parameters() == expected
