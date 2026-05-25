"""Tests for the masked self-prediction kill-switch."""

from __future__ import annotations

import pytest
import torch

from hylog.data.schema import LogSequence
from hylog.training.self_supervised import (
    MaskedLineLoss,
    TargetAugmentorConfig,
    UnsupervisedTargetAugmentor,
    make_mask,
    select_target_lines,
)


def _seq(group_id: str) -> LogSequence:
    return LogSequence(lines=("a", "b"), label=0, group_id=group_id, source="x")


def test_target_augmentor_config_validates() -> None:
    with pytest.raises(ValueError):
        TargetAugmentorConfig(mask_ratio=0.0)
    with pytest.raises(ValueError):
        TargetAugmentorConfig(mask_ratio=1.5)
    with pytest.raises(ValueError):
        TargetAugmentorConfig(sample_fraction=0.0)
    with pytest.raises(ValueError):
        TargetAugmentorConfig(lambda_unsup=-1.0)


def test_make_mask_has_correct_shape() -> None:
    g = torch.Generator().manual_seed(0)
    mask = make_mask(shape=(64, 8), ratio=0.3, generator=g)
    assert mask.shape == (64, 8)
    assert mask.dtype == torch.bool


def test_make_mask_rejects_out_of_range_ratio() -> None:
    with pytest.raises(ValueError):
        make_mask(shape=(2,), ratio=-0.1)
    with pytest.raises(ValueError):
        make_mask(shape=(2,), ratio=1.5)


def test_masked_line_loss_zero_when_predictions_match() -> None:
    loss_fn = MaskedLineLoss()
    pred = torch.ones(3, 4)
    target = torch.ones(3, 4)
    mask = torch.ones(3, 4, dtype=torch.bool)
    assert loss_fn(predicted=pred, target=target, mask=mask).item() == pytest.approx(0.0)


def test_masked_line_loss_ignores_unmasked_positions() -> None:
    loss_fn = MaskedLineLoss()
    pred = torch.zeros(2, 2)
    target = torch.ones(2, 2)  # MSE per element would be 1.0
    # Only first row is masked.
    mask = torch.tensor([[True, True], [False, False]])
    out = loss_fn(predicted=pred, target=target, mask=mask)
    assert out.item() == pytest.approx(1.0)


def test_masked_line_loss_shape_mismatch_raises() -> None:
    loss_fn = MaskedLineLoss()
    with pytest.raises(ValueError):
        loss_fn(
            predicted=torch.zeros(2, 3),
            target=torch.zeros(3, 3),
            mask=torch.ones(2, 3, dtype=torch.bool),
        )


def test_select_target_lines_deterministic_subset() -> None:
    seqs = [_seq(f"g{i}") for i in range(20)]
    a = select_target_lines(seqs, fraction=0.25, seed=7)
    b = select_target_lines(seqs, fraction=0.25, seed=7)
    assert [s.group_id for s in a] == [s.group_id for s in b]
    # Different seed -> different (or at least possibly different) sample.
    c = select_target_lines(seqs, fraction=0.25, seed=99)
    assert [s.group_id for s in a] != [s.group_id for s in c]


def test_select_target_lines_rejects_invalid_fraction() -> None:
    seqs = [_seq("g0")]
    with pytest.raises(ValueError):
        select_target_lines(seqs, fraction=0.0)
    with pytest.raises(ValueError):
        select_target_lines(seqs, fraction=2.0)


def test_augmentor_loss_smoke() -> None:
    cfg = TargetAugmentorConfig(mask_ratio=0.5, sample_fraction=0.1, lambda_unsup=0.1)
    aug = UnsupervisedTargetAugmentor(config=cfg)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(8, 16)
    recon = proj.clone() + 0.1
    loss = aug.compute_step_loss(projected_vectors=proj, reconstructed=recon, generator=g)
    assert loss.shape == ()
    assert loss.item() >= 0.0
