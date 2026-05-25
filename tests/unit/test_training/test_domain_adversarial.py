"""Tests for the domain-adversarial kill-switch."""

from __future__ import annotations

import pytest
import torch

from hylog.training.domain_adversarial import (
    DomainClassifier,
    combined_loss,
    grad_reverse,
)


def test_gradient_reverse_forward_is_identity() -> None:
    x = torch.randn(3, 4, requires_grad=True)
    y = grad_reverse(x, lambda_=2.0)
    assert torch.allclose(y, x)


def test_gradient_reverse_backward_flips_and_scales() -> None:
    x = torch.ones(4, requires_grad=True)
    y = grad_reverse(x, lambda_=2.5)
    loss = y.sum()
    loss.backward()
    # d(sum)/dx = 1 then reverse + scale -> grad = -2.5
    assert torch.allclose(x.grad, torch.full((4,), -2.5))


def test_domain_classifier_output_shape() -> None:
    clf = DomainClassifier(in_features=16, n_domains=3)
    x = torch.randn(5, 16)
    out = clf(x, lambda_=1.0)
    assert out.shape == (5, 3)


def test_domain_classifier_requires_min_two_domains() -> None:
    with pytest.raises(ValueError):
        DomainClassifier(in_features=8, n_domains=1)


def test_combined_loss_lambda_zero_returns_task_loss() -> None:
    task = torch.tensor(0.5)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    out = combined_loss(
        task_loss=task,
        domain_logits=logits,
        domain_targets=targets,
        lambda_domain=0.0,
    )
    assert torch.allclose(out, task)


def test_combined_loss_with_active_lambda_changes_value() -> None:
    task = torch.tensor(0.5)
    logits = torch.randn(4, 3, requires_grad=False)
    targets = torch.tensor([0, 1, 2, 0])
    out = combined_loss(
        task_loss=task,
        domain_logits=logits,
        domain_targets=targets,
        lambda_domain=0.5,
    )
    assert out.item() != task.item()


def test_combined_loss_xor_arguments_raises() -> None:
    task = torch.tensor(0.5)
    with pytest.raises(ValueError):
        combined_loss(
            task_loss=task,
            domain_logits=torch.randn(2, 3),
            domain_targets=None,
            lambda_domain=0.5,
        )
