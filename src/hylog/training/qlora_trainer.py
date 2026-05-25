"""QLoRA trainer for HyLogCore.

This module drives the three-stage protocol defined in the roadmap
(§4 training stages: projector warm-up → projector+LoRA+head → end-to-end
refinement at reduced lr). It is *separate* from the LogLLM baseline
trainer (`three_stage_trainer.py` plus upstream's ``train_helper``) because
the loss is fundamentally different:

- LogLLM: token-matching cross-entropy on "The sequence is normal." vs
  "anomalous." answer tokens.
- HyLog : straight cross-entropy on a 2-class classification head.

The simpler loss makes calibration (Phase 5) mathematically clean and is
the reason the HyLog story is "first LAD pipeline with usable uncertainty".

The trainer is **gradient-checkpoint-friendly** and supports gradient
accumulation, deterministic seeding, and class-weighted loss (capped at
10x per the roadmap §4 stage description).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from hylog.evaluation.metrics import MetricPanel, compute_metric_panel
from hylog.training.collator import HyLogBatch


@dataclass(frozen=True, slots=True)
class StageHyperparams:
    """Per-stage knobs."""

    name: str
    n_epochs: int = 1
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0


@dataclass(frozen=True, slots=True)
class QLoraTrainerConfig:
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    class_weight_cap: float = 10.0
    """Per the roadmap §4 stages, class weights are capped at 10x."""
    early_stop_patience: int | None = None
    seed: int = 42
    compute_dtype: str = "bfloat16"  # bf16 on GPU; trainer casts to float32 on CPU.

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


def _balanced_class_weights(labels: Iterable[int], cap: float) -> torch.Tensor:
    """Inverse-frequency class weights, capped at ``cap`` to avoid degenerate
    gradients when the minority class is essentially absent in a batch.
    """
    counts = [0, 0]
    for lab in labels:
        if lab not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {lab!r}")
        counts[int(lab)] += 1
    total = counts[0] + counts[1]
    if total == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    # Per-class weight: total / (2 * count_c). Cap at ``cap``.
    weights = []
    for c in counts:
        w = cap if c == 0 else min(total / (2.0 * c), cap)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass(slots=True)
class EpochSummary:
    """Per-epoch training+validation summary."""

    stage: str
    epoch: int
    train_loss: float
    val_loss: float | None
    val_panel: MetricPanel | None


@dataclass(slots=True)
class StageHistory:
    name: str
    summaries: list[EpochSummary] = field(default_factory=list)

    def train_losses(self) -> list[float]:
        return [s.train_loss for s in self.summaries]

    def val_losses(self) -> list[float | None]:
        return [s.val_loss for s in self.summaries]


def _tail_is_monotone_non_increasing(values: Sequence[float], tail_fraction: float) -> bool:
    """True if the last ``tail_fraction`` of the series is non-increasing."""
    if len(values) <= 1:
        return True
    k = max(2, math.ceil(len(values) * tail_fraction))
    tail = list(values)[-k:]
    return all(b <= a + 1e-9 for a, b in itertools.pairwise(tail))


@dataclass(slots=True)
class QLoraTrainer:
    """Trainer for HyLogCore.

    The trainer is intentionally small and explicit so it is easy to audit
    in code review: ``fit_stage`` is a vanilla train loop with grad accumulation,
    class-weighted cross-entropy, and per-epoch validation.
    """

    config: QLoraTrainerConfig

    def fit_stage(
        self,
        *,
        model: nn.Module,
        stage: StageHyperparams,
        train_batches: Iterable[HyLogBatch],
        val_batches: Iterable[HyLogBatch] | None = None,
        device: torch.device | str = "cpu",
    ) -> StageHistory:
        """Train ``model`` for one stage.

        The caller is responsible for switching the model's training mode
        BEFORE this method is invoked.
        """
        history = StageHistory(name=stage.name)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                f"stage {stage.name!r}: model has zero trainable parameters; "
                f"caller must invoke set_train_* before fit_stage"
            )
        optimizer = torch.optim.AdamW(
            trainable_params, lr=stage.lr, weight_decay=stage.weight_decay
        )
        train_batches_list = list(train_batches)
        val_batches_list = list(val_batches) if val_batches is not None else []

        for epoch in range(stage.n_epochs):
            train_loss = self._train_one_epoch(
                model=model,
                batches=train_batches_list,
                optimizer=optimizer,
                grad_clip_norm=stage.grad_clip_norm,
                device=device,
            )
            val_loss: float | None = None
            val_panel: MetricPanel | None = None
            if val_batches_list:
                val_loss, val_panel = self._evaluate(model, val_batches_list, device)
            history.summaries.append(
                EpochSummary(
                    stage=stage.name,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_panel=val_panel,
                )
            )
        return history

    def _train_one_epoch(
        self,
        *,
        model: nn.Module,
        batches: list[HyLogBatch],
        optimizer: torch.optim.Optimizer,
        grad_clip_norm: float | None,
        device: torch.device | str,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_n = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches):
            batch = batch.to(device)
            weights = _balanced_class_weights(
                batch.labels.tolist(), cap=self.config.class_weight_cap
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            logits = model(
                line_inputs=batch.line_inputs,
                sequence_lengths=batch.sequence_lengths,
            )
            loss = criterion(logits, batch.labels)
            (loss / self.config.grad_accum_steps).backward()
            total_loss += float(loss.item()) * batch.labels.size(0)
            total_n += batch.labels.size(0)

            stepping = (step + 1) % self.config.grad_accum_steps == 0 or (step + 1) == len(batches)
            if stepping:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=grad_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return total_loss / max(total_n, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        batches: list[HyLogBatch],
        device: torch.device | str,
    ) -> tuple[float, MetricPanel]:
        import numpy as np

        model.eval()
        total_loss = 0.0
        total_n = 0
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_scores: list[float] = []
        for batch in batches:
            batch = batch.to(device)
            logits = model(
                line_inputs=batch.line_inputs,
                sequence_lengths=batch.sequence_lengths,
            )
            loss = nn.functional.cross_entropy(logits, batch.labels)
            total_loss += float(loss.item()) * batch.labels.size(0)
            total_n += batch.labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = probs > 0.5
            all_preds.extend(preds.long().cpu().tolist())
            all_labels.extend(batch.labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
        panel = compute_metric_panel(
            y_true=np.asarray(all_labels),
            y_pred=np.asarray(all_preds),
            y_score=np.asarray(all_scores),
        )
        return total_loss / max(total_n, 1), panel


def default_hylog_stages() -> list[StageHyperparams]:
    """The HyLog three-stage schedule (roadmap §4)."""
    return [
        StageHyperparams(name="projector_warmup", n_epochs=1, lr=5e-4),
        StageHyperparams(name="joint_qlora", n_epochs=2, lr=5e-5),
        StageHyperparams(name="end_to_end_refine", n_epochs=1, lr=5e-6),
    ]


__all__ = [
    "EpochSummary",
    "QLoraTrainer",
    "QLoraTrainerConfig",
    "StageHistory",
    "StageHyperparams",
    "_tail_is_monotone_non_increasing",
    "default_hylog_stages",
]
