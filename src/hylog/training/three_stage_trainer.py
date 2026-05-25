"""Three-stage trainer used by both the LogLLM baseline and HyLog.

Stages (parity with upstream ``train.py:166-191``):

  1. Train only the decoder LoRA      (upstream ``set_train_only_Llama``).
  2. Train only the projector         (upstream ``set_train_only_projector``).
  3. Train projector + encoder LoRA   (upstream ``set_train_projectorAndBert``).
  4. Train everything                 (upstream ``set_finetuning_all``).

Upstream calls Stage 1 *before* the projector — HyLog follows the same
ordering to keep numerical parity. The roadmap §4 description lists three
stages, the upstream code uses four; both are supported via the ``stages``
argument so an experiment config can choose.

The trainer is *backend-agnostic*: it operates on a ``StageRunner`` callable
that the caller supplies. This lets the LogLLM baseline use upstream's
``train_helper`` while HyLog uses its classification-head loss without any
trainer-level if/else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from hylog.training.mlflow_logger import MLflowLogger


class _SwitchProto(Protocol):
    """Switching the model to a given training stage."""

    def __call__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One stage of the three-stage protocol."""

    name: str
    switch: _SwitchProto
    n_epochs: int
    lr: float


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Hyperparameters shared across stages."""

    batch_size: int = 16
    micro_batch_size: int = 4
    grad_clip_norm: float | None = 1.0
    scheduler_gamma: float = 0.7  # parity with upstream train.py:77
    seed: int = 42
    log_every: int = 50

    @property
    def grad_accum_steps(self) -> int:
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"batch_size {self.batch_size} must be divisible by "
                f"micro_batch_size {self.micro_batch_size}"
            )
        return self.batch_size // self.micro_batch_size


# Default 4-stage protocol matching upstream LogLLM. Phase 2B uses this.
def default_logllm_stages(model: object) -> list[StageSpec]:
    """Build the 4-stage upstream-equivalent schedule.

    Parameters are tied to upstream defaults
    (``third_party/LogLLM/train.py:13-26``).
    """
    return [
        StageSpec(
            name="decoder_lora_only",
            switch=model.set_train_only_decoder,  # type: ignore[attr-defined]
            n_epochs=1,
            lr=5e-4,
        ),
        StageSpec(
            name="projector_only",
            switch=model.set_train_only_projector,  # type: ignore[attr-defined]
            n_epochs=1,
            lr=5e-4,
        ),
        StageSpec(
            name="projector_and_encoder",
            switch=model.set_train_projector_and_encoder,  # type: ignore[attr-defined]
            n_epochs=1,
            lr=5e-5,
        ),
        StageSpec(
            name="finetune_all",
            switch=model.set_finetuning_all,  # type: ignore[attr-defined]
            n_epochs=2,
            lr=5e-5,
        ),
    ]


@dataclass(slots=True)
class ThreeStageTrainer:
    """Coordinator that walks a model through each stage in turn.

    Concrete optimization (forward/backward/optimizer.step) is delegated to
    a ``runner`` callable injected by the caller. The trainer's only job is
    to flip training modes, build optimizers, and emit MLflow events.
    """

    config: TrainerConfig
    logger: MLflowLogger = field(default_factory=lambda: MLflowLogger(track=False))

    def fit(
        self,
        *,
        run_name: str,
        stages: list[StageSpec],
        runner: Callable[[StageSpec], dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Run every stage in order and return per-stage final metrics."""
        results: dict[str, dict[str, float]] = {}
        with self.logger.start_run(run_name=run_name):
            self.logger.log_params(
                {
                    "batch_size": self.config.batch_size,
                    "micro_batch_size": self.config.micro_batch_size,
                    "seed": self.config.seed,
                }
            )
            for stage in stages:
                stage.switch()
                metrics = runner(stage)
                results[stage.name] = metrics
                self.logger.log_metrics({f"{stage.name}/{k}": v for k, v in metrics.items()})
        return results


__all__ = [
    "StageSpec",
    "ThreeStageTrainer",
    "TrainerConfig",
    "default_logllm_stages",
]
