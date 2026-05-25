"""Tests for the three-stage trainer."""

from __future__ import annotations

import pytest

from hylog.training.three_stage_trainer import (
    StageSpec,
    ThreeStageTrainer,
    TrainerConfig,
)


def test_grad_accum_steps_basic() -> None:
    cfg = TrainerConfig(batch_size=16, micro_batch_size=4)
    assert cfg.grad_accum_steps == 4


def test_grad_accum_divisibility_check() -> None:
    cfg = TrainerConfig(batch_size=10, micro_batch_size=4)
    with pytest.raises(ValueError):
        _ = cfg.grad_accum_steps


def test_fit_runs_every_stage_in_order() -> None:
    seen: list[str] = []

    def make_switch(name: str):
        def _switch() -> None:
            seen.append(name)

        return _switch

    stages = [
        StageSpec(name="a", switch=make_switch("a"), n_epochs=1, lr=1e-3),
        StageSpec(name="b", switch=make_switch("b"), n_epochs=1, lr=1e-3),
        StageSpec(name="c", switch=make_switch("c"), n_epochs=1, lr=1e-3),
    ]

    runner_calls: list[str] = []

    def runner(stage: StageSpec) -> dict[str, float]:
        runner_calls.append(stage.name)
        return {"loss": 0.5}

    trainer = ThreeStageTrainer(config=TrainerConfig())
    results = trainer.fit(run_name="test", stages=stages, runner=runner)

    assert seen == ["a", "b", "c"]
    assert runner_calls == ["a", "b", "c"]
    assert set(results) == {"a", "b", "c"}


def test_default_logllm_stages_uses_upstream_hyperparams() -> None:
    """The first stage trains the decoder (parity with upstream train.py:167-169)."""
    from hylog.training.three_stage_trainer import default_logllm_stages

    class _Stub:
        def set_train_only_decoder(self) -> None:
            pass

        def set_train_only_projector(self) -> None:
            pass

        def set_train_projector_and_encoder(self) -> None:
            pass

        def set_finetuning_all(self) -> None:
            pass

    stages = default_logllm_stages(_Stub())
    names = [s.name for s in stages]
    assert names == [
        "decoder_lora_only",
        "projector_only",
        "projector_and_encoder",
        "finetune_all",
    ]
    # Upstream hyperparameters (train.py:13-26).
    lrs = [s.lr for s in stages]
    assert lrs == [5e-4, 5e-4, 5e-5, 5e-5]
    epochs = [s.n_epochs for s in stages]
    assert epochs == [1, 1, 1, 2]
