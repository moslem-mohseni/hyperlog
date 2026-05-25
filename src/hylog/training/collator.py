"""Batch collator for HyLog: flattens a batch of LogSequence objects into
``(line_inputs, sequence_lengths, labels)`` tuples consumed by HyLogCore.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from hylog.data.schema import LogSequence
from hylog.models.encoder import LogLineEncoder


@dataclass(slots=True)
class HyLogBatch:
    line_inputs: dict[str, torch.Tensor]
    sequence_lengths: list[int]
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> HyLogBatch:
        line_inputs = {k: v.to(device) for k, v in self.line_inputs.items()}
        return HyLogBatch(
            line_inputs=line_inputs,
            sequence_lengths=self.sequence_lengths,
            labels=self.labels.to(device),
        )


@dataclass(slots=True)
class HyLogCollator:
    encoder: LogLineEncoder
    max_sequence_lines: int = 128

    def __call__(self, batch: Sequence[LogSequence]) -> HyLogBatch:
        if not batch:
            raise ValueError("HyLogCollator received an empty batch")
        flat_lines: list[str] = []
        seq_lens: list[int] = []
        labels: list[int] = []
        for seq in batch:
            truncated = seq.lines[: self.max_sequence_lines]
            flat_lines.extend(truncated)
            seq_lens.append(len(truncated))
            labels.append(int(seq.label))
        tokenized = self.encoder.tokenize(flat_lines)
        return HyLogBatch(
            line_inputs=tokenized,
            sequence_lengths=seq_lens,
            labels=torch.tensor(labels, dtype=torch.long),
        )


__all__ = ["HyLogBatch", "HyLogCollator"]
