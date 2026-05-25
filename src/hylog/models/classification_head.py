"""Binary classification head used by HyLog (Phase 3+).

Note: the LogLLM baseline does *not* use this module. Upstream emits answer
tokens autoregressively and scores against ``"The sequence is normal."`` vs
``"The sequence is anomalous."`` (``third_party/LogLLM/train.py:79-82``).
This head is the HyLog-side replacement that lets us:

- Apply temperature scaling on a single logit pair (Phase 5).
- Run risk-coverage selective prediction on a calibrated probability.

The head is part of the Phase 0 / Phase 2 deliverables because the Phase 3
trainer reuses it; Phase 2's LogLLM re-implementation keeps the upstream
token-matching mechanism for fidelity.
"""

from __future__ import annotations

import torch
from torch import nn


class BinaryClassificationHead(nn.Module):
    """Linear head over a single feature vector.

    Output shape: ``[batch, 2]`` (logits for {normal, anomaly}).
    """

    def __init__(self, in_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(feature))


__all__ = ["BinaryClassificationHead"]
