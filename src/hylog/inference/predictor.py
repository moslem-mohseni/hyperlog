"""Abstract predictor protocol + a mock implementation for CPU tests.

The FastAPI service depends on this protocol — *not* on HyLogCore
directly. The production server loads a real predictor from a model
directory; the test suite uses ``MockPredictor`` to exercise every
endpoint without touching torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """Per-sequence prediction row returned by a predictor."""

    p_anomaly: float
    p_anomaly_calibrated: float
    confidence: float


class PredictorProtocol(Protocol):
    """Minimal predictor interface used by the FastAPI server."""

    def predict_batch(self, sequences: Sequence[Sequence[str]]) -> list[PredictionRow]: ...

    def model_version(self) -> str: ...

    def selective_threshold(self) -> float: ...

    def calibration_info(self) -> dict[str, object]: ...


@dataclass(slots=True)
class MockPredictor:
    """Deterministic predictor used in tests + the dev server.

    The ``p_anomaly`` it emits is a pure function of (sequence content),
    so the FastAPI endpoint can be tested without GPU. Calibration is
    a no-op (``method='none'``); the FastAPI server still wraps the
    output with the §11.7 schema.
    """

    threshold: float = 0.7
    version: str = "mock-1.0"
    seed: int = 42
    _calibration: dict[str, object] = field(
        default_factory=lambda: {
            "method": "none",
            "fitted_on": "mock",
        }
    )

    def predict_batch(self, sequences: Sequence[Sequence[str]]) -> list[PredictionRow]:
        out: list[PredictionRow] = []
        for lines in sequences:
            content = "\n".join(lines)
            digest = hashlib.sha256(content.encode("utf-8")).digest()
            # Deterministic [0, 1] from the digest's first byte.
            raw = digest[0] / 255.0
            # Up-weight when the content mentions FATAL / panic / error.
            lower = content.lower()
            bias = 0.0
            for token in ("fatal", "panic", "kerndtlb", "exception"):
                if token in lower:
                    bias += 0.15
            p = max(0.0, min(1.0, raw + bias))
            confidence = max(p, 1.0 - p)
            out.append(
                PredictionRow(
                    p_anomaly=float(p),
                    p_anomaly_calibrated=float(p),
                    confidence=float(confidence),
                )
            )
        return out

    def model_version(self) -> str:
        return self.version

    def selective_threshold(self) -> float:
        return self.threshold

    def calibration_info(self) -> dict[str, object]:
        return dict(self._calibration)


__all__ = ["MockPredictor", "PredictionRow", "PredictorProtocol"]
