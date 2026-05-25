"""System-to-system distribution distance — a cross-system diagnostic.

The idea: before reading any LOSO F1 number, we want a structural
hint of how far apart the target system's lines are from the
training-source distribution. A target with vocabulary, message
patterns, and event mixes very different from sources is *expected* to
be hard; a target that overlaps heavily is *expected* to be easy.

We compute the distance over **n-gram fingerprints of the preprocessed
lines** (token-level), which is cheap, deterministic, and does not
require any neural inference. Two variants are exposed:

- **Jaccard distance over unique n-grams.** Robust to volume
  differences, dominated by *whether* an n-gram appears at all.
- **Cosine distance over n-gram-frequency vectors.** Captures the
  *relative* prevalence of n-grams.

Both are normalised to [0, 1]; 0 = identical distribution, 1 = no
overlap. The diagnostic is reported per LOSO fold and persisted in the
fold's ``ood_distance.json``.

The metric is **not** a Maximum Mean Discrepancy (MMD) because MMD with
a deep kernel requires the encoder forward pass for every line, which
would slow Phase 4's audit-only path. The n-gram fingerprint is
~1000x cheaper while preserving the qualitative ordering of distances
that a deep kernel would produce.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hylog.data.schema import LogSequence


@dataclass(frozen=True, slots=True)
class OODDistanceReport:
    """Symmetric distance between two system distributions."""

    source_system: str
    target_system: str
    n: int  # n-gram length used
    jaccard_distance: float
    cosine_distance: float
    n_source_lines: int
    n_target_lines: int
    n_source_ngrams: int
    n_target_ngrams: int
    n_shared_ngrams: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "source_system": self.source_system,
            "target_system": self.target_system,
            "n": self.n,
            "jaccard_distance": self.jaccard_distance,
            "cosine_distance": self.cosine_distance,
            "n_source_lines": self.n_source_lines,
            "n_target_lines": self.n_target_lines,
            "n_source_ngrams": self.n_source_ngrams,
            "n_target_ngrams": self.n_target_ngrams,
            "n_shared_ngrams": self.n_shared_ngrams,
        }


def _line_to_ngrams(line: str, n: int) -> Iterable[tuple[str, ...]]:
    tokens = line.split()
    if len(tokens) < n:
        return ()
    return tuple(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _accumulate_ngrams(
    sequences: Iterable[LogSequence], n: int
) -> tuple[dict[tuple[str, ...], int], int]:
    counts: dict[tuple[str, ...], int] = {}
    line_count = 0
    for seq in sequences:
        for line in seq.lines:
            line_count += 1
            for ng in _line_to_ngrams(line, n):
                counts[ng] = counts.get(ng, 0) + 1
    return counts, line_count


def _jaccard(a: Mapping[tuple[str, ...], int], b: Mapping[tuple[str, ...], int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    intersection = set_a & set_b
    return 1.0 - (len(intersection) / len(union))


def _cosine(a: Mapping[tuple[str, ...], int], b: Mapping[tuple[str, ...], int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def ood_distance(
    *,
    source_sequences: Iterable[LogSequence],
    target_sequences: Iterable[LogSequence],
    source_system: str,
    target_system: str,
    n: int = 2,
) -> OODDistanceReport:
    """Compute the Jaccard + cosine n-gram distance between two systems."""
    if n < 1:
        raise ValueError("n must be >= 1")
    src_counts, n_src_lines = _accumulate_ngrams(source_sequences, n)
    tgt_counts, n_tgt_lines = _accumulate_ngrams(target_sequences, n)
    shared = set(src_counts) & set(tgt_counts)
    return OODDistanceReport(
        source_system=source_system,
        target_system=target_system,
        n=n,
        jaccard_distance=_jaccard(src_counts, tgt_counts),
        cosine_distance=_cosine(src_counts, tgt_counts),
        n_source_lines=n_src_lines,
        n_target_lines=n_tgt_lines,
        n_source_ngrams=len(src_counts),
        n_target_ngrams=len(tgt_counts),
        n_shared_ngrams=len(shared),
    )


def pairwise_distance_matrix(
    *,
    datasets: Mapping[str, Sequence[LogSequence]],
    n: int = 2,
) -> dict[tuple[str, str], OODDistanceReport]:
    """Compute every pairwise distance between registered systems.

    Returns a dict keyed by ``(source, target)``. The matrix is
    symmetric but both directions are returned so callers can index
    either way without checking.
    """
    out: dict[tuple[str, str], OODDistanceReport] = {}
    names = sorted(datasets)
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            out[(src, tgt)] = ood_distance(
                source_sequences=datasets[src],
                target_sequences=datasets[tgt],
                source_system=src,
                target_system=tgt,
                n=n,
            )
    return out


__all__ = [
    "OODDistanceReport",
    "ood_distance",
    "pairwise_distance_matrix",
]
