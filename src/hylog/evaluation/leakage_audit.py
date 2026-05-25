"""Information-leakage audit for the cross-system protocol.

This module implements the methodological safeguard required by the
Phase 4 checklist: when we train on systems A, B, C and test on system D
with zero D-labels, we must mechanically verify that no preprocessed line
from D's test split ever appears in training. The audit operates over
SHA-256 fingerprints of preprocessed lines so identical raw lines that
differ only in volatile arguments (which the preprocessor masks) are
treated as the *same* line — that is the correct unit of leakage in a
masked-regex pipeline.

The audit returns a structured ``LeakageReport`` that:

- Is JSON-serialisable so every LOSO fold archives its audit alongside
  the metrics.
- Captures both the cardinalities (|train|, |test|, |intersection|) and
  a deterministic *sample* of leaked lines (capped at 16) so a reviewer
  can inspect the actual contaminating content rather than just a
  count.
- Distinguishes between two leakage modes:
    * exact line leakage (same masked string verbatim) — almost always
      catastrophic for the integrity of the cross-system claim.
    * group leakage (same group_id appears in both splits) — already
      enforced in the Phase 1 splitter but re-asserted here as a
      defence in depth.

The audit is the *load-bearing* test for novelty claim N2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hylog.data.dataset import LogDataset
from hylog.data.schema import LogSequence

LEAKAGE_SAMPLE_CAP = 16


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Structured outcome of a leakage audit run.

    Attributes:
        train_lines: Number of preprocessed lines in the training set.
        test_lines: Number of preprocessed lines in the held-out test
            set.
        train_unique_hashes: Cardinality of the train hash set.
        test_unique_hashes: Cardinality of the test hash set.
        line_intersection: Cardinality of the train ∩ test hash sets.
            **Must be zero** for the LOSO protocol to be valid.
        group_intersection: Number of group_ids appearing in both
            splits. Must be zero (defence in depth).
        leaked_line_samples: Up to ``LEAKAGE_SAMPLE_CAP`` of the actual
            leaked preprocessed lines, sorted for determinism. Empty
            when no leakage is detected.
        leaked_group_samples: Up to ``LEAKAGE_SAMPLE_CAP`` of the
            leaked group_ids, sorted.
        verdict: ``"clean"`` if both intersections are zero,
            ``"leakage"`` otherwise.
    """

    train_lines: int
    test_lines: int
    train_unique_hashes: int
    test_unique_hashes: int
    line_intersection: int
    group_intersection: int
    leaked_line_samples: tuple[str, ...] = field(default_factory=tuple)
    leaked_group_samples: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> str:
        return (
            "clean" if self.line_intersection == 0 and self.group_intersection == 0 else "leakage"
        )

    @property
    def is_clean(self) -> bool:
        return self.verdict == "clean"

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)


def _hash_line(text: str) -> bytes:
    """SHA-256 digest (16-byte prefix) of a preprocessed log line.

    16 bytes = 128 bits is overkill for collision-resistance in a
    single-system audit but keeps the in-memory cost predictable for
    very large datasets.
    """
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


def _iter_lines(sequences: Iterable[LogSequence]) -> Iterable[tuple[bytes, str]]:
    for seq in sequences:
        for line in seq.lines:
            yield _hash_line(line), line


def _collect_hashes_and_groups(
    sequences: Iterable[LogSequence],
) -> tuple[dict[bytes, str], set[str], int]:
    """Build the hash-to-line lookup and the group id set in one pass.

    The hash->line mapping retains the *first observed* line for each
    hash so the leaked-sample emission is deterministic and stable
    across re-runs.
    """
    lookup: dict[bytes, str] = {}
    groups: set[str] = set()
    line_count = 0
    for seq in sequences:
        groups.add(seq.group_id)
        for line in seq.lines:
            line_count += 1
            digest = _hash_line(line)
            if digest not in lookup:
                lookup[digest] = line
    return lookup, groups, line_count


def audit_leakage(
    *,
    train: Iterable[LogSequence] | LogDataset,
    test: Iterable[LogSequence] | LogDataset,
) -> LeakageReport:
    """Run the full leakage audit between two splits.

    The function consumes the iterables fully (it must materialise the
    hash sets), so callers that pass lazy iterables should expect them
    to be exhausted.
    """
    train_seqs = list(train)
    test_seqs = list(test)

    train_lookup, train_groups, train_lines = _collect_hashes_and_groups(train_seqs)
    test_lookup, test_groups, test_lines = _collect_hashes_and_groups(test_seqs)

    line_overlap = sorted(set(train_lookup) & set(test_lookup))
    group_overlap = sorted(train_groups & test_groups)

    sample_lines = tuple(sorted({train_lookup[h] for h in line_overlap[:LEAKAGE_SAMPLE_CAP]}))[
        :LEAKAGE_SAMPLE_CAP
    ]
    sample_groups = tuple(group_overlap[:LEAKAGE_SAMPLE_CAP])

    return LeakageReport(
        train_lines=train_lines,
        test_lines=test_lines,
        train_unique_hashes=len(train_lookup),
        test_unique_hashes=len(test_lookup),
        line_intersection=len(line_overlap),
        group_intersection=len(group_overlap),
        leaked_line_samples=sample_lines,
        leaked_group_samples=sample_groups,
    )


def write_report(report: LeakageReport, path: Path | str) -> Path:
    """Persist the audit to disk in a deterministic JSON form."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_json() + "\n", encoding="utf-8", newline="\n")
    return p


class LeakageError(RuntimeError):
    """Raised when ``assert_clean`` detects leakage."""


def assert_clean(report: LeakageReport) -> None:
    """Raise ``LeakageError`` if the audit found any leakage.

    This is the *strict* gatekeeper that wraps every LOSO fold. The
    error message contains a deterministic sample of the leaked content
    so debugging does not require re-running the audit.
    """
    if report.is_clean:
        return
    samples = "\n  ".join(report.leaked_line_samples[:4]) or "(no line samples)"
    raise LeakageError(
        f"cross-system leakage detected: "
        f"line_intersection={report.line_intersection}, "
        f"group_intersection={report.group_intersection}. "
        f"Sample leaked lines (up to 4):\n  {samples}"
    )


__all__ = [
    "LEAKAGE_SAMPLE_CAP",
    "LeakageError",
    "LeakageReport",
    "assert_clean",
    "audit_leakage",
    "write_report",
]
