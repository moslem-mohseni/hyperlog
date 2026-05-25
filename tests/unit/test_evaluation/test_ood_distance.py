"""Tests for the n-gram OOD distance diagnostic."""

from __future__ import annotations

import pytest

from hylog.data.schema import LogSequence
from hylog.evaluation.ood_distance import (
    ood_distance,
    pairwise_distance_matrix,
)


def _seq(*lines: str, source: str = "x") -> LogSequence:
    return LogSequence(lines=lines, label=0, group_id="g0", source=source)


def test_identical_sets_have_zero_distance() -> None:
    a = [_seq("alpha beta gamma", "delta epsilon"), _seq("alpha beta gamma")]
    report = ood_distance(
        source_sequences=a,
        target_sequences=a,
        source_system="x",
        target_system="y",
        n=2,
    )
    assert report.jaccard_distance == pytest.approx(0.0)
    assert report.cosine_distance == pytest.approx(0.0)


def test_disjoint_sets_have_distance_one() -> None:
    a = [_seq("alpha beta gamma")]
    b = [_seq("uno dos tres")]
    report = ood_distance(
        source_sequences=a,
        target_sequences=b,
        source_system="x",
        target_system="y",
        n=2,
    )
    assert report.jaccard_distance == pytest.approx(1.0)
    assert report.cosine_distance == pytest.approx(1.0)


def test_partial_overlap_is_between_0_and_1() -> None:
    a = [_seq("alpha beta gamma delta")]
    b = [_seq("alpha beta gamma omega")]
    report = ood_distance(
        source_sequences=a,
        target_sequences=b,
        source_system="x",
        target_system="y",
        n=2,
    )
    assert 0.0 < report.jaccard_distance < 1.0
    assert 0.0 < report.cosine_distance < 1.0


def test_pairwise_matrix_excludes_self_pairs() -> None:
    datasets = {
        "a": [_seq("alpha beta")],
        "b": [_seq("alpha beta")],
        "c": [_seq("gamma delta")],
    }
    matrix = pairwise_distance_matrix(datasets=datasets, n=2)
    assert ("a", "a") not in matrix
    assert ("a", "b") in matrix
    assert ("b", "a") in matrix
    # Symmetric.
    assert matrix[("a", "b")].jaccard_distance == matrix[("b", "a")].jaccard_distance


def test_report_to_dict_round_trip() -> None:
    a = [_seq("alpha beta")]
    b = [_seq("alpha gamma")]
    r = ood_distance(
        source_sequences=a,
        target_sequences=b,
        source_system="a",
        target_system="b",
        n=2,
    )
    d = r.to_dict()
    assert d["source_system"] == "a"
    assert d["target_system"] == "b"
    assert "jaccard_distance" in d
    assert "cosine_distance" in d


def test_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        ood_distance(
            source_sequences=[_seq("alpha")],
            target_sequences=[_seq("alpha")],
            source_system="a",
            target_system="b",
            n=0,
        )


def test_short_lines_skipped() -> None:
    """Lines shorter than n tokens contribute no n-grams."""
    a = [_seq("hi")]  # single token, no 2-grams
    b = [_seq("alpha beta")]  # one 2-gram
    report = ood_distance(
        source_sequences=a,
        target_sequences=b,
        source_system="a",
        target_system="b",
        n=2,
    )
    # a contributes 0 n-grams; jaccard is over union of 0 and 1 -> 1.0.
    assert report.n_source_ngrams == 0
    assert report.n_target_ngrams == 1
    assert report.jaccard_distance == pytest.approx(1.0)
